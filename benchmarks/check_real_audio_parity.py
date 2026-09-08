"""JAX-only, explicitly compiled real-audio parity and data-parallel HLO audit.

Prepare references with tests/fixtures/generate_real_whisper_reference.py first.
This program deliberately imports neither torch nor the OpenAI Whisper package.
"""

import argparse
import ast
import json
import platform
import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import WhisperConfig, WhisperFeatureExtractor

from whisper_jax.modeling_flax_whisper import FlaxWhisperForConditionalGeneration


def numerical_error(actual, expected):
    actual, expected = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f"Invalid numerical comparison: {actual.shape}, {expected.shape}")
    difference = actual - expected
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(difference), dtype=np.float64))),
        "relative_l2": float(np.linalg.norm(difference) / max(float(np.linalg.norm(expected)), 1e-12)),
    }


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        row = [i]
        for j, b in enumerate(right, 1):
            row.append(min(previous[j] + 1, row[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = row
    return previous[-1]


def load_model(directory, name, info, dtype):
    config = WhisperConfig(**info["config"])
    model = FlaxWhisperForConditionalGeneration(config, dtype=dtype, _do_init=False)
    with np.load(directory / f"{name}_params.npz", allow_pickle=False) as data:
        params = {}
        for key, value in data.items():
            assert value.dtype == np.float32, "Parameters must remain FP32"
            params[tuple(key.split("/"))] = jnp.array(value)
        params = unflatten_dict(params)
    generation = model.generation_config
    generation.is_multilingual = info["is_multilingual"]
    generation.no_timestamps_token_id = info["no_timestamps_token_id"]
    generation.suppress_tokens = info["suppress_tokens"]
    generation.begin_suppress_tokens = info["begin_suppress_tokens"]
    generation.max_initial_timestamp_index = 50
    if info["is_multilingual"]:
        generation.language = "en"
        generation.lang_to_id = {"<|en|>": info["language_id"]}
        generation.task = "transcribe"
        generation.task_to_id = {"transcribe": info["task_id"]}
    return model, params


def cached_reference_logits(model, params, encoder, prompt, reference_tokens, max_length=96):
    """Teacher forcing through the actual KV cache; every decoder step is in XLA."""
    cache = model.init_cache(prompt.shape[0], max_length, (encoder,))
    for path, value in flatten_dict(cache).items():
        if path[-1] in ("cached_key", "cached_value"):
            assert value.dtype == model.dtype, (path, value.dtype, model.dtype)
    mask = jnp.ones((prompt.shape[0], max_length), dtype=jnp.int32)
    first = model.decode(
        prompt,
        (encoder,),
        params=params,
        past_key_values=cache,
        decoder_position_ids=jnp.arange(prompt.shape[1])[None],
        decoder_attention_mask=mask,
    )

    def step(past, token_and_position):
        token, position = token_and_position
        output = model.decode(
            token.reshape(1, 1),
            (encoder,),
            params=params,
            past_key_values=past,
            decoder_position_ids=position.reshape(1, 1),
            decoder_attention_mask=mask,
        )
        return output.past_key_values, output.logits[0, -1]

    _, logits = jax.lax.scan(
        step,
        first.past_key_values,
        (reference_tokens, jnp.arange(reference_tokens.shape[0]) + prompt.shape[1]),
    )
    return jnp.concatenate((first.logits[0, -1][None], logits), axis=0)


def hlo_summary(executable, path):
    text = executable.as_text()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    operations = (
        "all-reduce",
        "all-gather",
        "reduce-scatter",
        "all-to-all",
        "collective-permute",
        "custom-call",
        "while",
        "dot",
        "gather",
        "convolution",
        "copy",
        "convert",
    )
    memory = executable.memory_analysis()
    # Use the actual computation location, not op_name: XLA retains the original
    # while/body metadata even on loop-invariant projections hoisted into ENTRY.
    cross_kv = {"entry": 0, "other_computations": 0}
    embedding_dots = 0
    in_entry = False
    for line in text.splitlines():
        if line.startswith("ENTRY "):
            in_entry = True
        elif line and not line.startswith(" "):
            in_entry = False
        if " dot(" in line:
            if re.search(r"encoder_attn/[kv]_proj/", line):
                cross_kv["entry" if in_entry else "other_computations"] += 1
            if re.search(r"embed_(tokens|positions)/", line):
                embedding_dots += 1
    summary = {
        "operations": {operation: len(re.findall(r"\b" + operation + r"\(", text)) for operation in operations},
        "custom_call_targets": sorted(set(re.findall(r'custom_call_target="([^"]+)"', text))),
        "memory_bytes": {
            field: int(getattr(memory, field))
            for field in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
            )
        },
        "has_host_callback": any(
            term in text.lower() for term in ("xla_python", "python_cpu_callback", "host_callback")
        ),
        "hlo_file": path.name,
        "cross_attention_kv_projection_dots": cross_kv,
        "embedding_lookup_dots": embedding_dots,
    }
    return summary


def audit_data_parallel(model, params, features, prompt, directory, name):
    devices = jax.local_device_count()
    if devices < 2:
        raise ValueError("Use at least two devices (on CPU, set XLA_FLAGS=--xla_force_host_platform_device_count=2)")
    # Match the pipeline's pmap data parallelism: complete replicated weights and
    # one local audio example per device, with no model-parallel sharding.
    # Different inputs catch accidental broadcast/cross-replica mixing.
    batch_features = jnp.stack([jnp.roll(features, i * 10, axis=-1) for i in range(devices)])
    batch_prompt = jnp.broadcast_to(prompt, (devices,) + prompt.shape)

    def forward(p, x, ids):
        return model(x, ids, params=p).logits

    compiled = jax.pmap(forward, in_axes=(None, 0, 0)).lower(params, batch_features, batch_prompt).compile()
    actual = compiled(params, batch_features, batch_prompt)
    single = jax.jit(forward).lower(params, features, prompt).compile()
    expected = jnp.stack([single(params, batch_features[i], prompt) for i in range(devices)])
    np.testing.assert_allclose(actual, expected, atol=2e-2, rtol=1e-3)
    result = hlo_summary(compiled, directory / f"{name}_forward_dp.hlo")
    result["devices"] = devices
    result["distinct_replica_inputs"] = True
    result["replica_output_max_difference"] = float(jnp.max(jnp.abs(actual - expected)))
    assert not result["has_host_callback"]
    for op in ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute"):
        assert result["operations"][op] == 0, f"Unexpected forward collective: {op}"
    return result


def check_native_sources():
    root = Path(__file__).resolve().parents[1] / "whisper_jax"
    imports = []
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names if alias.name.split(".")[0] == "torch")
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "torch":
                imports.append(node.module)
    assert not imports, imports
    assert "torch" not in sys.modules, "Run JAX validation in an environment without importing the Torch reference"
    return {"torch_imports_in_runtime_package": imports, "torch_loaded_in_jax_process": False}


def evaluate(directory, dtypes=("float32", "float16", "bfloat16"), hlo_directory=None):
    if jax.config.jax_disable_jit:
        raise ValueError("Parity validation requires JIT enabled")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["torch_activation_dtype"] == "float16"
    assert manifest["torch_params_dtype"] == "float32"
    report = {
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "jit": True,
        "reference": {key: manifest[key] for key in manifest if key != "models"},
        "native_runtime": check_native_sources(),
        "cases": [],
        "hlo": {},
    }
    for name, info in manifest["models"].items():
        pieces = [bytes.fromhex(piece) for piece in json.loads((directory / f"{name}_token_bytes.json").read_text())]
        extractor = WhisperFeatureExtractor(feature_size=info["config"]["num_mel_bins"])
        for dtype_name in dtypes:
            dtype = getattr(jnp, dtype_name)
            print(f"Loading {name}: {dtype_name} computation / float32 parameters", flush=True)
            model, params = load_model(directory, name, info, dtype)
            encode = jax.jit(lambda p, x: model.encode(x, params=p).last_hidden_state)
            teacher_force = jax.jit(lambda p, enc, ids, tokens: cached_reference_logits(model, p, enc, ids, tokens))
            generators = {}
            for case in info["cases"]:
                with np.load(directory / (case["name"] + ".npz"), allow_pickle=False) as data:
                    # In the JAX-only process Transformers uses its NumPy STFT path.
                    features = jnp.array(
                        extractor(data["waveform"], sampling_rate=16000, return_tensors="np").input_features
                    )
                    feature_error = numerical_error(features, data["features"])
                    assert feature_error["max_abs"] < 1e-4, feature_error
                    encoder = encode.lower(params, features).compile()(params, features)
                    assert encoder.dtype == dtype
                    prompt, tokens = jnp.array(data["prompt"]), jnp.array(data["tokens"])
                    compiled_teacher = teacher_force.lower(params, encoder, prompt, tokens).compile()
                    logits = compiled_teacher(params, encoder, prompt, tokens)
                    assert logits.dtype == dtype
                    encoder_error = numerical_error(encoder, data["encoder"])
                    logits_error = numerical_error(logits, data["logits"])
                    top1_agreement = float(np.mean(np.asarray(logits).argmax(-1) == data["logits"].argmax(-1)))
                timestamps = case["timestamps"]
                if timestamps not in generators:
                    generate = jax.jit(
                        lambda p, x: model.generate(x, params=p, return_timestamps=timestamps, max_length=96).sequences
                    )
                    generators[timestamps] = generate.lower(params, features).compile()
                    if hlo_directory is not None and dtype_name == "bfloat16":
                        report["hlo"][f"{name}_generate_{timestamps}"] = hlo_summary(
                            generators[timestamps], hlo_directory / f"{name}_generate_{timestamps}.hlo"
                        )
                compiled_generate = generators[timestamps]
                compiled_generate(params, features).block_until_ready()  # warm up; exclude compilation from timing
                start = time.perf_counter()
                sequence = np.asarray(compiled_generate(params, features))[0].tolist()
                seconds = time.perf_counter() - start
                eos = model.config.eos_token_id
                sequence = sequence[prompt.shape[1] :]
                assert eos in sequence, "JAX must reach EOS before max_length"
                sequence = sequence[: sequence.index(eos)]
                text = (
                    b"".join(pieces[token] for token in sequence if token < info["timestamp_begin"])
                    .decode("utf-8", errors="replace")
                    .strip()
                )
                entry = {
                    "checkpoint": name,
                    "checkpoint_sha256": info["checkpoint_sha256"],
                    "case": case["name"],
                    "jax_activation_dtype": dtype_name,
                    "jax_params_dtype": "float32",
                    "jax_kv_cache_dtype": dtype_name,
                    "reference_text": case["text"],
                    "jax_text": text,
                    "reference_tokens": case["tokens"],
                    "jax_tokens": sequence,
                    "tokens_exact": sequence == case["tokens"],
                    "text_exact": text == case["text"],
                    "token_edit_distance": edit_distance(sequence, case["tokens"]),
                    "word_edit_distance": edit_distance(text.split(), case["text"].split()),
                    "features_error": feature_error,
                    "encoder_error": encoder_error,
                    "logits_error": logits_error,
                    "raw_logit_argmax_agreement": top1_agreement,
                    "jax_warm_seconds": seconds,
                }
                report["cases"].append(entry)
                print(
                    case["name"],
                    dtype_name,
                    "exact",
                    entry["tokens_exact"],
                    "logit MAE",
                    logits_error["mean_abs"],
                    flush=True,
                )
            if hlo_directory is not None and dtype_name == "bfloat16":
                report["hlo"][f"{name}_forward_dp"] = audit_data_parallel(
                    model, params, features, prompt, hlo_directory, name
                )
            del params, generators
            jax.clear_caches()
    report["native_runtime"] = check_native_sources()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--dtypes", nargs="+", choices=["float32", "float16", "bfloat16"], default=["float32", "float16", "bfloat16"]
    )
    parser.add_argument("--hlo-dir", type=Path)
    parser.add_argument("--strict-tokens", action="store_true", help="Exit nonzero on any token mismatch")
    args = parser.parse_args()
    result = evaluate(args.reference_dir, args.dtypes, args.hlo_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    mismatches = [case for case in result["cases"] if not case["tokens_exact"]]
    print(f"{len(result['cases']) - len(mismatches)}/{len(result['cases'])} token sequences match exactly")
    if args.strict_tokens and mismatches:
        raise SystemExit(1)
