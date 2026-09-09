"""Compile and audit encoder K/V placement in a data-parallel decoder while loop.

Run from the repository root (no weights or audio downloads are needed)::

    PYTHONPATH=. python benchmarks/check_cross_attention_cache_hlo.py --platform cpu
    PYTHONPATH=. python benchmarks/check_cross_attention_cache_hlo.py \
        --platform gpu --attention-backend cudnn --dtype float16 \
        --preset large-v3 --batch-size 32 --devices 8

Repeat with --dtype bfloat16 for BF16. Parameters are abstract FP32 inputs,
replicated across devices; encoder outputs are abstract activation-dtype inputs,
sharded on the batch axis. Compilation does not allocate full model weights.
The cross-cache is explicitly initialized inside shard_map before while_loop.

Both pre-optimization and optimized HLO are saved. The first establishes that
caching does not depend on compiler hoisting; the second checks actual compiled
placement, including projections inside called/fused computations. CPU results
are a structural check only, and do not establish CUDA or cuDNN behavior.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from transformers import WhisperConfig

from whisper_jax import FlaxWhisperForConditionalGeneration


_COMPUTATION = re.compile(r"^(?:ENTRY )?%?([\w.-]+)\s*(?:\(.*\).*?)?\s*\{$")
_REFERENCE = re.compile(r"\b(?:calls|to_apply|body|condition|branches|called_computations)=({[^}]*}|%?[\w.-]+)")
_PROJECTION = re.compile(r"encoder_attn(?:\.[^/\"]+)?/(?:[^/\"]+/)*([kv]_proj)/")
_COLLECTIVES = (
    "all-reduce",
    "all-gather",
    "reduce-scatter",
    "all-to-all",
    "collective-permute",
    "collective-broadcast",
    "ragged-all-to-all",
)


def summarize_hlo(hlo):
    """Follow computation edges rather than trusting possibly hoisted op_names."""
    computations = {}
    current = None
    for line in hlo.splitlines():
        match = _COMPUTATION.match(line)
        if match:
            current = match.group(1)
            computations[current] = []
        elif line == "}":
            current = None
        elif current is not None:
            computations[current].append(line)
    if not computations:
        raise ValueError("Could not parse HLO computations; inspect the saved HLO manually.")

    def references(line):
        return {
            name
            for match in _REFERENCE.finditer(line)
            for name in re.findall(r"[\w.-]+", match.group(1))
            if name in computations
        }

    token_loop_bodies = set()
    for lines in computations.values():
        for line in lines:
            if re.search(r"\bwhile\(", line) and "decoder_token_loop" in line:
                body = re.search(r"\bbody=%?([\w.-]+)", line)
                if body:
                    token_loop_bodies.add(body.group(1))
    if not token_loop_bodies:
        raise ValueError("Could not identify the decoder token while_loop; inspect the saved HLO manually.")

    inside_loop = set()
    pending = list(token_loop_bodies)
    while pending:
        name = pending.pop()
        if name not in inside_loop:
            inside_loop.add(name)
            for line in computations[name]:
                pending.extend(references(line) - inside_loop)

    projections = {"inside_token_loop": [], "outside_token_loop": []}
    collectives = {}
    for name, lines in computations.items():
        for line in lines:
            projection = _PROJECTION.search(line)
            # GPU GEMMs may be custom calls; Triton fusions retain a dot in the
            # called computation. Avoid counting reshape/convert/bias operations
            # that also retain the projection's source metadata.
            if projection and re.search(r"\b(?:dot|custom-call)\(", line):
                location = "inside_token_loop" if name in inside_loop else "outside_token_loop"
                layer = re.search(r"layers/(\d+)/", line)
                projections[location].append(
                    {
                        "layer": int(layer.group(1)) if layer else None,
                        "projection": projection.group(1),
                        "computation": name,
                    }
                )
            for operation in _COLLECTIVES:
                # Include asynchronous start/done forms.
                if re.search(r"\b" + operation + r"(?:-start|-done)?\(", line):
                    collectives[operation] = collectives.get(operation, 0) + 1
    return {
        "token_loop_bodies": sorted(token_loop_bodies),
        "cross_attention_kv_projections": projections,
        "projection_counts": {location: len(values) for location, values in projections.items()},
        "collectives": collectives,
    }


def make_config(preset):
    large = preset == "large-v3"
    return WhisperConfig(
        vocab_size=51866 if large else 64,
        num_mel_bins=128 if large else 4,
        d_model=1280 if large else 128,
        encoder_layers=32 if large else 1,
        decoder_layers=32 if large else 2,
        encoder_attention_heads=20 if large else 2,
        decoder_attention_heads=20 if large else 2,
        encoder_ffn_dim=5120 if large else 256,
        decoder_ffn_dim=5120 if large else 256,
        max_source_positions=1500 if large else 128,
        max_target_positions=448 if large else 16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        dropout=0.0,
        attention_dropout=0.0,
    )


def make_decoder_loop(model, use_cross_cache):
    def decode_loop(params, encoder_hidden_states, token_ids, num_steps):
        config = model.config
        batch_size = token_ids.shape[0]
        encoder_outputs = (encoder_hidden_states,)
        past = model.init_cache(batch_size, config.max_target_positions, encoder_outputs)
        # Zero-filled cache arrays begin replicated but become device-varying
        # after the first token. Declare that carry type before entering while.
        past = jax.tree_util.tree_map(
            lambda value: jax.lax.pcast(value, ("data",), to="varying") if value.ndim else value, past
        )
        mask = jnp.ones((batch_size, config.max_target_positions), dtype=jnp.int32)
        cross_cache = model.init_cross_attention_cache(encoder_outputs, params=params) if use_cross_cache else None

        def step(state):
            index, tokens, self_cache, _ = state
            outputs = model.decode(
                tokens,
                encoder_outputs,
                decoder_attention_mask=mask,
                decoder_position_ids=jnp.full(tokens.shape, index, dtype=jnp.int32),
                past_key_values=self_cache,
                cross_attention_cache=cross_cache,
                params=params,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            next_tokens = jnp.argmax(logits, axis=-1).astype(jnp.int32)[:, None]
            return index + 1, next_tokens, outputs.past_key_values, logits

        initial_logits = jax.lax.pcast(
            jnp.zeros((batch_size, config.vocab_size), dtype=model.dtype), ("data",), to="varying"
        )
        initial = (jnp.int32(0), token_ids, past, initial_logits)
        with jax.named_scope("decoder_token_loop"):
            return jax.lax.while_loop(lambda state: state[0] < num_steps, step, initial)[-1]

    return decode_loop


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--attention-backend", choices=("default", "cudnn"), default="default")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--preset", choices=("tiny", "large-v3"), default="tiny")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device.")
    parser.add_argument("--devices", type=int, default=1, help="Number of local devices in the data-parallel mesh.")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/whisper-cross-attention-hlo"))
    args = parser.parse_args()
    if args.batch_size < 1 or args.devices < 1:
        parser.error("--batch-size and --devices must be positive")
    if args.attention_backend == "cudnn" and args.platform != "gpu":
        parser.error("--attention-backend cudnn requires --platform gpu")
    devices = jax.local_devices(backend=args.platform)
    if len(devices) < args.devices:
        parser.error(
            f"Requested {args.devices} devices, but only {len(devices)} {args.platform} devices are available"
        )
    devices = devices[: args.devices]
    mesh = Mesh(np.asarray(devices), ("data",))
    config = make_config(args.preset)
    dtype = getattr(jnp, args.dtype)
    model = FlaxWhisperForConditionalGeneration(
        config, dtype=dtype, params_dtype=jnp.float32, attention_backend=args.attention_backend, _do_init=False
    )
    global_batch = args.batch_size * args.devices
    inputs = (
        model.params_shape_tree,
        jax.ShapeDtypeStruct((global_batch, config.max_source_positions, config.d_model), dtype),
        jax.ShapeDtypeStruct((global_batch, 1), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "jax_version": jax.__version__,
        "platform": args.platform,
        "platform_version": devices[0].client.platform_version,
        "devices": [str(device) for device in devices],
        "attention_backend": args.attention_backend,
        "dtype": args.dtype,
        "preset": args.preset,
        "batch_size_per_device": args.batch_size,
        "scope": "Compiled abstract inputs only; numerical parity is covered by the cache tests.",
        "variants": {},
    }
    failures = []
    expected_projections = {
        (layer, projection) for layer in range(config.decoder_layers) for projection in ("k_proj", "v_proj")
    }
    for use_cache in (False, True):
        variant = "cached" if use_cache else "uncached"
        print(f"Compiling {variant} on {args.platform} ({args.dtype}, {args.devices} devices)...", file=sys.stderr)
        mapped = jax.shard_map(
            make_decoder_loop(model, use_cache),
            mesh=mesh,
            in_specs=(P(), P("data"), P("data"), P()),
            out_specs=P("data"),
        )
        lowered = jax.jit(mapped).lower(*inputs)
        # as_hlo_text() omits the operation metadata needed to identify K/V.
        before = lowered.compiler_ir(dialect="hlo").get_hlo_module().to_string()
        (args.output_dir / f"{variant}.before_optimizations.hlo").write_text(before)
        compiled = lowered.compile()
        optimized = compiled.as_text()
        (args.output_dir / f"{variant}.optimized.hlo").write_text(optimized)
        report["variants"][variant] = {
            "before_optimizations": summarize_hlo(before),
            "optimized": summarize_hlo(optimized),
        }
        for stage, summary in report["variants"][variant].items():
            counts = summary["projection_counts"]
            observed_projections = {
                (projection["layer"], projection["projection"])
                for location in summary["cross_attention_kv_projections"].values()
                for projection in location
            }
            if observed_projections != expected_projections:
                failures.append(
                    f"{variant}/{stage}: incomplete or unrecognized per-layer K/V projection metadata; "
                    "inspect the HLO manually"
                )
            if use_cache and counts["inside_token_loop"]:
                failures.append(f"{variant}/{stage}: encoder K/V projections remain inside the token loop")
            if use_cache and summary["collectives"]:
                failures.append(f"{variant}/{stage}: unexpected collectives: {summary['collectives']}")
    uncached_before = report["variants"]["uncached"]["before_optimizations"]
    uncached_loop_projections = {
        (projection["layer"], projection["projection"])
        for projection in uncached_before["cross_attention_kv_projections"]["inside_token_loop"]
    }
    if uncached_loop_projections != expected_projections:
        failures.append("uncached/before_optimizations: expected per-layer K/V projections in the token loop")
    report["passed"] = not failures
    report["failures"] = failures
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"HLO and report saved in {args.output_dir}", file=sys.stderr)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
