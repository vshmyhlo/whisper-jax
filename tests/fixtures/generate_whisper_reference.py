"""Regenerate synthetic reference outputs; see README.md in this directory."""

import ast
import hashlib
import json
import linecache
import subprocess
import sys
import types
import urllib.request
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import WhisperConfig

from whisper_jax.modeling_flax_whisper import FlaxWhisperForConditionalGeneration


OPENAI_ROOT = "https://raw.githubusercontent.com/openai/whisper/v20250625/whisper/"
SOURCE_HASHES = {
    "model.py": "473491a97554ad1d0beaaef6fe91d965f12785b466b93ebb61024307b50ad683",
    "decoding.py": "94084fedc5af74cabf8e1883a85552700057081f00ea669ad49435ecb29039db",
}
LEGACY_COMMIT = "f983178a80ad37cf2f655777c26a74438b5d8690"
LEGACY_HASH = "b6cd88a89ea78f4c6f63e30ac0fac8dbfadffc93a8087110a9c965979f6301a1"
CONFIG = {
    "vocab_size": 32,
    "num_mel_bins": 4,
    "d_model": 8,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "encoder_attention_heads": 2,
    "decoder_attention_heads": 2,
    "encoder_ffn_dim": 32,
    "decoder_ffn_dim": 32,
    "max_source_positions": 4,
    "max_target_positions": 8,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "decoder_start_token_id": 1,
}


def load_references():
    sources = {}
    for filename, checksum in SOURCE_HASHES.items():
        source = urllib.request.urlopen(OPENAI_ROOT + filename, timeout=30).read()
        if hashlib.sha256(source).hexdigest() != checksum:
            raise ValueError(f"Unexpected upstream source: {filename}")
        sources[filename] = ast.parse(source)

    # Run the upstream model classes verbatim, omitting only imports of audio/tokenizer
    # entry points which these tests do not call. No model or attention code is rewritten.
    module = types.ModuleType("reference_openai_whisper")
    module.__dict__.update(decode_function=None, detect_language_function=None, transcribe_function=None)
    sys.modules[module.__name__] = module
    tree = sources["model.py"]
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.level)]
    exec(compile(tree, OPENAI_ROOT + "model.py", "exec"), module.__dict__)

    tree = sources["decoding.py"]
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in ("LogitFilter", "ApplyTimestampRules")
    ]
    namespace = {"np": np, "torch": torch, "F": F, "Tensor": torch.Tensor, "Tokenizer": object, "Optional": Optional}
    exec(compile(tree, OPENAI_ROOT + "decoding.py", "exec"), namespace)

    repository = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ["git", "show", f"{LEGACY_COMMIT}:whisper_jax/modeling_flax_whisper.py"], cwd=repository
    )
    if hashlib.sha256(source).hexdigest() != LEGACY_HASH:
        raise ValueError("Unexpected original whisper-jax source")
    legacy = types.ModuleType("reference_original_whisper_jax")
    legacy.__file__ = "<original_whisper_jax>"
    # Transformers' docstring decorators inspect the source during class construction.
    linecache.cache[legacy.__file__] = (len(source), None, source.decode().splitlines(keepends=True), legacy.__file__)
    sys.modules[legacy.__name__] = legacy
    exec(compile(source, legacy.__file__, "exec"), legacy.__dict__)
    return module, namespace["ApplyTimestampRules"], legacy


def torch_parameter_name(key):
    parts = list(key[1:])
    if parts[1] == "embed_positions":
        return parts[0] + ".positional_embedding"
    renames = {
        "layers": "blocks",
        "self_attn": "attn",
        "encoder_attn": "cross_attn",
        "self_attn_layer_norm": "attn_ln",
        "encoder_attn_layer_norm": "cross_attn_ln",
        "final_layer_norm": "mlp_ln",
        "q_proj": "query",
        "k_proj": "key",
        "v_proj": "value",
        "out_proj": "out",
        "fc1": "mlp.0",
        "fc2": "mlp.2",
        "embed_tokens": "token_embedding",
        "embedding": "weight",
        "kernel": "weight",
        "scale": "weight",
        "layer_norm": "ln_post" if parts[0] == "encoder" else "ln",
    }
    return ".".join(renames.get(part, part) for part in parts)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    reference, timestamp_class, legacy = load_references()
    config = WhisperConfig(**CONFIG)
    rng = np.random.default_rng(42)
    # Only parameter shapes come from the current model; every expected output below
    # comes from upstream PyTorch or the original JAX implementation.
    shapes = flatten_dict(FlaxWhisperForConditionalGeneration(config).params)
    flat = {
        key: rng.normal(1 if key[-1] == "scale" else 0, 0.15, value.shape).astype(np.float32)
        for key, value in shapes.items()
    }
    params = unflatten_dict({key: jnp.array(value) for key, value in flat.items()})
    model = reference.Whisper(reference.ModelDimensions(4, 4, 8, 2, 2, 32, 8, 8, 2, 2)).eval()
    state = {}
    for key, value in flat.items():
        if key[-1] == "kernel":
            value = value.T if value.ndim == 2 else value.transpose(2, 1, 0)
        state[torch_parameter_name(key)] = torch.tensor(value.copy())
    model.load_state_dict(state, strict=True)
    reference.MultiHeadAttention.use_sdpa = False
    features = rng.normal(size=(2, 4, 8)).astype(np.float32)
    ids = np.array([[1, 3, 4, 7, 8, 9], [1, 5, 6, 8, 7, 3]], dtype=np.int32)
    encoder = model.encoder(torch.tensor(features))
    logits = model.decoder(torch.tensor(ids), encoder)
    data = {"params/" + "/".join(key): value for key, value in flat.items()}
    data.update(features=features, ids=ids, encoder=encoder.numpy(), logits=logits.numpy())

    old_model = legacy.FlaxWhisperForConditionalGeneration(config)
    old_encoder = old_model.encode(jnp.array(features), params=params)
    old_logits = old_model.decode(jnp.array(ids), old_encoder, params=params).logits
    data.update(legacy_encoder=np.array(old_encoder[0]), legacy_logits=np.array(old_logits))
    np.testing.assert_allclose(old_logits, logits.numpy(), rtol=1e-5, atol=2e-6)
    for prompt_length in (1, 3):
        cache, hooks = model.install_kv_cache_hooks()
        try:
            cached = [model.decoder(torch.tensor(ids[:, :prompt_length]), encoder, kv_cache=cache)]
            for index in range(prompt_length, ids.shape[1]):
                cached.append(model.decoder(torch.tensor(ids[:, index : index + 1]), encoder, kv_cache=cache))
            np.testing.assert_allclose(torch.cat(cached, dim=1).numpy(), logits.numpy(), rtol=1e-5, atol=2e-6)
        finally:
            for hook in hooks:
                hook.remove()
        sequence = ids[:, :prompt_length].copy()
        while sequence.shape[-1] < CONFIG["max_target_positions"]:
            next_token = model.decoder(torch.tensor(sequence), encoder)[:, -1].argmax(-1).numpy()
            sequence = np.concatenate((sequence, next_token[:, None]), axis=-1)
        data[f"sequences_{prompt_length}"] = sequence
        old_model.generation_config.suppress_tokens = []
        old_model.generation_config.begin_suppress_tokens = []
        old_sequence = old_model.generate(
            jnp.array(features),
            decoder_input_ids=jnp.array(ids[:, :prompt_length]),
            params=params,
            max_length=8,
            eos_token_id=32,
        ).sequences
        np.testing.assert_array_equal(old_sequence, sequence)

    processor = timestamp_class(types.SimpleNamespace(no_timestamps=10, timestamp_begin=11, eot=9), 3, 2)
    for length in range(3, 8):
        tokens = rng.choice([7, 11, 12, 20, 31], size=(64, length)).astype(np.int32)
        tokens[:, :3] = np.array([1, 4, 5])
        scores = rng.normal(0, 2, size=(64, 32)).astype(np.float32)
        data[f"timestamp_ids_{length}"] = np.pad(tokens, ((0, 0), (0, 8 - length)))
        data[f"timestamp_scores_{length}"] = scores
        for dtype in (torch.float32, torch.bfloat16):
            expected = torch.tensor(scores, dtype=dtype)
            processor.apply(expected, torch.tensor(tokens))
            data[f"timestamp_mask_{length}_{str(dtype).split('.')[-1]}"] = np.isneginf(expected.float().numpy())

    destination = Path(__file__).parent
    np.savez_compressed(destination / "whisper_reference.npz", **data)
    (destination / "whisper_reference.json").write_text(
        json.dumps(
            {
                "config": CONFIG,
                "openai_root": OPENAI_ROOT,
                "source_hashes": SOURCE_HASHES,
                "legacy_commit": LEGACY_COMMIT,
                "legacy_hash": LEGACY_HASH,
                "torch_version": torch.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    print("Saved upstream model, cache, token and timestamp references.")


if __name__ == "__main__":
    main()
