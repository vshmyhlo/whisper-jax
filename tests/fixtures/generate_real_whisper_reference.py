"""Prepare real OpenAI checkpoints/audio and native FP16 decoding references.

Run in an environment with openai-whisper==20250625, torch and ffmpeg installed.
Large artifacts go in --output, never in the source tree unless explicitly requested.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import whisper
from flax.traverse_util import flatten_dict
from generate_whisper_reference import torch_parameter_name
from transformers import WhisperConfig

from whisper_jax.modeling_flax_whisper import FlaxWhisperForConditionalGeneration


AUDIO = {
    "jfk": (
        "https://raw.githubusercontent.com/openai/whisper/v20250625/tests/jfk.flac",
        "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715",
    ),
    "librispeech_1": (
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/8d141c84e3f84c54cd7bbaa851d24edd0f559734/1.flac",
        "30885601173f96b0d8ddd020dc959b055c6c1582b85a33e3fcab8c4b08ed94c2",
    ),
    "librispeech_2": (
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/8d141c84e3f84c54cd7bbaa851d24edd0f559734/2.flac",
        "3fc09ec6d4cc496c530b2019b17bd8fc8ef8a43d6697090971dd1d52cc3d4d89",
    ),
}


def download(url, target, checksum):
    if not target.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            target.write_bytes(response.read())
    if hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
        raise ValueError(f"Checksum mismatch: {target}")


@torch.no_grad()
def prepare(output, checkpoints, device):
    if whisper.__version__ != "20250625":
        raise ValueError("Use openai-whisper==20250625 to keep the reference implementation fixed")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "whisper_version": whisper.__version__,
        "torch_version": torch.__version__,
        "torch_device": device,
        "torch_params_dtype": "float32",
        "torch_activation_dtype": "float16",
        "torch_precision_policy": "Unchanged DecodingOptions.fp16=True default; no model.half() or BF16 conversion",
        "audio": {name: {"url": url, "sha256": checksum} for name, (url, checksum) in AUDIO.items()},
        "models": {},
    }
    waves = {}
    for name, (url, checksum) in AUDIO.items():
        path = output / f"{name}.flac"
        download(url, path, checksum)
        waves[name] = whisper.load_audio(str(path))

    for checkpoint in checkpoints:
        model = whisper.load_model(checkpoint, device=device, download_root=str(output)).eval()
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
        tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language="en", task="transcribe")
        dims = model.dims
        config = WhisperConfig(
            vocab_size=dims.n_vocab,
            num_mel_bins=dims.n_mels,
            d_model=dims.n_text_state,
            encoder_layers=dims.n_audio_layer,
            decoder_layers=dims.n_text_layer,
            encoder_attention_heads=dims.n_audio_head,
            decoder_attention_heads=dims.n_text_head,
            encoder_ffn_dim=4 * dims.n_audio_state,
            decoder_ffn_dim=4 * dims.n_text_state,
            max_source_positions=dims.n_audio_ctx,
            max_target_positions=dims.n_text_ctx,
            pad_token_id=tokenizer.eot,
            eos_token_id=tokenizer.eot,
            bos_token_id=tokenizer.sot,
            decoder_start_token_id=tokenizer.sot,
        )
        shapes = flatten_dict(FlaxWhisperForConditionalGeneration(config, _do_init=False).params_shape_tree)
        params = {}
        state = model.state_dict()
        for key, shape in shapes.items():
            value = state[torch_parameter_name(key)].float().cpu().numpy()
            if key[-1] == "kernel":
                value = value.T if value.ndim == 2 else value.transpose(2, 1, 0)
            assert value.shape == shape.shape, (key, value.shape, shape.shape)
            params["/".join(key)] = value
        np.savez_compressed(output / f"{checkpoint}_params.npz", **params)
        model_info = {
            "config": config.to_dict(),
            "checkpoint_url": whisper._MODELS[checkpoint],
            "checkpoint_sha256": hashlib.sha256((output / f"{checkpoint}.pt").read_bytes()).hexdigest(),
            "is_multilingual": model.is_multilingual,
            "no_timestamps_token_id": tokenizer.no_timestamps,
            "timestamp_begin": tokenizer.timestamp_begin,
            "language_id": tokenizer.language_token if model.is_multilingual else None,
            "task_id": tokenizer.transcribe,
            "begin_suppress_tokens": [tokenizer.encode(" ")[0], tokenizer.eot],
            "cases": [],
        }
        # Export byte pieces so the JAX-only evaluator can decode mismatches without
        # importing the PyTorch reference or downloading a second tokenizer.
        pieces = [tokenizer.encoding.decode_single_token_bytes(i).hex() for i in range(dims.n_vocab)]
        (output / f"{checkpoint}_token_bytes.json").write_text(json.dumps(pieces))
        for audio_name, waveform in waves.items():
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(waveform)).to(device)
            for timestamps in (False, True):
                options = whisper.DecodingOptions(language="en", without_timestamps=not timestamps, sample_len=92)
                assert options.fp16  # Keep the reference's original precision in every comparison.
                task = whisper.decoding.DecodingTask(model, options)
                step_logits = []
                original_logits = task.inference.logits

                def record_logits(tokens, features):
                    logits = original_logits(tokens, features)
                    step_logits.append(logits[:, -1].float().cpu().numpy().copy())
                    return logits

                task.inference.logits = record_logits
                start = time.perf_counter()
                result = task.run(mel[None])[0]
                elapsed = time.perf_counter() - start
                assert result.audio_features.dtype == torch.float16
                assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
                assert (
                    len(step_logits) == len(result.tokens) + 1
                ), "Reference must finish with EOS before the length limit"
                case_name = f"{checkpoint}_{audio_name}_{'timestamps' if timestamps else 'text'}"
                np.savez_compressed(
                    output / f"{case_name}.npz",
                    waveform=waveform,
                    features=mel.float().cpu().numpy()[None],
                    encoder=result.audio_features.float().cpu().numpy()[None],
                    prompt=np.array(task.initial_tokens, dtype=np.int32)[None],
                    tokens=np.array(result.tokens, dtype=np.int32),
                    logits=np.concatenate(step_logits, axis=0),
                )
                model_info["suppress_tokens"] = list(task._get_suppress_tokens())
                model_info["cases"].append(
                    {
                        "name": case_name,
                        "timestamps": timestamps,
                        "text": result.text,
                        "tokens": result.tokens,
                        "seconds": elapsed,
                        "audio_seconds": len(waveform) / 16000,
                    }
                )
                print(case_name, result.text, flush=True)
        manifest["models"][checkpoint] = model_info
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"References saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", default=["tiny.en", "tiny"])
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    prepare(arguments.output, arguments.checkpoints, arguments.device)
