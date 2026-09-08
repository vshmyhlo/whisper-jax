# Whisper generation reference data

`whisper_reference.npz` contains synthetic weights, input features, encoder outputs,
decoder logits, greedy tokens, and timestamp masks. Normal tests load this fixture
offline; they require neither PyTorch nor downloads.

The references are OpenAI Whisper **v20250625** (`model.py` and `decoding.py`) and
original whisper-jax commit **f983178a80ad37cf2f655777c26a74438b5d8690**, before this
repository's dependency and attention-backend updates. Source SHA-256 hashes and
the tiny two-layer model configuration are in `whisper_reference.json`.

The generator executes the upstream model and timestamp-rule classes without
rewriting their computations. It omits unused audio/tokenizer entry-point imports.
It maps identical random weights between frameworks, checks the original PyTorch
KV-cache hooks, and verifies that original whisper-jax logits and greedy sequences
agree with the PyTorch outputs before saving them. The current model provides only
parameter shapes; it does not produce expected outputs. The legacy model runs with
this repository's installed JAX/Transformers dependencies and layer utilities.

To regenerate, install PyTorch in a development environment, then run from the
repository root (network access and the original Git commit are required):

```sh
PYTHONPATH=. python tests/fixtures/generate_whisper_reference.py
python -m unittest discover -s tests -v
```

Tests compare FP32 encoder/logit outputs within `atol=2e-6, rtol=1e-5`, token IDs
exactly, and timestamp masks exactly for FP32 and BF16. Separate hand-constructed
tests cover BF16 probability boundaries. The decoder tests exercise both prompt
prefill and chunk appends; the original JAX chunk-offset bug is intentionally fixed.
Timestamp prompt gating, monotonicity, and initial-token suppression also intentionally
follow OpenAI's rules where the old Flax implementation was incorrect.

These fixtures test network arithmetic and decoding constraints on a tiny model.
They do not establish real-checkpoint transcription quality, accelerator kernel
parity, or equivalence of the complete audio-processing and search APIs.

## Real checkpoints and audio

`generate_real_whisper_reference.py` adds a separate integration suite using original
OpenAI `tiny.en` and `tiny` checkpoints, the upstream JFK test recording, and two
LibriSpeech recordings from a pinned `Narsil/asr_dummy` revision. Audio downloads are
SHA-256 checked, and the upstream checkpoint loader verifies its checkpoint hashes.
The generator saves the original waveforms, native log-Mel features, FP32 mapped
parameters, encoder features, cached logits for every reference-prefix decision
(including EOS), prompt/generated tokens, and tokenizer byte pieces. Large files
are external; they are not required for offline unit tests.

The original OpenAI `DecodingTask` runs unchanged, with default FP16 computation
and FP32 parameters. A recording wrapper observes logits without modifying them.
All JAX precision comparisons use these same Torch outputs, including the BF16
comparison. JAX never produces the expected outputs. The JAX runner computes its
own features from the waveforms and explicitly compiles encoder, cached decoder
scan and complete generation with JIT, in a process without Torch imports.

See the main README's [precision results and commands](../../README.md#generation-parity-and-precision)
and the [full numerical report](../../benchmarks/reports/real_audio_parity.json).
`tests/test_real_audio_parity.py` is enabled by `WHISPER_REAL_PARITY_DIR`; it checks
exact FP32/FP16 token parity, numerical tolerances, and BF16 drift ceilings per case.
BF16 currently matches 9/12 sequences exactly, so its bounded-error test must not
be interpreted as exact parity. Two-device runs additionally inspect data-parallel
forward HLO, check separate replica results, and reject unexpected collectives,
host callbacks, or dense embedding lookups.
