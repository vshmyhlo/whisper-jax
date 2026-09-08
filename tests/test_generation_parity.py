"""Offline numerical comparisons with independently generated upstream outputs."""

import copy
import json
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import unflatten_dict
from transformers import WhisperConfig

from whisper_jax.modeling_flax_whisper import FlaxWhisperForConditionalGeneration, FlaxWhisperTimeStampLogitsProcessor


class GenerationParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(__file__).parent / "fixtures"
        metadata = json.loads((directory / "whisper_reference.json").read_text())
        cls.config = WhisperConfig(**metadata["config"])
        with np.load(directory / "whisper_reference.npz", allow_pickle=False) as fixture:
            cls.reference = dict(fixture)
        cls.params = unflatten_dict(
            {
                tuple(key.removeprefix("params/").split("/")): jnp.array(value)
                for key, value in cls.reference.items()
                if key.startswith("params/")
            }
        )
        cls.model = FlaxWhisperForConditionalGeneration(cls.config, _do_init=False)
        cls.features = jnp.array(cls.reference["features"])
        cls.ids = jnp.array(cls.reference["ids"])

    def test_encoder_and_logits_match_openai_and_original_whisper_jax(self):
        def forward(params, features, ids):
            encoder = self.model.encode(features, params=params)
            return encoder[0], self.model.decode(ids, encoder, params=params).logits

        encoder, logits = (
            jax.jit(forward)
            .lower(self.params, self.features, self.ids)
            .compile()(self.params, self.features, self.ids)
        )
        for prefix in ("", "legacy_"):
            with self.subTest(reference=prefix or "openai"):
                np.testing.assert_allclose(encoder, self.reference[prefix + "encoder"], rtol=1e-5, atol=2e-6)
                np.testing.assert_allclose(logits, self.reference[prefix + "logits"], rtol=1e-5, atol=2e-6)

    def test_cached_chunks_match_openai_logits(self):
        encoder = jax.jit(lambda params, features: self.model.encode(features, params=params))(
            self.params, self.features
        )
        decode = jax.jit(lambda ids, kwargs: self.model.decode(ids, params=self.params, **kwargs))
        # Exercise prefill, single-token appends, and chunks appended at nonzero offsets.
        for chunks in ((1, 1, 1, 1, 1, 1), (3, 1, 1, 1), (2, 2, 2), (6,)):
            with self.subTest(chunks=chunks):
                kwargs = self.model.prepare_inputs_for_generation(self.ids[:, : chunks[0]], 8, encoder_outputs=encoder)
                outputs = []
                offset = 0
                for chunk in chunks:
                    kwargs["decoder_position_ids"] = jnp.broadcast_to(jnp.arange(offset, offset + chunk), (2, chunk))
                    result = decode(self.ids[:, offset : offset + chunk], kwargs)
                    outputs.append(result.logits)
                    kwargs["past_key_values"] = result.past_key_values
                    offset += chunk
                np.testing.assert_allclose(
                    jnp.concatenate(outputs, axis=1), self.reference["logits"], rtol=1e-5, atol=2e-6
                )

    def test_greedy_tokens_match_both_original_implementations(self):
        config = copy.deepcopy(self.model.generation_config)
        config.suppress_tokens = []
        config.begin_suppress_tokens = []
        for prompt_length in (1, 3):
            with self.subTest(prompt_length=prompt_length):
                generate = jax.jit(
                    lambda params, features, prompt: self.model.generate(
                        features,
                        generation_config=config,
                        decoder_input_ids=prompt,
                        params=params,
                        max_length=8,
                        eos_token_id=32,
                    ).sequences
                )
                prompt = self.ids[:, :prompt_length]
                actual = generate.lower(self.params, self.features, prompt).compile()(
                    self.params, self.features, prompt
                )
                np.testing.assert_array_equal(actual, self.reference[f"sequences_{prompt_length}"])

    def test_timestamp_masks_match_openai_reference(self):
        config = copy.deepcopy(self.model.generation_config)
        config.eos_token_id = 9
        config.no_timestamps_token_id = 10
        config.is_multilingual = False
        config.max_initial_timestamp_index = 2
        processor = FlaxWhisperTimeStampLogitsProcessor(config, self.config, 1)
        processor.begin_index = 3
        apply = jax.jit(processor)
        for length in range(3, 8):
            for dtype in (jnp.float32, jnp.bfloat16):
                with self.subTest(length=length, dtype=dtype):
                    scores = jnp.array(self.reference[f"timestamp_scores_{length}"], dtype=dtype)
                    actual = apply(jnp.array(self.reference[f"timestamp_ids_{length}"]), scores, jnp.array(length))
                    expected_mask = self.reference[f"timestamp_mask_{length}_{jnp.dtype(dtype).name}"]
                    np.testing.assert_array_equal(jnp.isneginf(actual), expected_mask)
                    # Unmasked logits must retain their exact original values and dtype.
                    self.assertEqual(actual.dtype, scores.dtype)
                    np.testing.assert_array_equal(
                        np.asarray(actual)[~expected_mask], np.asarray(scores)[~expected_mask]
                    )


if __name__ == "__main__":
    unittest.main()
