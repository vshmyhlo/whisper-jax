import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from transformers import WhisperConfig

from whisper_jax.modeling_flax_whisper import (
    FlaxWhisperAttention,
    FlaxWhisperForConditionalGeneration,
    FlaxWhisperForConditionalGenerationModule,
)
from whisper_jax.pipeline import FlaxWhisperPipline


class AttentionBackendTest(unittest.TestCase):
    def setUp(self):
        self.config = WhisperConfig(
            vocab_size=32,
            num_mel_bins=4,
            d_model=8,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=16,
            decoder_ffn_dim=16,
            max_source_positions=4,
            max_target_positions=6,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=1,
            dropout=0.0,
            attention_dropout=0.0,
        )
        self.inputs = {
            "input_features": jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 32,
            "decoder_input_ids": jnp.array([[1, 3, 4]], dtype=jnp.int32),
            "decoder_attention_mask": jnp.ones((1, 3), dtype=jnp.int32),
            "decoder_position_ids": jnp.arange(3, dtype=jnp.int32)[None],
            "output_attentions": False,
        }
        self.dtype = jnp.bfloat16
        self.default_module = FlaxWhisperForConditionalGenerationModule(self.config, dtype=self.dtype)
        self.variables = self.default_module.init(jax.random.PRNGKey(0), **self.inputs)

    def test_cudnn_backend_dispatches_all_attention_blocks(self):
        expected = self.default_module.apply(self.variables, **self.inputs).logits
        original_dot_product_attention = jax.nn.dot_product_attention
        calls = []

        def fake_cudnn_attention(*args, **kwargs):
            calls.append((args, kwargs.copy()))
            kwargs["implementation"] = "xla"
            return original_dot_product_attention(*args, **kwargs)

        cudnn_module = FlaxWhisperForConditionalGenerationModule(
            self.config,
            dtype=self.dtype,
            attention_backend="cudnn",
        )
        with patch.object(jax.nn, "dot_product_attention", fake_cudnn_attention):
            actual = cudnn_module.apply(self.variables, **self.inputs).logits

        self.assertEqual(len(calls), 3)
        self.assertTrue(all(kwargs["implementation"] == "cudnn" for _, kwargs in calls))
        self.assertTrue(all(args[0].dtype == self.dtype for args, _ in calls))
        self.assertTrue(all(kwargs["mask"] is None or kwargs["mask"].dtype == jnp.bool_ for _, kwargs in calls))
        self.assertTrue(jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5))

    def test_model_constructor_accepts_cudnn_backend(self):
        original_dot_product_attention = jax.nn.dot_product_attention

        def fake_cudnn_attention(*args, **kwargs):
            kwargs["implementation"] = "xla"
            return original_dot_product_attention(*args, **kwargs)

        with patch.object(jax.nn, "dot_product_attention", fake_cudnn_attention):
            model = FlaxWhisperForConditionalGeneration(
                self.config,
                _do_init=False,
                dtype=self.dtype,
                attention_backend="cudnn",
            )

        self.assertEqual(model.module.attention_backend, "cudnn")

    def test_output_attentions_uses_default_backend(self):
        cudnn_module = FlaxWhisperForConditionalGenerationModule(
            self.config,
            dtype=self.dtype,
            attention_backend="cudnn",
        )
        with patch.object(
            jax.nn,
            "dot_product_attention",
            side_effect=AssertionError("cuDNN must not be used when attention weights are requested"),
        ):
            outputs = cudnn_module.apply(
                self.variables,
                **{**self.inputs, "output_attentions": True},
            )

        self.assertEqual(outputs.encoder_attentions[0].shape, (1, 2, 4, 4))
        self.assertEqual(outputs.decoder_attentions[0].shape, (1, 2, 3, 3))
        self.assertEqual(outputs.cross_attentions[0].shape, (1, 2, 3, 4))

    def test_cudnn_backend_broadcasts_padding_mask(self):
        hidden_states = jnp.arange(24, dtype=self.dtype).reshape(1, 3, 8)
        attention_mask = jnp.array([[1, 1, 0]], dtype=jnp.int32)
        default_attention = FlaxWhisperAttention(
            self.config,
            embed_dim=8,
            num_heads=2,
            dtype=self.dtype,
        )
        variables = default_attention.init(
            jax.random.PRNGKey(0),
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
        )
        expected, _ = default_attention.apply(
            variables,
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
        )

        original_dot_product_attention = jax.nn.dot_product_attention
        calls = []

        def fake_cudnn_attention(*args, **kwargs):
            calls.append(kwargs.copy())
            kwargs["implementation"] = "xla"
            return original_dot_product_attention(*args, **kwargs)

        cudnn_attention = FlaxWhisperAttention(
            self.config,
            embed_dim=8,
            num_heads=2,
            dtype=self.dtype,
            attention_backend="cudnn",
        )
        with patch.object(jax.nn, "dot_product_attention", fake_cudnn_attention):
            actual, _ = cudnn_attention.apply(
                variables,
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=False,
            )

        self.assertEqual(calls[0]["mask"].shape, (1, 1, 3, 3))
        self.assertTrue(jnp.allclose(actual, expected, rtol=1e-2, atol=1e-2))

    def test_attention_dropout_uses_default_backend(self):
        attention = FlaxWhisperAttention(
            self.config,
            embed_dim=8,
            num_heads=2,
            dropout=0.1,
            dtype=self.dtype,
            attention_backend="cudnn",
        )
        hidden_states = jnp.ones((1, 3, 8), dtype=self.dtype)
        rngs = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}

        with patch.object(
            jax.nn,
            "dot_product_attention",
            side_effect=AssertionError("cuDNN must not be used when attention dropout is active"),
        ):
            attention.init(
                rngs,
                hidden_states,
                deterministic=False,
                output_attentions=False,
            )

    def test_cudnn_backend_preserves_cached_decoding(self):
        hidden_states = jnp.ones((1, self.config.max_target_positions, 8), dtype=self.dtype)
        default_attention = FlaxWhisperAttention(
            self.config,
            embed_dim=8,
            num_heads=2,
            causal=True,
            dtype=self.dtype,
        )
        variables = default_attention.init(
            jax.random.PRNGKey(0),
            hidden_states,
            init_cache=True,
            output_attentions=False,
        )

        original_dot_product_attention = jax.nn.dot_product_attention

        def fake_cudnn_attention(*args, **kwargs):
            kwargs["implementation"] = "xla"
            return original_dot_product_attention(*args, **kwargs)

        cudnn_attention = FlaxWhisperAttention(
            self.config,
            embed_dim=8,
            num_heads=2,
            causal=True,
            dtype=self.dtype,
            attention_backend="cudnn",
        )
        expected_variables = variables
        actual_variables = variables
        for index in range(3):
            next_hidden_state = (jnp.arange(8, dtype=self.dtype) + index).reshape(1, 1, 8)
            expected, expected_state = default_attention.apply(
                expected_variables,
                next_hidden_state,
                output_attentions=False,
                mutable=["cache"],
            )
            with patch.object(jax.nn, "dot_product_attention", fake_cudnn_attention):
                actual, actual_state = cudnn_attention.apply(
                    actual_variables,
                    next_hidden_state,
                    output_attentions=False,
                    mutable=["cache"],
                )

            self.assertTrue(jnp.allclose(actual[0], expected[0], rtol=1e-2, atol=1e-2))
            self.assertIsNone(actual[1])
            expected_leaves = jax.tree_util.tree_leaves(expected_state)
            actual_leaves = jax.tree_util.tree_leaves(actual_state)
            self.assertEqual(len(actual_leaves), len(expected_leaves))
            self.assertTrue(
                all(jnp.array_equal(actual, expected) for actual, expected in zip(actual_leaves, expected_leaves))
            )
            expected_variables = {"params": variables["params"], "cache": expected_state["cache"]}
            actual_variables = {"params": variables["params"], "cache": actual_state["cache"]}

    def test_invalid_backend_is_rejected(self):
        attention = FlaxWhisperAttention(
            self.config,
            embed_dim=8,
            num_heads=2,
            attention_backend="unknown",
        )
        with self.assertRaisesRegex(ValueError, "Unsupported attention backend"):
            attention.init(jax.random.PRNGKey(0), jnp.ones((1, 3, 8)))

    def test_cudnn_backend_rejects_unsupported_dtype(self):
        attention = FlaxWhisperAttention(
            self.config,
            embed_dim=8,
            num_heads=2,
            dtype=jnp.float32,
            attention_backend="cudnn",
        )
        with self.assertRaisesRegex(ValueError, "requires.*float16.*bfloat16"):
            attention.init(jax.random.PRNGKey(0), jnp.ones((1, 3, 8)))

    def test_pipeline_rejects_unsupported_cudnn_dtype_before_loading_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "requires.*float16.*bfloat16"):
            FlaxWhisperPipline(attention_backend="cudnn")


if __name__ == "__main__":
    unittest.main()
