"""Native encoder K/V caching must preserve decoding and self-attention state."""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import unfreeze
from transformers import WhisperConfig

from whisper_jax.modeling_flax_whisper import (
    FlaxWhisperAttention,
    FlaxWhisperForConditionalGeneration,
    FlaxWhisperModel,
)


class CrossAttentionCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = WhisperConfig(
            vocab_size=32,
            num_mel_bins=4,
            d_model=8,
            encoder_layers=1,
            decoder_layers=2,
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
        initialized = FlaxWhisperForConditionalGeneration(cls.config, seed=17)
        cls.params = unfreeze(initialized.params)
        # Exercise projection biases and supplied parameters distinct from init_cache's dummy parameters.
        for index in range(cls.config.decoder_layers):
            attention_params = cls.params["model"]["decoder"]["layers"][str(index)]["encoder_attn"]
            attention_params["k_proj"]["kernel"] *= 0.75 + index / 8
            attention_params["v_proj"]["bias"] = jnp.arange(cls.config.d_model, dtype=jnp.float32) / 31 + index / 7
        cls.features = jax.random.normal(jax.random.PRNGKey(5), (2, 4, 8))
        cls.ids = jnp.array([[1, 3, 4, 5], [1, 7, 8, 9]], dtype=jnp.int32)
        cls.mask = jnp.array([[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]], dtype=jnp.int32)

    @contextmanager
    def _backend(self, backend):
        """Exercise cuDNN dispatch on CPU using JAX's equivalent XLA attention."""
        original_attention = jax.nn.dot_product_attention
        calls = []

        def cpu_cudnn(query, key, value, **kwargs):
            self.assertEqual(kwargs["implementation"], "cudnn")
            calls.append((query.shape, key.shape, value.shape, query.dtype))
            # The CPU XLA backend cannot lower JAX's F16_F16_F32 precision preset.
            return original_attention(
                query.astype(jnp.float32),
                key.astype(jnp.float32),
                value.astype(jnp.float32),
                **{**kwargs, "implementation": "xla"},
            ).astype(query.dtype)

        if backend == "cudnn":
            with patch.object(jax.nn, "dot_product_attention", cpu_cudnn):
                yield calls
        else:
            yield calls

    def _model(self, dtype, backend="default"):
        return FlaxWhisperForConditionalGeneration(
            self.config, dtype=dtype, params_dtype=jnp.float32, attention_backend=backend, _do_init=False
        )

    def assert_tree_equal(self, actual, expected):
        self.assertEqual(jax.tree_util.tree_structure(actual), jax.tree_util.tree_structure(expected))
        for actual_leaf, expected_leaf in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
            self.assertEqual(actual_leaf.dtype, expected_leaf.dtype)
            np.testing.assert_array_equal(actual_leaf, expected_leaf)

    def assert_tree_close(self, actual, expected):
        self.assertEqual(jax.tree_util.tree_structure(actual), jax.tree_util.tree_structure(expected))
        for actual_leaf, expected_leaf in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
            self.assertEqual(actual_leaf.dtype, expected_leaf.dtype)
            if jnp.issubdtype(actual_leaf.dtype, jnp.inexact):
                tolerance = {jnp.dtype(jnp.float16): 1e-3, jnp.dtype(jnp.bfloat16): 1e-2}.get(actual_leaf.dtype, 1e-6)
                # Separate projection compilation can change CUDA fusion and its final rounding.
                np.testing.assert_allclose(
                    actual_leaf.astype(jnp.float32),
                    expected_leaf.astype(jnp.float32),
                    rtol=tolerance,
                    atol=tolerance,
                )
            else:
                np.testing.assert_array_equal(actual_leaf, expected_leaf)

    def test_initialization_uses_actual_parameters_and_preserves_checkpoint_layout(self):
        params_structure = jax.tree_util.tree_structure(self.params)
        for dtype in (jnp.float16, jnp.bfloat16):
            with self.subTest(dtype=dtype):
                model = self._model(dtype)
                encoder = model.encode(self.features, params=self.params, return_dict=True)
                cache = model.init_cross_attention_cache(encoder, params=self.params)
                self.assertIsInstance(cache, tuple)
                self.assertEqual(len(cache), self.config.decoder_layers)
                for index, layer in enumerate(cache):
                    self.assertEqual(set(layer), {"cached_key", "cached_value"})
                    attention_params = self.params["model"]["decoder"]["layers"][str(index)]["encoder_attn"]
                    for name, projection in (("cached_key", "k_proj"), ("cached_value", "v_proj")):
                        expected = encoder[0].astype(dtype) @ attention_params[projection]["kernel"].astype(dtype)
                        if projection == "v_proj":
                            expected = expected + attention_params[projection]["bias"].astype(dtype)
                        self.assertEqual(layer[name].shape, (2, 4, 2, 4))
                        self.assertEqual(layer[name].dtype, dtype)
                        self.assert_tree_close(layer[name], expected.reshape(2, 4, 2, 4))
                self.assertEqual(jax.tree_util.tree_structure(self.params), params_structure)
                self.assert_tree_equal(cache, model.init_cross_attention_cache((encoder[0],), params=self.params))

    def test_cached_logits_and_self_cache_match_uncached_fp16_bf16(self):
        for dtype in (jnp.float16, jnp.bfloat16):
            for backend in ("default", "cudnn"):
                with self.subTest(dtype=dtype, backend=backend), self._backend(backend) as attention_calls:
                    model = self._model(dtype, backend)
                    encoder = model.encode(self.features, params=self.params)
                    cross_cache = model.init_cross_attention_cache(encoder, params=self.params)
                    original_cross_cache = jax.tree_util.tree_map(jnp.copy, cross_cache)
                    actual_cache = expected_cache = model.init_cache(2, 6, encoder)
                    decode = jax.jit(
                        lambda ids, positions, self_cache, encoder_cache: model.decode(
                            ids,
                            encoder,
                            decoder_attention_mask=self.mask,
                            decoder_position_ids=positions,
                            past_key_values=self_cache,
                            cross_attention_cache=encoder_cache,
                            params=self.params,
                            return_dict=True,
                        )
                    )
                    # Include a prompt and two single-token appends at nonzero cache offsets.
                    for start, end in ((0, 2), (2, 3), (3, 4)):
                        positions = jnp.arange(start, end, dtype=jnp.int32)[None]
                        expected = decode(self.ids[:, start:end], positions, expected_cache, None)
                        actual = decode(self.ids[:, start:end], positions, actual_cache, cross_cache)
                        self.assertEqual(actual.logits.dtype, dtype)
                        self.assert_tree_close(actual.logits, expected.logits)
                        self.assert_tree_close(actual.past_key_values, expected.past_key_values)
                        actual_cache, expected_cache = actual.past_key_values, expected.past_key_values
                    self.assert_tree_equal(cross_cache, original_cross_cache)
                    if backend == "cudnn":
                        self.assertTrue(attention_calls)
                        self.assertTrue(all(call[-1] == dtype for call in attention_calls))

    def test_teacher_forced_outputs_and_attention_fallback_skip_encoder_projections(self):
        for dtype in (jnp.float16, jnp.bfloat16):
            with self.subTest(dtype=dtype):
                with self._backend("cudnn"):
                    model = self._model(dtype, "cudnn")
                encoder = (jax.random.normal(jax.random.PRNGKey(6), (2, 4, 8)).astype(dtype),)
                cross_cache = model.init_cross_attention_cache(encoder, params=self.params)
                projection_calls = []

                def observe_projection(next_fun, args, kwargs, context):
                    path = context.module.scope.path
                    if context.method_name == "__call__" and path[-2:] in (
                        ("encoder_attn", "k_proj"),
                        ("encoder_attn", "v_proj"),
                    ):
                        projection_calls.append(path)
                    return next_fun(*args, **kwargs)

                kwargs = {
                    "decoder_attention_mask": self.mask[:, :4],
                    "decoder_position_ids": jnp.arange(4, dtype=jnp.int32)[None],
                    "params": self.params,
                    "output_attentions": True,
                    "output_hidden_states": True,
                    "return_dict": True,
                }
                with patch.object(jax.nn, "dot_product_attention", side_effect=AssertionError("Expected fallback")):
                    with nn.intercept_methods(observe_projection):
                        expected = model.decode(self.ids, encoder, **kwargs)
                    self.assertEqual(len(projection_calls), 2 * self.config.decoder_layers)
                    projection_calls.clear()
                    with nn.intercept_methods(observe_projection):
                        actual = model.decode(self.ids, encoder, cross_attention_cache=cross_cache, **kwargs)
                    self.assertEqual(projection_calls, [])
                self.assert_tree_close(actual, expected)
                self.assertIsNone(actual.past_key_values)
                for weights in actual.attentions:
                    # Row one's first token is padding. Check subsequent queries, which have valid keys.
                    np.testing.assert_array_equal(weights[1, :, 1:, 0], 0)
                for weights in actual.cross_attentions:
                    self.assertEqual(weights.shape, (2, 2, 4, 4))
                    self.assertEqual(weights.dtype, dtype)

    def test_cross_attention_padding_mask_is_preserved(self):
        for dtype in (jnp.float16, jnp.bfloat16):
            with self.subTest(dtype=dtype):
                attention = FlaxWhisperAttention(self.config, embed_dim=8, num_heads=2, dtype=dtype)
                hidden = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 8)).astype(dtype)
                encoder = jax.random.normal(jax.random.PRNGKey(3), (2, 4, 8)).astype(dtype)
                mask = jnp.array([[1, 1, 0, 0], [0, 1, 1, 0]])
                variables = attention.init(jax.random.PRNGKey(4), hidden, key_value_states=encoder)
                cache = attention.apply(variables, encoder, method=attention.init_cross_attention_cache)
                expected = attention.apply(variables, hidden, key_value_states=encoder, attention_mask=mask)
                actual = attention.apply(
                    variables, hidden, key_value_states=encoder, attention_mask=mask, cross_attention_cache=cache
                )
                self.assert_tree_close(actual, expected)
                np.testing.assert_array_equal(actual[1][0, :, :, 2:], 0)
                np.testing.assert_array_equal(actual[1][1, :, :, (0, 3)], 0)

    def test_rejects_incompatible_cache_shapes_and_dtypes(self):
        model = self._model(jnp.float16)
        encoder = (jnp.ones((2, 4, 8), dtype=jnp.float16),)
        cache = model.init_cross_attention_cache(encoder, params=self.params)
        malformed = (
            (cache[:1], "2 layers"),
            (({"cached_key": cache[0]["cached_key"]}, cache[1]), "cached_value"),
            (({**cache[0], "cached_key": cache[0]["cached_key"][:, :-1]}, cache[1]), "shape"),
            (({**cache[0], "cached_value": cache[0]["cached_value"].astype(jnp.float32)}, cache[1]), "dtype"),
        )
        for bad_cache, message in malformed:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                model.decode(self.ids, encoder, cross_attention_cache=bad_cache, params=self.params)
        with self.assertRaisesRegex(ValueError, "encoder_outputs.*shape"):
            model.init_cross_attention_cache((jnp.ones((2, 4, 7)),), params=self.params)

    def test_cache_is_fixed_while_loop_carry_and_initializes_inside_jit(self):
        for dtype in (jnp.float16, jnp.bfloat16):
            with self.subTest(dtype=dtype):
                model = self._model(dtype)

                def run(params, features, ids, use_cross_cache):
                    encoder = model.encode(features, params=params)
                    cross_cache = model.init_cross_attention_cache(encoder, params=params) if use_cross_cache else None
                    self_cache = model.init_cache(ids.shape[0], 6, encoder)
                    logits = jnp.zeros(ids.shape + (self.config.vocab_size,), dtype=dtype)

                    def step(state):
                        index, self_cache, cross_cache, logits = state
                        token = jax.lax.dynamic_slice_in_dim(ids, index, 1, axis=1)
                        outputs = model.decode(
                            token,
                            encoder,
                            decoder_attention_mask=self.mask,
                            decoder_position_ids=index.reshape(1, 1),
                            past_key_values=self_cache,
                            cross_attention_cache=cross_cache,
                            params=params,
                            return_dict=True,
                        )
                        logits = jax.lax.dynamic_update_slice(logits, outputs.logits, (0, index, 0))
                        return index + 1, outputs.past_key_values, cross_cache, logits

                    return jax.lax.while_loop(
                        lambda state: state[0] < ids.shape[1], step, (jnp.array(0), self_cache, cross_cache, logits)
                    )

                compiled = jax.jit(run, static_argnums=3)
                expected = compiled(self.params, self.features, self.ids, False)
                actual = compiled(self.params, self.features, self.ids, True)
                self.assert_tree_close(actual[3], expected[3])
                self.assert_tree_close(actual[1], expected[1])
                self.assertEqual(len(jax.tree_util.tree_leaves(actual[2])), 2 * self.config.decoder_layers)
                for layer in actual[2]:
                    self.assertEqual(layer["cached_key"].shape, (2, 4, 2, 4))

    def test_default_parameters_and_base_model_tuple_outputs(self):
        encoder = (jax.random.normal(jax.random.PRNGKey(9), (2, 4, 8)),)
        for model_class in (FlaxWhisperForConditionalGeneration, FlaxWhisperModel):
            with self.subTest(model_class=model_class):
                model = model_class(self.config, seed=19)
                cross_cache = model.init_cross_attention_cache(encoder)
                self.assert_tree_equal(cross_cache, model.init_cross_attention_cache(encoder, params=model.params))
                self_cache = model.init_cache(2, 6, encoder)
                kwargs = {
                    "decoder_attention_mask": self.mask,
                    "decoder_position_ids": jnp.array([[0, 1]], dtype=jnp.int32),
                    "past_key_values": self_cache,
                    "return_dict": False,
                }
                expected = model.decode(self.ids[:, :2], encoder, **kwargs)
                actual = model.decode(self.ids[:, :2], encoder, cross_attention_cache=cross_cache, **kwargs)
                self.assert_tree_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
