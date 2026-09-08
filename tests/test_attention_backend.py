import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from transformers import WhisperConfig

from whisper_jax.layers import LayerNorm
from whisper_jax.modeling_flax_whisper import (
    FlaxStaticForceTokensLogitsProcessor,
    FlaxWhisperAttention,
    FlaxWhisperForConditionalGeneration,
    FlaxWhisperForConditionalGenerationModule,
    FlaxWhisperModel,
    FlaxWhisperTimeStampLogitsProcessor,
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

    @staticmethod
    def _with_identity_attention_params(variables):
        variables = unfreeze(variables)
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            kernel = variables["params"][projection]["kernel"]
            variables["params"][projection]["kernel"] = jnp.eye(kernel.shape[0], kernel.shape[1])
        return freeze(variables)

    @staticmethod
    def _random_params(params):
        parameter_leaves, parameter_tree = jax.tree_util.tree_flatten(params)
        parameter_keys = jax.random.split(jax.random.PRNGKey(7), len(parameter_leaves))
        return jax.tree_util.tree_unflatten(
            parameter_tree,
            [
                jax.random.normal(key, parameter.shape, dtype=parameter.dtype) * 0.1
                for key, parameter in zip(parameter_keys, parameter_leaves)
            ],
        )

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
        # FP32 softmax must cast back before the probability/value matmul.
        self.assertEqual(outputs.encoder_attentions[0].dtype, self.dtype)
        self.assertEqual(outputs.decoder_attentions[0].dtype, self.dtype)
        self.assertEqual(outputs.cross_attentions[0].dtype, self.dtype)

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

    def test_bfloat16_layer_norm_preserves_float32_affine_parameters(self):
        layer = LayerNorm(dtype=jnp.bfloat16, epsilon=1e-5)
        params = {"scale": jnp.full((2,), 1.002, dtype=jnp.float32), "bias": jnp.full((2,), 0.003, dtype=jnp.float32)}
        inputs = jnp.array([[1.0, -1.0]], dtype=jnp.bfloat16)
        actual = jax.jit(lambda x: layer.apply({"params": params}, x))(inputs)
        # Native Torch FP32 normalization/affine followed by BF16 output rounding.
        # Rounding the scale to BF16 first incorrectly gives 1.0 for the first element.
        self.assertTrue(jnp.array_equal(actual, jnp.array([[1.0078125, -1.0]], dtype=jnp.bfloat16)))

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
        variables = self._with_identity_attention_params(variables)

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

    def test_cudnn_autoregressive_generation_uses_kv_cache(self):
        params = self._random_params(self.variables["params"])
        features = self.inputs["input_features"]
        prompt = self.inputs["decoder_input_ids"]
        max_length = self.config.max_target_positions
        default_model = FlaxWhisperForConditionalGeneration(self.config, dtype=self.dtype, _do_init=False)
        generation_kwargs = {
            "params": params,
            "decoder_input_ids": prompt,
            "max_length": max_length,
            "eos_token_id": self.config.vocab_size,
            "suppress_tokens": [],
            "begin_suppress_tokens": [],
        }
        expected = default_model.generate(features, **generation_kwargs).sequences
        original_attention = jax.nn.dot_product_attention
        cached_calls = []

        def record_cache(query_length, key, value, mask):
            cached_calls.append((int(query_length), key.copy(), value.copy(), mask.copy()))

        def cpu_cudnn_standin(query, key, value, **kwargs):
            self.assertEqual(kwargs["implementation"], "cudnn")
            self.assertEqual(query.dtype, self.dtype)
            # Exclude the dummy full-length decoder call used to allocate the cache.
            # Callbacks observe each executed iteration, including inside while_loop/pmap.
            if key.shape[1] == max_length and query.shape[1] < max_length:
                jax.debug.callback(record_cache, query.shape[1], key, value, kwargs["mask"])
            return original_attention(query, key, value, **{**kwargs, "implementation": "xla"})

        with patch.object(jax.nn, "dot_product_attention", cpu_cudnn_standin):
            model = FlaxWhisperForConditionalGeneration(
                self.config, dtype=self.dtype, attention_backend="cudnn", _do_init=False
            )
            for mode in ("eager", "traced", "pmap"):
                with self.subTest(mode=mode), patch(
                    "whisper_jax.modeling_flax_whisper.dot_product_attention_weights",
                    side_effect=AssertionError("Generation must not fall back from the requested cuDNN backend"),
                ):
                    cached_calls.clear()
                    replicas = jax.local_device_count() if mode == "pmap" else 1
                    if mode == "pmap":
                        generate = jax.pmap(
                            lambda batch, forced: model.pipeline_generate(
                                batch, forced, **generation_kwargs
                            ).sequences,
                            in_axes=(0, None),
                        )
                        actual = generate(
                            jnp.broadcast_to(features, (replicas,) + features.shape),
                            jnp.empty((0, 2), dtype=jnp.int32),
                        )
                        expected_sequences = jnp.broadcast_to(expected, actual.shape)
                    else:
                        actual = model.generate(features, trace=mode == "traced", **generation_kwargs).sequences
                        expected_sequences = expected
                    actual.block_until_ready()
                    jax.effects_barrier()
                    self.assertTrue(jnp.array_equal(actual, expected_sequences))
                    self.assertEqual(len(cached_calls), 3 * replicas)

                    # Callback ordering across devices is unspecified. Sort by populated
                    # cache length; identical inputs and parameters give identical replicas.
                    cached_calls.sort(key=lambda call: int(call[3][0, 0, -1].sum()))
                    previous_key = previous_value = None
                    start = 0
                    for step, query_length in enumerate((3, 1, 1)):
                        end = start + query_length
                        for observed_length, key, value, mask in cached_calls[step * replicas : (step + 1) * replicas]:
                            self.assertEqual(observed_length, query_length)
                            self.assertEqual(key.shape, (1, max_length, 2, 4))
                            self.assertEqual(value.shape, key.shape)
                            self.assertEqual(mask.shape, (1, 1, query_length, max_length))
                            causal_mask = jnp.arange(max_length)[None, :] <= jnp.arange(start, end)[:, None]
                            self.assertTrue(jnp.array_equal(mask[0, 0], causal_mask))
                            self.assertTrue(jnp.any(key[:, start:end] != 0))
                            self.assertTrue(jnp.any(value[:, start:end] != 0))
                            self.assertTrue(jnp.all(key[:, end:] == 0))
                            self.assertTrue(jnp.all(value[:, end:] == 0))
                            if previous_key is not None:
                                self.assertTrue(jnp.array_equal(key[:, :start], previous_key[:, :start]))
                                self.assertTrue(jnp.array_equal(value[:, :start], previous_value[:, :start]))
                        previous_key, previous_value = key, value
                        start = end

    def test_chunked_cached_decoding_matches_full_attention(self):
        chunk_patterns = ((4,), (1, 3), (2, 2), (3, 1), (1, 1, 2), (1, 1, 1, 1))
        for dtype, tolerance in ((jnp.float32, 1e-5), (jnp.bfloat16, 1e-2)):
            hidden_states = (jnp.arange(64, dtype=jnp.float32).reshape(2, 4, 8) / 64).astype(dtype)
            attention = FlaxWhisperAttention(
                self.config,
                embed_dim=8,
                num_heads=2,
                causal=True,
                dtype=dtype,
            )
            variables = attention.init(
                jax.random.PRNGKey(0),
                jnp.zeros((2, self.config.max_target_positions, 8), dtype=dtype),
                init_cache=True,
                output_attentions=False,
            )
            variables = self._with_identity_attention_params(variables)
            expected, _ = attention.apply(
                {"params": variables["params"]},
                hidden_states,
                output_attentions=False,
            )
            expected_cached_key = jnp.moveaxis(hidden_states.reshape(2, 4, 2, 4), -3, -1)
            cached_apply = jax.jit(
                lambda cached_variables, chunk: attention.apply(
                    cached_variables,
                    chunk,
                    output_attentions=False,
                    mutable=["cache"],
                )
            )

            for chunk_sizes in chunk_patterns:
                with self.subTest(dtype=dtype, chunk_sizes=chunk_sizes):
                    cached_variables = variables
                    chunk_outputs = []
                    chunk_start = 0
                    for chunk_size in chunk_sizes:
                        chunk_end = chunk_start + chunk_size
                        (chunk_output, _), cache_state = cached_apply(
                            cached_variables,
                            hidden_states[:, chunk_start:chunk_end],
                        )
                        chunk_outputs.append(chunk_output)
                        cached_variables = {"params": variables["params"], "cache": cache_state["cache"]}
                        chunk_start = chunk_end

                    actual = jnp.concatenate(chunk_outputs, axis=1)
                    cached_key = cache_state["cache"]["cached_key"]
                    self.assertTrue(jnp.allclose(actual, expected, rtol=tolerance, atol=tolerance))
                    self.assertTrue(
                        jnp.allclose(
                            cached_key[..., : hidden_states.shape[1]],
                            expected_cached_key,
                            rtol=tolerance,
                            atol=tolerance,
                        )
                    )
                    self.assertTrue(jnp.all(cached_key[..., hidden_states.shape[1] :] == 0))
                    self.assertEqual(int(cache_state["cache"]["cache_index"]), hidden_states.shape[1])

    def test_model_decode_cache_matches_full_decode(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        params = self._random_params(model.params)
        encoder_outputs = (jax.random.normal(jax.random.PRNGKey(8), (1, 4, self.config.d_model)),)
        decoder_input_ids = jnp.array([[1, 3, 4, 5]], dtype=jnp.int32)
        expected = model.decode(decoder_input_ids, encoder_outputs, params=params).logits

        past_key_values = model.init_cache(1, self.config.max_target_positions, encoder_outputs)
        attention_mask = jnp.ones((1, self.config.max_target_positions), dtype=jnp.int32)
        prefill = model.decode(
            decoder_input_ids[:, :3],
            encoder_outputs,
            decoder_attention_mask=attention_mask,
            decoder_position_ids=jnp.array([[0, 1, 2]], dtype=jnp.int32),
            past_key_values=past_key_values,
            params=params,
        )
        decode_step = model.decode(
            decoder_input_ids[:, 3:],
            encoder_outputs,
            decoder_attention_mask=attention_mask,
            decoder_position_ids=jnp.array([[3]], dtype=jnp.int32),
            past_key_values=prefill.past_key_values,
            params=params,
        )

        actual = jnp.concatenate((prefill.logits, decode_step.logits), axis=1)
        self.assertTrue(jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5))

    def test_padded_prompt_cache_matches_full_decode(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        params = self._random_params(model.params)
        encoder_outputs = (jax.random.normal(jax.random.PRNGKey(8), (2, 4, self.config.d_model)),)
        prompt_ids = jnp.array([[0, 1, 3], [1, 3, 4]], dtype=jnp.int32)
        prompt_mask = jnp.array([[0, 1, 1], [1, 1, 1]], dtype=jnp.int32)
        next_token = jnp.array([[4], [5]], dtype=jnp.int32)
        full_ids = jnp.concatenate((prompt_ids, next_token), axis=-1)
        full_mask = jnp.array([[0, 1, 1, 1], [1, 1, 1, 1]], dtype=jnp.int32)
        expected = model.decode(
            full_ids,
            encoder_outputs,
            decoder_attention_mask=full_mask,
            params=params,
        ).logits

        model_kwargs = model.prepare_inputs_for_generation(
            prompt_ids,
            self.config.max_target_positions,
            decoder_attention_mask=prompt_mask,
            encoder_outputs=encoder_outputs,
        )
        prefill = model.decode(prompt_ids, params=params, **model_kwargs)
        model_kwargs = model.update_inputs_for_generation(prefill, model_kwargs)
        decode_step = model.decode(next_token, params=params, **model_kwargs)
        actual = jnp.concatenate((prefill.logits, decode_step.logits), axis=1)

        # The masked padding query itself is irrelevant; all real-token logits, including the cached step, must match.
        self.assertTrue(jnp.allclose(actual[0, 1:], expected[0, 1:], rtol=1e-5, atol=1e-5))
        self.assertTrue(jnp.allclose(actual[1], expected[1], rtol=1e-5, atol=1e-5))

    def test_traced_generation_cache_matches_uncached_decode(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        params = self._random_params(model.params)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        decoder_input_ids = jnp.array([[1, 3, 4]], dtype=jnp.int32)
        input_features = jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 32

        encoder_outputs = model.encode(input_features, params=params)
        expected = decoder_input_ids
        while expected.shape[1] < self.config.max_target_positions:
            logits = model.decode(expected, encoder_outputs, params=params).logits[:, -1]
            expected = jnp.concatenate((expected, jnp.argmax(logits, axis=-1)[:, None]), axis=1)

        actual = model.generate(
            input_features,
            decoder_input_ids=decoder_input_ids,
            max_length=self.config.max_target_positions,
            eos_token_id=self.config.vocab_size,
            trace=True,
            params=params,
        ).sequences
        eager = model.generate(
            input_features,
            decoder_input_ids=decoder_input_ids,
            max_length=self.config.max_target_positions,
            eos_token_id=self.config.vocab_size,
            trace=False,
            params=params,
        ).sequences

        self.assertTrue(jnp.array_equal(actual, expected))
        self.assertTrue(jnp.array_equal(actual, eager))

    def test_generation_prefill_at_cache_capacity_boundary(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        params = self._random_params(model.params)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        decoder_input_ids = jnp.array([[1, 3, 4, 5, 7]], dtype=jnp.int32)
        input_features = jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 32
        encoder_outputs = model.encode(input_features, params=params)
        expected_token = jnp.argmax(
            model.decode(decoder_input_ids, encoder_outputs, params=params).logits[:, -1],
            axis=-1,
        )

        actual = model.generate(
            input_features,
            decoder_input_ids=decoder_input_ids,
            max_length=self.config.max_target_positions,
            eos_token_id=self.config.vocab_size,
            params=params,
        ).sequences

        self.assertTrue(jnp.array_equal(actual[:, -1], expected_token))

    def test_generation_modes_match_with_traced_cache(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        params = self._random_params(model.params)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        decoder_input_ids = jnp.array([[1, 3, 4]], dtype=jnp.int32)
        input_features = jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 32

        generation_modes = (
            {"do_sample": True, "prng_key": jax.random.PRNGKey(11)},
            {"num_beams": 2},
        )
        for generation_kwargs in generation_modes:
            with self.subTest(generation_kwargs=generation_kwargs):
                traced = model.generate(
                    input_features,
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.config.max_target_positions,
                    trace=True,
                    params=params,
                    **generation_kwargs,
                ).sequences
                eager = model.generate(
                    input_features,
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.config.max_target_positions,
                    trace=False,
                    params=params,
                    **generation_kwargs,
                ).sequences
                self.assertTrue(jnp.array_equal(traced, eager))

    def test_beam_search_cache_reordering_matches_uncached_reference(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        params = self._random_params(model.params)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        decoder_input_ids = jnp.array([[0, 1, 4]], dtype=jnp.int32)
        decoder_attention_mask = jnp.array([[0, 1, 1]], dtype=jnp.int32)
        input_features = jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 32
        encoder_outputs = model.encode(input_features, params=params)

        first_log_probs = jax.nn.log_softmax(
            model.decode(
                decoder_input_ids,
                encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                params=params,
            ).logits[:, -1],
            axis=-1,
        )
        first_scores, first_tokens = jax.lax.top_k(first_log_probs[0], 2)
        beam_prompts = jnp.repeat(decoder_input_ids, 2, axis=0)
        beam_sequences = jnp.concatenate((beam_prompts, first_tokens[:, None]), axis=-1)
        beam_attention_mask = jnp.concatenate(
            (jnp.repeat(decoder_attention_mask, 2, axis=0), jnp.ones((2, 1), dtype=jnp.int32)),
            axis=-1,
        )
        beam_encoder_outputs = (jnp.repeat(encoder_outputs[0], 2, axis=0),)
        second_log_probs = jax.nn.log_softmax(
            model.decode(
                beam_sequences,
                beam_encoder_outputs,
                decoder_attention_mask=beam_attention_mask,
                params=params,
            ).logits[:, -1],
            axis=-1,
        )
        combined_scores = first_scores[:, None] + second_log_probs
        best_candidate = jnp.argmax(combined_scores)
        best_beam = best_candidate // self.config.vocab_size
        best_token = best_candidate % self.config.vocab_size
        expected = jnp.concatenate((beam_sequences[best_beam], best_token[None]))

        actual = model.generate(
            input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            max_length=5,
            eos_token_id=self.config.vocab_size,
            num_beams=2,
            params=params,
        ).sequences

        self.assertTrue(jnp.array_equal(actual[0], expected))

    def test_static_force_tokens_supports_multilingual_no_timestamps_position(self):
        processor = FlaxStaticForceTokensLogitsProcessor([(1, 4), (2, 5), (3, 6)])
        input_ids = jnp.ones((2, 4), dtype=jnp.int32)
        scores = jnp.zeros((2, self.config.vocab_size), dtype=jnp.float32)

        actual = processor(input_ids, scores, jnp.array(3, dtype=jnp.int32))

        self.assertTrue(jnp.array_equal(jnp.argmax(actual, axis=-1), jnp.array([6, 6])))

    def test_pipeline_generate_accepts_dynamic_forced_tokens_under_jit(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        model.generation_config.forced_decoder_ids = [[1, 9]]
        input_features = jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 32
        forced_decoder_ids = jnp.array([(1, 4), (2, 5), (3, 6)], dtype=jnp.int32)
        pipeline_generate = jax.jit(
            lambda features, forced_tokens: model.pipeline_generate(
                features,
                forced_tokens,
                max_length=5,
            ).sequences
        )

        actual = pipeline_generate(input_features, forced_decoder_ids)

        self.assertTrue(jnp.array_equal(actual[0, :4], jnp.array([1, 4, 5, 6])))
        self.assertEqual(model.generation_config.forced_decoder_ids, [[1, 9]])

    def test_pipeline_generate_accepts_dynamic_forced_tokens_under_pmap(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        input_features = jnp.broadcast_to(
            jnp.arange(32, dtype=jnp.float32).reshape(1, 1, 4, 8) / 32,
            (jax.local_device_count(), 1, 4, 8),
        )
        forced_decoder_ids = jnp.array([(1, 4), (2, 5), (3, 6)], dtype=jnp.int32)
        pipeline_generate = jax.pmap(
            lambda features, forced_tokens: model.pipeline_generate(
                features,
                forced_tokens,
                max_length=5,
            ).sequences,
            in_axes=(0, None),
        )

        actual = pipeline_generate(input_features, forced_decoder_ids)

        expected_prompt = jnp.broadcast_to(jnp.array([1, 4, 5, 6]), actual[:, 0, :4].shape)
        self.assertTrue(jnp.array_equal(actual[:, 0, :4], expected_prompt))

    def test_pipeline_timestamp_generation_handles_empty_forced_tokens_under_jit(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        model.generation_config.no_timestamps_token_id = 10
        model.generation_config.is_multilingual = False
        input_features = jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 32
        pipeline_generate = jax.jit(
            lambda features: model.pipeline_generate(
                features,
                [],
                return_timestamps=True,
                max_length=5,
            ).sequences
        )

        actual = pipeline_generate(input_features)

        self.assertGreaterEqual(int(actual[0, 1]), 11)

    def test_generate_forces_standard_whisper_prompt_without_mutating_config(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        model.generation_config.forced_decoder_ids = None
        model.generation_config.no_timestamps_token_id = 6
        model.generation_config.is_multilingual = True
        model.generation_config.language = "en"
        model.generation_config.lang_to_id = {"<|en|>": 4}
        model.generation_config.task = None
        model.generation_config.task_to_id = {"transcribe": 5}
        input_features = jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 32

        actual = model.generate(input_features, max_length=5).sequences

        self.assertTrue(jnp.array_equal(actual[0, :4], jnp.array([1, 4, 5, 6])))
        self.assertIsNone(model.generation_config.forced_decoder_ids)
        self.assertIsNone(model.generation_config.task)

    def test_generate_preserves_configured_multilingual_forced_decoder_ids(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        model.generation_config.forced_decoder_ids = [[1, 4], [2, 7], [3, 6]]
        model.generation_config.no_timestamps_token_id = 6
        model.generation_config.is_multilingual = True
        model.generation_config.language = None
        model.generation_config.lang_to_id = {"<|en|>": 4}
        model.generation_config.task = None
        model.generation_config.task_to_id = {"transcribe": 5, "translate": 7}
        input_features = jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 32

        actual = model.generate(input_features, max_length=5).sequences

        self.assertTrue(jnp.array_equal(actual[0, :4], jnp.array([1, 4, 7, 6])))
        self.assertEqual(model.generation_config.forced_decoder_ids, [[1, 4], [2, 7], [3, 6]])

    def test_generate_offsets_forced_tokens_after_custom_decoder_prompt(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        model.generation_config.forced_decoder_ids = None
        model.generation_config.no_timestamps_token_id = 6
        model.generation_config.is_multilingual = True
        model.generation_config.language = "english"
        model.generation_config.lang_to_id = {"<|en|>": 4}
        model.generation_config.task = "transcribe"
        model.generation_config.task_to_id = {"transcribe": 5}
        input_features = jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 32
        decoder_input_ids = jnp.array([[1, 8]], dtype=jnp.int32)

        actual = model.generate(
            input_features,
            decoder_input_ids=decoder_input_ids,
            max_length=6,
        ).sequences

        self.assertTrue(jnp.array_equal(actual[0, :5], jnp.array([1, 8, 4, 5, 6])))

    def test_generate_forces_no_timestamps_for_english_only_checkpoint(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        model.generation_config.forced_decoder_ids = None
        model.generation_config.no_timestamps_token_id = 6
        model.generation_config.is_multilingual = False
        input_features = jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 32

        actual = model.generate(input_features, max_length=2).sequences

        self.assertEqual(int(actual[0, 1]), 6)
        self.assertIsNone(model.generation_config.forced_decoder_ids)

    def test_timestamp_generation_uses_prompt_length_and_preserves_custom_processors(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        model.generation_config.forced_decoder_ids = None
        model.generation_config.no_timestamps_token_id = 10
        model.generation_config.is_multilingual = False
        input_features = jnp.broadcast_to(
            jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 32,
            (3, 4, 8),
        )
        decoder_input_ids = jnp.broadcast_to(jnp.array([[1, 3]], dtype=jnp.int32), (3, 2))
        timestamp_processors = []

        class RecordingTimestampProcessor:
            def __init__(self, generation_config, model_config, decoder_input_length):
                self.begin_index = decoder_input_length + 1 + (2 if generation_config.is_multilingual else 0)
                self.calls = 0
                timestamp_processors.append(self)

            def __call__(self, input_ids, scores, cur_len):
                self.calls += 1
                return scores

        class RecordingCustomProcessor:
            def __init__(self):
                self.calls = 0

            def __call__(self, input_ids, scores, cur_len):
                self.calls += 1
                return scores

        custom_processor = RecordingCustomProcessor()
        with patch(
            "whisper_jax.modeling_flax_whisper.FlaxWhisperTimeStampLogitsProcessor",
            RecordingTimestampProcessor,
        ):
            model.generate(
                input_features,
                decoder_input_ids=decoder_input_ids,
                max_length=3,
                return_timestamps=True,
                logits_processor=[custom_processor],
                trace=False,
            )

        self.assertEqual(timestamp_processors[0].begin_index, decoder_input_ids.shape[-1])
        self.assertEqual(timestamp_processors[0].calls, 1)
        self.assertEqual(custom_processor.calls, 1)
        self.assertFalse(getattr(model.generation_config, "return_timestamps", False))

    def test_timestamp_processor_starts_after_prompt_and_prevents_decreasing_timestamps(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.no_timestamps_token_id = 10
        model.generation_config.is_multilingual = True
        model.generation_config.max_initial_timestamp_index = 2
        processor = FlaxWhisperTimeStampLogitsProcessor(model.generation_config, self.config, 1)
        processor.begin_index = 3
        input_ids = jnp.array([[1, 4, 5, 14, 7, 0], [1, 4, 5, 7, 14, 0]], dtype=jnp.int32)
        scores = jnp.zeros((1, self.config.vocab_size), dtype=jnp.float32).at[:, 1].set(100.0)
        scores = jnp.broadcast_to(scores, (2, self.config.vocab_size))

        before_prompt_end = processor(input_ids, scores, jnp.array(2, dtype=jnp.int32))
        initial_timestamp = processor(input_ids, scores, jnp.array(3, dtype=jnp.int32))
        after_timestamp = processor(input_ids, scores, jnp.array(5, dtype=jnp.int32))

        self.assertTrue(jnp.array_equal(before_prompt_end, scores))
        self.assertTrue(jnp.all(jnp.isneginf(initial_timestamp[:, :11])))
        self.assertTrue(jnp.all(jnp.isfinite(initial_timestamp[:, 11:14])))
        self.assertTrue(jnp.all(jnp.isneginf(initial_timestamp[:, 14:])))
        # After a timestamp followed by text, the next timestamp must advance.
        self.assertTrue(jnp.all(jnp.isneginf(after_timestamp[0, 11:15])))
        self.assertTrue(jnp.isfinite(after_timestamp[0, 15]))
        # A timestamp following text may repeat once to close the timestamp pair.
        self.assertTrue(jnp.all(jnp.isneginf(after_timestamp[1, 11:14])))
        self.assertTrue(jnp.isfinite(after_timestamp[1, 14]))

    def test_timestamp_processor_enforces_pair_structure(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.eos_token_id = 9
        model.generation_config.no_timestamps_token_id = 10
        model.generation_config.is_multilingual = True
        processor = FlaxWhisperTimeStampLogitsProcessor(model.generation_config, self.config, 1)
        processor.begin_index = 3
        input_ids = jnp.array([[1, 4, 5, 14, 15, 0], [1, 4, 5, 7, 14, 0]], dtype=jnp.int32)
        scores = jnp.zeros((2, self.config.vocab_size), dtype=jnp.float32)

        actual = processor(input_ids, scores, jnp.array(5, dtype=jnp.int32))

        # Consecutive timestamps must be followed by text.
        self.assertTrue(jnp.all(jnp.isneginf(actual[0, 11:])))
        self.assertTrue(jnp.isfinite(actual[0, 7]))
        # A timestamp following text must be closed by another timestamp (or EOS).
        self.assertTrue(jnp.all(jnp.isneginf(actual[1, :9])))
        self.assertTrue(jnp.isfinite(actual[1, 14]))

    def test_timestamp_generation_matches_between_traced_and_eager_execution(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        model.generation_config.forced_decoder_ids = None
        model.generation_config.no_timestamps_token_id = 10
        model.generation_config.is_multilingual = False
        input_features = jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 32

        traced = model.generate(
            input_features,
            max_length=5,
            return_timestamps=True,
            trace=True,
        ).sequences
        eager = model.generate(
            input_features,
            max_length=5,
            return_timestamps=True,
            trace=False,
        ).sequences

        self.assertTrue(jnp.array_equal(traced, eager))
        self.assertGreaterEqual(int(traced[0, 1]), 11)

    def test_generation_validates_cache_bounds_and_mask_shape(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        input_features = jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 32
        encoder_outputs = model.encode(input_features)
        decoder_input_ids = jnp.array([[1, 3]], dtype=jnp.int32)

        with self.assertRaisesRegex(ValueError, "greater than the decoder prompt length"):
            model.prepare_inputs_for_generation(decoder_input_ids, 2, encoder_outputs=encoder_outputs)
        with self.assertRaisesRegex(ValueError, "cannot exceed.*max_target_positions"):
            model.prepare_inputs_for_generation(
                decoder_input_ids,
                self.config.max_target_positions + 1,
                encoder_outputs=encoder_outputs,
            )
        with self.assertRaisesRegex(ValueError, "same shape"):
            model.prepare_inputs_for_generation(
                decoder_input_ids,
                3,
                decoder_attention_mask=jnp.ones((1, 1), dtype=jnp.int32),
                encoder_outputs=encoder_outputs,
            )
        with self.assertRaisesRegex(ValueError, "decoder_position_ids.*same shape"):
            model.decode(
                decoder_input_ids,
                encoder_outputs,
                decoder_position_ids=jnp.zeros((1, 1), dtype=jnp.int32),
            )
        with self.assertRaisesRegex(ValueError, "non-empty cache"):
            model.decode(
                decoder_input_ids,
                encoder_outputs,
                decoder_position_ids=jnp.zeros_like(decoder_input_ids),
                past_key_values={},
            )

    def test_timestamp_probability_ignores_forbidden_timestamps(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.no_timestamps_token_id = 10
        model.generation_config.is_multilingual = False
        processor = FlaxWhisperTimeStampLogitsProcessor(model.generation_config, self.config, 1)
        processor.begin_index = 1
        # A forbidden earlier timestamp dominates the raw distribution. Valid text must survive,
        # including when the last timestamp is already the final entry in the vocabulary.
        input_ids = jnp.array([[1, 20, 7, 0], [1, 31, 7, 0]], dtype=jnp.int32)
        scores = jnp.full((2, 32), -20.0).at[:, 12].set(100.0).at[:, 7].set(10.0)
        for apply in (processor, jax.jit(processor)):
            actual = apply(input_ids, scores, jnp.array(3))
            self.assertTrue(jnp.all(jnp.isfinite(actual[:, 7])))
            self.assertTrue(jnp.array_equal(jnp.argmax(actual, axis=-1), jnp.array([7, 7])))

    def test_timestamp_probability_uses_float32_with_bfloat16_logits(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.no_timestamps_token_id = 10
        model.generation_config.is_multilingual = False
        processor = FlaxWhisperTimeStampLogitsProcessor(model.generation_config, self.config, 1)
        processor.begin_index = 1
        ids = jnp.array([[1, 7, 0]])
        # exp(0) + exp(0) exceeds exp(0.69140625), so timestamps must win.
        # Conversely, 21 * exp(0) is below exp(3.046875), so text must survive.
        for timestamp_count, text_score, suppress_text in ((2, 0.69140625, True), (21, 3.046875, False)):
            scores = jnp.full((1, 32), -jnp.inf, dtype=jnp.bfloat16)
            scores = scores.at[:, 11 : 11 + timestamp_count].set(0).at[:, 7].set(text_score)
            actual = jax.jit(processor)(ids, scores, jnp.array(2))
            self.assertEqual(actual.dtype, scores.dtype)
            self.assertEqual(bool(jnp.isneginf(actual[0, 7])), suppress_text)

    def test_timestamp_generation_normalizes_keyword_forced_tokens(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        model.generation_config.no_timestamps_token_id = 10
        model.generation_config.max_initial_timestamp_index = 2
        model.generation_config.is_multilingual = True
        model.generation_config.task_to_id = {"transcribe": 5}
        input_features = self.inputs["input_features"]
        actual = model.generate(
            input_features,
            forced_decoder_ids=[[1, 4], [2, 5], [3, 10]],
            return_timestamps=True,
            max_initial_timestamp_index=0,
            max_length=5,
        ).sequences
        self.assertTrue(jnp.array_equal(actual[0, :4], jnp.array([1, 4, 5, 11])))
        self.assertIsNone(getattr(model.generation_config, "forced_decoder_ids", None))

    def test_pipeline_suppresses_tokens_after_forced_prompt(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        # Zero logits make token 0 the greedy choice unless the processor suppresses it.
        params = jax.tree_util.tree_map(jnp.zeros_like, model.params)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = [0, 2]
        model.generation_config.no_timestamps_token_id = 6
        model.generation_config.is_multilingual = False
        actual = jax.jit(
            lambda features, forced: model.pipeline_generate(features, forced, params=params, max_length=4).sequences
        )(self.inputs["input_features"], jnp.array([[1, 6]]))
        expected = model.generate(self.inputs["input_features"], params=params, max_length=4).sequences
        self.assertTrue(jnp.array_equal(actual, expected))
        self.assertTrue(jnp.array_equal(actual[0], jnp.array([1, 6, 1, 0])))

    def test_pipeline_offsets_dynamic_prompt_and_timestamp_rules(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        model.generation_config.no_timestamps_token_id = 10
        model.generation_config.max_initial_timestamp_index = 2
        model.generation_config.is_multilingual = True
        prompt = jnp.array([[1, 8]], dtype=jnp.int32)
        generate = jax.jit(
            lambda features, forced: model.pipeline_generate(
                features,
                forced,
                decoder_input_ids=prompt,
                return_timestamps=True,
                max_initial_timestamp_index=0,
                max_length=6,
            ).sequences
        )
        for language_token in (4, 7):
            actual = generate(self.inputs["input_features"], jnp.array([[1, language_token], [2, 5]]))
            self.assertTrue(jnp.array_equal(actual[0, :5], jnp.array([1, 8, language_token, 5, 11])))

    def test_timestamp_generation_respects_eos_override(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        model.generation_config.no_timestamps_token_id = 10
        model.generation_config.is_multilingual = False

        class SegmentScores:
            def __call__(self, input_ids, scores, cur_len):
                # Open a segment, emit text, close it, then prefer illegal text over EOS.
                token = jnp.array([0, 11, 7, 12, 7, 7])[cur_len]
                return jnp.full_like(scores, -20).at[:, 9].set(10).at[:, token].set(100)

        for pipeline in (False, True):
            with self.subTest(pipeline=pipeline):
                generate = model.pipeline_generate if pipeline else model.generate
                kwargs = {"forced_decoder_ids": []} if pipeline else {}
                actual = generate(
                    self.inputs["input_features"],
                    eos_token_id=9,
                    return_timestamps=True,
                    logits_processor=[SegmentScores()],
                    max_length=6,
                    **kwargs,
                ).sequences
                self.assertTrue(jnp.array_equal(actual[0], jnp.array([1, 11, 7, 12, 9, 0])))
        self.assertEqual(model.generation_config.eos_token_id, 2)

    def test_batched_generation_pads_finished_rows(self):
        model = FlaxWhisperForConditionalGeneration(self.config)
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        features = jnp.repeat(self.inputs["input_features"], 2, axis=0)

        class StaggeredEos:
            def __call__(self, input_ids, scores, cur_len):
                tokens = jnp.where(cur_len >= jnp.array([1, 3]), 2, 7)
                return jnp.full_like(scores, -jnp.inf).at[jnp.arange(2), tokens].set(0)

        expected = jnp.array([[1, 2, 0, 0, 0, 0], [1, 7, 7, 2, 0, 0]])
        for trace in (False, True):
            for do_sample in (False, True):
                with self.subTest(trace=trace, do_sample=do_sample):
                    actual = model.generate(
                        features,
                        max_new_tokens=5,
                        trace=trace,
                        do_sample=do_sample,
                        logits_processor=[StaggeredEos()],
                    ).sequences
                    self.assertTrue(jnp.array_equal(actual, expected))

    def test_cached_tuple_outputs_match_full_decode_for_both_model_classes(self):
        ids = jnp.array([[1, 3, 4]], dtype=jnp.int32)
        encoder_outputs = (jax.random.normal(jax.random.PRNGKey(8), (1, 4, self.config.d_model)),)
        for model_class in (FlaxWhisperModel, FlaxWhisperForConditionalGeneration):
            with self.subTest(model_class=model_class):
                model = model_class(self.config)
                params = self._random_params(model.params)
                expected = model.decode(ids, encoder_outputs, params=params, return_dict=False)[0]
                kwargs = {
                    "encoder_outputs": encoder_outputs,
                    "params": params,
                    "return_dict": False,
                    "decoder_attention_mask": jnp.ones((1, 6), dtype=jnp.int32),
                }
                prefill = model.decode(
                    ids[:, :2],
                    decoder_position_ids=jnp.array([[0, 1]]),
                    past_key_values=model.init_cache(1, 6, encoder_outputs),
                    **kwargs,
                )
                step = model.decode(
                    ids[:, 2:],
                    decoder_position_ids=jnp.array([[2]]),
                    past_key_values=prefill[1],
                    **kwargs,
                )
                actual = jnp.concatenate((prefill[0], step[0]), axis=1)
                self.assertTrue(jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5))
                for max_length in (0, self.config.max_target_positions + 1):
                    with self.assertRaisesRegex(ValueError, "max_length.*between"):
                        model.init_cache(1, max_length, encoder_outputs)

    def test_decode_preserves_shared_position_ids_across_batch(self):
        ids = jnp.array([[1, 3, 4], [1, 5, 6]])
        encoder_outputs = (jax.random.normal(jax.random.PRNGKey(8), (2, 4, self.config.d_model)),)
        for model_class in (FlaxWhisperModel, FlaxWhisperForConditionalGeneration):
            model = model_class(self.config)
            params = self._random_params(model.params)
            expected = model.decode(ids, encoder_outputs, params=params)[0]
            for positions in (jnp.arange(3), jnp.arange(3)[None]):
                with self.subTest(model_class=model_class, positions=positions.shape):
                    actual = model.decode(ids, encoder_outputs, decoder_position_ids=positions, params=params)[0]
                    self.assertTrue(jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5))

    def test_pipeline_forced_decoder_ids_cover_english_and_multilingual_prompts(self):
        pipeline = object.__new__(FlaxWhisperPipline)
        model = FlaxWhisperForConditionalGeneration(self.config)
        generation_config = model.generation_config
        generation_config.no_timestamps_token_id = 6
        generation_config.task_to_id = {"transcribe": 5}

        generation_config.is_multilingual = False
        english_ids = pipeline.get_forced_decoder_ids(generation_config=generation_config)
        self.assertEqual(english_ids, [(1, 6)])

        generation_config.is_multilingual = True
        multilingual_ids = pipeline.get_forced_decoder_ids(generation_config=generation_config)
        timestamp_ids = pipeline.get_forced_decoder_ids(
            generation_config=generation_config,
            return_timestamps=True,
        )
        self.assertEqual(multilingual_ids, [(2, 5), (3, 6)])
        self.assertEqual(timestamp_ids, [(2, 5)])

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
