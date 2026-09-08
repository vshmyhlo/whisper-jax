"""Embedding gathers preserve the previous one-hot lookup semantics."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from whisper_jax.layers import Embed


class EmbeddingTest(unittest.TestCase):
    def test_compiled_gather_matches_one_hot_including_padding(self):
        weights = jax.random.normal(jax.random.PRNGKey(21), (16, 8))
        for dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
            for ids in (jnp.arange(16), jnp.array([[-1, 0, 7], [15, 16, 3]])):
                with self.subTest(dtype=dtype, shape=ids.shape):
                    gather = Embed(16, 8, dtype=dtype, one_hot=False)
                    original = Embed(16, 8, dtype=dtype, one_hot=True)
                    params = {"params": {"embedding": weights}}
                    actual = jax.jit(gather.apply)(params, ids)
                    expected = jax.jit(original.apply)(params, ids)
                    self.assertEqual(actual.dtype, dtype)
                    np.testing.assert_array_equal(actual, expected)

    def test_gather_gradient_matches_one_hot_for_repeated_and_invalid_ids(self):
        ids = jnp.array([[-1, 0, 7, 7, 16]])
        weights = jnp.arange(128, dtype=jnp.float32).reshape(16, 8) / 128

        def gradient(one_hot):
            layer = Embed(16, 8, one_hot=one_hot)
            return jax.jit(jax.grad(lambda w: jnp.square(layer.apply({"params": {"embedding": w}}, ids)).sum()))(
                weights
            )

        np.testing.assert_array_equal(gradient(False), gradient(True))

    @unittest.skipUnless(jax.local_device_count() >= 2, "Requires two devices; use XLA_FLAGS on CPU")
    def test_gather_with_vocabulary_sharded_parameters(self):
        mesh = Mesh(np.array(jax.local_devices()[:2]), ("model",))
        weights = jnp.arange(128, dtype=jnp.float32).reshape(16, 8) / 128
        ids = jnp.array([[0, 7, 8, 15, -1, 16]])
        gather = Embed(16, 8, dtype=jnp.bfloat16, one_hot=False)
        original = Embed(16, 8, dtype=jnp.bfloat16, one_hot=True)
        sharded_weights = jax.device_put(weights, NamedSharding(mesh, PartitionSpec("model", None)))
        apply = jax.jit(
            lambda w, x: gather.apply({"params": {"embedding": w}}, x),
            in_shardings=(NamedSharding(mesh, PartitionSpec("model", None)), NamedSharding(mesh, PartitionSpec())),
            out_shardings=NamedSharding(mesh, PartitionSpec()),
        )
        expected = original.apply({"params": {"embedding": weights}}, ids)
        np.testing.assert_array_equal(apply(sharded_weights, ids), expected)


if __name__ == "__main__":
    unittest.main()
