"""Opt-in integration tests against native OpenAI checkpoint/audio references.

Set WHISPER_REAL_PARITY_DIR to the output of generate_real_whisper_reference.py.
Expected values come from original Torch decoding, never from this JAX model.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import jax

from benchmarks.check_real_audio_parity import evaluate


@unittest.skipUnless(
    os.environ.get("WHISPER_REAL_PARITY_DIR"), "Set WHISPER_REAL_PARITY_DIR for real-checkpoint tests"
)
class RealAudioParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hlo = tempfile.TemporaryDirectory(prefix="whisper-parity-hlo-")
        cls.addClassCleanup(cls.hlo.cleanup)
        cls.report = evaluate(
            Path(os.environ["WHISPER_REAL_PARITY_DIR"]),
            hlo_directory=Path(cls.hlo.name) if jax.local_device_count() >= 2 else None,
        )
        cls.baseline = json.loads(
            (Path(__file__).resolve().parents[1] / "benchmarks/reports/real_audio_parity.json").read_text()
        )

    def test_real_checkpoint_audio_and_precision_coverage(self):
        self.assertEqual(len(self.report["cases"]), 36)
        names = {
            f"{model}_{audio}_{mode}"
            for model in ("tiny.en", "tiny")
            for audio in ("jfk", "librispeech_1", "librispeech_2")
            for mode in ("text", "timestamps")
        }
        for dtype in ("float32", "float16", "bfloat16"):
            cases = [case for case in self.report["cases"] if case["jax_activation_dtype"] == dtype]
            self.assertEqual({case["case"] for case in cases}, names)
            self.assertTrue(all(case["jax_params_dtype"] == "float32" for case in cases))
            self.assertTrue(all(case["jax_kv_cache_dtype"] == dtype for case in cases))
        self.assertTrue(self.report["jit"])
        self.assertEqual(self.report["reference"]["torch_activation_dtype"], "float16")
        self.assertEqual(self.report["reference"]["torch_params_dtype"], "float32")
        self.assertFalse(self.report["native_runtime"]["torch_loaded_in_jax_process"])
        self.assertEqual(self.report["native_runtime"]["torch_imports_in_runtime_package"], [])

    def test_fp32_and_fp16_tokens_match_native_torch_exactly(self):
        for case in self.report["cases"]:
            if case["jax_activation_dtype"] == "bfloat16":
                continue
            with self.subTest(case=case["case"], dtype=case["jax_activation_dtype"]):
                self.assertEqual(case["jax_tokens"], case["reference_tokens"])
                self.assertEqual(case["jax_text"], case["reference_text"])
                self.assertLess(case["logits_error"]["max_abs"], 0.2)
                self.assertLess(case["logits_error"]["mean_abs"], 0.04)
                self.assertLess(case["logits_error"]["relative_l2"], 0.004)
                self.assertLess(case["encoder_error"]["relative_l2"], 0.012)

    def test_bf16_numerical_error_and_known_token_drift_do_not_regress(self):
        # BF16 is not token-identical to native FP16. Enforce the measured per-case
        # drift ceilings while allowing improvements; do not call bounded drift
        # exact parity or require JAX to reproduce its previous wrong tokens.
        baseline = {
            case["case"]: case for case in self.baseline["cases"] if case["jax_activation_dtype"] == "bfloat16"
        }
        for case in self.report["cases"]:
            if case["jax_activation_dtype"] != "bfloat16":
                continue
            with self.subTest(case=case["case"]):
                original = baseline[case["case"]]
                self.assertEqual(case["checkpoint_sha256"], original["checkpoint_sha256"])
                self.assertEqual(case["reference_tokens"], original["reference_tokens"])
                self.assertLessEqual(case["token_edit_distance"], original["token_edit_distance"])
                self.assertLessEqual(case["word_edit_distance"], original["word_edit_distance"])
                self.assertLess(case["logits_error"]["max_abs"], 1.2)
                self.assertLess(case["logits_error"]["mean_abs"], 0.22)
                self.assertLess(case["logits_error"]["relative_l2"], 0.025)
                self.assertLess(case["encoder_error"]["relative_l2"], 0.06)

    @unittest.skipUnless(jax.local_device_count() >= 2, "Requires two devices; use XLA_FLAGS on CPU")
    def test_data_parallel_forward_hlo_has_no_collectives_or_embedding_matmuls(self):
        for name, audit in self.report["hlo"].items():
            with self.subTest(executable=name):
                self.assertFalse(audit["has_host_callback"])
                self.assertEqual(audit["embedding_lookup_dots"], 0)
                for op in ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute"):
                    self.assertEqual(audit["operations"][op], 0)
                if "forward_dp" in name:
                    self.assertGreaterEqual(audit["devices"], 2)
                    self.assertTrue(audit["distinct_replica_inputs"])
                elif jax.default_backend() == "cpu":
                    self.assertEqual(audit["operations"]["while"], 1)
                    self.assertEqual(
                        audit["cross_attention_kv_projection_dots"], {"entry": 8, "other_computations": 0}
                    )


if __name__ == "__main__":
    unittest.main()
