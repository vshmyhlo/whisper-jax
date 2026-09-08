"""Opt-in large-v3 comparisons against unchanged native Torch FP16 decoding."""

import json
import os
import unittest
from pathlib import Path

from benchmarks.check_real_audio_parity import evaluate


@unittest.skipUnless(
    os.environ.get("WHISPER_LARGE_V3_PARITY_DIR"), "Set WHISPER_LARGE_V3_PARITY_DIR for large-v3 checkpoint tests"
)
class LargeV3ParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = evaluate(Path(os.environ["WHISPER_LARGE_V3_PARITY_DIR"]))
        cls.baseline = json.loads(
            (Path(__file__).resolve().parents[1] / "benchmarks/reports/large_v3_audio_parity.json").read_text()
        )

    def test_real_audio_matrix_uses_jit_and_original_reference_precision(self):
        expected = {
            (f"large-v3_{audio}_{mode}", dtype)
            for audio in ("jfk", "librispeech_1", "librispeech_2")
            for mode in ("text", "timestamps")
            for dtype in ("float32", "float16", "bfloat16")
        }
        self.assertEqual(len(self.report["cases"]), 18)
        self.assertEqual({(c["case"], c["jax_activation_dtype"]) for c in self.report["cases"]}, expected)
        self.assertTrue(self.report["jit"])
        self.assertEqual(self.report["reference"]["torch_activation_dtype"], "float16")
        self.assertEqual(self.report["reference"]["torch_params_dtype"], "float32")
        self.assertFalse(self.report["native_runtime"]["torch_loaded_in_jax_process"])
        for case in self.report["cases"]:
            self.assertEqual(case["checkpoint"], "large-v3")
            self.assertEqual(case["jax_params_dtype"], "float32")
            self.assertEqual(case["jax_kv_cache_dtype"], case["jax_activation_dtype"])
            self.assertLess(case["features_error"]["max_abs"], 1e-4)

    def test_token_and_numerical_differences_do_not_regress(self):
        baseline = {(c["case"], c["jax_activation_dtype"]): c for c in self.baseline["cases"]}
        for case in self.report["cases"]:
            with self.subTest(case=case["case"], dtype=case["jax_activation_dtype"]):
                previous = baseline[case["case"], case["jax_activation_dtype"]]
                self.assertEqual(case["checkpoint_sha256"], previous["checkpoint_sha256"])
                self.assertEqual(case["reference_tokens"], previous["reference_tokens"])
                # A previously exact case must remain exact. Known differences may
                # improve; never require JAX to reproduce its prior differing tokens.
                self.assertLessEqual(case["token_edit_distance"], previous["token_edit_distance"])
                self.assertLessEqual(case["word_edit_distance"], previous["word_edit_distance"])
                for stage in ("encoder_error", "logits_error"):
                    for metric in ("max_abs", "mean_abs", "rmse", "relative_l2"):
                        self.assertLessEqual(case[stage][metric], previous[stage][metric] * 1.25 + 1e-6)


if __name__ == "__main__":
    unittest.main()
