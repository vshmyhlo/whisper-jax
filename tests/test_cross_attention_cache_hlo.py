"""HLO auditing follows execution edges, including while conditions and fusions."""

import unittest

from benchmarks.check_cross_attention_cache_hlo import summarize_hlo


_HLO = """HloModule cache_audit
%body_projection {
  ROOT %k = f16[1,1] dot(%p, %w), metadata={op_name="decoder/layers/0/encoder_attn/k_proj/dot_general"}
}
%body_fusion {
  ROOT %fused = f16[1,1] fusion(%p), kind=kCustom, calls=%body_projection
}
%condition_projection {
  ROOT %k = f16[1,1] dot(%p, %w), metadata={op_name="decoder/layers/1/encoder_attn/k_proj/dot_general"}
}
%body {
  %k = f16[1,1] fusion(%p), kind=kCustom, calls=%body_fusion
  %v = f16[1,1] custom-call(%p, %w), custom_call_target="__cublas$gemm", metadata={op_name="decoder/layers/0/encoder_attn/v_proj/dot_general"}
  %start = (f16[1,1], f16[1,1]) all-reduce-start(%v)
  ROOT %done = f16[1,1] all-reduce-done(%start)
}
%condition {
  %k = f16[1,1] fusion(%p), kind=kCustom, calls=%condition_projection
  ROOT %c = pred[] compare(%index, %limit), direction=LT
}
ENTRY %main {
  %v = f16[1,1] dot(%p, %w), metadata={op_name="layers/1/encoder_attn/encoder_attn.init_cross_attention_cache/v_proj/dot_general"}
  ROOT %loop = f16[1,1] while(%v), condition=%condition, body=%body, metadata={op_name="decoder_token_loop/while"}
}
"""


class CrossAttentionCacheHloTest(unittest.TestCase):
    def test_follows_fusions_in_both_body_and_condition(self):
        summary = summarize_hlo(_HLO)
        self.assertEqual(summary["token_loop_bodies"], ["body"])
        self.assertEqual(summary["token_loop_conditions"], ["condition"])
        self.assertEqual(summary["projection_counts"], {"inside_token_loop": 3, "outside_token_loop": 1})
        locations = summary["cross_attention_kv_projections"]
        self.assertEqual(
            {(item["layer"], item["projection"], item["computation"]) for item in locations["inside_token_loop"]},
            {(0, "k_proj", "body_projection"), (0, "v_proj", "body"), (1, "k_proj", "condition_projection")},
        )
        self.assertEqual(
            locations["outside_token_loop"], [{"layer": 1, "projection": "v_proj", "computation": "main"}]
        )

    def test_counts_asynchronous_collectives(self):
        self.assertEqual(summarize_hlo(_HLO)["collectives"], {"all-reduce": 2})


if __name__ == "__main__":
    unittest.main()
