"""ex014 第一阶段教师测试：整数账本精确相等，不测硬件耗时。

参照从 ex013 实际权重 shape 按输出元素计数，不调用待测计算辅助函数。
模型仅在 CPU float32 下构造，种子 1401；无需执行前向或设置数值容差。
空框架：1 项明确失败、9 项跳过；实现完成后必须全部运行，无跳过。
"""
import unittest

import torch

from exercises.ex013_llama_style.model import LlamaLM
from exercises.ex014_model_cost import cost as learner


CONFIG = dict(B=2, C=24, Hq=6, Hkv=2, G=64, layers=2, N=17)
PARTS = ("q_proj", "kv_proj", "score", "value_read", "out_proj", "ffn", "vocab")


def call_learner(**changes):
    return learner.estimate_matmul_flops(**(CONFIG | dict(n=7, S=7) | changes))


def reference_from_weights(config):
    """教师参照：逐层枚举实际矩阵，每个输出元素数输入宽度次乘加。"""
    with torch.random.fork_rng():
        torch.manual_seed(1401)
        model = LlamaLM(
            config["N"], config["C"], config["Hq"], config["Hkv"],
            config["G"], config["layers"], max(32, config["S"]), dtype=torch.float32,
        )
    B, n, S = (config[key] for key in ("B", "n", "S"))
    counts = dict.fromkeys(PARTS, 0)

    def count_projection(weight):
        input_width, output_width = weight.shape
        return sum(2 * input_width for _ in range(B * n * output_width))

    for block in model.blocks:
        counts["q_proj"] += count_projection(block.Wq)
        counts["kv_proj"] += count_projection(block.Wk) + count_projection(block.Wv)
        counts["out_proj"] += count_projection(block.Wo)
        for weight in (block.ffn.Wgate, block.ffn.Wup, block.ffn.Wdown):
            counts["ffn"] += count_projection(weight)
        for _batch in range(B):
            for _head in range(block.num_query_heads):
                for _query in range(n):
                    counts["score"] += sum(2 * block.head_dim for _key in range(S))
                    counts["value_read"] += sum(2 * S for _feature in range(block.head_dim))
    counts["vocab"] = count_projection(model.vocab_proj)
    return counts | {"total": sum(counts.values())}


class ImplementationStatusTest(unittest.TestCase):
    def test_implemented(self):
        try:
            call_learner()
        except NotImplementedError:
            self.fail("请先填写 exercises/ex014_model_cost/cost.py 的 estimate_matmul_flops")


class MatmulCostTest(unittest.TestCase):
    def setUp(self):
        try:
            call_learner()
        except NotImplementedError:
            self.skipTest("成本账本尚未实现；另有实现状态测试明确失败")

    def assert_ledger(self, actual, expected):
        self.assertIsInstance(actual, dict)
        self.assertEqual(set(actual), set(PARTS) | {"total"})
        for key in (*PARTS, "total"):
            self.assertIs(type(actual[key]), int, key)
            self.assertEqual(actual[key], expected[key], key)
        self.assertEqual(actual["total"], sum(actual[key] for key in PARTS))

    def test_known_prefill_breakdown(self):
        self.assert_ledger(call_learner(), dict(
            q_proj=32256, kv_proj=21504, score=9408, value_read=9408,
            out_proj=32256, ffn=258048, vocab=11424, total=374304,
        ))

    def test_known_decode_breakdown(self):
        self.assert_ledger(call_learner(n=1, S=8), dict(
            q_proj=4608, kv_proj=3072, score=1536, value_read=1536,
            out_proj=4608, ffn=36864, vocab=1632, total=53856,
        ))

    def test_actual_weights_across_distinct_configs(self):
        cases = [
            dict(B=3, n=4, S=11, C=24, Hq=6, Hkv=2, G=31, layers=3, N=13),
            dict(B=1, n=5, S=5, C=16, Hq=4, Hkv=4, G=19, layers=1, N=7),
            dict(B=2, n=2, S=9, C=12, Hq=3, Hkv=1, G=17, layers=4, N=11),
        ]
        for config in cases:
            with self.subTest(config=config):
                self.assert_ledger(learner.estimate_matmul_flops(**config), reference_from_weights(config))

    def test_longer_history_changes_only_attention_reads(self):
        short, long = call_learner(n=1, S=8), call_learner(n=1, S=24)
        for key in PARTS:
            multiplier = 3 if key in ("score", "value_read") else 1
            self.assertEqual(long[key], multiplier * short[key], key)

    def test_chunk_size_at_fixed_total_length(self):
        single, chunk = call_learner(n=1, S=12), call_learner(n=3, S=12)
        for key in (*PARTS, "total"):
            self.assertEqual(chunk[key], 3 * single[key], key)

    def test_prefill_length_scales_dense_attention_quadratically(self):
        short, long = call_learner(n=4, S=4), call_learner(n=8, S=8)
        for key in PARTS:
            multiplier = 4 if key in ("score", "value_read") else 2
            self.assertEqual(long[key], multiplier * short[key], key)

    def test_kv_heads_do_not_reduce_query_head_reads(self):
        mqa = call_learner(Hkv=1)
        for heads in (2, 3, 6):
            actual = call_learner(Hkv=heads)
            for key in PARTS:
                multiplier = heads if key == "kv_proj" else 1
                self.assertEqual(actual[key], multiplier * mqa[key], (heads, key))

    def test_layers_do_not_repeat_vocabulary_projection(self):
        one, three = call_learner(layers=1), call_learner(layers=3)
        for key in PARTS:
            self.assertEqual(three[key], one[key] * (1 if key == "vocab" else 3), key)

    def test_batch_scales_every_part(self):
        one, three = call_learner(B=1), call_learner(B=3)
        for key in (*PARTS, "total"):
            self.assertEqual(three[key], 3 * one[key], key)


if __name__ == "__main__":
    unittest.main()
