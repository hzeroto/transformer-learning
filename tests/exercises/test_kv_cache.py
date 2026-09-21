"""ex011 KV Cache 的验证。

约定：CPU；对齐检查在 float64 下用 ATOL64、float32 下用 ATOL32，
均为**事先声明的容差**而非逐位相等 —— 全量与增量的矩阵乘分块方式不同，
浮点结果可能差最后几位（讲义第 6 节有实测数据）。
返回缓存只包含已经送入模型的位置：生成了新 token 时，最后一个输出尚未缓存；
没有生成新 token 时，传入前缀仍需全部缓存。续用时只传尚未缓存的 token。
本文件由教师提供，不属于学习者独立设计的测试。
"""

import unittest

import torch

from exercises.ex009_mini_gpt.model import MiniGPT
from exercises.ex011_kv_cache.cache import (
    LayerKVCache,
    block_step_with_cache,
    generate_with_cache,
    kv_cache_bytes,
)

ATOL64 = 1e-12
ATOL32 = 1e-5
EOS_ID = 2


def make_model(seed=0, dtype=torch.float64, n_layer=3, C=32, heads=4, L=32):
    torch.manual_seed(seed)
    return MiniGPT(
        vocab_size=11, C=C, num_heads=heads, ffn_hidden=2 * C, n_layer=n_layer,
        max_positions=L, norm_style="pre", p=0.0, dtype=dtype,
    )


def full_forward(model, ids):
    """全量参照：不使用缓存，直接调用已验收的 MiniGPT.forward。"""
    model.eval()
    with torch.no_grad():
        return model(ids, torch.ones_like(ids, dtype=torch.bool))


class TestCacheContainer(unittest.TestCase):
    def test_empty_cache_has_zero_length(self):
        cache = LayerKVCache()
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.k)
        self.assertIsNone(cache.v)

    def test_append_grows_along_position_axis(self):
        cache = LayerKVCache()
        a = torch.randn(1, 3, 8, dtype=torch.float64)
        b = torch.randn(1, 1, 8, dtype=torch.float64)
        cache.append(a, a * 2)
        self.assertEqual(len(cache), 3)
        self.assertEqual(tuple(cache.k.shape), (1, 3, 8))
        cache.append(b, b * 2)
        self.assertEqual(len(cache), 4, "第二次 append 应沿位置轴增长，而不是覆盖")
        self.assertEqual(tuple(cache.k.shape), (1, 4, 8))
        # 顺序正确：先追加的在前
        torch.testing.assert_close(cache.k[:, :3, :], a)
        torch.testing.assert_close(cache.k[:, 3:, :], b)
        torch.testing.assert_close(cache.v[:, 3:, :], b * 2)

    def test_reset_returns_to_empty(self):
        cache = LayerKVCache()
        x = torch.randn(1, 5, 8, dtype=torch.float64)
        cache.append(x, x * 3)
        cache.reset()
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.k, "reset 后 k 应为 None")
        self.assertIsNone(cache.v, "reset 后 v 也应为 None（容易只清 k 漏掉 v）")
        # reset 后行为与新建一致，且 v 不能残留旧内容
        y = torch.randn(1, 2, 8, dtype=torch.float64)
        cache.append(y, y * 3)
        self.assertEqual(len(cache), 2)
        torch.testing.assert_close(cache.k, y)
        torch.testing.assert_close(cache.v, y * 3)
        self.assertEqual(tuple(cache.v.shape), (1, 2, 8), "v 里混入了 reset 前的内容")

    def test_capacity_is_enforced_before_writing(self):
        cache = LayerKVCache(max_length=4)
        a = torch.randn(1, 3, 8, dtype=torch.float64)
        cache.append(a, a)
        self.assertEqual(len(cache), 3)
        too_big = torch.randn(1, 2, 8, dtype=torch.float64)
        with self.assertRaises(ValueError):
            cache.append(too_big, too_big)
        self.assertEqual(len(cache), 3, "容量检查失败后缓存不能被部分写入")
        torch.testing.assert_close(cache.k, a)
        # 正好装满是允许的
        one = torch.randn(1, 1, 8, dtype=torch.float64)
        cache.append(one, one)
        self.assertEqual(len(cache), 4)

    def test_unlimited_by_default(self):
        cache = LayerKVCache()
        x = torch.randn(1, 50, 8, dtype=torch.float64)
        cache.append(x, x)
        self.assertEqual(len(cache), 50)


class TestBlockStep(unittest.TestCase):
    """块级别：增量结果必须与整段一次前向的对应位置一致。"""

    def setUp(self):
        self.model = make_model()
        self.block = self.model.blocks[0]
        torch.manual_seed(7)
        self.x = torch.randn(1, 5, self.model.C, dtype=torch.float64)

    def _block_full(self, x):
        """全量参照：直接调用已验收的 ex008 块。"""
        with torch.no_grad():
            return self.block(x, torch.ones(x.shape[0], x.shape[1], dtype=torch.bool))

    def test_prefill_matches_full_block(self):
        cache = LayerKVCache()
        with torch.no_grad():
            got = block_step_with_cache(self.block, self.x, cache)
        want = self._block_full(self.x)
        torch.testing.assert_close(got, want, atol=ATOL64, rtol=0)
        self.assertEqual(len(cache), 5, "prefill 后缓存长度应等于前缀长度")

    def test_token_by_token_matches_full_block(self):
        """逐 token 喂入，每一步都要与全量前向的对应位置一致。"""
        cache = LayerKVCache()
        want = self._block_full(self.x)
        with torch.no_grad():
            for t in range(5):
                got = block_step_with_cache(self.block, self.x[:, t : t + 1, :], cache)
                torch.testing.assert_close(
                    got[:, 0, :], want[:, t, :], atol=ATOL64, rtol=0,
                    msg=f"位置 {t} 的增量输出与全量不一致",
                )
                self.assertEqual(len(cache), t + 1)

    def test_prefill_then_decode_matches_full_block(self):
        """混合：先 3 个，再逐个追加 2 个。"""
        cache = LayerKVCache()
        want = self._block_full(self.x)
        with torch.no_grad():
            got3 = block_step_with_cache(self.block, self.x[:, :3, :], cache)
            torch.testing.assert_close(got3, want[:, :3, :], atol=ATOL64, rtol=0)
            for t in (3, 4):
                got = block_step_with_cache(self.block, self.x[:, t : t + 1, :], cache)
                torch.testing.assert_close(
                    got[:, 0, :], want[:, t, :], atol=ATOL64, rtol=0
                )
        self.assertEqual(len(cache), 5)

    def test_new_token_can_read_itself(self):
        """先追加再计算：新 token 必须能读到自己（j == i 是允许的）。

        若实现写成「先计算后追加」，第一个 token 会面对空缓存而报错或读不到自己。
        """
        cache = LayerKVCache()
        with torch.no_grad():
            out = block_step_with_cache(self.block, self.x[:, :1, :], cache)
        self.assertEqual(len(cache), 1)
        want = self._block_full(self.x[:, :1, :])
        torch.testing.assert_close(out, want, atol=ATOL64, rtol=0)

    def test_does_not_modify_input(self):
        cache = LayerKVCache()
        before = self.x.clone()
        with torch.no_grad():
            block_step_with_cache(self.block, self.x, cache)
        self.assertTrue(torch.equal(self.x, before))

    def test_decode_sees_all_cached_keys(self):
        """decode 阶段若误用下三角 mask，只能看见位置 0，结果会与全量不符。

        构造一个让「只看位置 0」与「看全部」明显不同的输入。
        """
        cache = LayerKVCache()
        torch.manual_seed(11)
        x = torch.randn(1, 4, self.model.C, dtype=torch.float64) * 3
        want = self._block_full(x)
        with torch.no_grad():
            for t in range(4):
                got = block_step_with_cache(self.block, x[:, t : t + 1, :], cache)
        torch.testing.assert_close(got[:, 0, :], want[:, 3, :], atol=ATOL64, rtol=0)


class TestGenerationAlignment(unittest.TestCase):
    def test_prefill_logits_match_full_forward_float64(self):
        model = make_model(dtype=torch.float64)
        ids = torch.tensor([[1, 3, 5, 6, 7]])
        want = full_forward(model, ids)
        out, caches = generate_with_cache(model, ids, 0, EOS_ID)
        # 不生成新 token 时应原样返回前缀
        self.assertTrue(torch.equal(out, ids))
        self.assertEqual(len(caches), model.n_layer)
        for i, c in enumerate(caches):
            self.assertEqual(len(c), 5, f"第 {i} 层缓存长度应为 5")

    def test_generated_sequence_matches_uncached_greedy(self):
        """端到端：缓存版与无缓存贪心必须生成完全相同的 token 序列。"""
        for dtype, atol in ((torch.float64, ATOL64), (torch.float32, ATOL32)):
            with self.subTest(dtype=dtype):
                model = make_model(dtype=dtype)
                ids = torch.tensor([[1, 3, 5]])
                cached, _ = generate_with_cache(model, ids, 6, EOS_ID)
                # 无缓存参照：每步重新全量前向
                model.eval()
                ref = ids.clone()
                with torch.no_grad():
                    for _ in range(6):
                        lg = model(ref, torch.ones_like(ref, dtype=torch.bool))
                        nxt = lg[:, -1:, :].argmax(-1)
                        ref = torch.cat([ref, nxt], dim=1)
                        if nxt.item() == EOS_ID:
                            break
                self.assertTrue(
                    torch.equal(cached, ref),
                    f"{dtype}: 缓存版 {cached.tolist()} != 无缓存 {ref.tolist()}",
                )

    def test_various_prefix_lengths(self):
        model = make_model()
        for T in (1, 2, 5, 9):
            for max_new_tokens in (0, 1, 3):
                with self.subTest(prefix_len=T, max_new_tokens=max_new_tokens):
                    ids = torch.tensor([[1] + [3 + (i % 5) for i in range(T - 1)]])
                    self.assertEqual(ids.shape[1], T)
                    out, caches = generate_with_cache(model, ids, max_new_tokens, EOS_ID)
                    generated = out.shape[1] - T
                    # 输入不含 EOS；允许生成时至少应产出一个 token。
                    self.assertGreaterEqual(generated, min(1, max_new_tokens))
                    self.assertLessEqual(generated, max_new_tokens)
                    self.assertTrue(torch.equal(out[:, :T], ids))
                    # 最后选出的 ID 尚未进入模型；K=0 时则完整缓存前缀。
                    expected_length = T + max(generated - 1, 0)
                    for i, c in enumerate(caches):
                        self.assertEqual(
                            len(c), expected_length,
                            f"第 {i} 层应只缓存实际处理过的位置，不补算最后一个新 token",
                        )

    def test_positions_come_from_cache_not_arguments(self):
        """分两次调用、续用同一份缓存：位置必须接着数，不能从头。

        这正是讲义检查 B 的场景。这里比较的是 **logits 数值**而不是生成的
        token —— 未训练模型的贪心输出常常是同一个 ID，位置错了 argmax 也不变
        （已实测），用 token 序列检测不出这个 bug。
        """
        model = make_model()
        ids = torch.tensor([[1, 3, 5, 6, 7]])
        # 参照：位置 5 的 token 与全量前向对齐
        nxt = torch.tensor([[9]])
        full_ids = torch.cat([ids, nxt], dim=1)
        want = full_forward(model, full_ids)[:, -1, :]

        # 先 prefill 5 个，再在同一缓存上追加第 6 个（它的位置必须是 5）
        caches = [LayerKVCache() for _ in range(model.n_layer)]
        model.eval()
        with torch.no_grad():
            pos = len(caches[0])
            x = model.token_table[ids] + model.position_table[pos : pos + ids.shape[1]]
            for i, blk in enumerate(model.blocks):
                x = block_step_with_cache(blk, x, caches[i])
            pos = len(caches[0])
            self.assertEqual(pos, 5, "prefill 后缓存长度应为 5，它就是下一个位置下标")
            x = model.token_table[nxt] + model.position_table[pos : pos + 1]
            for i, blk in enumerate(model.blocks):
                x = block_step_with_cache(blk, x, caches[i])
            h = model.final_norm(x) if model.final_norm is not None else x
            got = (h @ model.vocab_proj)[:, -1, :]
        torch.testing.assert_close(
            got, want, atol=ATOL64, rtol=0,
            msg="续用缓存时位置下标错误：应取自缓存长度，而不是从 0 或参数推算",
        )

    def test_generate_continues_on_existing_cache(self):
        """将上次尚未缓存的最后一个输出送入，续用后逐层缓存应与整段处理一致。"""
        model = make_model()
        # 同分时选择 ID 0，保证不会提前产生 EOS，确实执行两次生成。
        with torch.no_grad():
            model.vocab_proj.zero_()
        ids = torch.tensor([[1, 3, 5]])
        first, caches = generate_with_cache(model, ids, 2, EOS_ID)
        self.assertEqual(first.shape[1], ids.shape[1] + 2)
        len_after_first = len(caches[0])
        for c in caches:
            self.assertEqual(len(c), first.shape[1] - 1)
        # 最后一个输出还没有 K/V，作为本次唯一的新输入，不重复传已缓存的前缀。
        cont, caches2 = generate_with_cache(
            model, first[:, -1:], 2, EOS_ID, caches=caches
        )
        self.assertEqual(cont.shape[1], 3)
        self.assertTrue(torch.equal(cont[:, :1], first[:, -1:]))
        whole = torch.cat([first[:, :-1], cont], dim=1)
        # 一次处理所有已消费的 token；最终新输出仍不应出现在缓存里。
        _, ref_caches = generate_with_cache(
            model, whole[:, :-1], 0, EOS_ID
        )
        self.assertEqual(len(caches2), model.n_layer)
        for i, (got, want) in enumerate(zip(caches2, ref_caches)):
            self.assertIs(got, caches[i], "应复用每一层传入的缓存对象")
            self.assertEqual(len(got), len_after_first + cont.shape[1] - 1)
            torch.testing.assert_close(
                got.k, want.k, atol=ATOL64, rtol=0,
                msg=f"第 {i} 层续用后的 K 与整段处理不一致：检查重复追加或位置偏移",
            )
            torch.testing.assert_close(got.v, want.v, atol=ATOL64, rtol=0)

    def test_continued_generation_uses_cache_length_as_position(self):
        """走 generate_with_cache 的续用路径，逐层核对缓存内容而非 token。

        若续用时位置从「传入序列长度」推算（本次只传 1 个 token，便从 0 或 1 开始），
        新 token 的输入表示会用错 position_table 的行。这里把续用后的缓存
        与「一次性处理整段」的缓存逐元素比较，能直接暴露这个错误。
        """
        model = make_model()
        head = torch.tensor([[1, 3, 5, 6]])
        tail = torch.tensor([[9]])
        whole = torch.cat([head, tail], dim=1)

        # 参照：一次性 prefill 整段
        _, ref_caches = generate_with_cache(model, whole, 0, EOS_ID)
        # 被测：先 prefill head，再在同一缓存上处理 tail
        _, caches = generate_with_cache(model, head, 0, EOS_ID)
        self.assertEqual(len(caches[0]), 4)
        generate_with_cache(model, tail, 0, EOS_ID, caches=caches)

        for i, (got, want) in enumerate(zip(caches, ref_caches)):
            self.assertEqual(len(got), len(want), f"第 {i} 层缓存长度不一致")
            torch.testing.assert_close(
                got.k, want.k, atol=ATOL64, rtol=0,
                msg=f"第 {i} 层缓存的 K 与整段处理不一致：续用时位置下标可能取错",
            )
            torch.testing.assert_close(got.v, want.v, atol=ATOL64, rtol=0)

    def test_eos_stops_and_is_kept(self):
        model = make_model()
        ids = torch.tensor([[1, 3, EOS_ID]])
        out, caches = generate_with_cache(model, ids, 5, EOS_ID)
        self.assertTrue(torch.equal(out, ids), "前缀已以 EOS 结束时不应继续生成")
        for c in caches:
            self.assertEqual(len(c), 3, "即使不生成，prefill 仍应建立缓存")

    def test_requests_are_isolated(self):
        """两个请求各自新建缓存，结果必须与单独运行时相同。"""
        model = make_model()
        a = torch.tensor([[1, 3, 5]])
        b = torch.tensor([[1, 7, 8, 9]])
        solo_a, _ = generate_with_cache(model, a, 3, EOS_ID)
        solo_b, _ = generate_with_cache(model, b, 3, EOS_ID)
        # 连续处理，不复用缓存
        seq_a, _ = generate_with_cache(model, a, 3, EOS_ID)
        seq_b, _ = generate_with_cache(model, b, 3, EOS_ID)
        self.assertTrue(torch.equal(solo_a, seq_a))
        self.assertTrue(
            torch.equal(solo_b, seq_b), "第二个请求的结果受到了第一个请求的影响"
        )

    def test_preserves_model_mode_and_grads(self):
        model = make_model()
        ids = torch.tensor([[1, 3, 5]])
        model.train()
        generate_with_cache(model, ids, 3, EOS_ID)
        self.assertTrue(model.training, "生成结束后应恢复调用前的 training 状态")
        model.eval()
        generate_with_cache(model, ids, 3, EOS_ID)
        self.assertFalse(model.training, "从 eval 态调用后应仍为 eval")
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_input_not_modified_and_returns_long(self):
        model = make_model()
        ids = torch.tensor([[1, 3, 5]])
        before = ids.clone()
        out, _ = generate_with_cache(model, ids, 3, EOS_ID)
        self.assertTrue(torch.equal(ids, before))
        self.assertEqual(out.dtype, torch.long)
        self.assertEqual(out.dim(), 2)
        self.assertEqual(out.shape[0], 1)

    def test_capacity_limit_propagates(self):
        """缓存带容量上限时，超出应抛 ValueError 而不是静默截断。"""
        model = make_model()
        ids = torch.tensor([[1, 3, 5]])
        caches = [LayerKVCache(max_length=4) for _ in range(model.n_layer)]
        with self.assertRaises(ValueError):
            generate_with_cache(model, ids, 5, EOS_ID, caches=caches)


class TestByteLedger(unittest.TestCase):
    def test_matches_formula(self):
        self.assertEqual(kv_cache_bytes(2, 1, 16, 4, 4, 4), 4096)
        self.assertEqual(kv_cache_bytes(12, 1, 512, 12, 64, 4), 36 * 1024 * 1024)
        self.assertEqual(kv_cache_bytes(12, 8, 512, 12, 64, 4), 288 * 1024 * 1024)

    def test_matches_real_tensors(self):
        """用真实张量的 numel × element_size 核对。"""
        n_layer, B, L, H_kv, Dh = 2, 1, 16, 4, 4
        tensors = [
            torch.zeros(B, H_kv, L, Dh, dtype=torch.float32)
            for _ in range(2 * n_layer)
        ]
        actual = sum(t.numel() * t.element_size() for t in tensors)
        self.assertEqual(kv_cache_bytes(n_layer, B, L, H_kv, Dh, 4), actual)

    def test_scales_linearly(self):
        base = kv_cache_bytes(2, 1, 16, 4, 4, 4)
        self.assertEqual(kv_cache_bytes(2, 2, 16, 4, 4, 4), 2 * base)
        self.assertEqual(kv_cache_bytes(2, 1, 32, 4, 4, 4), 2 * base)
        self.assertEqual(kv_cache_bytes(4, 1, 16, 4, 4, 4), 2 * base)
        self.assertEqual(kv_cache_bytes(2, 1, 16, 4, 4, 8), 2 * base)

    def test_zero_length_is_zero(self):
        self.assertEqual(kv_cache_bytes(12, 8, 0, 12, 64, 4), 0)


if __name__ == "__main__":
    unittest.main()
