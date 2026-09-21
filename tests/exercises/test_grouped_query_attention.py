"""ex012 教师测试；参照仅用于验收，不能从作业导入。

CPU；float64 rtol=1e-8/atol=1e-10，float32 rtol=1e-5/atol=1e-6。
随机种子 1201/1207/1213。参照逐头切片，独立于待测拆头与分组广播路径。
"""
import copy
import math
import unittest

import torch
from torch.nn import functional as F

from exercises.ex007_multi_head_attention.attention import multi_head_attention
from exercises.ex012_grouped_query_attention import gqa as learner


def setUpModule():
    global _old_threads
    _old_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_old_threads)


def sample(B=2, Hq=6, Hkv=2, Tq=3, Tk=5, D=2,
           dtype=torch.float64, layout="contiguous", grad=False):
    generator = torch.Generator().manual_seed(1201)
    C = Hq * D
    bases, values = [], []
    for shape in ((B, Tq, C), (B, Tk, Hkv * D),
                  (B, Tk, Hkv * D), (C, C)):
        if layout == "transpose":
            raw_shape = (*shape[:-2], shape[-1], shape[-2])
        elif layout == "slice":
            raw_shape = (*shape[:-1], shape[-1] * 2 + 1)
        else:
            raw_shape = shape
        base = (torch.randn(raw_shape, generator=generator, dtype=dtype) * .5).requires_grad_(grad)
        if layout == "transpose":
            value = base.transpose(-2, -1)
        elif layout == "slice":
            value = base[..., 1:2 * shape[-1] + 1:2]
        else:
            value = base
        bases.append(base)
        values.append(value)
    allowed = torch.rand(B, Tq, Tk, generator=generator) > .4
    allowed[:, :, 0] = True
    return bases, (*values, Hq, Hkv, allowed)


def reference(Q, K, V, Wo, Hq, Hkv, allowed=None):
    """教师逐头数学参照；不复用学习者的拆合头、广播或分组 helper。"""
    D = Q.shape[-1] // Hq
    outputs, weights = [], []
    for h in range(Hq):
        g = h // (Hq // Hkv)
        q = Q[..., h * D:(h + 1) * D]
        k = K[..., g * D:(g + 1) * D]
        v = V[..., g * D:(g + 1) * D]
        scores = q @ k.transpose(-1, -2) / math.sqrt(D)
        if allowed is not None:
            scores = scores.masked_fill(~allowed, -torch.inf)
        weight = torch.softmax(scores, dim=-1)
        outputs.append(weight @ v)
        weights.append(weight)
    return torch.cat(outputs, dim=-1) @ Wo, torch.stack(weights, dim=1)


def causal_reference(valid):
    B, T = valid.shape
    result = torch.zeros(B, T, T, dtype=torch.bool)
    for b in range(B):
        for i in range(T):
            result[b, i, :i + 1] = valid[b, :i + 1]
    return result


def block_sample(dtype=torch.float64, layout="contiguous", grad=False, Hkv=2):
    # 独立生成器与 fork_rng 避免影响调用方的随机序列。
    with torch.random.fork_rng():
        torch.manual_seed(1207)
        block = learner.GQABlock(12, 6, Hkv, 17, eps=.01, dtype=dtype)
    generator = torch.Generator().manual_seed(1213)
    # 非零偏置、不同 norm 参数和足够大的分支，避免错误被微小初始化掩盖。
    with torch.no_grad():
        for name, parameter in block.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=dtype) * .25)
            if name.endswith("gamma"):
                parameter.add_(1.0)
    shape = (2, 12, 5) if layout == "transpose" else (2, 5, 12)
    base = torch.randn(shape, generator=generator, dtype=dtype).requires_grad_(grad)
    x = base.transpose(-2, -1) if layout == "transpose" else base
    valid = torch.tensor([[True, True, False, True, False],
                          [True, False, True, True, True]])
    return block, base, x, valid


def block_reference(block, x, valid):
    n1 = F.layer_norm(x, (block.C,), block.norm1.gamma,
                      block.norm1.beta, block.norm1.eps)
    attention, _ = reference(n1 @ block.Wq, n1 @ block.Wk, n1 @ block.Wv,
                             block.Wo, block.num_query_heads, block.num_kv_heads,
                             causal_reference(valid))
    u = x + attention
    n2 = F.layer_norm(u, (block.C,), block.norm2.gamma,
                      block.norm2.beta, block.norm2.eps)
    return u + F.relu(n2 @ block.ffn.W1 + block.ffn.b1) @ block.ffn.W2 + block.ffn.b2


class TensorAssertions(unittest.TestCase):
    def assert_tensor(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        tolerance = dict(rtol=1e-5, atol=1e-6) if expected.dtype == torch.float32 else dict(rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(actual, expected, **tolerance)

    def assert_result(self, actual, expected):
        self.assertIsInstance(actual, tuple)
        self.assertEqual(len(actual), 2)
        for value, target in zip(actual, expected):
            self.assert_tensor(value, target)


class ImplementationStatusTest(unittest.TestCase):
    def test_grouped_query_attention_is_implemented(self):
        _, args = sample()
        try:
            learner.grouped_query_attention(*args)
        except NotImplementedError as error:
            self.fail(str(error))

    def test_gqa_block_is_implemented(self):
        block, _, x, valid = block_sample()
        try:
            block(x, valid)
        except NotImplementedError as error:
            self.fail(str(error))


class GroupedQueryAttentionTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        _, args = sample()
        try:
            learner.grouped_query_attention(*args)
        except NotImplementedError:
            raise unittest.SkipTest("先完成 grouped_query_attention")

    def test_mha_mqa_gqa_dtypes_layouts_and_no_mutation(self):
        for Hq, Hkv, D in ((1, 1, 1), (4, 4, 2), (6, 1, 2), (6, 2, 2), (6, 3, 3)):
            for dtype in (torch.float32, torch.float64):
                for layout in ("contiguous", "transpose", "slice"):
                    for masked in (True, False):
                        with self.subTest(Hq=Hq, Hkv=Hkv, D=D, dtype=dtype, layout=layout, masked=masked):
                            bases, args = sample(Hq=Hq, Hkv=Hkv, D=D, dtype=dtype, layout=layout)
                            args = (*args[:-1], args[-1] if masked else None)
                            saved = [base.clone() for base in bases]
                            old_mask = None if args[-1] is None else args[-1].clone()
                            result = learner.grouped_query_attention(*args)
                            self.assert_result(result, reference(*args))
                            if Hkv == Hq:
                                self.assert_result(result, multi_head_attention(*args[:4], Hq, args[-1]))
                            for base, before in zip(bases, saved):
                                self.assertTrue(torch.equal(base, before))
                            if old_mask is not None:
                                self.assertTrue(torch.equal(args[-1], old_mask))

    def test_contiguous_groups_and_distinct_queries_with_shared_kv(self):
        # 前三头读取 KV0，后三头读取 KV1；h%Hkv 会给出明显不同答案。
        q = torch.tensor([[[3., -3., 0., 2., -2., 0.]]], dtype=torch.float64)
        k = torch.tensor([[[1., 2.], [-1., -2.]]], dtype=q.dtype)
        v = torch.tensor([[[10., 100.], [20., 200.]]], dtype=q.dtype)
        args = (q, k, v, torch.eye(6, dtype=q.dtype), 6, 2)
        output, weights = learner.grouped_query_attention(*args)
        self.assert_result((output, weights), reference(*args))
        self.assertGreater(weights[0, 0, 0, 0].item(), .99)
        self.assertLess(weights[0, 1, 0, 0].item(), .01)
        self.assertAlmostEqual(output[0, 0, 2].item(), 15.)
        self.assertAlmostEqual(output[0, 0, 5].item(), 150.)

    def test_scaling_uses_head_dimension_not_total_or_kv_width(self):
        q = torch.ones(1, 1, 12, dtype=torch.float64)
        k = torch.tensor([[[0.] * 4, [1.] * 4]], dtype=q.dtype)
        v = torch.tensor([[[2.] * 4, [8.] * 4]], dtype=q.dtype)
        actual = learner.grouped_query_attention(q, k, v, torch.eye(12, dtype=q.dtype), 6, 2)
        p = math.exp(math.sqrt(2)) / (1 + math.exp(math.sqrt(2)))
        weights = torch.tensor([1 - p, p], dtype=q.dtype).expand(1, 6, 1, 2)
        self.assert_tensor(actual[1], weights)
        self.assert_tensor(actual[0], torch.full_like(q, 2 * (1 - p) + 8 * p))

    def test_batch_equal_heads_mask_broadcast_and_normalization(self):
        _, args = sample(B=4, Hq=4, Hkv=2)
        q, k, v, wo, hq, hkv, allowed = args
        allowed[0, :, 1:] = False
        allowed[1] = True
        allowed[2, :, 0] = False
        allowed[2, :, -1] = True
        actual = learner.grouped_query_attention(*args)
        self.assert_result(actual, reference(*args))
        weights = actual[1]
        self.assertTrue((weights.masked_select(~allowed[:, None]) == 0).all().item())
        self.assert_tensor(weights.sum(dim=-1), torch.ones(4, 4, 3, dtype=q.dtype))
        self.assertTrue((weights[1] > 0).all().item())

    def test_head_independence_and_output_projection(self):
        _, args = sample()
        q, k, v, wo, hq, hkv, allowed = args
        eye = torch.eye(q.shape[-1], dtype=q.dtype)
        baseline = learner.grouped_query_attention(q, k, v, eye, hq, hkv, allowed)
        changed_q = q.clone()
        changed_q[..., 2:4] += 3.
        changed = learner.grouped_query_attention(changed_q, k, v, eye, hq, hkv, allowed)
        heads = [0, 2, 3, 4, 5]
        features = [0, 1, *range(4, 12)]
        self.assert_tensor(changed[1][:, heads], baseline[1][:, heads])
        self.assert_tensor(changed[0][..., features], baseline[0][..., features])
        self.assertGreater((changed[1][:, 1] - baseline[1][:, 1]).abs().max().item(), 1e-3)
        self.assert_result(learner.grouped_query_attention(*args), (baseline[0] @ wo, baseline[1]))

        # 只改 KV0，应只影响连续的 Q0/Q1/Q2，另一组的分布和内容均保持不变。
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[:, 1:, :2] += 2.
        changed_v[..., :2] += 3.
        changed_group = learner.grouped_query_attention(
            q, changed_k, changed_v, eye, hq, hkv, allowed)
        self.assert_tensor(changed_group[1][:, 3:], baseline[1][:, 3:])
        self.assert_tensor(changed_group[0][..., 6:], baseline[0][..., 6:])
        self.assertGreater((changed_group[0][..., :6] - baseline[0][..., :6]).abs().max().item(), 1.)

    def test_all_blocked_row_and_invalid_heads_raise_value_error(self):
        _, args = sample()
        q, k, v, wo, hq, hkv, allowed = args
        allowed[1, 1] = False
        before = allowed.clone()
        with self.assertRaises(ValueError):
            learner.grouped_query_attention(*args)
        self.assertTrue(torch.equal(allowed, before))
        for bad_hq, bad_hkv in ((0, 2), (-2, 2), (6, 0), (6, -1), (5, 1), (6, 4)):
            with self.subTest(Hq=bad_hq, Hkv=bad_hkv), self.assertRaises(ValueError):
                learner.grouped_query_attention(q, k, v, wo, bad_hq, bad_hkv)

    def test_both_outputs_gradients_and_existing_gradients_are_preserved(self):
        for layout in ("contiguous", "transpose", "slice"):
            with self.subTest(layout=layout):
                bases, args = sample(layout=layout, grad=True)
                ref_bases, ref_args = sample(layout=layout, grad=True)
                saved = [base.detach().clone() for base in bases]
                for base in (*bases, *ref_bases):
                    base.grad = torch.full_like(base, .125)
                result = learner.grouped_query_attention(*args)
                expected = reference(*ref_args)
                for base, before in zip(bases, saved):
                    self.assertTrue(torch.equal(base, before))
                    self.assert_tensor(base.grad, torch.full_like(base, .125))
                generator = torch.Generator().manual_seed(1213)
                seeds = [torch.randn(x.shape, generator=generator, dtype=x.dtype) for x in expected]
                sum((x * seed).sum() for x, seed in zip(result, seeds)).backward()
                sum((x * seed).sum() for x, seed in zip(expected, seeds)).backward()
                for name, base, target in zip(("Q", "K", "V", "Wo"), bases, ref_bases):
                    with self.subTest(tensor=name):
                        self.assertIsNotNone(base.grad)
                        self.assert_tensor(base.grad, target.grad)

    def test_shared_value_gradient_sums_all_query_heads(self):
        # 一个 key 时所有权重为 1；每个 KV 的梯度应累加所属的 3 个 Q 头。
        q = torch.ones(1, 1, 6, dtype=torch.float64, requires_grad=True)
        k = torch.ones(1, 1, 2, dtype=q.dtype, requires_grad=True)
        v = torch.tensor([[[10., 20.]]], dtype=q.dtype, requires_grad=True)
        output, _ = learner.grouped_query_attention(q, k, v, torch.eye(6, dtype=q.dtype), 6, 2)
        (output * torch.arange(1., 7., dtype=q.dtype)).sum().backward()
        self.assert_tensor(v.grad, torch.tensor([[[6., 15.]]], dtype=q.dtype))

    def test_large_finite_scores_remain_finite(self):
        q = torch.full((1, 2, 4), 100., dtype=torch.float64)
        k = torch.tensor([[[100., 100.], [99., 99.], [-100., -100.]]], dtype=q.dtype)
        v = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]], dtype=q.dtype)
        args = (q, k, v, torch.eye(4, dtype=q.dtype), 2, 1)
        result = learner.grouped_query_attention(*args)
        self.assertTrue(all(torch.isfinite(value).all().item() for value in result))
        self.assert_result(result, reference(*args))


class GQABlockTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        block, _, x, valid = block_sample()
        try:
            block(x, valid)
        except NotImplementedError:
            raise unittest.SkipTest("先完成 GQABlock.forward 及其 GQA 依赖")

    def test_full_block_dtypes_layouts_modes_and_parameter_identity(self):
        for dtype in (torch.float32, torch.float64):
            for layout in ("contiguous", "transpose"):
                for hkv in (1, 2, 6):
                    with self.subTest(dtype=dtype, layout=layout, Hkv=hkv):
                        block, base, x, valid = block_sample(dtype, layout, Hkv=hkv)
                        values = [base, valid, *block.parameters()]
                        before = [value.detach().clone() for value in values]
                        ids = {name: id(p) for name, p in block.named_parameters()}
                        for training in (True, False):
                            block.train(training)
                            modes = {name: part.training for name, part in block.named_modules()}
                            self.assert_tensor(block(x, valid), block_reference(block, x, valid))
                            self.assertEqual(modes, {name: part.training for name, part in block.named_modules()})
                        self.assertEqual(ids, {name: id(p) for name, p in block.named_parameters()})
                        for value, saved in zip(values, before):
                            self.assertTrue(torch.equal(value, saved))

    def test_input_and_all_parameter_gradients(self):
        block, base, x, valid = block_sample(layout="transpose", grad=True)
        ref_block = copy.deepcopy(block)
        ref_base = base.detach().clone().requires_grad_()
        ref_x = ref_base.transpose(-2, -1)
        values = [base, *block.parameters()]
        ref_values = [ref_base, *ref_block.parameters()]
        for value in (*values, *ref_values):
            value.grad = torch.full_like(value, .125)
        result = block(x, valid)
        expected = block_reference(ref_block, ref_x, valid)
        self.assert_tensor(result, expected)
        for value in values:
            self.assert_tensor(value.grad, torch.full_like(value, .125))
        generator = torch.Generator().manual_seed(1213)
        seed = torch.randn(result.shape, generator=generator, dtype=result.dtype)
        (result * seed).sum().backward()
        (expected * seed).sum().backward()
        names = ["input", *dict(block.named_parameters())]
        for name, value, target in zip(names, values, ref_values):
            with self.subTest(parameter=name):
                self.assertIsNotNone(value.grad)
                self.assert_tensor(value.grad, target.grad)

    def test_future_padding_holes_and_prefix_visibility(self):
        block, _, x, valid = block_sample()
        baseline = block(x, valid)
        changed = x.clone()
        changed[:, 3:] += torch.linspace(-10., 10., x.shape[-1], dtype=x.dtype)
        self.assert_tensor(block(changed, valid)[:, :3], baseline[:, :3])
        changed = x.clone()
        changed[~valid] += torch.linspace(-10., 10., x.shape[-1], dtype=x.dtype)
        self.assert_tensor(block(changed, valid)[valid], baseline[valid])
        self.assert_tensor(block(x[:, :3], valid[:, :3]), baseline[:, :3])
        self.assertTrue((baseline[~valid].abs().sum(dim=-1) > 0).all().item())
        bad_valid = valid.clone()
        bad_valid[1, 0] = False
        with self.assertRaises(ValueError):
            block(x, bad_valid)

    def test_zero_branches_preserve_residual_and_input_gradient(self):
        block, base, x, valid = block_sample(grad=True)
        with torch.no_grad():
            block.Wo.zero_()
            block.ffn.W2.zero_()
            block.ffn.b2.zero_()
        self.assert_tensor(block(x, valid), x)
        block(x, valid).sum().backward()
        self.assert_tensor(base.grad, torch.ones_like(base))


if __name__ == "__main__":
    unittest.main()
