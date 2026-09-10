"""完整 MHA 综合测试：按 head 切片并选取可读 key 独立构造参照。"""

import math
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from exercises.ex006_single_head_attention.attention import single_head_self_attention
    from exercises.ex007_multi_head_attention import attention as learner
    from tests.exercises.test_single_head_attention import attention_reference, causal_reference


def sample(shape=(3, 2, 3, 5, 6), dtype=None, layout="contiguous", requires_grad=False):
    if dtype is None:
        dtype = torch.float64
    b, h, tq, tk, c = shape
    generator = torch.Generator().manual_seed(83)
    bases, values = [], []
    for logical in ((b, tq, c), (b, tk, c), (b, tk, c), (c, c)):
        if layout == "sliced":
            raw = (*logical[:-1], 2 * logical[-1] + 1)
        elif layout == "transposed":
            raw = (*logical[:-2], logical[-1], logical[-2])
        else:
            raw = logical
        base = (torch.randn(raw, generator=generator, dtype=dtype) * 0.5).requires_grad_(requires_grad)
        if layout == "sliced":
            value = base[..., 1:1 + 2 * logical[-1]:2]
        elif layout == "transposed":
            value = base.transpose(-2, -1)
        else:
            value = base
        bases.append(base)
        values.append(value)
    mask = torch.rand(b, tq, tk, generator=generator) > 0.4
    mask[:, :, 0] = True
    return bases, (*values, h, mask)


def reference(q, k, v, wo, h, allowed=None):
    # 不调用待测拆合头，也不使用相同的四维广播路径。
    dh = q.shape[-1] // h
    outputs, weights = [], []
    for head in range(h):
        start, stop = head * dh, (head + 1) * dh
        out, weight = attention_reference(
            q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop], allowed
        )
        outputs.append(out)
        weights.append(weight)
    return torch.cat(outputs, dim=-1) @ wo, torch.stack(weights, dim=1)


def self_sample(requires_grad=False):
    generator = torch.Generator().manual_seed(89)
    values = [
        (torch.randn(size, generator=generator, dtype=torch.float64) * 0.5).requires_grad_(requires_grad)
        for size in ((2, 5, 6), (6, 6), (6, 6), (6, 6), (6, 6))
    ]
    valid = torch.tensor([[True, True, True, True, False], [True, True, False, False, False]])
    return (*values, valid, 3)


def self_reference(x, wq, wk, wv, wo, valid, h):
    return reference(x @ wq, x @ wk, x @ wv, wo, h, causal_reference(valid))


class ImplementationStatusTest(unittest.TestCase):
    def test_multi_head_is_implemented(self):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境。")
        _, args = sample()
        try:
            learner.multi_head_attention(*args)
        except NotImplementedError as error:
            self.fail(str(error))

    def test_self_attention_is_implemented(self):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境。")
        try:
            learner.multi_head_self_attention(*self_sample())
        except NotImplementedError as error:
            self.fail(str(error))


class TensorAssertions(unittest.TestCase):
    def assert_tensor(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        tolerance = (
            dict(rtol=1e-5, atol=1e-6)
            if expected.dtype == torch.float32
            else dict(rtol=1e-10, atol=1e-12)
        )
        torch.testing.assert_close(actual, expected, **tolerance)

    def assert_result(self, actual, expected):
        self.assertIsInstance(actual, tuple)
        self.assertEqual(len(actual), 2)
        for value, target in zip(actual, expected):
            self.assert_tensor(value, target)


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class MultiHeadTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        _, args = sample()
        try:
            learner.multi_head_attention(*args)
        except NotImplementedError:
            raise unittest.SkipTest("先完成 multi_head_attention")

    def test_forward_dtypes_lengths_layouts_and_no_mutation(self):
        cases = ((3, 2, 3, 5, 6), (2, 2, 4, 3, 4), (1, 1, 1, 1, 1), (2, 4, 2, 3, 4))
        for dtype in (torch.float32, torch.float64):
            for shape in cases:
                for layout in ("contiguous", "transposed", "sliced"):
                    for masked in (False, True):
                        with self.subTest(dtype=dtype, shape=shape, layout=layout, masked=masked):
                            bases, args = sample(shape, dtype, layout)
                            if not masked:
                                args = (*args[:-1], None)
                            before = [base.clone() for base in bases]
                            mask_before = None if args[-1] is None else args[-1].clone()
                            self.assert_result(learner.multi_head_attention(*args), reference(*args))
                            for base, saved in zip(bases, before):
                                self.assertTrue(torch.equal(base, saved))
                            if mask_before is not None:
                                self.assertTrue(torch.equal(args[-1], mask_before))

    def test_scaling_uses_per_head_width(self):
        q = torch.ones(1, 1, 4, dtype=torch.float64)
        k = torch.tensor([[[0., 0., 0., 0.], [1., 1., 1., 1.]]], dtype=q.dtype)
        v = torch.tensor([[[2., 3., 4., 5.], [8., 9., 10., 11.]]], dtype=q.dtype)
        out, weights = learner.multi_head_attention(q, k, v, torch.eye(4, dtype=q.dtype), 2)
        p = math.exp(math.sqrt(2)) / (1 + math.exp(math.sqrt(2)))
        expected = torch.tensor([[[[1 - p, p]], [[1 - p, p]]]], dtype=q.dtype)
        self.assert_tensor(weights, expected)
        self.assert_tensor(out, v[:, :1] * (1 - p) + v[:, 1:] * p)

    def test_batch_mask_is_shared_across_heads_not_batches(self):
        # B==H 时漏掉 unsqueeze(1) 也可能不报错，但会把样本权限当作头权限。
        _, args = sample((2, 2, 3, 4, 6))
        q, k, v, wo, h, mask = args
        mask[0, :, 1:] = False
        mask[1] = True
        result = learner.multi_head_attention(q, k, v, wo, h, mask)
        self.assert_result(result, reference(q, k, v, wo, h, mask))
        weights = result[1]
        self.assert_tensor(weights.sum(dim=-1), torch.ones(2, 2, 3, dtype=q.dtype))
        self.assertTrue((weights >= 0).all().item())
        self.assertTrue((weights[0, :, :, 1:] == 0).all().item())
        self.assertTrue((weights[1] > 0).all().item())

    def test_heads_can_use_different_reading_distributions(self):
        q = torch.ones(1, 1, 2, dtype=torch.float64)
        k = torch.tensor([[[3., 0.], [0., 3.]]], dtype=q.dtype)
        v = torch.tensor([[[2., 20.], [8., 80.]]], dtype=q.dtype)
        args = (q, k, v, torch.eye(2, dtype=q.dtype), 2)
        result = learner.multi_head_attention(*args)
        self.assert_result(result, reference(*args))
        self.assertGreater(result[1][0, 0, 0, 0].item(), 0.9)
        self.assertGreater(result[1][0, 1, 0, 1].item(), 0.9)

    def test_output_projection_changes_content_not_weights(self):
        _, args = sample()
        q, k, v, wo, h, mask = args
        joined, weights = learner.multi_head_attention(q, k, v, torch.eye(6, dtype=q.dtype), h, mask)
        actual = learner.multi_head_attention(*args)
        self.assert_result(actual, (joined @ wo, weights))
        self.assertGreater((actual[0] - joined).abs().max().item(), 1e-3)
        zero = learner.multi_head_attention(q, k, v, torch.zeros_like(wo), h, mask)
        self.assert_result(zero, (torch.zeros_like(joined), weights))

    def test_one_head_query_change_does_not_change_other_heads(self):
        _, args = sample((2, 3, 3, 4, 6))
        q, k, v, _, h, mask = args
        eye = torch.eye(6, dtype=q.dtype)
        baseline = learner.multi_head_attention(q, k, v, eye, h, mask)
        changed_q = q.clone()
        changed_q[:, :, 2:4] += 3
        changed = learner.multi_head_attention(changed_q, k, v, eye, h, mask)
        self.assert_tensor(changed[1][:, [0, 2]], baseline[1][:, [0, 2]])
        self.assert_tensor(changed[0][:, :, [0, 1, 4, 5]], baseline[0][:, :, [0, 1, 4, 5]])
        self.assertGreater((changed[1][:, 1] - baseline[1][:, 1]).abs().max().item(), 1e-3)

    def test_all_blocked_row_rejected_before_softmax(self):
        _, args = sample()
        q, k, v, wo, h, mask = args
        mask[1, 2] = False
        before = mask.clone()
        with self.assertRaises(ValueError):
            learner.multi_head_attention(q, k, v, wo, h, mask)
        self.assertTrue(torch.equal(mask, before))

    def test_invalid_head_counts_raise_value_error(self):
        _, args = sample()
        q, k, v, wo, _, mask = args
        for h in (0, -1, 4, 7):
            with self.subTest(h=h), self.assertRaises(ValueError):
                learner.multi_head_attention(q, k, v, wo, h, mask)

    def test_gradients_for_q_k_v_wo_reach_underlying_tensors(self):
        for layout in ("contiguous", "transposed", "sliced"):
            with self.subTest(layout=layout):
                bases, args = sample(layout=layout, requires_grad=True)
                ref_bases, ref_args = sample(layout=layout, requires_grad=True)
                before = [base.detach().clone() for base in bases]
                for base in (*bases, *ref_bases):
                    base.grad = torch.full_like(base, 0.125)
                result = learner.multi_head_attention(*args)
                expected = reference(*ref_args)
                self.assert_result(result, expected)
                for base, saved in zip(bases, before):
                    self.assertTrue(torch.equal(base, saved))
                    self.assert_tensor(base.grad, torch.full_like(base, 0.125))
                generator = torch.Generator().manual_seed(97)
                seeds = [torch.randn(x.shape, generator=generator, dtype=x.dtype) for x in expected]
                sum((x * seed).sum() for x, seed in zip(result, seeds)).backward()
                sum((x * seed).sum() for x, seed in zip(expected, seeds)).backward()
                for base, ref_base in zip(bases, ref_bases):
                    self.assert_tensor(base.grad, ref_base.grad)


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class SelfAttentionTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        try:
            learner.multi_head_self_attention(*self_sample())
        except NotImplementedError:
            raise unittest.SkipTest("先完成 multi_head_self_attention")

    def test_full_projection_mask_reading_and_output_projection(self):
        args = self_sample()
        before = [x.clone() for x in args[:-1]]
        result = learner.multi_head_self_attention(*args)
        self.assert_result(result, self_reference(*args))
        for x, saved in zip(args[:-1], before):
            self.assertTrue(torch.equal(x, saved))
        # PAD query 仍能读到合法历史；不能把该行权重强行清零。
        self.assert_tensor(result[1][:, :, -1].sum(dim=-1), torch.ones(2, 3, dtype=args[0].dtype))

    def test_one_head_matches_existing_single_head_then_wo(self):
        x, wq, wk, wv, wo, valid, _ = self_sample()
        single, weights = single_head_self_attention(x, wq, wk, wv, valid)
        actual = learner.multi_head_self_attention(x, wq, wk, wv, wo, valid, 1)
        self.assert_result(actual, (single @ wo, weights.unsqueeze(1)))

    def test_future_padding_and_prefix_visibility(self):
        args = self_sample()
        x, wq, wk, wv, wo, valid, h = args
        baseline = learner.multi_head_self_attention(*args)
        changed = x.clone()
        changed[:, 3:] += 100
        future = learner.multi_head_self_attention(changed, wq, wk, wv, wo, valid, h)
        self.assert_tensor(future[0][:, :3], baseline[0][:, :3])
        self.assert_tensor(future[1][:, :, :3], baseline[1][:, :, :3])
        changed = x.clone()
        changed[~valid] -= 100
        padded = learner.multi_head_self_attention(changed, wq, wk, wv, wo, valid, h)
        self.assert_tensor(padded[0][valid], baseline[0][valid])
        prefix = learner.multi_head_self_attention(x[:, :3], wq, wk, wv, wo, valid[:, :3], h)
        self.assert_result(prefix, (baseline[0][:, :3], baseline[1][:, :, :3, :3]))
        bad_valid = valid.clone()
        bad_valid[1, 0] = False
        with self.assertRaises(ValueError):
            learner.multi_head_self_attention(x, wq, wk, wv, wo, bad_valid, h)

    def test_gradients_for_input_and_all_four_parameter_matrices(self):
        args = self_sample(requires_grad=True)
        ref_args = self_sample(requires_grad=True)
        for x in (*args[:5], *ref_args[:5]):
            x.grad = torch.full_like(x, 0.125)
        result = learner.multi_head_self_attention(*args)
        expected = self_reference(*ref_args)
        self.assert_result(result, expected)
        for x in args[:5]:
            self.assert_tensor(x.grad, torch.full_like(x, 0.125))
        generator = torch.Generator().manual_seed(101)
        seeds = [torch.randn(x.shape, generator=generator, dtype=x.dtype) for x in expected]
        sum((x * seed).sum() for x, seed in zip(result, seeds)).backward()
        sum((x * seed).sum() for x, seed in zip(expected, seeds)).backward()
        for actual, target in zip(args[:5], ref_args[:5]):
            self.assert_tensor(actual.grad, target.grad)


if __name__ == "__main__":
    unittest.main()
