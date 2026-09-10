"""作业 006 综合测试；参照按 query 选出允许候选，独立验证向量化实现。"""

import math
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from exercises.ex006_single_head_attention import attention as learner


def sample(dtype=None, shape=(2, 3, 5, 4, 2)):
    if dtype is None:
        dtype = torch.float64
    b, tq, tk, dk, dv = shape
    generator = torch.Generator().manual_seed(53)
    q, k, v = (
        torch.randn(size, generator=generator, dtype=dtype)
        for size in ((b, tq, dk), (b, tk, dk), (b, tk, dv))
    )
    allowed = torch.rand(b, tq, tk, generator=generator) > 0.4
    allowed[:, :, 0] = True
    return q, k, v, allowed


def causal_reference(valid):
    b, t = valid.shape
    return torch.tensor(
        [[[j <= i and bool(valid[n, j]) for j in range(t)]
          for i in range(t)] for n in range(b)],
        dtype=torch.bool,
    )


def attention_reference(q, k, v, allowed=None):
    # 先实际选出可读候选，再归一化；不依赖学习者的矩阵 mask 实现。
    b, tq, dk = q.shape
    tk = k.shape[1]
    if allowed is None:
        allowed = torch.ones(b, tq, tk, dtype=torch.bool)
    outputs, weights = [], []
    for n in range(b):
        batch_outputs, batch_weights = [], []
        for i in range(tq):
            indices = allowed[n, i].nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                raise ValueError("每个 query 至少需要一个允许 key")
            scores = (k[n, indices] * q[n, i]).sum(dim=-1) / math.sqrt(dk)
            probs = torch.softmax(scores, dim=0)
            row = torch.zeros(tk, dtype=q.dtype).index_copy(0, indices, probs)
            batch_weights.append(row)
            batch_outputs.append((probs.unsqueeze(-1) * v[n, indices]).sum(dim=0))
        outputs.append(torch.stack(batch_outputs))
        weights.append(torch.stack(batch_weights))
    return torch.stack(outputs), torch.stack(weights)


def self_sample():
    generator = torch.Generator().manual_seed(59)
    values = tuple(
        torch.randn(shape, generator=generator, dtype=torch.float64) * 0.4
        for shape in ((2, 4, 3), (3, 2), (3, 2), (3, 5))
    )
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    return values, valid


class ImplementationStatusTest(unittest.TestCase):
    def require_implemented(self, function_name, *args):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境。")
        try:
            getattr(learner, function_name)(*args)
        except NotImplementedError as error:
            self.fail(str(error))

    def test_mask_is_implemented(self):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境。")
        self.require_implemented("make_causal_allowed", torch.ones(1, 2, dtype=torch.bool))

    def test_attention_is_implemented(self):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境。")
        self.require_implemented("scaled_dot_product_attention", *sample())

    def test_composition_is_implemented(self):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境。")
        values, valid = self_sample()
        self.require_implemented("single_head_self_attention", *values, valid)


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
        for result, reference in zip(actual, expected):
            self.assert_tensor(result, reference)


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class CausalAllowedTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        try:
            learner.make_causal_allowed(torch.ones(1, 2, dtype=torch.bool))
        except NotImplementedError:
            raise unittest.SkipTest("先完成 make_causal_allowed")

    def test_exact_causal_and_key_padding_axes(self):
        valid = torch.tensor([[True, True, True], [True, True, False]])
        expected = torch.tensor([
            [[True, False, False], [True, True, False], [True, True, True]],
            [[True, False, False], [True, True, False], [True, True, False]],
        ])
        before = valid.clone()
        self.assert_tensor(learner.make_causal_allowed(valid), expected)
        self.assertTrue(torch.equal(valid, before))

    def test_invalid_queries_are_not_automatically_cleared(self):
        valid = torch.tensor([[True, False, True, False], [False, False, False, False]])
        actual = learner.make_causal_allowed(valid)
        self.assert_tensor(actual, causal_reference(valid))
        self.assertTrue(actual[0, 3, 0].item())
        self.assertFalse(actual[1].any().item())  # 构造器可以返回全屏蔽行。

    def test_single_position_and_batch(self):
        valid = torch.tensor([[True]])
        self.assert_tensor(learner.make_causal_allowed(valid), torch.tensor([[[True]]]))


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class ScaledAttentionTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        try:
            learner.scaled_dot_product_attention(*sample())
        except NotImplementedError:
            raise unittest.SkipTest("先完成 scaled_dot_product_attention")

    def test_forward_shapes_dtypes_masks_and_no_input_mutation(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((2, 3, 5, 4, 2), (1, 1, 1, 1, 3), (2, 4, 2, 3, 5)):
                for masked in (False, True):
                    with self.subTest(dtype=dtype, shape=shape, masked=masked):
                        q, k, v, allowed = sample(dtype, shape)
                        mask = allowed if masked else None
                        before = tuple(x.clone() for x in (q, k, v, allowed))
                        actual = learner.scaled_dot_product_attention(q, k, v, mask)
                        self.assert_result(actual, attention_reference(q, k, v, mask))
                        weights = actual[1]
                        self.assert_tensor(weights.sum(dim=-1), torch.ones(shape[:2], dtype=dtype))
                        self.assertTrue((weights >= 0).all().item())
                        if masked:
                            self.assertTrue((weights[~allowed] == 0).all().item())
                        for x, original in zip((q, k, v, allowed), before):
                            self.assertTrue(torch.equal(x, original))

    def test_scaling_uses_dk_not_tk_or_dv(self):
        q = torch.ones(1, 1, 4, dtype=torch.float64)
        k = torch.tensor([[[0., 0., 0., 0.], [1., 1., 1., 1.]]], dtype=q.dtype)
        v = torch.tensor([[[2.], [8.]]], dtype=q.dtype)
        p = math.exp(2) / (1 + math.exp(2))  # 原始分数 4 / sqrt(4) = 2。
        expected = (
            torch.tensor([[[2 * (1 - p) + 8 * p]]], dtype=q.dtype),
            torch.tensor([[[1 - p, p]]], dtype=q.dtype),
        )
        self.assert_result(learner.scaled_dot_product_attention(q, k, v), expected)

    def test_mask_precedes_softmax_even_with_dominant_forbidden_score(self):
        for dtype in (torch.float32, torch.float64):
            q = torch.ones(1, 1, 1, dtype=dtype)
            k = torch.tensor([[[0.], [0.], [10000.]]], dtype=dtype)
            v = torch.tensor([[[2., 4.], [4., 8.], [900., -900.]]], dtype=dtype)
            allowed = torch.tensor([[[True, True, False]]])
            expected = (
                torch.tensor([[[3., 6.]]], dtype=dtype),
                torch.tensor([[[0.5, 0.5, 0.]]], dtype=dtype),
            )
            self.assert_result(learner.scaled_dot_product_attention(q, k, v, allowed), expected)

    def test_extreme_finite_unmasked_scores_are_stable(self):
        for dtype in (torch.float32, torch.float64):
            q = torch.ones(1, 1, 1, dtype=dtype)
            k = torch.tensor([[[1000.], [1001.], [-1000.]]], dtype=dtype)
            v = torch.tensor([[[2.], [4.], [8.]]], dtype=dtype)
            result = learner.scaled_dot_product_attention(q, k, v)
            self.assert_result(result, attention_reference(q, k, v))
            self.assertTrue(all(torch.isfinite(x).all().item() for x in result))

    def test_forbidden_keys_and_values_cannot_change_outputs(self):
        q, k, v, allowed = sample()
        allowed[:, :, -1] = False
        baseline = learner.scaled_dot_product_attention(q, k, v, allowed)
        changed_k = k.clone()
        changed_k[:, -1] = q[:, 0] * 1000
        changed_v = v.clone()
        changed_v[:, -1] += 1000
        self.assert_result(learner.scaled_dot_product_attention(q, changed_k, v, allowed), baseline)
        self.assert_result(learner.scaled_dot_product_attention(q, k, changed_v, allowed), baseline)

    def test_key_value_pairing_and_v_does_not_change_weights(self):
        q, k, v, allowed = sample()
        order = torch.tensor([2, 4, 0, 1, 3])
        for mask in (None, allowed):
            baseline = learner.scaled_dot_product_attention(q, k, v, mask)
            reordered_mask = None if mask is None else mask[:, :, order]
            changed = learner.scaled_dot_product_attention(q, k[:, order], v[:, order], reordered_mask)
            self.assert_tensor(changed[0], baseline[0])
            self.assert_tensor(changed[1], baseline[1][:, :, order])
        baseline = learner.scaled_dot_product_attention(q, k, v)
        wrong_pairing = learner.scaled_dot_product_attention(q, k[:, order], v)
        self.assertGreater((wrong_pairing[0] - baseline[0]).abs().max().item(), 1e-3)
        changed_v = learner.scaled_dot_product_attention(q, k, v + 2)
        self.assert_tensor(changed_v[1], baseline[1])
        self.assert_tensor(changed_v[0], baseline[0] + 2)

    def test_batch_and_query_isolation(self):
        q, k, v, allowed = sample()
        baseline = learner.scaled_dot_product_attention(q, k, v, allowed)
        q2 = q.clone()
        q2[1, 2] += 3
        changed = learner.scaled_dot_product_attention(q2, k, v, allowed)
        for result, original in zip(changed, baseline):
            self.assert_tensor(result[0], original[0])
            self.assert_tensor(result[1, :2], original[1, :2])
        self.assert_result(changed, attention_reference(q2, k, v, allowed))

    def test_any_all_blocked_row_is_rejected_without_mutation(self):
        q, k, v, allowed = sample()
        for all_rows in (False, True):
            mask = allowed.clone()
            if all_rows:
                mask[:] = False
            else:
                mask[1, 2] = False
            before = tuple(x.clone() for x in (q, k, v, mask))
            with self.assertRaises(ValueError):
                learner.scaled_dot_product_attention(q, k, v, mask)
            for actual, original in zip((q, k, v, mask), before):
                self.assertTrue(torch.equal(actual, original))

    def test_gradients_of_both_returns_and_blocked_candidates(self):
        for masked in (False, True):
            q, k, v, allowed = sample()
            allowed[:, :, -1] = False
            mask = allowed if masked else None
            values = tuple(x.requires_grad_(True) for x in (q, k, v))
            references = tuple(x.detach().clone().requires_grad_(True) for x in values)
            # 前向不能清理/改写调用者已有的梯度。
            for x in values + references:
                x.grad = torch.full_like(x, 0.125)
            actual = learner.scaled_dot_product_attention(*values, mask)
            expected = attention_reference(*references, mask)
            self.assert_result(actual, expected)
            for x in values:
                self.assert_tensor(x.grad, torch.full_like(x, 0.125))
                x.grad = None
            for x in references:
                x.grad = None
            generator = torch.Generator().manual_seed(61)
            coefficients = tuple(torch.randn(x.shape, generator=generator, dtype=x.dtype) for x in actual)
            self.assertTrue(all(x.requires_grad for x in actual))
            sum((x * c).sum() for x, c in zip(actual, coefficients)).backward()
            sum((x * c).sum() for x, c in zip(expected, coefficients)).backward()
            for x, reference in zip(values, references):
                self.assertIsNotNone(x.grad)
                self.assert_tensor(x.grad, reference.grad)
                self.assertTrue(torch.equal(x.detach(), reference.detach()))
            if masked:
                self.assert_tensor(k.grad[:, -1], torch.zeros_like(k.grad[:, -1]))
                self.assert_tensor(v.grad[:, -1], torch.zeros_like(v.grad[:, -1]))


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class SelfAttentionTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        try:
            values, valid = self_sample()
            learner.single_head_self_attention(*values, valid)
        except NotImplementedError:
            raise unittest.SkipTest("先完成组合函数及其依赖")

    def test_composed_forward_and_parameter_gradients(self):
        samples, valid = self_sample()
        values = tuple(x.requires_grad_(True) for x in samples)
        refs = tuple(x.detach().clone().requires_grad_(True) for x in values)
        valid_before = valid.clone()
        actual = learner.single_head_self_attention(*values, valid)
        xr, wqr, wkr, wvr = refs
        expected = attention_reference(xr @ wqr, xr @ wkr, xr @ wvr, causal_reference(valid))
        self.assert_result(actual, expected)
        generator = torch.Generator().manual_seed(67)
        coefficients = tuple(torch.randn(x.shape, generator=generator, dtype=x.dtype) for x in actual)
        sum((x * c).sum() for x, c in zip(actual, coefficients)).backward()
        sum((x * c).sum() for x, c in zip(expected, coefficients)).backward()
        for x, reference in zip(values, refs):
            self.assertIsNotNone(x.grad)
            self.assert_tensor(x.grad, reference.grad)
            self.assertTrue(torch.equal(x.detach(), reference.detach()))
        self.assertTrue(torch.equal(valid, valid_before))

    def test_future_perturbation_and_prefix_equivalence(self):
        values, _ = self_sample()
        x, wq, wk, wv = values
        valid = torch.ones(x.shape[:2], dtype=torch.bool)
        baseline = learner.single_head_self_attention(*values, valid)
        changed_x = x.clone()
        changed_x[:, 2:] += 7
        changed = learner.single_head_self_attention(changed_x, wq, wk, wv, valid)
        self.assert_tensor(changed[0][:, :2], baseline[0][:, :2])
        self.assert_tensor(changed[1][:, :2], baseline[1][:, :2])
        prefix = learner.single_head_self_attention(x[:, :2], wq, wk, wv, valid[:, :2])
        self.assert_tensor(prefix[0], baseline[0][:, :2])
        self.assert_tensor(prefix[1], baseline[1][:, :2, :2])

    def test_pad_perturbation_and_invalid_query_contract(self):
        values, valid = self_sample()
        x, wq, wk, wv = values
        baseline = learner.single_head_self_attention(*values, valid)
        changed_x = x.clone()
        changed_x[~valid] += 100
        changed = learner.single_head_self_attention(changed_x, wq, wk, wv, valid)
        self.assert_tensor(changed[0][valid], baseline[0][valid])
        self.assert_tensor(changed[1][valid], baseline[1][valid])
        for output, weights in (baseline, changed):
            self.assertTrue(torch.isfinite(output).all().item())
            self.assert_tensor(weights.sum(dim=-1), torch.ones(valid.shape, dtype=x.dtype))
        invalid = valid.clone()
        invalid[1, 0] = False  # 至少 query 0 无法读取任何 key。
        with self.assertRaises(ValueError):
            learner.single_head_self_attention(*values, invalid)

    def test_same_final_query_can_read_different_context(self):
        x = torch.tensor([
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            [[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]],
        ], dtype=torch.float64)
        wq = torch.eye(3, dtype=x.dtype)
        wk = wq.clone()
        wv = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=x.dtype)
        valid = torch.ones(2, 3, dtype=torch.bool)
        result = learner.single_head_self_attention(x, wq, wk, wv, valid)
        self.assert_tensor(result[1][0, -1], result[1][1, -1])
        self.assertGreater((result[0][0, -1] - result[0][1, -1]).abs().max().item(), 0.1)


if __name__ == "__main__":
    unittest.main()
