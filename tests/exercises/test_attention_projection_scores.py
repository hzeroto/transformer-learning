"""作业 006 第一部分；教师提供的数值、性质和梯度测试。"""

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from exercises.ex006_single_head_attention.attention import (
        project_qkv,
        raw_attention_scores,
    )


def projection_reference(X, Wq, Wk, Wv):
    return tuple(torch.einsum("btc,cd->btd", X, W) for W in (Wq, Wk, Wv))


def scores_reference(Q, K):
    return torch.einsum("bid,bjd->bij", Q, K)


def sample(dtype=None, shape=(2, 3, 4, 2, 5)):
    if dtype is None:
        dtype = torch.float64
    b, t, c, dk, dv = shape
    generator = torch.Generator().manual_seed(29)
    return tuple(
        torch.randn(size, dtype=dtype, generator=generator)
        for size in ((b, t, c), (c, dk), (c, dk), (c, dv))
    )


class AttentionImplementationStatusTest(unittest.TestCase):
    def test_projection_is_implemented(self):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境。")
        try:
            project_qkv(*sample())
        except NotImplementedError as error:
            self.fail(str(error))

    def test_scores_are_implemented(self):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境。")
        q = torch.ones(1, 2, 3)
        k = torch.ones(1, 4, 3)
        try:
            raw_attention_scores(q, k)
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


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class QKVProjectionTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        try:
            project_qkv(*sample())
        except NotImplementedError:
            raise unittest.SkipTest("先完成 project_qkv")

    def test_values_shapes_dtypes_and_inputs_unchanged(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((2, 3, 4, 2, 5), (1, 1, 1, 1, 2)):
                with self.subTest(dtype=dtype, shape=shape):
                    values = sample(dtype, shape)
                    before = tuple(x.clone() for x in values)
                    expected = projection_reference(*before)
                    actual = project_qkv(*values)
                    self.assertIsInstance(actual, tuple)
                    self.assertEqual(len(actual), 3)
                    for result, reference in zip(actual, expected):
                        self.assert_tensor(result, reference)
                    for value, original in zip(values, before):
                        self.assertTrue(torch.equal(value, original))

    def test_three_parameter_matrices_have_independent_roles(self):
        values = sample()
        baseline = project_qkv(*values)
        for parameter_index in range(3):
            with self.subTest(parameter_index=parameter_index):
                changed = list(values)
                changed[parameter_index + 1] = changed[parameter_index + 1] * 2
                outputs = project_qkv(*changed)
                for output_index in range(3):
                    factor = 2 if output_index == parameter_index else 1
                    self.assert_tensor(outputs[output_index], baseline[output_index] * factor)
                self.assertGreater(
                    (outputs[parameter_index] - baseline[parameter_index]).abs().max().item(),
                    1e-3,
                )


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class RawScoresTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        try:
            raw_attention_scores(torch.ones(1, 2, 3), torch.ones(1, 4, 3))
        except NotImplementedError:
            raise unittest.SkipTest("先完成 raw_attention_scores")

    def test_exact_pairing_with_different_lengths_and_signed_scores(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                q = torch.tensor([[[1, 2], [-1, 3]]], dtype=dtype)
                k = torch.tensor([[[2, 0], [0, -1], [1, 1]]], dtype=dtype)
                q_before, k_before = q.clone(), k.clone()
                expected = torch.tensor([[[2, -2, 3], [-2, -3, 2]]], dtype=dtype)
                self.assert_tensor(raw_attention_scores(q, k), expected)
                self.assertTrue(torch.equal(q, q_before))
                self.assertTrue(torch.equal(k, k_before))

    def test_key_order_controls_columns_and_batches_stay_independent(self):
        generator = torch.Generator().manual_seed(29)
        q = torch.randn(2, 3, 4, generator=generator, dtype=torch.float64)
        k = torch.randn(2, 5, 4, generator=generator, dtype=torch.float64)
        baseline = raw_attention_scores(q, k)
        self.assert_tensor(baseline, scores_reference(q, k))
        order = torch.tensor([3, 0, 4, 1, 2])
        self.assert_tensor(raw_attention_scores(q, k[:, order]), baseline[:, :, order])
        changed = q.clone()
        changed[1, 2] += 1
        actual = raw_attention_scores(changed, k)
        self.assert_tensor(actual, scores_reference(changed, k))
        self.assert_tensor(actual[0], baseline[0])
        self.assert_tensor(actual[1, :2], baseline[1, :2])

    def test_equal_lengths_do_not_imply_symmetric_scores(self):
        q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
        k = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64)
        expected = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]], dtype=torch.float64)
        self.assert_tensor(raw_attention_scores(q, k), expected)


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class ProjectionScoresGradientTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        try:
            q, k, _ = project_qkv(*sample())
            raw_attention_scores(q, k)
        except NotImplementedError:
            raise unittest.SkipTest("先完成两个函数")

    def test_gradients_reach_input_and_all_three_parameters(self):
        values = tuple(x.requires_grad_(True) for x in sample())
        references = tuple(x.detach().clone().requires_grad_(True) for x in values)
        q, k, v = project_qkv(*values)
        scores = raw_attention_scores(q, k)
        qr, kr, vr = projection_reference(*references)
        expected = scores_reference(qr, kr)
        self.assert_tensor(scores, expected)
        self.assertTrue(scores.requires_grad)
        self.assertTrue(v.requires_grad)
        generator = torch.Generator().manual_seed(29)
        score_weights = torch.randn(scores.shape, generator=generator, dtype=scores.dtype)
        value_weights = torch.randn(v.shape, generator=generator, dtype=v.dtype)
        # 任意上游梯度同时检验匹配路径与 V 路径，不预先实现后续加权读取。
        ((scores * score_weights).sum() + (v * value_weights).sum()).backward()
        ((expected * score_weights).sum() + (vr * value_weights).sum()).backward()
        for value, reference in zip(values, references):
            self.assertIsNotNone(value.grad)
            self.assert_tensor(value.grad, reference.grad)
            self.assert_tensor(value.detach(), reference.detach())


if __name__ == "__main__":
    unittest.main()
