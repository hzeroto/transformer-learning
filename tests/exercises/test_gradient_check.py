"""有限差分的行为测试；autograd 只在教师验收侧作为独立参考。"""

import math
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from exercises.ex005_training_loop import gradient_check as exercise
    from exercises.ex005_training_loop.loss import masked_cross_entropy
    from exercises.ex005_training_loop.training import forward_logits


def sample():
    generator = torch.Generator().manual_seed(17)
    E = (0.15 * torch.randn(5, 4, generator=generator, dtype=torch.float64)).requires_grad_(True)
    W = (0.15 * torch.randn(4, 5, generator=generator, dtype=torch.float64)).requires_grad_(True)
    inputs = torch.tensor([[1, 2, 3], [2, 3, 4]])
    targets = torch.tensor([[2, 3, 4], [3, 4, 0]])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    return inputs, targets, valid, E, W


def reference_loss(inputs, targets, valid, E, W):
    logits = torch.nn.functional.linear(
        torch.nn.functional.embedding(inputs, E), W.transpose(0, 1)
    )
    return torch.nn.functional.cross_entropy(logits[valid], targets[valid])


def reference_gradient(data, name, index):
    inputs, targets, valid, E, W = data
    ref_E = E.detach().clone().requires_grad_(True)
    ref_W = W.detach().clone().requires_grad_(True)
    reference_loss(inputs, targets, valid, ref_E, ref_W).backward()
    parameter = ref_E if name == "E" else ref_W
    return parameter.grad[index].item()


def probe():
    return exercise.finite_difference_one(*sample(), "W", (0, 2))


class GradientCheckImplementationStatusTest(unittest.TestCase):
    def test_function_is_implemented(self):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境。")
        try:
            probe()
        except NotImplementedError as error:
            self.fail(str(error))


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class FiniteDifferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            probe()
        except NotImplementedError as error:
            raise unittest.SkipTest(str(error))

    def assert_gradient_close(self, actual, expected):
        self.assertIsInstance(actual, float)
        self.assertTrue(math.isfinite(actual))
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-8),
            f"中心差分 {actual} 与参考梯度 {expected} 不符",
        )

    def test_checks_coordinates_in_both_parameter_matrices(self):
        data = sample()
        for name, index in (("E", (2, 0)), ("E", (1, 2)), ("W", (0, 2)), ("W", (3, 4))):
            with self.subTest(parameter=name, index=index):
                expected = reference_gradient(data, name, index)
                actual = exercise.finite_difference_one(*data, name, index, h=1e-5)
                self.assert_gradient_close(actual, expected)

    def test_repeated_token_updates_all_uses_not_one_position(self):
        _, _, _, E, W = sample()
        # ID 2 在位置 1、3，位置 2 则是 ID 3。
        inputs = torch.tensor([[1, 2, 3, 2]])
        targets = torch.tensor([[2, 3, 4, 1]])
        valid = torch.ones((1, 4), dtype=torch.bool)
        data = (inputs, targets, valid, E, W)
        expected = reference_gradient(data, "E", (2, 0))
        actual = exercise.finite_difference_one(*data, "E", (2, 0))
        self.assert_gradient_close(actual, expected)

        # 分开计算中间结果的局部梯度，确认共享参数汇集两次使用的贡献。
        x = E.detach()[inputs].clone().requires_grad_(True)
        torch.nn.functional.cross_entropy(
            (x @ W.detach())[valid], targets[valid]
        ).backward()
        summed = (x.grad[0, 1, 0] + x.grad[0, 3, 0]).item()
        self.assertAlmostEqual(expected, summed, places=12)
        self.assertFalse(math.isclose(actual, x.grad[0, 1, 0].item(), rel_tol=1e-5, abs_tol=1e-8))

    def test_works_without_gradients_and_does_not_call_autograd(self):
        inputs, targets, valid, E, W = sample()
        data = (inputs, targets, valid, E.detach(), W.detach())
        expected = reference_gradient(data, "W", (0, 2))
        observed_grad_modes = []

        def checked_forward(*args):
            observed_grad_modes.append(torch.is_grad_enabled())
            return forward_logits(*args)

        with (
            patch.object(exercise, "forward_logits", side_effect=checked_forward),
            patch.object(torch.autograd, "backward", side_effect=AssertionError("差分禁止反向求导")),
            patch.object(torch.autograd, "grad", side_effect=AssertionError("差分禁止反向求导")),
        ):
            actual = exercise.finite_difference_one(*data, "W", (0, 2))
        self.assert_gradient_close(actual, expected)
        self.assertTrue(observed_grad_modes, "请复用现有 forward_logits")
        self.assertFalse(any(observed_grad_modes), "数值探测应在 no_grad 范围内")

    def test_preserves_inputs_parameters_and_existing_gradients(self):
        for name in ("E", "W"):
            for has_grad in (False, True):
                with self.subTest(parameter=name, existing_grad=has_grad):
                    data = sample()
                    E, W = data[-2:]
                    if has_grad:
                        E.grad = torch.full_like(E, 12.0)
                        W.grad = torch.full_like(W, -7.0)
                    values = tuple(x.detach().clone() for x in data)
                    grad_objects = (E.grad, W.grad)
                    grad_values = tuple(None if x is None else x.clone() for x in grad_objects)
                    exercise.finite_difference_one(*data, name, (2, 0))
                    for actual, expected in zip(data, values):
                        self.assertTrue(torch.equal(actual, expected), "检查不能修改原输入或参数")
                    for parameter, old_grad, old_value in zip((E, W), grad_objects, grad_values):
                        self.assertTrue(parameter.requires_grad)
                        self.assertIs(parameter.grad, old_grad, "不能替换原有梯度对象")
                        if old_value is not None:
                            self.assertTrue(torch.equal(parameter.grad, old_value))

    def test_embedding_used_only_at_ignored_position_has_zero_gradient(self):
        data = sample()
        # ID 4 的唯一一次输入使用对应无效目标。
        actual = exercise.finite_difference_one(*data, "E", (4, 0))
        self.assertAlmostEqual(actual, 0.0, places=12)

    def test_exception_does_not_leave_modified_parameters(self):
        inputs, targets, valid, E, W = sample()
        E.grad = torch.full_like(E, 3.0)
        W.grad = torch.full_like(W, -2.0)
        old_E, old_W = E.detach().clone(), W.detach().clone()
        old_E_grad, old_W_grad = E.grad.clone(), W.grad.clone()
        for name in ("E", "W"):
            with self.assertRaises(ValueError):
                exercise.finite_difference_one(
                    inputs, targets, torch.zeros_like(valid), E, W, name, (2, 0)
                )
            self.assertTrue(torch.equal(E, old_E))
            self.assertTrue(torch.equal(W, old_W))
            self.assertTrue(torch.equal(E.grad, old_E_grad))
            self.assertTrue(torch.equal(W.grad, old_W_grad))

    def test_uses_requested_step_size_in_central_difference(self):
        inputs, targets, valid, E, W = sample()
        # 大 h 用于区分“指定差分公式”和“直接返回精确梯度”，不是精度推荐值。
        h = 0.2
        with torch.no_grad():
            plus, minus = W.clone(), W.clone()
            plus[0, 2] += h
            minus[0, 2] -= h
            expected = (
                reference_loss(inputs, targets, valid, E, plus).item()
                - reference_loss(inputs, targets, valid, E, minus).item()
            ) / (2 * h)
        exact = reference_gradient((inputs, targets, valid, E, W), "W", (0, 2))
        self.assertGreater(abs(expected - exact), 1e-8)
        actual = exercise.finite_difference_one(
            inputs, targets, valid, E, W, "W", (0, 2), h=h
        )
        self.assertIsInstance(actual, float)
        self.assertAlmostEqual(actual, expected, delta=1e-10)

    def test_single_position_with_different_matrix_shapes(self):
        E = torch.tensor([[0.1, 0.2], [0.3, -0.1], [-0.2, 0.4]], dtype=torch.float64)
        W = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.2, -0.1]], dtype=torch.float64)
        data = (torch.tensor([[2]]), torch.tensor([[1]]), torch.tensor([[True]]), E, W)
        for name, index in (("E", (2, 1)), ("W", (1, 0))):
            actual = exercise.finite_difference_one(*data, name, index)
            self.assert_gradient_close(actual, reference_gradient(data, name, index))


if __name__ == "__main__":
    unittest.main()
