"""作业 004 的行为测试；官方 Softmax 仅作为验收参照。"""

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from exercises.ex004_stable_softmax.stable_softmax import stable_softmax


class SoftmaxImplementationStatusTest(unittest.TestCase):
    def test_required_function_is_implemented(self):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境运行测试。")
        try:
            stable_softmax(torch.tensor([0.0, 1.0]))
        except NotImplementedError as error:
            self.fail(str(error))


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class StableSoftmaxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            stable_softmax(torch.tensor([0.0, 1.0]))
        except NotImplementedError:
            raise unittest.SkipTest("先完成 stable_softmax.py 中的函数")

    def assert_distribution(self, scores):
        actual = stable_softmax(scores)
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, scores.shape)
        self.assertEqual(actual.dtype, scores.dtype)
        self.assertEqual(actual.device, scores.device)
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue((actual >= 0).all().item())
        torch.testing.assert_close(actual, torch.softmax(scores, dim=-1))
        torch.testing.assert_close(
            actual.sum(dim=-1), torch.ones_like(actual.sum(dim=-1))
        )
        return actual

    def test_equal_scores_give_uniform_weights(self):
        scores = torch.tensor([[7.0, 7.0, 7.0], [-20.0, -20.0, -20.0]])
        actual = self.assert_distribution(scores)
        torch.testing.assert_close(actual, torch.full_like(scores, 1.0 / 3.0))

    def test_single_group_and_float64(self):
        self.assert_distribution(
            torch.tensor([-2.0, 0.0, 1.0, 3.0], dtype=torch.float64)
        )

    def test_large_positive_and_negative_rows_stay_independent(self):
        scores = torch.tensor([[1000.0, 1001.0], [-1000.0, -999.0]])
        actual = self.assert_distribution(scores)
        torch.testing.assert_close(actual[0], actual[1])

    def test_tiny_probabilities_may_underflow_without_nan(self):
        scores = torch.tensor([[0.0, -2000.0], [-10000.0, -12000.0]])
        actual = self.assert_distribution(scores)
        torch.testing.assert_close(actual, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))

    def test_three_axes_normalize_only_candidates(self):
        scores = torch.tensor(
            [
                [[1, 2, 3, 4], [-4, -3, -2, -1], [7, 0, -2, 1]],
                [[5, -1, 0, 2], [0, 0, 0, 0], [-8, 4, 3, -2]],
            ],
            dtype=torch.float64,
        )
        self.assert_distribution(scores)

    def test_each_group_can_be_shifted_by_a_different_constant(self):
        scores = torch.tensor(
            [[[1, 3, 0], [2, -1, 4]], [[-2, 1, 5], [6, 0, -3]]],
            dtype=torch.float64,
        )
        shifts = torch.tensor([[[1000], [-1000]], [[300], [-500]]], dtype=scores.dtype)
        original = self.assert_distribution(scores)
        shifted = self.assert_distribution(scores + shifts)
        torch.testing.assert_close(original, shifted)

    def test_single_candidate_gets_all_weight(self):
        scores = torch.tensor([[[1000.0], [-1000.0]], [[3.0], [0.0]]])
        actual = self.assert_distribution(scores)
        torch.testing.assert_close(actual, torch.ones_like(scores))

    def test_does_not_modify_input(self):
        scores = torch.tensor([[1.0, 5.0, -2.0], [3.0, -4.0, 6.0]])
        original = scores.clone()
        stable_softmax(scores)
        self.assertTrue(torch.equal(scores, original))

    def test_keeps_autograd_path_and_matches_reference_gradient(self):
        scores = torch.tensor(
            [[1.0, 2.0, -1.0], [0.0, 0.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        reference_scores = scores.detach().clone().requires_grad_(True)
        coefficients = torch.tensor([[1.0, -2.0, 4.0], [3.0, 1.0, -1.0]])
        output = stable_softmax(scores)
        self.assertTrue(output.requires_grad, "不能取出 Python 数字后重建结果")
        (output * coefficients).sum().backward()
        (torch.softmax(reference_scores, dim=-1) * coefficients).sum().backward()
        self.assertIsNotNone(scores.grad)
        torch.testing.assert_close(scores.grad, reference_scores.grad)


if __name__ == "__main__":
    unittest.main()
