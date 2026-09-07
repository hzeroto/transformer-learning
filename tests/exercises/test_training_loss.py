"""作业 005 第一部分的行为测试；参考封装仅用于验收。"""

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from exercises.ex005_training_loop.loss import masked_cross_entropy


class TrainingLossImplementationStatusTest(unittest.TestCase):
    def test_required_function_is_implemented(self):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境运行测试。")
        try:
            masked_cross_entropy(
                torch.tensor([[[0.0, 1.0]]]),
                torch.tensor([[0]]),
                torch.tensor([[True]]),
            )
        except NotImplementedError as error:
            self.fail(str(error))


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class TrainingLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            masked_cross_entropy(
                torch.tensor([[[0.0, 1.0]]]),
                torch.tensor([[0]]),
                torch.tensor([[True]]),
            )
        except NotImplementedError:
            raise unittest.SkipTest("先完成 ex005_training_loop/loss.py 中的函数")

    def example(self, dtype=None):
        if dtype is None:
            dtype = torch.float64
        logits = torch.tensor(
            [
                [[3, 0, 1, -2], [-1, 2, 0.5, 0], [4, 0, 2, -1]],
                [[-2, 0, 3, 1], [1, 4, -2, 0], [0, 0, 0, 0]],
            ],
            dtype=dtype,
        )
        targets = torch.tensor([[2, 1, 0], [0, 3, 1]], dtype=torch.long)
        valid = torch.tensor([[True, True, False], [True, False, False]])
        return logits, targets, valid

    def assert_matches_reference(self, logits, targets, valid):
        actual = masked_cross_entropy(logits, targets, valid)
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, torch.Size([]))
        self.assertEqual(actual.dtype, logits.dtype)
        self.assertEqual(actual.device, logits.device)
        self.assertTrue(torch.isfinite(actual).item())
        expected = torch.nn.functional.cross_entropy(logits[valid], targets[valid])
        torch.testing.assert_close(actual, expected)
        return actual

    def test_three_axes_and_different_labels(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                logits, targets, valid = self.example(dtype)
                self.assert_matches_reference(logits, targets, torch.ones_like(valid))

    def test_changing_correct_label_changes_objective(self):
        logits = torch.tensor([[[3.0, 0.0, -1.0]]])
        valid = torch.tensor([[True]])
        favored = self.assert_matches_reference(logits, torch.tensor([[0]]), valid)
        unfavored = self.assert_matches_reference(logits, torch.tensor([[2]]), valid)
        self.assertGreater(unfavored.item(), favored.item())

    def test_extreme_wrong_predictions_still_have_finite_loss(self):
        logits = torch.tensor([[[1000.0, -1000.0]], [[-1000.0, 1000.0]]])
        targets = torch.tensor([[1], [0]])
        valid = torch.tensor([[True], [True]])
        actual = self.assert_matches_reference(logits, targets, valid)
        torch.testing.assert_close(actual, torch.tensor(2000.0))

    def test_each_valid_token_has_equal_weight(self):
        logits, targets, valid = self.example()
        actual = self.assert_matches_reference(logits, targets, valid)
        first = torch.nn.functional.cross_entropy(logits[0, :2], targets[0, :2])
        second = torch.nn.functional.cross_entropy(logits[1, :1], targets[1, :1])
        torch.testing.assert_close(actual, (2 * first + second) / 3)
        self.assertFalse(torch.isclose(actual, (first + second) / 2).item())

    def test_extra_padding_does_not_change_loss(self):
        logits, targets, valid = self.example()
        original = self.assert_matches_reference(logits, targets, valid)
        extra_logits = torch.full((2, 2, 4), 10000.0, dtype=logits.dtype)
        extra_targets = torch.zeros((2, 2), dtype=targets.dtype)
        extra_valid = torch.zeros((2, 2), dtype=torch.bool)
        padded = self.assert_matches_reference(
            torch.cat((logits, extra_logits), dim=1),
            torch.cat((targets, extra_targets), dim=1),
            torch.cat((valid, extra_valid), dim=1),
        )
        torch.testing.assert_close(original, padded)

    def test_ignored_values_and_fully_ignored_sequence_do_not_contribute(self):
        logits, targets, valid = self.example()
        valid[0] = False
        original = self.assert_matches_reference(logits, targets, valid)
        changed_logits = logits.clone()
        changed_targets = targets.clone()
        changed_logits[~valid] = torch.tensor(
            [1000.0, -1000.0, 2000.0, -2000.0], dtype=logits.dtype
        )
        changed_targets[~valid] = 3
        changed = self.assert_matches_reference(changed_logits, changed_targets, valid)
        torch.testing.assert_close(original, changed)

    def test_each_group_can_be_shifted_independently(self):
        logits, targets, valid = self.example()
        offsets = torch.tensor(
            [[[1000], [-1000], [300]], [[-500], [2000], [-2000]]],
            dtype=logits.dtype,
        )
        original = self.assert_matches_reference(logits, targets, valid)
        shifted = self.assert_matches_reference(logits + offsets, targets, valid)
        torch.testing.assert_close(original, shifted)

    def test_single_position_single_candidate_has_zero_loss(self):
        logits = torch.tensor([[[-1000.0]]], dtype=torch.float64)
        actual = self.assert_matches_reference(
            logits, torch.tensor([[0]]), torch.tensor([[True]])
        )
        torch.testing.assert_close(actual, torch.tensor(0.0, dtype=logits.dtype))

    def test_rejects_batch_without_valid_labels(self):
        logits, targets, valid = self.example()
        with self.assertRaises(ValueError):
            masked_cross_entropy(logits, targets, torch.zeros_like(valid))

    def test_does_not_modify_inputs(self):
        inputs = self.example()
        originals = tuple(x.clone() for x in inputs)
        masked_cross_entropy(*inputs)
        for actual, original in zip(inputs, originals):
            self.assertTrue(torch.equal(actual, original))

    def test_gradient_matches_reference_and_ignored_positions_are_zero(self):
        logits, targets, valid = self.example()
        logits[0, 0] = torch.tensor([1000, 0, -1000, -2000], dtype=logits.dtype)
        logits.requires_grad_(True)
        reference_logits = logits.detach().clone().requires_grad_(True)
        loss = masked_cross_entropy(logits, targets, valid)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.requires_grad, "返回值必须保留自动求导路径")
        loss.backward()
        torch.nn.functional.cross_entropy(
            reference_logits[valid], targets[valid]
        ).backward()
        self.assertIsNotNone(logits.grad)
        torch.testing.assert_close(logits.grad, reference_logits.grad)
        self.assertTrue((logits.grad[~valid] == 0).all().item())


if __name__ == "__main__":
    unittest.main()
