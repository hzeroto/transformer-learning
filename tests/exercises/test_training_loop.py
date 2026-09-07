"""训练闭环第二部分；官方计算仅用作验收参考，不替代学习者实现。"""

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from exercises.ex005_training_loop.training import (
        forward_logits,
        prepare_next_token_batch,
        train_step,
    )


def sample_data():
    ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 0]])
    valid = torch.tensor([[True, True, True, True], [True, True, True, False]])
    return ids, valid


def parameters(dtype=None):
    if dtype is None:
        dtype = torch.float64
    generator = torch.Generator().manual_seed(17)
    E = (0.15 * torch.randn(5, 4, generator=generator, dtype=dtype)).requires_grad_(True)
    W = (0.15 * torch.randn(4, 5, generator=generator, dtype=dtype)).requires_grad_(True)
    return E, W


def training_data():
    ids, valid = sample_data()
    # 单独测试 train_step 时不依赖尚未完成的 prepare_next_token_batch。
    return ids[:, :-1], ids[:, 1:], valid[:, 1:]


def reference_loss(input_ids, targets, valid, E, W):
    logits = torch.nn.functional.linear(
        torch.nn.functional.embedding(input_ids, E), W.transpose(0, 1)
    )
    return torch.nn.functional.cross_entropy(logits[valid], targets[valid])


def reference_step(input_ids, targets, valid, E, W, lr):
    # 为一次参考更新创建独立参数，避免继承旧梯度。
    ref_E = E.detach().clone().requires_grad_(True)
    ref_W = W.detach().clone().requires_grad_(True)
    loss = reference_loss(input_ids, targets, valid, ref_E, ref_W)
    loss.backward()
    return (
        ref_E.detach() - lr * ref_E.grad,
        ref_W.detach() - lr * ref_W.grad,
        loss.item(),
    )


def probe_prepare():
    return prepare_next_token_batch(*sample_data())


def probe_forward():
    E, W = parameters()
    return forward_logits(training_data()[0], E, W)


def probe_step():
    E, W = parameters()
    return train_step(*training_data(), E, W, lr=0.1)


class TrainingLoopImplementationStatusTest(unittest.TestCase):
    def check_implemented(self, probe):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境。")
        try:
            probe()
        except NotImplementedError as error:
            self.fail(str(error))

    def test_prepare_is_implemented(self):
        self.check_implemented(probe_prepare)

    def test_forward_is_implemented(self):
        self.check_implemented(probe_forward)

    def test_step_is_implemented(self):
        self.check_implemented(probe_step)


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class NextTokenBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            probe_prepare()
        except NotImplementedError as error:
            raise unittest.SkipTest(str(error))

    def test_shifted_labels_keep_eos_but_exclude_its_padding_successor(self):
        ids, valid = sample_data()
        input_ids, targets, target_valid = prepare_next_token_batch(ids, valid)
        expected = (
            torch.tensor([[1, 2, 3], [2, 3, 4]]),
            torch.tensor([[2, 3, 4], [3, 4, 0]]),
            torch.tensor([[True, True, True], [True, True, False]]),
        )
        for actual, wanted in zip((input_ids, targets, target_valid), expected):
            torch.testing.assert_close(actual, wanted)

    def test_validity_comes_from_mask_not_a_particular_id(self):
        ids = torch.tensor([[3, 0, 2, 4], [0, 2, 4, 4]])
        valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
        inputs, targets, mask = prepare_next_token_batch(ids, valid)
        torch.testing.assert_close(inputs, ids[:, :-1])
        torch.testing.assert_close(targets, ids[:, 1:])
        torch.testing.assert_close(mask, valid[:, 1:])
        self.assertTrue(mask[0, 0].item(), "真实 ID 0 也可以是有效答案")

    def test_length_two_preserves_axes_and_all_invalid_targets_are_allowed(self):
        ids = torch.tensor([[2, 4]])
        for valid in (torch.tensor([[True, True]]), torch.tensor([[True, False]])):
            outputs = prepare_next_token_batch(ids, valid)
            for output in outputs:
                self.assertEqual(output.shape, (1, 1))
            torch.testing.assert_close(outputs[2], valid[:, 1:])

    def test_does_not_modify_inputs(self):
        ids, valid = sample_data()
        old_ids, old_valid = ids.clone(), valid.clone()
        prepare_next_token_batch(ids, valid)
        torch.testing.assert_close(ids, old_ids)
        torch.testing.assert_close(valid, old_valid)


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class ForwardLogitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            probe_forward()
        except NotImplementedError as error:
            raise unittest.SkipTest(str(error))

    def test_lookup_and_shared_projection_match_reference(self):
        for dtype in (torch.float32, torch.float64):
            E, W = parameters(dtype)
            ids = training_data()[0]
            actual = forward_logits(ids, E, W)
            expected = torch.nn.functional.linear(
                torch.nn.functional.embedding(ids, E), W.transpose(0, 1)
            )
            self.assertEqual(actual.shape, (2, 3, 5))
            torch.testing.assert_close(actual, expected)
            self.assertTrue(actual.requires_grad)
            torch.testing.assert_close(
                forward_logits(ids, E.detach(), W.detach()), expected.detach()
            )

    def test_single_position_keeps_batch_and_time_axes(self):
        ids = torch.tensor([[2]])
        E = torch.tensor([[0.1, 0.2], [0.3, -0.1], [-0.2, 0.4]])
        W = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.2, -0.1]])
        actual = forward_logits(ids, E, W)
        self.assertEqual(actual.shape, (1, 1, 3))
        torch.testing.assert_close(actual[0, 0], E[2] @ W)

    def test_does_not_modify_inputs_or_parameters(self):
        E, W = parameters()
        ids = training_data()[0]
        originals = (ids.clone(), E.detach().clone(), W.detach().clone())
        forward_logits(ids, E, W)
        for actual, expected in zip((ids, E, W), originals):
            torch.testing.assert_close(actual, expected)

    def test_token_renumbering_permutes_embedding_rows_and_output_columns(self):
        E, W = parameters()
        ids = torch.tensor([[0, 2, 3], [2, 1, 4]])
        swap = torch.tensor([2, 1, 0, 3, 4])
        original = forward_logits(ids, E, W)
        remapped = forward_logits(swap[ids], E[swap], W[:, swap])
        torch.testing.assert_close(remapped, original[:, :, swap])

    def test_other_positions_do_not_affect_current_position(self):
        E, W = parameters()
        ids = training_data()[0]
        original = forward_logits(ids, E, W)
        changed = ids.clone()
        changed[0, 2] = 0
        actual = forward_logits(changed, E, W)
        torch.testing.assert_close(actual[0, :2], original[0, :2])
        torch.testing.assert_close(actual[1], original[1])


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class SGDStepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            probe_step()
        except NotImplementedError as error:
            raise unittest.SkipTest(str(error))

    def assert_parameters_close(self, E, W, expected):
        torch.testing.assert_close(E, expected[0])
        torch.testing.assert_close(W, expected[1])

    def test_one_update_matches_reference_and_returns_pre_update_float(self):
        for dtype in (torch.float32, torch.float64):
            E, W = parameters(dtype)
            data = training_data()
            old_E, old_W = E.detach().clone(), W.detach().clone()
            expected = reference_step(*data, E, W, lr=0.2)
            before = train_step(*data, E, W, lr=0.2)
            self.assertIsInstance(before, float)
            tolerance = 1e-6 if dtype == torch.float32 else 1e-12
            self.assertAlmostEqual(before, expected[2], delta=tolerance)
            self.assert_parameters_close(E, W, expected)
            self.assertFalse(torch.equal(E, old_E), "E 必须实际更新")
            self.assertFalse(torch.equal(W, old_W), "W 必须实际更新")
            after = reference_loss(*data, E, W).item()
            self.assertNotAlmostEqual(before, after, places=6)

    def test_existing_gradients_do_not_enter_this_update(self):
        E, W = parameters()
        data = training_data()
        expected = reference_step(*data, E, W, lr=0.1)
        E.grad = torch.full_like(E, 123.0)
        W.grad = torch.full_like(W, -71.0)
        train_step(*data, E, W, lr=0.1)
        self.assert_parameters_close(E, W, expected)

    def test_two_different_batches_use_fresh_gradients_and_updated_parameters(self):
        E, W = parameters()
        first = training_data()
        second = (
            torch.tensor([[3, 2]]),
            torch.tensor([[4, 3]]),
            torch.tensor([[True, True]]),
        )
        ref_E, ref_W = E.detach().clone(), W.detach().clone()
        for data in (first, second):
            expected = reference_step(*data, ref_E, ref_W, lr=0.3)
            train_step(*data, E, W, lr=0.3)
            self.assert_parameters_close(E, W, expected)
            ref_E, ref_W = expected[:2]

    def test_learning_rate_scales_the_same_starting_gradient(self):
        E1, W1 = parameters()
        E2, W2 = parameters()
        old_E, old_W = E1.detach().clone(), W1.detach().clone()
        data = training_data()
        train_step(*data, E1, W1, lr=0.05)
        train_step(*data, E2, W2, lr=0.15)
        torch.testing.assert_close(E2 - old_E, 3 * (E1 - old_E))
        torch.testing.assert_close(W2 - old_W, 3 * (W1 - old_W))

    def test_extra_padding_does_not_change_the_update(self):
        E1, W1 = parameters()
        E2, W2 = parameters()
        inputs, targets, valid = training_data()
        padded = (
            torch.cat((inputs, torch.tensor([[4, 0], [0, 4]])), dim=1),
            torch.cat((targets, torch.tensor([[1, 2], [3, 4]])), dim=1),
            torch.cat((valid, torch.zeros((2, 2), dtype=torch.bool)), dim=1),
        )
        a = train_step(inputs, targets, valid, E1, W1, lr=0.2)
        b = train_step(*padded, E2, W2, lr=0.2)
        self.assertAlmostEqual(a, b, places=12)
        torch.testing.assert_close(E1, E2)
        torch.testing.assert_close(W1, W2)

    def test_invalid_batch_raises_without_changing_parameter_values(self):
        E, W = parameters()
        inputs, targets, valid = training_data()
        old_E, old_W = E.detach().clone(), W.detach().clone()
        with self.assertRaises(ValueError):
            train_step(inputs, targets, torch.zeros_like(valid), E, W, lr=0.1)
        torch.testing.assert_close(E, old_E)
        torch.testing.assert_close(W, old_W)

    def test_data_is_unchanged_and_parameters_remain_trainable(self):
        E, W = parameters()
        data = training_data()
        original_data = tuple(x.clone() for x in data)
        addresses = (E.data_ptr(), W.data_ptr())
        train_step(*data, E, W, lr=0.1)
        for actual, expected in zip(data, original_data):
            torch.testing.assert_close(actual, expected)
        self.assertEqual((E.data_ptr(), W.data_ptr()), addresses)
        for parameter in (E, W):
            self.assertTrue(parameter.requires_grad)
            self.assertTrue(parameter.is_leaf)
            self.assertIsNone(parameter.grad_fn)
            parameter.grad = None
        forward_logits(data[0], E, W).sum().backward()
        self.assertIsNotNone(E.grad)
        self.assertIsNotNone(W.grad)


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class TinyTrainingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            probe_prepare()
            probe_forward()
            probe_step()
        except NotImplementedError as error:
            raise unittest.SkipTest(str(error))

    def test_learns_consistent_successors(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            E, W = parameters()
            data = prepare_next_token_batch(*sample_data())
            initial = reference_loss(*data, E, W).item()
            for _ in range(160):
                train_step(*data, E, W, lr=0.5)
            final = reference_loss(*data, E, W).item()
            self.assertLess(final, 0.1)
            self.assertLess(final, initial / 10)
            inputs, targets, valid = data
            predictions = forward_logits(inputs, E, W).argmax(dim=-1)
            torch.testing.assert_close(predictions[valid], targets[valid])
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
