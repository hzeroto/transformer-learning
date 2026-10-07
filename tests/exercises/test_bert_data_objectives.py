"""ex016 数据与目标的教师测试，不是可从作业导入的模型答案。

CPU，固定种子 1603；默认 float64，rtol=1e-8 / atol=1e-10。
极端 logits 额外覆盖 float32，rtol=1e-5 / atol=1e-6。
空脚手架真实抛出 NotImplementedError，不跳过行为测试。
"""

import unittest

import torch

from exercises.ex016_mini_bert import data, training


def fixture_inputs():
    """人为指定三种替换；不是在短句上统计 80/10/10 的比例。"""
    clean = torch.tensor([[1, 4, 5, 6, 2, 7, 2, 0],
                          [1, 8, 2, 9, 10, 11, 2, 0]])
    valid = clean != data.PAD_ID
    segments = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 0],
                             [0, 0, 0, 1, 1, 1, 1, 0]])
    selected = torch.tensor([[False, True, True, True, False, False, False, False],
                             [False, False, False, False, True, False, False, False]])
    # 未选中位置的 kind 不参与策略；99 故意不是有效的替换策略。
    kinds = torch.full_like(clean, 99)
    kinds[0, 1:4] = torch.tensor([0, 1, 2])
    kinds[1, 4] = 1
    random_ids = torch.full_like(clean, 11)
    random_ids[1, 4] = 10  # 随机结果允许碰巧等于原词，仍是监督位置。
    return clean, valid, segments, selected, kinds, random_ids


def independent_batch():
    """不调用待测数据入口，让目标测试独立定位训练入口错误。"""
    clean, valid, segments, selected, _, _ = fixture_inputs()
    corrupted = torch.tensor([[1, 3, 11, 6, 2, 7, 2, 0],
                              [1, 8, 2, 9, 10, 11, 2, 0]])
    return dict(input_ids=corrupted, targets=clean, input_valid=valid,
                segment_ids=segments, target_valid=selected)


def strided_copy(value):
    """相同元素、非连续步长，不依赖待测 reshape/广播路径。"""
    backing = torch.zeros((*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype)
    backing[..., ::2] = value
    return backing[..., ::2]


def row_cross_entropy(scores, target):
    # 先平移，避免 float32 的大绝对偏移使参照梯度本身丢精度。
    shifted = scores - scores.max()
    return torch.logsumexp(shifted, dim=0) - shifted[int(target)]


def independent_losses(mlm_logits, nsp_logits, batch, nsp_labels):
    """逐位置参照；不复用被测代码的 mask、gather 或 reduction。"""
    terms = []
    for b in range(batch["targets"].shape[0]):
        for t in range(batch["targets"].shape[1]):
            if bool(batch["target_valid"][b, t]):
                terms.append(row_cross_entropy(mlm_logits[b, t], batch["targets"][b, t]))
    mlm = torch.stack(terms).mean()
    nsp = torch.stack([row_cross_entropy(scores, label)
                       for scores, label in zip(nsp_logits, nsp_labels)]).mean()
    return mlm, nsp


class LogitsSpy(torch.nn.Module):
    """只记录真正收到的输入并返回独立 logits，不提供 Encoder 实现。"""

    def __init__(self, dtype=torch.float64, noncontiguous=False, extreme=False):
        super().__init__()
        generator = torch.Generator().manual_seed(1603)
        mlm = torch.randn(2, 8, data.VOCAB_SIZE, dtype=dtype, generator=generator)
        nsp = torch.tensor([[2., -1.], [-3., .5]], dtype=dtype)
        if extreme:
            mlm *= 1000.
            nsp *= 1000.
        if noncontiguous:
            mlm, nsp = strided_copy(mlm), strided_copy(nsp)
        self.mlm_logits = torch.nn.Parameter(mlm)
        self.nsp_logits = torch.nn.Parameter(nsp)
        self.seen = []

    def forward(self, input_ids, segment_ids, input_valid):
        self.seen.append(tuple(value.detach().clone()
                               for value in (input_ids, segment_ids, input_valid)))
        return dict(mlm_logits=self.mlm_logits, nsp_logits=self.nsp_logits)


class TensorAssertions(unittest.TestCase):
    def assert_tensor(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        tolerance = (dict(rtol=1e-5, atol=1e-6) if expected.dtype == torch.float32
                     else dict(rtol=1e-8, atol=1e-10))
        torch.testing.assert_close(actual, expected, **tolerance)


class TestMLMBatch(TensorAssertions):
    def test_three_replacements_keep_original_unshifted_targets(self):
        result = data.make_mlm_batch(*fixture_inputs())
        expected = independent_batch()
        self.assertEqual(set(result), set(expected))
        for name in expected:
            with self.subTest(field=name):
                self.assert_tensor(result[name], expected[name])
        self.assertTrue(result["input_valid"][0, 1].item(), "MASK 仍是有效输入，不是 PAD")
        self.assertEqual(result["target_valid"].sum().item(), 4,
                         "MASK、随机替换、保持原样，三类选中位置全部受监督")

    def test_no_input_mutation_and_all_outputs_have_independent_storage(self):
        values = fixture_inputs()
        snapshots = [value.clone() for value in values]
        result = data.make_mlm_batch(*values)
        input_storage = {value.untyped_storage().data_ptr() for value in values}
        output_storage = []
        for name, value in result.items():
            pointer = value.untyped_storage().data_ptr()
            self.assertNotIn(pointer, input_storage, f"{name} 不应与任一入参共享存储")
            output_storage.append(pointer)
        self.assertEqual(len(set(output_storage)), len(output_storage),
                         "返回的 input_ids / targets 等对象必须各自独立")
        for value, snapshot in zip(values, snapshots):
            self.assert_tensor(value, snapshot)
        result["input_ids"][0, 1] = 9
        self.assertEqual(result["targets"][0, 1].item(), 4)
        result["target_valid"].zero_()
        for value, snapshot in zip(values, snapshots):
            self.assert_tensor(value, snapshot)

    def test_noncontiguous_inputs(self):
        values = tuple(strided_copy(value) for value in fixture_inputs())
        self.assertTrue(all(not value.is_contiguous() for value in values))
        result = data.make_mlm_batch(*values)
        for name, expected in independent_batch().items():
            with self.subTest(field=name):
                self.assert_tensor(result[name], expected)

    def test_rejects_selected_special_token_or_invalid_input(self):
        for token in (data.PAD_ID, data.CLS_ID, data.SEP_ID, data.MASK_ID, 4):
            with self.subTest(token=token):
                values = list(fixture_inputs())
                values[0][0, 1] = token
                # 最后一例：ID 是普通词，但读取有效性明确为 False。
                if token == 4:
                    values[1][0, 1] = False
                with self.assertRaises(ValueError):
                    data.make_mlm_batch(*values)

    def test_rejects_batch_without_any_selected_position(self):
        values = list(fixture_inputs())
        values[3].zero_()
        with self.assertRaises(ValueError):
            data.make_mlm_batch(*values)
        # 约束是整个 batch 非空，不是要求每句话都选到目标。
        values = list(fixture_inputs())
        values[3][0].zero_()
        result = data.make_mlm_batch(*values)
        self.assertEqual(result["target_valid"].sum().item(), 1)
        self.assert_tensor(result["input_ids"][0], values[0][0])

    def test_rejects_invalid_kind_only_where_selected(self):
        for kind in (-1, 3, 99):
            with self.subTest(kind=kind):
                values = list(fixture_inputs())
                values[4][0, 1] = kind
                with self.assertRaises(ValueError):
                    data.make_mlm_batch(*values)
        # fixture 在未选中位置使用 99；那些位置必须被忽略。
        data.make_mlm_batch(*fixture_inputs())


class TestPretrainingObjectives(TensorAssertions):
    def test_model_receives_corrupted_inputs_not_clean_targets(self):
        batch, model = independent_batch(), LogitsSpy()
        training.pretraining_loss(model, batch, torch.tensor([0, 1]))
        self.assertTrue(model.seen, "必须经过模型调用入口，不能只对标签构造 loss")
        for observed in model.seen:
            for actual, name in zip(observed, ("input_ids", "segment_ids", "input_valid")):
                self.assert_tensor(actual, batch[name])
        self.assertEqual(model.seen[-1][0][0, 1].item(), data.MASK_ID,
                         "应遮盖的位置实际进入 Encoder 的必须是 MASK，不是原文")

    def test_unshifted_global_selected_mean_and_unweighted_sum_of_two_losses(self):
        batch, model = independent_batch(), LogitsSpy()
        labels = torch.tensor([1, 0])
        total, mlm, nsp = training.pretraining_loss(model, batch, labels)
        expected_mlm, expected_nsp = independent_losses(model.mlm_logits, model.nsp_logits,
                                                       batch, labels)
        for actual in (total, mlm, nsp):
            self.assertEqual(actual.shape, torch.Size([]))
            self.assertTrue(actual.requires_grad)
        self.assert_tensor(mlm, expected_mlm)
        self.assert_tensor(nsp, expected_nsp)
        self.assert_tensor(total, expected_mlm + expected_nsp)
        # 两行分别有 3 和 1 个目标，不能先做各句均值再平均。
        sentence_means = []
        for b in range(2):
            terms = [row_cross_entropy(model.mlm_logits[b, t], batch["targets"][b, t])
                     for t in range(8) if batch["target_valid"][b, t]]
            sentence_means.append(torch.stack(terms).mean())
        self.assertGreater(abs((torch.stack(sentence_means).mean() - expected_mlm).item()), .01,
                           "教师夹具必须能区分逐句均值和全 batch 目标均值")

    def test_both_losses_keep_exact_gradients_and_all_three_selected_branches(self):
        batch, model = independent_batch(), LogitsSpy()
        labels = torch.tensor([1, 0])
        total, _, _ = training.pretraining_loss(model, batch, labels)
        expected_mlm, expected_nsp = independent_losses(model.mlm_logits, model.nsp_logits,
                                                       batch, labels)
        parameters = (model.mlm_logits, model.nsp_logits)
        expected_grads = torch.autograd.grad(expected_mlm + expected_nsp, parameters)
        actual_grads = torch.autograd.grad(total, parameters)
        for actual, expected in zip(actual_grads, expected_grads):
            self.assert_tensor(actual, expected)
        mlm_gradient = actual_grads[0]
        self.assert_tensor(mlm_gradient[~batch["target_valid"]],
                           torch.zeros_like(mlm_gradient[~batch["target_valid"]]))
        for b, t in ((0, 1), (0, 2), (0, 3), (1, 4)):
            with self.subTest(batch=b, token=t):
                self.assertGreater(mlm_gradient[b, t].abs().sum().item(), 0.,
                                   "保持/随机替换后的普通 token 也不能漏算监督")

    def test_unsupervised_logits_and_labels_do_not_change_mlm_loss(self):
        batch, model = independent_batch(), LogitsSpy()
        labels = torch.tensor([0, 1])
        original = training.pretraining_loss(model, batch, labels)
        with torch.no_grad():
            model.mlm_logits[~batch["target_valid"]] = torch.arange(
                data.VOCAB_SIZE, dtype=model.mlm_logits.dtype) * 500.
        changed = {name: value.clone() for name, value in batch.items()}
        changed["targets"][~batch["target_valid"]] = 11
        result = training.pretraining_loss(model, changed, labels)
        for actual, expected in zip(result, original):
            self.assert_tensor(actual, expected)

    def test_loss_call_does_not_update_inputs_parameters_existing_gradients_or_mode(self):
        for train_mode in (True, False):
            with self.subTest(train_mode=train_mode):
                batch, model = independent_batch(), LogitsSpy()
                model.train(train_mode)
                labels = torch.tensor([1, 0])
                snapshots = {name: value.clone() for name, value in batch.items()}
                label_snapshot = labels.clone()
                before = [parameter.detach().clone() for parameter in model.parameters()]
                for parameter in model.parameters():
                    parameter.grad = torch.full_like(parameter, .125)
                training.pretraining_loss(model, batch, labels)
                self.assertEqual(model.training, train_mode)
                self.assert_tensor(labels, label_snapshot)
                for name, snapshot in snapshots.items():
                    self.assert_tensor(batch[name], snapshot)
                for parameter, snapshot in zip(model.parameters(), before):
                    self.assert_tensor(parameter, snapshot)
                    self.assert_tensor(parameter.grad, torch.full_like(parameter, .125))

    def test_classification_loss_extreme_logits_noncontiguous_and_gradient(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                logits = strided_copy(torch.tensor([[1000., 1001., -1000.],
                                                    [-999., -1000., 1000.]], dtype=dtype))
                logits.requires_grad_()
                labels = torch.tensor([2, 0])
                self.assertFalse(logits.is_contiguous())
                actual = training.classification_loss(logits, labels)
                expected = torch.stack([row_cross_entropy(row, label)
                                        for row, label in zip(logits, labels)]).mean()
                self.assertTrue(torch.isfinite(actual).item())
                self.assert_tensor(actual, expected)
                actual_gradient, = torch.autograd.grad(actual, logits)
                expected_gradient, = torch.autograd.grad(expected, logits)
                self.assert_tensor(actual_gradient, expected_gradient)

    def test_pretraining_noncontiguous_extreme_logits_are_finite_and_differentiable(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                batch = {name: strided_copy(value) for name, value in independent_batch().items()}
                model = LogitsSpy(dtype=dtype, noncontiguous=True, extreme=True)
                labels = torch.tensor([1, 0])
                self.assertFalse(model.mlm_logits.is_contiguous())
                self.assertFalse(model.nsp_logits.is_contiguous())
                total, mlm, nsp = training.pretraining_loss(model, batch, labels)
                expected_mlm, expected_nsp = independent_losses(model.mlm_logits, model.nsp_logits,
                                                               batch, labels)
                self.assert_tensor(mlm, expected_mlm)
                self.assert_tensor(nsp, expected_nsp)
                self.assert_tensor(total, expected_mlm + expected_nsp)
                total.backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())


if __name__ == "__main__":
    unittest.main()
