"""ex015 教师测试：同一精度策略的参照；不要求 BF16 与 FP32 逐位一致。

CPU FP32 参数，BF16 autocast，seed=1503/1511/1517，单线程。
教师数值参照复用已验收模型，但用官方交叉熵与 autograd.grad 独立核对 loss/梯度。
未实现时两个状态测试明确失败，其余对应行为测试只捕获 NotImplementedError 后跳过。
"""

import copy
import math
import unittest

import torch
from torch.nn import functional as F

from exercises.ex015_mixed_precision import precision as learner
from exercises.ex015_mixed_precision.fixtures import make_case


def setUpModule():
    global _old_threads
    _old_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_old_threads)


def require_cpu_bf16(test):
    """能力不足时明确跳过相关行为，不能伪装为 FP32 的 AMP 成功。"""
    try:
        w = torch.ones(1, 1, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = torch.ones(1, 1) @ w
        y.float().sum().backward()
        if y.dtype != torch.bfloat16:
            test.skipTest("当前 CPU autocast 未产生 BF16；需记录该后端限制")
    except RuntimeError as error:
        test.skipTest(f"当前 CPU BF16 前向/反向不支持：{error}")


def reference_pass(model, batch, use_bf16):
    """教师参照：不调用本练习的两个目标函数或 ex005 的交叉熵。"""
    reference_model = copy.deepcopy(model)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
        logits = reference_model(batch["input_ids"], batch["input_valid"])
        mask = batch["target_valid"]
        loss = F.cross_entropy(logits.float()[mask], batch["targets"][mask])
    named = dict(reference_model.named_parameters())
    values = torch.autograd.grad(loss, tuple(named.values()))
    return dict(logits=logits.detach().clone(), loss=loss.detach().clone(),
                grads={name: grad.detach().clone() for name, grad in zip(named, values)})


def tiny_result():
    return dict(logits=torch.tensor([[[1.0, -2.0, 0.0]]]),
                loss=torch.tensor(2.0),
                grads={"early": torch.tensor([1.0, -1.0]), "late": torch.tensor([[0.0]])})


class ImplementationStatusTest(unittest.TestCase):
    def test_run_precision_pass_is_implemented(self):
        model, batch = make_case(B=1, T=2)
        try:
            learner.run_precision_pass(model, **batch, use_bf16=False)
        except NotImplementedError:
            self.fail("请填写 precision.py 的 run_precision_pass")

    def test_compare_results_is_implemented(self):
        try:
            learner.compare_results(tiny_result(), tiny_result())
        except NotImplementedError:
            self.fail("请填写 precision.py 的 compare_results")


class PrecisionPassTest(unittest.TestCase):
    def setUp(self):
        self.model, self.batch = make_case()
        try:
            learner.run_precision_pass(self.model, **self.batch, use_bf16=False)
        except NotImplementedError:
            self.skipTest("run_precision_pass 尚未实现；另有状态测试明确失败")

    def assert_result(self, actual, expected, model, use_bf16):
        self.assertEqual(set(actual), {"logits", "loss", "grads"})
        self.assertEqual(set(actual["grads"]), set(dict(model.named_parameters())))
        self.assertEqual(actual["logits"].dtype,
                         torch.bfloat16 if use_bf16 else torch.float32)
        self.assertEqual(actual["loss"].shape, torch.Size([]))
        self.assertEqual(actual["loss"].dtype, torch.float32)
        pairs = [("logits", actual["logits"], expected["logits"]),
                 ("loss", actual["loss"], expected["loss"])]
        pairs += [(name, actual["grads"][name], expected["grads"][name])
                  for name in expected["grads"]]
        for name, value, reference in pairs:
            with self.subTest(object=name):
                self.assertEqual(value.shape, reference.shape)
                self.assertEqual(value.device.type, "cpu")
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)
                self.assertTrue(torch.isfinite(value).all())
                torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)
        self.assertTrue(all(g.dtype == torch.float32 for g in actual["grads"].values()))

    def test_fp32_and_bf16_match_same_policy_reference(self):
        require_cpu_bf16(self)
        for seed, B, T, Hkv in ((1503, 2, 6, 2), (1511, 1, 1, 1), (1517, 2, 4, 6)):
            for enabled in (False, True):
                with self.subTest(seed=seed, B=B, T=T, Hkv=Hkv, bf16=enabled):
                    model, batch = make_case(seed, B=B, T=T, Hkv=Hkv)
                    expected = reference_pass(model, batch, enabled)
                    actual = learner.run_precision_pass(model, **batch, use_bf16=enabled)
                    self.assert_result(actual, expected, model, enabled)

    def test_old_gradients_are_cleared_for_each_run(self):
        require_cpu_bf16(self)
        for p in self.model.parameters():
            p.grad = torch.full_like(p, 7.0)
        for enabled in (False, True, True):
            expected = reference_pass(self.model, self.batch, enabled)
            actual = learner.run_precision_pass(self.model, **self.batch, use_bf16=enabled)
            self.assert_result(actual, expected, self.model, enabled)

    def test_parameters_inputs_and_training_mode_are_unchanged(self):
        require_cpu_bf16(self)
        weights = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        inputs = {n: t.clone() for n, t in self.batch.items()}
        for training in (True, False):
            self.model.train(training)
            for enabled in (False, True):
                learner.run_precision_pass(self.model, **self.batch, use_bf16=enabled)
                self.assertTrue(all(m.training == training for m in self.model.modules()))
                for name, p in self.model.named_parameters():
                    self.assertEqual(p.dtype, torch.float32, name)
                    self.assertTrue(p.requires_grad, name)
                    self.assertTrue(torch.equal(p, weights[name]), f"参数被改变：{name}")
                for name, value in self.batch.items():
                    self.assertTrue(torch.equal(value, inputs[name]), name)

    def test_snapshots_have_no_graph_and_do_not_alias_live_results(self):
        captured = []
        handle = self.model.register_forward_hook(
            lambda module, args, output: captured.append(output))
        try:
            result = learner.run_precision_pass(self.model, **self.batch, use_bf16=False)
        finally:
            handle.remove()
        self.assertEqual(len(captured), 1)
        self.assertFalse(result["logits"].requires_grad)
        self.assertFalse(result["loss"].requires_grad)
        self.assertNotEqual(result["logits"].untyped_storage().data_ptr(),
                            captured[0].untyped_storage().data_ptr())
        for name, p in self.model.named_parameters():
            value = result["grads"][name]
            self.assertFalse(value.requires_grad)
            self.assertIsNotNone(p.grad, name)
            before = value.clone()
            self.assertNotEqual(value.untyped_storage().data_ptr(),
                                p.grad.untyped_storage().data_ptr(), name)
            p.grad.fill_(99.0)
            self.assertTrue(torch.equal(value, before), f"快照被后续梯度修改污染：{name}")

    def test_target_mask_is_distinct_from_key_visibility(self):
        require_cpu_bf16(self)
        batch = {n: t.clone() for n, t in self.batch.items()}
        batch["input_valid"][1, 2] = False
        batch["target_valid"][0, :] = False  # 该序列作为输入仍然有效。
        batch["target_valid"][1, 4:] = False
        changed = {n: t.clone() for n, t in batch.items()}
        invalid = ~batch["target_valid"]
        changed["targets"][invalid] = (changed["targets"][invalid] + 5) % self.model.N
        for enabled in (False, True):
            expected = reference_pass(self.model, batch, enabled)
            original = learner.run_precision_pass(self.model, **batch, use_bf16=enabled)
            altered = learner.run_precision_pass(self.model, **changed, use_bf16=enabled)
            self.assert_result(original, expected, self.model, enabled)
            self.assert_result(altered, expected, self.model, enabled)

    def test_autocast_changes_internal_ffn_without_casting_parameters(self):
        require_cpu_bf16(self)
        observed = []

        def inspect(module, args, output):
            observed.append((output.dtype, {p.dtype for p in module.parameters()}))

        handle = self.model.blocks[0].ffn.register_forward_hook(inspect)
        try:
            for enabled in (True, False):
                learner.run_precision_pass(self.model, **self.batch, use_bf16=enabled)
        finally:
            handle.remove()
        self.assertEqual(observed, [(torch.bfloat16, {torch.float32}),
                                    (torch.float32, {torch.float32})],
                         "需要真正改变内部运算精度，不能只在最后转换 logits")


class CompareResultsTest(unittest.TestCase):
    def setUp(self):
        try:
            learner.compare_results(tiny_result(), tiny_result())
        except NotImplementedError:
            self.skipTest("compare_results 尚未实现；另有状态测试明确失败")

    def assert_report(self, actual, expected):
        self.assertEqual(set(actual), {"logits_max_abs", "loss_abs", "grad_max_abs", "all_finite"})
        self.assertIs(type(actual["all_finite"]), bool)
        self.assertEqual(actual["all_finite"], expected["all_finite"])
        for key in ("logits_max_abs", "loss_abs", "grad_max_abs"):
            self.assertIs(type(actual[key]), float, key)
            if math.isinf(expected[key]):
                self.assertEqual(actual[key], float("inf"), key)
            else:
                self.assertAlmostEqual(actual[key], expected[key], places=7, msg=key)

    def test_known_errors_include_every_parameter(self):
        ref, candidate = tiny_result(), tiny_result()
        candidate["logits"] = torch.tensor([[[1.5, -3.0, 0.25]]], dtype=torch.bfloat16)
        candidate["loss"] = torch.tensor(2.125)
        candidate["grads"]["early"] = torch.tensor([-3.0, -1.0])
        candidate["grads"]["late"] = torch.tensor([[0.5]])
        self.assert_report(learner.compare_results(ref, candidate),
                           dict(logits_max_abs=1.0, loss_abs=0.125,
                                grad_max_abs=4.0, all_finite=True))

    def test_identical_and_zero_results(self):
        for zero in (False, True):
            ref = tiny_result()
            if zero:
                ref["logits"].zero_()
                ref["loss"].zero_()
                for g in ref["grads"].values():
                    g.zero_()
            self.assert_report(learner.compare_results(ref, copy.deepcopy(ref)),
                               dict(logits_max_abs=0.0, loss_abs=0.0,
                                    grad_max_abs=0.0, all_finite=True))

    def test_comparison_preserves_fp32_reference_detail(self):
        ref, candidate = tiny_result(), tiny_result()
        ref["logits"] = torch.tensor([[[1.003]]], dtype=torch.float32)
        candidate["logits"] = torch.tensor([[[1.0]]], dtype=torch.bfloat16)
        expected = abs(float(ref["logits"].item()) - 1.0)
        self.assert_report(learner.compare_results(ref, candidate),
                           dict(logits_max_abs=expected, loss_abs=0.0,
                                grad_max_abs=0.0, all_finite=True))

    def test_nonfinite_values_in_either_side_and_any_group_are_reported(self):
        for side in (0, 1):
            for group, metric in (("logits", "logits_max_abs"),
                                  ("loss", "loss_abs"), ("grads", "grad_max_abs")):
                for bad in (float("nan"), float("inf"), -float("inf")):
                    with self.subTest(side=side, group=group, value=bad):
                        inputs = [tiny_result(), tiny_result()]
                        value = (inputs[side]["grads"]["early"] if group == "grads"
                                 else inputs[side][group])
                        value.reshape(-1)[0] = bad
                        expected = dict(logits_max_abs=0.0, loss_abs=0.0,
                                        grad_max_abs=0.0, all_finite=False)
                        expected[metric] = float("inf")
                        self.assert_report(learner.compare_results(*inputs), expected)

    def test_comparison_does_not_modify_inputs(self):
        ref, candidate = tiny_result(), tiny_result()
        candidate["logits"] = candidate["logits"].bfloat16()
        before = copy.deepcopy([ref, candidate])
        learner.compare_results(ref, candidate)
        for after, original in zip((ref, candidate), before):
            for key in ("logits", "loss"):
                self.assertEqual(after[key].dtype, original[key].dtype)
                self.assertTrue(torch.equal(after[key], original[key]))
            for key in original["grads"]:
                self.assertTrue(torch.equal(after["grads"][key], original["grads"][key]))


if __name__ == "__main__":
    unittest.main()
