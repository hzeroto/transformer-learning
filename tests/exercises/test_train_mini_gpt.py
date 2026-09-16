"""ex010 训练闭环的验证。

约定：CPU、float32、torch.manual_seed 固定；容差在下方声明。
本文件由教师提供，不属于学习者独立设计的测试。
"""

import io
import unittest

import torch

from exercises.ex005_training_loop.loss import masked_cross_entropy
from exercises.ex009_mini_gpt.model import MiniGPT
from exercises.ex010_train_mini_gpt.dataset import (
    BOS_ID,
    CONTENT_IDS,
    EOS_ID,
    MODE_A,
    MODE_B,
    N,
    OUTPUT_SLICE,
    split_by_triple,
    split_random_labels,
)
from exercises.ex010_train_mini_gpt.training import (
    evaluate,
    greedy_generate,
    load_checkpoint,
    make_supervised_batch,
    save_checkpoint,
    train_model,
)

ATOL = 1e-10
DTYPE = torch.float32
CONFIG = dict(
    vocab_size=N, C=64, num_heads=4, ffn_hidden=128, n_layer=2, max_positions=16
)


def make_model(seed=0, p=0.0, **overrides):
    cfg = {**CONFIG, **overrides}
    torch.manual_seed(seed)
    return MiniGPT(**cfg, norm_style="pre", p=p, dtype=DTYPE)


def trained(seed=0, steps=400, p=0.0):
    """教师侧的共享夹具：训练一个模型供多个用例复用。"""
    train_ids, val_ids = split_by_triple()
    model = make_model(seed=seed, p=p)
    opt = train_model(model, train_ids, steps=steps, seed=seed)
    return model, opt, train_ids, val_ids


class TestDataset(unittest.TestCase):
    """数据集由教师提供，这里只锁定它的关键性质，避免后续用例建立在错误前提上。"""

    def test_split_shapes_and_disjoint_triples(self):
        train_ids, val_ids = split_by_triple()
        self.assertEqual(tuple(train_ids.shape), (346, 9))
        self.assertEqual(tuple(val_ids.shape), (86, 9))
        train_triples = {tuple(row[2:5].tolist()) for row in train_ids}
        val_triples = {tuple(row[2:5].tolist()) for row in val_ids}
        self.assertEqual(train_triples & val_triples, set())

    def test_rule_holds_in_both_modes(self):
        train_ids, val_ids = split_by_triple()
        for row in torch.cat([train_ids, val_ids]):
            p, q, r = row[2].item(), row[3].item(), row[4].item()
            outputs = row[5:8].tolist()
            if row[1].item() == MODE_A:
                self.assertEqual(outputs, [p, q, r])
            else:
                self.assertEqual(outputs, [r, q, p])
            self.assertEqual(row[0].item(), BOS_ID)
            self.assertEqual(row[8].item(), EOS_ID)


class TestSupervisedBatch(unittest.TestCase):
    def test_shapes_and_shift(self):
        ids, _ = split_by_triple()
        batch = ids[:5]
        inputs, targets, input_valid, target_valid = make_supervised_batch(batch)
        for tensor in (inputs, targets, input_valid, target_valid):
            self.assertEqual(tuple(tensor.shape), (5, 8))
        self.assertTrue(torch.equal(inputs, batch[:, :-1]))
        self.assertTrue(torch.equal(targets, batch[:, 1:]))
        self.assertEqual(inputs.dtype, torch.long)
        self.assertEqual(targets.dtype, torch.long)

    def test_only_output_span_is_supervised(self):
        ids, _ = split_by_triple()
        _, _, input_valid, target_valid = make_supervised_batch(ids[:4])
        self.assertEqual(input_valid.dtype, torch.bool)
        self.assertEqual(target_valid.dtype, torch.bool)
        # 输入端无 PAD，全部可读
        self.assertTrue(input_valid.all())
        expected = torch.zeros(4, 8, dtype=torch.bool)
        expected[:, OUTPUT_SLICE] = True
        self.assertTrue(
            torch.equal(target_valid, expected),
            f"target_valid 应只在 {OUTPUT_SLICE} 为 True，实际每行 True 数="
            f"{target_valid[0].sum().item()}",
        )
        # 两种有效性不能是同一个对象，也不应数值相同
        self.assertFalse(torch.equal(input_valid, target_valid))

    def test_targets_align_with_output_tokens(self):
        """错位方向写反时，这一项会失败。"""
        ids, _ = split_by_triple()
        row = ids[0:1]
        _, targets, _, target_valid = make_supervised_batch(row)
        supervised = targets[0][target_valid[0]].tolist()
        self.assertEqual(supervised, row[0, 5:9].tolist())

    def test_input_not_modified(self):
        ids, _ = split_by_triple()
        batch = ids[:8].clone()
        before = batch.clone()
        make_supervised_batch(batch)
        self.assertTrue(torch.equal(batch, before))


class TestEvaluate(unittest.TestCase):
    def test_returns_python_floats_and_matches_reference_loss(self):
        model = make_model()
        _, val_ids = split_by_triple()
        loss, acc = evaluate(model, val_ids)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        # 独立复算参照损失，不调用待测 evaluate
        inputs, targets, input_valid, target_valid = make_supervised_batch(val_ids)
        model.eval()
        with torch.no_grad():
            expected = masked_cross_entropy(
                model(inputs, input_valid), targets, target_valid
            ).item()
        self.assertAlmostEqual(loss, expected, delta=1e-6)

    def test_untrained_loss_near_uniform(self):
        """随机初始化时，logits 接近 0，损失应接近 ln(N)。"""
        import math

        model = make_model()
        _, val_ids = split_by_triple()
        loss, acc = evaluate(model, val_ids)
        self.assertLess(abs(loss - math.log(N)), 0.5, f"初始损失 {loss} 偏离 ln(N)")
        self.assertLess(acc, 0.2)

    def test_accuracy_requires_whole_output_span(self):
        """逐位置正确率会给出更高的数字；这一项要求的是整条全对。"""
        model = make_model()
        _, val_ids = split_by_triple()
        inputs, targets, input_valid, target_valid = make_supervised_batch(val_ids)
        model.eval()
        with torch.no_grad():
            pred = model(inputs, input_valid).argmax(-1)
        per_position = (pred == targets)[target_valid].float().mean().item()
        whole = ((pred == targets) | ~target_valid).all(dim=1).float().mean().item()
        _, acc = evaluate(model, val_ids)
        self.assertAlmostEqual(acc, whole, delta=1e-9)
        if abs(per_position - whole) > 1e-9:
            self.assertNotAlmostEqual(acc, per_position, delta=1e-9)

    def test_does_not_leave_model_in_eval_or_touch_grads(self):
        model = make_model()
        model.train()
        _, val_ids = split_by_triple()
        evaluate(model, val_ids)
        self.assertTrue(model.training, "evaluate 结束后应恢复调用前的 training 状态")
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_no_grad_path(self):
        """evaluate 内部不应建图；参数不应因评估而获得梯度。"""
        model = make_model()
        _, val_ids = split_by_triple()
        evaluate(model, val_ids[:16])
        self.assertTrue(all(p.grad is None for p in model.parameters()))


class TestTraining(unittest.TestCase):
    def test_returns_adam_and_learns_the_rule(self):
        model, opt, train_ids, val_ids = trained(steps=400)
        self.assertIsInstance(opt, torch.optim.Adam)
        train_loss, train_acc = evaluate(model, train_ids)
        val_loss, val_acc = evaluate(model, val_ids)
        self.assertLess(train_loss, 0.05, f"训练损失仍为 {train_loss}")
        self.assertGreater(train_acc, 0.95)
        # 关键：验证集三元组训练时从未出现，高准确率说明学到的是规则
        self.assertGreater(
            val_acc, 0.9, f"验证准确率仅 {val_acc}，模型可能只记住了训练样本"
        )

    def test_all_parameters_move(self):
        """漏掉某组参数（例如块没被注册）时，它们会原封不动。"""
        model = make_model()
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        train_ids, _ = split_by_triple()
        train_model(model, train_ids, steps=30)
        for name, param in model.named_parameters():
            self.assertFalse(
                torch.equal(param, before[name]), f"参数 {name} 训练后没有变化"
            )

    def test_gradients_are_cleared_each_step(self):
        """忘记 zero_grad 时梯度会累积，训练无法收敛。"""
        model = make_model()
        train_ids, _ = split_by_triple()
        train_model(model, train_ids, steps=200)
        loss, acc = evaluate(model, train_ids)
        self.assertLess(loss, 0.2, f"200 步后损失 {loss}，检查是否每步都清了梯度")
        self.assertGreater(acc, 0.8)

    def test_seed_makes_batches_reproducible(self):
        """相同 seed 必须给出相同的批次序列。

        这里比较 1 步之后的参数而不是 40 步：float32 下逐步累积的浮点误差
        会让同一份代码两次运行也产生 ~1e-3 的差异（已实测），用长训练比较
        参数无法区分「批次不同」和「浮点抖动」。单步的差异仍在 1e-7 量级。
        """
        train_ids, _ = split_by_triple()
        a, b = make_model(seed=0), make_model(seed=0)
        train_model(a, train_ids, steps=1, seed=7)
        train_model(b, train_ids, steps=1, seed=7)
        for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
            torch.testing.assert_close(pa, pb, atol=1e-5, rtol=1e-4, msg=name)

    def test_different_seed_gives_different_batches(self):
        train_ids, _ = split_by_triple()
        a, b = make_model(seed=0), make_model(seed=0)
        train_model(a, train_ids, steps=40, seed=1)
        train_model(b, train_ids, steps=40, seed=2)
        diffs = [
            (pa - pb).abs().max().item()
            for pa, pb in zip(a.parameters(), b.parameters())
        ]
        # 实测同 seed 的浮点抖动约 5e-3，不同 seed 约 1.4e-1，留出量级间隔
        self.assertGreater(max(diffs), 3e-2, "不同 seed 应产生不同的批次序列")

    def test_model_left_in_train_mode_is_not_required_but_grads_exist(self):
        """训练后参数应带有梯度，说明反向确实跑过。"""
        model = make_model()
        train_ids, _ = split_by_triple()
        train_model(model, train_ids, steps=5)
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_grad_clip_is_applied(self):
        """裁剪生效时，SGD 式的爆炸会被抑制；这里只验证参数被接受且训练仍可收敛。"""
        model = make_model()
        train_ids, _ = split_by_triple()
        train_model(model, train_ids, steps=200, grad_clip=1.0)
        loss, _ = evaluate(model, train_ids)
        self.assertLess(loss, 0.3, f"带裁剪训练 200 步后损失 {loss}")


class TestMemorizationContrast(unittest.TestCase):
    """同一套代码，两份数据：区分「训练损失下降」与「学会规则」。"""

    def test_random_labels_memorized_but_do_not_generalize(self):
        train_ids, val_ids = split_random_labels()
        model = make_model()
        train_model(model, train_ids, steps=900)
        train_loss, train_acc = evaluate(model, train_ids)
        val_loss, val_acc = evaluate(model, val_ids)
        # 训练集：能背下来
        self.assertLess(train_loss, 0.5, f"训练损失 {train_loss}，应能记住无规则数据")
        self.assertGreater(train_acc, 0.7)
        # 验证集：没有规则可迁移
        self.assertGreater(
            val_loss,
            train_loss * 3,
            f"无规则数据上验证损失({val_loss})应明显高于训练损失({train_loss})",
        )
        self.assertLess(val_acc, 0.2, f"无规则数据的验证准确率应接近 0，实际 {val_acc}")


class TestCheckpoint(unittest.TestCase):
    def test_roundtrip_is_bitwise_identical(self):
        model, opt, _, val_ids = trained(steps=120)
        inputs, _, input_valid, _ = make_supervised_batch(val_ids[:8])
        model.eval()
        with torch.no_grad():
            before = model(inputs, input_valid).clone()

        buffer = io.BytesIO()
        save_checkpoint(buffer, model, opt, dict(CONFIG))
        buffer.seek(0)
        restored, restored_opt, config = load_checkpoint(buffer)

        with torch.no_grad():
            after = restored(inputs, input_valid)
        self.assertTrue(
            torch.equal(before, after),
            f"恢复后 logits 最大差 {(before - after).abs().max().item()}",
        )
        self.assertIsInstance(restored_opt, torch.optim.Adam)
        self.assertEqual(config["n_layer"], CONFIG["n_layer"])
        self.assertFalse(restored.training, "load_checkpoint 应返回 eval 态模型")

    def test_checkpoint_contains_three_parts(self):
        model, opt, _, _ = trained(steps=20)
        buffer = io.BytesIO()
        save_checkpoint(buffer, model, opt, dict(CONFIG))
        buffer.seek(0)
        blob = torch.load(buffer, weights_only=False)
        self.assertEqual(set(blob.keys()), {"model", "opt", "config"})
        self.assertIn("state", blob["opt"])
        self.assertIn("param_groups", blob["opt"])

    def test_optimizer_state_carries_adam_moments(self):
        model, opt, _, _ = trained(steps=20)
        buffer = io.BytesIO()
        save_checkpoint(buffer, model, opt, dict(CONFIG))
        buffer.seek(0)
        _, restored_opt, _ = load_checkpoint(buffer)
        state = restored_opt.state_dict()["state"]
        self.assertGreater(len(state), 0, "优化器状态为空，可能没有 load_state_dict")
        first = state[next(iter(state))]
        self.assertIn("exp_avg", first)
        self.assertIn("exp_avg_sq", first)

    def test_load_actually_loads_rather_than_rebuilds(self):
        """只按 config 重新构造会得到随机参数；这一项能区分。"""
        model, opt, _, val_ids = trained(steps=120)
        buffer = io.BytesIO()
        save_checkpoint(buffer, model, opt, dict(CONFIG))
        buffer.seek(0)
        restored, _, _ = load_checkpoint(buffer)
        for (name, a), (_, b) in zip(
            model.named_parameters(), restored.named_parameters()
        ):
            self.assertTrue(torch.equal(a, b), f"参数 {name} 与存档不一致")


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.model, _, _, self.val_ids = trained(steps=400)

    def test_follows_copy_and_reverse_rules(self):
        p, q, r = CONTENT_IDS[0], CONTENT_IDS[1], CONTENT_IDS[2]
        for mode, expected in (
            (MODE_A, [p, q, r, EOS_ID]),
            (MODE_B, [r, q, p, EOS_ID]),
        ):
            prefix = torch.tensor([[BOS_ID, mode, p, q, r]])
            out = greedy_generate(self.model, prefix, max_new_tokens=6)
            self.assertEqual(out[0, :5].tolist(), prefix[0].tolist())
            self.assertEqual(out[0, 5:].tolist(), expected)

    def test_generalizes_on_unseen_triples(self):
        correct = 0
        for row in self.val_ids:
            prefix = row[:5].unsqueeze(0)
            out = greedy_generate(self.model, prefix, max_new_tokens=6)
            correct += out[0].tolist() == row.tolist()
        ratio = correct / len(self.val_ids)
        self.assertGreater(
            ratio, 0.9, f"未见三元组上的逐条生成正确率仅 {ratio:.3f}"
        )

    def test_stops_at_eos_and_keeps_it(self):
        p, q, r = CONTENT_IDS[3], CONTENT_IDS[4], CONTENT_IDS[5]
        prefix = torch.tensor([[BOS_ID, MODE_A, p, q, r]])
        out = greedy_generate(self.model, prefix, max_new_tokens=6)
        self.assertEqual(out[0, -1].item(), EOS_ID)
        self.assertEqual((out[0] == EOS_ID).sum().item(), 1)
        self.assertLess(out.shape[1], prefix.shape[1] + 6)

    def test_zero_budget_and_eos_prefix_return_copy(self):
        prefix = torch.tensor([[BOS_ID, MODE_A, CONTENT_IDS[0]]])
        out = greedy_generate(self.model, prefix, max_new_tokens=0)
        self.assertTrue(torch.equal(out, prefix))
        self.assertIsNot(out, prefix)

        ended = torch.tensor([[BOS_ID, MODE_A, CONTENT_IDS[0], EOS_ID]])
        out2 = greedy_generate(self.model, ended, max_new_tokens=5)
        self.assertTrue(torch.equal(out2, ended))

    def test_does_not_modify_inputs_or_model_state(self):
        prefix = torch.tensor([[BOS_ID, MODE_B, *CONTENT_IDS[:3]]])
        before = prefix.clone()
        self.model.train()
        # 模型已训练过，.grad 非空；这里要求生成不改变它们，而不是要求为 None
        grads_before = {
            name: (None if p.grad is None else p.grad.detach().clone())
            for name, p in self.model.named_parameters()
        }
        greedy_generate(self.model, prefix, max_new_tokens=6)
        self.assertTrue(torch.equal(prefix, before))
        self.assertTrue(
            self.model.training, "生成结束后应恢复调用前的 training 状态"
        )
        for name, param in self.model.named_parameters():
            old = grads_before[name]
            if old is None:
                self.assertIsNone(param.grad, f"{name} 的梯度被生成过程创建了")
            else:
                self.assertTrue(
                    torch.equal(param.grad, old), f"{name} 的梯度被生成过程改动了"
                )

    def test_deterministic_across_calls(self):
        prefix = torch.tensor([[BOS_ID, MODE_A, *CONTENT_IDS[:3]]])
        a = greedy_generate(self.model, prefix, max_new_tokens=6)
        b = greedy_generate(self.model, prefix, max_new_tokens=6)
        self.assertTrue(torch.equal(a, b), "贪心生成应完全确定")

    def test_returns_long_tensor_with_batch_axis(self):
        prefix = torch.tensor([[BOS_ID, MODE_A, *CONTENT_IDS[:3]]])
        out = greedy_generate(self.model, prefix, max_new_tokens=6)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.dtype, torch.long)
        self.assertEqual(out.dim(), 2)
        self.assertEqual(out.shape[0], 1)

    def test_switches_to_eval_even_when_called_in_train_mode(self):
        """带 Dropout 的模型停在 train 态生成时，结果会随机波动。

        上面的用例都用 p=0.0 的模型，Dropout 是恒等映射，检测不到这个错误。
        这里用 p=0.3 且**只训练 30 步**的检查点：充分训练后模型对每步的选择
        非常确定，Dropout 的扰动不足以让 argmax 翻转（实测 200 步时 12 次
        生成结果完全一致），只有在欠训练的检查点上这个错误才可观察。
        """
        train_ids, _ = split_by_triple()
        model = make_model(seed=0, p=0.3)
        train_model(model, train_ids, steps=30, seed=0)
        prefix = torch.tensor([[BOS_ID, MODE_B, *CONTENT_IDS[:3]]])
        model.train()
        runs = [
            tuple(greedy_generate(model, prefix, max_new_tokens=6)[0].tolist())
            for _ in range(12)
        ]
        self.assertEqual(
            len(set(runs)),
            1,
            f"train 态下调用得到 {len(set(runs))} 种不同结果，说明生成没有切到 eval",
        )
        self.assertTrue(model.training, "生成结束后应恢复调用前的 training 状态")


class TestEvalModeMatters(unittest.TestCase):
    def test_dropout_model_is_deterministic_only_in_eval(self):
        model, _, _, val_ids = trained(steps=200, p=0.3)
        inputs, _, input_valid, _ = make_supervised_batch(val_ids[:16])
        model.eval()
        with torch.no_grad():
            a, b = model(inputs, input_valid), model(inputs, input_valid)
        self.assertTrue(torch.equal(a, b))
        model.train()
        with torch.no_grad():
            c, d = model(inputs, input_valid), model(inputs, input_valid)
        self.assertFalse(torch.equal(c, d))
        # evaluate 必须自己切 eval，即使调用前模型处于 train 态
        model.train()
        first, _ = evaluate(model, val_ids)
        model.train()
        second, _ = evaluate(model, val_ids)
        self.assertAlmostEqual(first, second, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
