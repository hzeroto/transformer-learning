"""教师 MoE 实现的正确性测试，不用于自动上报学习者掌握。

CPU；float64 atol=1e-10/rtol=1e-8，float32 atol=2e-6/rtol=2e-5。
密集参照故意先计算所有专家再加权，用独立路径检查稀疏分发的前向/梯度。
"""

import copy
import io
import math
import unittest

import torch
from torch.nn import functional as F

from exercises.ex019_moe.moe import (
    CausalSelfAttention,
    SparseMoE,
    TinyMoELM,
    greedy_generate,
    make_inspection_case,
    make_training_batch,
    next_token_loss,
)


def setUpModule():
    global old_threads
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(old_threads)


def dense_reference(moe, x):
    """全专家参照：全概率→排序→截取→重归一；不使用待测分组循环。"""
    flat = x.reshape(-1, x.shape[-1])
    scores = flat @ moe.W_g
    probabilities = torch.softmax(scores, dim=-1)
    ids = torch.argsort(scores, descending=True, dim=-1)[:, :moe.top_k]
    selected = probabilities.gather(1, ids)
    selected = selected / selected.sum(dim=-1, keepdim=True)
    gates = torch.zeros_like(probabilities).scatter(1, ids, selected)
    all_outputs = torch.stack([
        F.relu(flat @ expert.W1 + expert.b1) @ expert.W2 + expert.b2
        for expert in moe.experts
    ], dim=1)
    output = (gates.unsqueeze(-1) * all_outputs).sum(dim=1).reshape_as(x)
    counts = torch.bincount(ids.reshape(-1), minlength=moe.num_experts)
    fractions = counts.to(x.dtype) / ids.numel()
    balance = moe.num_experts * (fractions * probabilities.mean(dim=0)).sum()
    return output, balance


class SparseMoETests(unittest.TestCase):
    def test_exact_two_token_walkthrough(self):
        moe, x = make_inspection_case()
        output, stats = moe(x)
        torch.testing.assert_close(stats.expert_ids, torch.tensor([[0, 1], [0, 2]]))
        torch.testing.assert_close(stats.weights, x.new_tensor([[2/3, 1/3], [8/13, 5/13]]))
        torch.testing.assert_close(output, x.new_tensor([[[4/3, 0], [0, 3/13]]]))
        torch.testing.assert_close(stats.counts, torch.tensor([2, 1, 1, 0]))
        torch.testing.assert_close(stats.fractions, x.new_tensor([.5, .25, .25, 0]))
        torch.testing.assert_close(stats.mean_probabilities, x.new_tensor([.5, .25, .15, .1]))
        self.assertAlmostEqual(stats.balance_loss.item(), 1.4)

    def test_sparse_forward_and_all_gradients_match_dense_reference(self):
        for dtype in (torch.float32, torch.float64):
            for top_k in (2, 4):
                with self.subTest(dtype=dtype, top_k=top_k):
                    torch.manual_seed(7301)
                    moe = SparseMoE(5, 7, 4, top_k).to(dtype=dtype)
                    reference = copy.deepcopy(moe)
                    # 转置后的非连续输入，避免把 view 的限制漏掉。
                    x = torch.randn(2, 5, 3, dtype=dtype).transpose(1, 2).requires_grad_()
                    ref_x = x.detach().clone().requires_grad_()
                    output, stats = moe(x)
                    ref_output, ref_balance = dense_reference(reference, ref_x)
                    atol, rtol = ((2e-6, 2e-5) if dtype == torch.float32 else (1e-10, 1e-8))
                    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)
                    torch.testing.assert_close(stats.balance_loss, ref_balance, atol=atol, rtol=rtol)
                    upstream = torch.randn_like(output)
                    ((output * upstream).sum() + .07 * stats.balance_loss).backward()
                    ((ref_output * upstream).sum() + .07 * ref_balance).backward()
                    torch.testing.assert_close(x.grad, ref_x.grad, atol=atol, rtol=rtol)
                    for (name, p), (_, q) in zip(moe.named_parameters(), reference.named_parameters()):
                        actual = torch.zeros_like(p) if p.grad is None else p.grad
                        torch.testing.assert_close(actual, q.grad, atol=atol, rtol=rtol, msg=name)

    def test_unselected_expert_is_not_called_and_has_no_task_gradient(self):
        moe, x = make_inspection_case()
        rows_seen = {}
        handles = []
        for e, expert in enumerate(moe.experts):
            def record(module, args, output, e=e):
                rows_seen[e] = args[0].shape[0]
            handles.append(expert.register_forward_hook(record))
        try:
            output, stats = moe(x)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(rows_seen, {0: 2, 1: 1, 2: 1})
        self.assertEqual(sum(rows_seen.values()), x.shape[0] * x.shape[1] * moe.top_k)
        (output.square().sum() + .01 * stats.balance_loss).backward()
        self.assertTrue(all(p.grad is None for p in moe.experts[3].parameters()))
        self.assertTrue(all(p.grad is not None for p in moe.experts[0].parameters()))

    def test_router_task_and_auxiliary_gradient_paths(self):
        moe, x = make_inspection_case()
        output, stats = moe(x)
        task_gradient = torch.autograd.grad(output.square().sum(), moe.W_g, retain_graph=True)[0]
        aux_gradient = torch.autograd.grad(stats.balance_loss, moe.W_g)[0]
        torch.testing.assert_close(task_gradient[:, 3], torch.zeros(2, dtype=x.dtype))
        self.assertGreater(task_gradient[:, :3].norm().item(), 0)
        self.assertTrue(bool((aux_gradient[:, 3].abs() > 1e-8).all()))
        # 独立核对 dL/dscores = E/N * p * (f - sum(p*f))。
        p, f = stats.probabilities.detach(), stats.fractions
        expected_scores_gradient = 4/2 * p * (f - (p * f).sum(-1, keepdim=True))
        expected_W_gradient = x.reshape(2, 2).T @ expected_scores_gradient
        torch.testing.assert_close(aux_gradient, expected_W_gradient)

    def test_identical_experts_remove_task_signal_through_mixing_weights(self):
        torch.manual_seed(7302)
        moe = SparseMoE(3, 5, 4, 2).double()
        for expert in moe.experts[1:]:
            expert.load_state_dict(moe.experts[0].state_dict())
        x = torch.randn(2, 4, 3, dtype=torch.float64)
        output, _ = moe(x)
        torch.testing.assert_close(output, moe.experts[0](x))
        router_gradient = torch.autograd.grad(output.square().sum(), moe.W_g)[0]
        torch.testing.assert_close(router_gradient, torch.zeros_like(router_gradient), atol=1e-12, rtol=0)

    def test_token_permutation_and_chunking_preserve_output(self):
        torch.manual_seed(7303)
        moe = SparseMoE(3, 5).double()
        x = torch.randn(1, 6, 3, dtype=torch.float64)
        output, stats = moe(x)
        permutation = torch.tensor([4, 1, 0, 5, 3, 2])
        permuted, permuted_stats = moe(x[:, permutation])
        torch.testing.assert_close(permuted, output[:, permutation])
        torch.testing.assert_close(permuted_stats.counts, stats.counts)
        chunks = torch.cat([moe(x[:, i:i+1])[0] for i in range(6)], dim=1)
        torch.testing.assert_close(chunks, output)
        # 辅助项按整批统计，不声称“分块辅助项的平均”等于整批辅助项。

    def test_large_logits_and_top_k_weights_stay_finite(self):
        moe = SparseMoE(1, 3, 4, 2).double()
        with torch.no_grad():
            moe.W_g.copy_(torch.tensor([[10000., 9999., -10000., -9999.]]))
        output, stats = moe(torch.ones(1, 2, 1, dtype=torch.float64))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(bool(torch.isfinite(stats.balance_loss)))
        torch.testing.assert_close(stats.weights.sum(-1), torch.ones(2, dtype=torch.float64))
        torch.testing.assert_close(stats.probabilities.sum(-1), torch.ones(2, dtype=torch.float64))
        (output.sum() + stats.balance_loss).backward()
        self.assertTrue(bool(torch.isfinite(moe.W_g.grad).all()))

    def test_rejects_top_one_and_invalid_input(self):
        for kwargs in ({"top_k": 1}, {"top_k": 5}, {"num_experts": 0}, {"top_k": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SparseMoE(3, 5, **kwargs)
        moe = SparseMoE(3, 5)
        for x in (torch.ones(2, 3), torch.ones(1, 2, 4), torch.ones(1, 0, 3),
                  torch.ones(1, 2, 3, dtype=torch.long)):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                moe(x)


class ModelIntegrationTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(7310)
        return TinyMoELM(5, max_length=12, d_model=8, num_heads=2, d_ff=13, num_layers=2).double()

    def test_attention_matches_independent_per_head_reference(self):
        torch.manual_seed(7311)
        attention = CausalSelfAttention(6, 3).double()
        x = torch.randn(2, 4, 6, dtype=torch.float64)
        results = []
        for h in range(3):
            q = x @ attention.Wq[:, h*2:(h+1)*2]
            k = x @ attention.Wk[:, h*2:(h+1)*2]
            v = x @ attention.Wv[:, h*2:(h+1)*2]
            scores = q @ k.transpose(-1, -2) / math.sqrt(2)
            mask = torch.ones(4, 4, dtype=torch.bool).tril()
            results.append(torch.softmax(scores.masked_fill(~mask, -torch.inf), -1) @ v)
        expected = torch.cat(results, -1) @ attention.Wo
        torch.testing.assert_close(attention(x), expected)

    def test_future_and_other_batch_rows_do_not_change_prefix_logits(self):
        model = self.model()
        inputs = torch.tensor([[0, 2, 3, 2, 3], [1, 2, 4, 2, 4]])
        original, _, _ = model(inputs)
        changed = inputs.clone()
        changed[0, 3:] = torch.tensor([1, 0])
        changed[1] = torch.tensor([4, 3, 1, 0, 2])
        actual, _, _ = model(changed)
        torch.testing.assert_close(actual[0, :3], original[0, :3])
        prefix, _, _ = model(inputs[:, :3])
        torch.testing.assert_close(prefix, original[:, :3])

    def test_loss_matches_cross_entropy_and_has_finite_extreme_gradients(self):
        logits = torch.tensor([[[10000., -10000., 0.], [-9999., 9999., 0.]]],
                              dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[1, 2]])
        actual = next_token_loss(logits, targets)
        expected = F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1))
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_all_experts_registered_and_save_restore_matches(self):
        model = self.model()
        inputs, _, _ = make_training_batch()
        expected, aux, _ = model(inputs)
        state = model.state_dict()
        for layer in range(2):
            for expert in range(4):
                self.assertIn(f"blocks.{layer}.moe.experts.{expert}.W1", state)
            self.assertIn(f"blocks.{layer}.moe.W_g", state)
        self.assertNotEqual(model.blocks[0].moe.W_g.data_ptr(), model.blocks[1].moe.W_g.data_ptr())
        pointers = [expert.W1.data_ptr() for expert in model.blocks[0].moe.experts]
        self.assertEqual(len(set(pointers)), 4)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        restored = self.model()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        actual, actual_aux, routing = restored(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_aux, aux, rtol=0, atol=0)
        torch.testing.assert_close(actual_aux, torch.stack([s.balance_loss for s in routing]).mean())

    def test_training_decreases_loss_and_updates_router(self):
        torch.manual_seed(42)
        model = TinyMoELM(5)
        inputs, targets, complete = make_training_batch()
        with torch.no_grad():
            before = next_token_loss(model(inputs)[0], targets).item()
        router_before = model.blocks[0].moe.W_g.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            logits, aux, _ = model(inputs)
            (next_token_loss(logits, targets) + .01 * aux).backward()
            optimizer.step()
        with torch.no_grad():
            after = next_token_loss(model(inputs)[0], targets).item()
        self.assertLess(after, before * .1)
        self.assertGreater((model.blocks[0].moe.W_g - router_before).abs().max().item(), 1e-5)
        generated = greedy_generate(model, complete[:, :2], 7)
        torch.testing.assert_close(generated, complete)

    def test_generation_preserves_mode_and_checks_length(self):
        model = self.model()
        prefix = torch.tensor([[0, 2]])
        for training in (True, False):
            model.train(training)
            calls = []
            def observe(module, args, output):
                calls.append((module.training, torch.is_grad_enabled(), output[0].requires_grad))
            handle = model.register_forward_hook(observe)
            try:
                generated = greedy_generate(model, prefix, 3)
            finally:
                handle.remove()
            self.assertEqual(calls, [(False, False, False)] * 3)
            self.assertEqual(model.training, training)
            self.assertEqual(generated.shape, (1, 5))
            torch.testing.assert_close(generated[:, :2], prefix)
        with self.assertRaises(ValueError):
            greedy_generate(model, prefix, 11)
        with self.assertRaises(ValueError):
            greedy_generate(model, prefix.float(), 0)
        torch.testing.assert_close(greedy_generate(model, prefix, 0), prefix)


if __name__ == "__main__":
    unittest.main()
