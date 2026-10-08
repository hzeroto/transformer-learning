"""ex017 教师模型测试；参照仅供验收，不得从作业导入作为实现。

CPU、固定种子 1703/1717；float64 rtol=1e-8/atol=1e-10，
float32 rtol=2e-5/atol=2e-6。逐头读取和正弦参照不调用作业 helper。
直接验证教师导读版实现；测试通过不自动代表学习者已独立掌握实现。
"""

import copy
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from exercises.ex017_original_transformer import model as learner


def setUpModule():
    global _old_threads
    _old_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_old_threads)


def fill_parameters(module):
    generator = torch.Generator().manual_seed(1717)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator,
                                        dtype=parameter.dtype) * .23)
            if name.endswith("gamma"):
                parameter.add_(1.)


def reference_allowed(valid, query_length, causal=False):
    return torch.tensor([[[bool(valid[batch, key]) and (not causal or key <= query)
                           for key in range(valid.shape[1])]
                          for query in range(query_length)]
                         for batch in range(valid.shape[0])], dtype=torch.bool)


def reference_attention(attention, query_states, memory, allowed):
    head_width = query_states.shape[-1] // attention.num_heads
    outputs, weights = [], []
    for head in range(attention.num_heads):
        part = slice(head * head_width, (head + 1) * head_width)
        query = query_states @ attention.Wq[:, part]
        key = memory @ attention.Wk[:, part]
        value = memory @ attention.Wv[:, part]
        samples, sample_weights = [], []
        for batch in range(query_states.shape[0]):
            scores = query[batch] @ key[batch].T / math.sqrt(head_width)
            probability = scores.masked_fill(~allowed[batch], -torch.inf).softmax(-1)
            samples.append(probability @ value[batch])
            sample_weights.append(probability)
        outputs.append(torch.stack(samples))
        weights.append(torch.stack(sample_weights))
    return torch.cat(outputs, -1) @ attention.Wo, torch.stack(weights, 1)


def reference_norm(states, norm):
    centered = states - states.mean(-1, keepdim=True)
    return centered / (centered.square().mean(-1, keepdim=True) + norm.eps).sqrt() * norm.gamma + norm.beta


def reference_ffn(states, ffn):
    return (states @ ffn.W1 + ffn.b1).clamp_min(0) @ ffn.W2 + ffn.b2


def reference_encoder(block, states, src_valid):
    allowed = reference_allowed(src_valid, states.shape[1])
    update, _ = reference_attention(block.self_attn, states, states, allowed)
    states = reference_norm(states + block.drop1(update), block.norm1)
    return reference_norm(states + block.drop2(reference_ffn(states, block.ffn)), block.norm2)


def reference_decoder(block, states, memory, src_valid, tgt_valid):
    allowed = reference_allowed(tgt_valid, states.shape[1], causal=True)
    update, _ = reference_attention(block.self_attn, states, states, allowed)
    states = reference_norm(states + block.drop1(update), block.norm1)
    allowed = reference_allowed(src_valid, states.shape[1])
    update, _ = reference_attention(block.cross_attn, states, memory, allowed)
    states = reference_norm(states + block.drop2(update), block.norm2)
    return reference_norm(states + block.drop3(reference_ffn(states, block.ffn)), block.norm3)


def reference_positions(length, width, dtype):
    return torch.tensor([[math.sin(position / 10000 ** (2 * (channel // 2) / width))
                          if channel % 2 == 0 else
                          math.cos(position / 10000 ** (2 * (channel // 2) / width))
                          for channel in range(width)] for position in range(length)], dtype=dtype)


def reference_encode(model, src_ids, src_valid):
    states = math.sqrt(model.C) * model.src_table[src_ids]
    states = model.src_drop(states + reference_positions(src_ids.shape[1], model.C, states.dtype))
    for block in model.encoders:
        states = reference_encoder(block, states, src_valid)
    return states


def reference_decode(model, tgt_ids, memory, src_valid, tgt_valid):
    states = math.sqrt(model.C) * model.tgt_table[tgt_ids]
    states = model.tgt_drop(states + reference_positions(tgt_ids.shape[1], model.C, states.dtype))
    for block in model.decoders:
        states = reference_decoder(block, states, memory, src_valid, tgt_valid)
    return states @ model.W_vocab + model.b_vocab


def cross_sample(dtype=torch.float64, equal_lengths=False, layout="contiguous", grad=False):
    with torch.random.fork_rng():
        torch.manual_seed(1703)
        attention = learner.CrossAttention(8, 2, dtype=dtype)
    fill_parameters(attention)
    generator = torch.Generator().manual_seed(1703)
    target_length = 5 if equal_lengths else 3
    query_base = torch.randn(2, 8, target_length, generator=generator, dtype=dtype).requires_grad_(grad)
    memory_base = torch.randn(2, 5, 16, generator=generator, dtype=dtype).requires_grad_(grad)
    query, memory = query_base.transpose(1, 2), memory_base[..., ::2]
    if layout == "contiguous":
        query, memory = query.contiguous(), memory.contiguous()
    src_valid = torch.tensor([[True, True, False, True, False],
                              [True, False, True, False, True]])
    return attention, query_base, memory_base, query, memory, src_valid


def decoder_sample(dtype=torch.float64, p=0.):
    _, _, _, states, memory, src_valid = cross_sample(dtype, layout="noncontiguous")
    with torch.random.fork_rng():
        torch.manual_seed(1703)
        block = learner.DecoderBlock(8, 2, 13, p=p, eps=.04, dtype=dtype)
    fill_parameters(block)
    tgt_valid = torch.tensor([[True, True, False], [True, False, True]])
    return block, states, memory, src_valid, tgt_valid


def model_sample(dtype=torch.float64):
    with torch.random.fork_rng():
        torch.manual_seed(1703)
        model = learner.MiniTransformer(19, 17, 8, 2, 13, 2, 3, eps=.04, dtype=dtype)
    fill_parameters(model)
    src_ids = torch.tensor([[0, 4, 7, 8, 13], [3, 5, 9, 6, 11]])
    tgt_ids = torch.tensor([[1, 6, 3, 9], [1, 5, 7, 12]])
    src_valid = torch.tensor([[True, True, False, True, False],
                              [True, False, True, True, True]])
    tgt_valid = torch.tensor([[True, True, True, False], [True, False, True, True]])
    return model, src_ids, tgt_ids, src_valid, tgt_valid


class TensorAssertions(unittest.TestCase):
    def assert_tensor(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        tolerance = dict(rtol=2e-5, atol=2e-6) if expected.dtype == torch.float32 else dict(rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(actual, expected, **tolerance)


class SourceAllowedTest(TensorAssertions):
    def test_keys_only_rectangular_and_square_masks_preserve_pad_query_rows(self):
        valid = torch.tensor([[True, False, True, False], [False, True, False, True]])
        before = valid.clone()
        for query_length in (1, 3, 4, 6):
            with self.subTest(query_length=query_length):
                self.assert_tensor(learner.source_allowed(valid, query_length),
                                   reference_allowed(valid, query_length))
        self.assertTrue(torch.equal(valid, before))

    def test_rejects_any_sample_without_source_keys(self):
        valid = torch.tensor([[True, True], [False, False]])
        with self.assertRaises(ValueError):
            learner.source_allowed(valid, 3)


class CrossAttentionTest(TensorAssertions):
    def test_independent_sources_dtypes_lengths_and_noncontiguous_layouts(self):
        for dtype in (torch.float32, torch.float64):
            for equal_lengths in (False, True):
                for layout in ("contiguous", "noncontiguous"):
                    with self.subTest(dtype=dtype, equal_lengths=equal_lengths, layout=layout):
                        attention, _, _, query, memory, valid = cross_sample(dtype, equal_lengths, layout)
                        actual = attention(query, memory, valid)
                        expected = reference_attention(attention, query, memory, reference_allowed(valid, query.shape[1]))
                        for result, reference in zip(actual, expected):
                            self.assert_tensor(result, reference)

    def test_first_target_can_read_rightmost_source_but_cannot_read_pad(self):
        attention = learner.CrossAttention(2, 1, dtype=torch.float64)
        with torch.no_grad():
            attention.Wq.zero_()
            for parameter in (attention.Wk, attention.Wv, attention.Wo):
                parameter.copy_(torch.eye(2, dtype=torch.float64))
        query = torch.tensor([[[7., 2.], [3., -1.]]], dtype=torch.float64)
        memory = torch.tensor([[[1., 0.], [0., 1.], [2., 3.], [90., -90.]]], dtype=torch.float64)
        valid = torch.tensor([[True, True, True, False]])
        output, weights = attention(query, memory, valid)
        expected = torch.tensor([[[1., 4 / 3], [1., 4 / 3]]], dtype=torch.float64)
        self.assert_tensor(output, expected)
        self.assert_tensor(weights, torch.tensor([[[[1 / 3, 1 / 3, 1 / 3, 0.],
                                                     [1 / 3, 1 / 3, 1 / 3, 0.]]]], dtype=torch.float64))

    def test_query_memory_all_parameter_and_weight_gradients_match_reference(self):
        attention, query_base, memory_base, query, memory, valid = cross_sample(layout="noncontiguous", grad=True)
        reference = copy.deepcopy(attention)
        ref_query_base = query_base.detach().clone().requires_grad_(True)
        ref_memory_base = memory_base.detach().clone().requires_grad_(True)
        actual = attention(query, memory, valid)
        expected = reference_attention(reference, ref_query_base.transpose(1, 2), ref_memory_base[..., ::2],
                                       reference_allowed(valid, query.shape[1]))
        actual_loss, expected_loss = 0., 0.
        for result, reference_result in zip(actual, expected):
            weight = torch.linspace(-.7, 1.3, result.numel(), dtype=result.dtype).reshape(result.shape)
            actual_loss = actual_loss + (result * weight).sum()
            expected_loss = expected_loss + (reference_result * weight).sum()
        actual_grads = torch.autograd.grad(actual_loss, (query_base, memory_base, *attention.parameters()))
        expected_grads = torch.autograd.grad(expected_loss, (ref_query_base, ref_memory_base, *reference.parameters()))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assert_tensor(actual_grad, expected_grad)

    def test_source_pad_invariance_and_valid_source_sensitivity(self):
        attention, _, _, query, memory, valid = cross_sample()
        before = attention(query, memory, valid)[0]
        altered = memory.clone()
        altered[~valid] += 50.
        self.assert_tensor(attention(query, altered, valid)[0], before)
        altered = memory.clone()
        altered[:, 3] += torch.arange(8, dtype=memory.dtype) - 3.
        after = attention(query, altered, valid)[0]
        self.assertGreater((after[0, 0] - before[0, 0]).abs().max().item(), .01)
        self.assert_tensor(after[1], before[1])

    def test_all_invalid_source_is_rejected(self):
        attention, _, _, query, memory, valid = cross_sample()
        valid[1].fill_(False)
        with self.assertRaises(ValueError):
            attention(query, memory, valid)


class DecoderBlockTest(TensorAssertions):
    def test_three_post_norm_residuals_match_independent_reference(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                block, states, memory, src_valid, tgt_valid = decoder_sample(dtype)
                self.assert_tensor(block(states, memory, src_valid, tgt_valid),
                                   reference_decoder(block, states, memory, src_valid, tgt_valid))

    def test_zero_updates_still_apply_all_three_post_norms(self):
        block, states, memory, src_valid, tgt_valid = decoder_sample()
        with torch.no_grad():
            block.self_attn.Wo.zero_()
            block.cross_attn.Wo.zero_()
            block.ffn.W2.zero_()
            block.ffn.b2.zero_()
        expected = reference_norm(reference_norm(reference_norm(states, block.norm1), block.norm2), block.norm3)
        self.assert_tensor(block(states, memory, src_valid, tgt_valid), expected)

    def test_nonzero_dropout_runs_on_each_branch_before_residual_and_norm(self):
        block, states, memory, src_valid, tgt_valid = decoder_sample(p=.35)
        block.train()
        with torch.random.fork_rng():
            torch.manual_seed(1731)
            actual = block(states, memory, src_valid, tgt_valid)
            torch.manual_seed(1731)
            expected = reference_decoder(block, states, memory, src_valid, tgt_valid)
        self.assert_tensor(actual, expected)
        block.eval()
        deterministic = block(states, memory, src_valid, tgt_valid)
        self.assertGreater((actual - deterministic).abs().max().item(), .01)

    def test_inputs_and_all_block_parameters_keep_gradients(self):
        block, states, memory, src_valid, tgt_valid = decoder_sample()
        states = states.detach().requires_grad_(True)
        memory = memory.detach().requires_grad_(True)
        reference = copy.deepcopy(block)
        ref_states = states.detach().clone().requires_grad_(True)
        ref_memory = memory.detach().clone().requires_grad_(True)
        actual = block(states, memory, src_valid, tgt_valid)
        expected = reference_decoder(reference, ref_states, ref_memory, src_valid, tgt_valid)
        weights = torch.linspace(-1.2, .9, actual.numel(), dtype=actual.dtype).reshape(actual.shape)
        actual_grads = torch.autograd.grad((actual * weights).sum(), (states, memory, *block.parameters()))
        expected_grads = torch.autograd.grad((expected * weights).sum(), (ref_states, ref_memory, *reference.parameters()))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assert_tensor(actual_grad, expected_grad)

    def test_future_and_pad_target_keys_do_not_leak_and_pad_queries_are_kept(self):
        block, states, memory, src_valid, tgt_valid = decoder_sample()
        before = block(states, memory, src_valid, tgt_valid)
        altered = states.clone()
        altered[:, 2] += torch.arange(8, dtype=states.dtype) - 3.
        after = block(altered, memory, src_valid, tgt_valid)
        self.assert_tensor(after[:, :2], before[:, :2])
        altered = states.clone()
        altered[~tgt_valid] *= -11.
        after = block(altered, memory, src_valid, tgt_valid)
        self.assert_tensor(after[tgt_valid], before[tgt_valid])
        self.assertGreater(before[~tgt_valid].abs().max().item(), .1)

    def test_target_without_readable_first_key_is_rejected(self):
        block, states, memory, src_valid, tgt_valid = decoder_sample()
        tgt_valid[0, 0] = False
        with self.assertRaises(ValueError):
            block(states, memory, src_valid, tgt_valid)


class MiniTransformerTest(TensorAssertions):
    def test_full_model_matches_independent_reference_dtypes_and_sliced_inputs(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                model, *inputs = model_sample(dtype)
                inputs = [value.repeat_interleave(2, dim=1)[:, ::2] for value in inputs]
                src_ids, tgt_ids, src_valid, tgt_valid = inputs
                self.assertFalse(src_ids.is_contiguous())
                memory = reference_encode(model, src_ids, src_valid)
                self.assert_tensor(model.encode(src_ids, src_valid), memory)
                expected = reference_decode(model, tgt_ids, memory, src_valid, tgt_valid)
                self.assert_tensor(model(src_ids, tgt_ids, src_valid, tgt_valid), expected)

    def test_encoder_layers_are_bidirectional_and_ignore_pad_keys(self):
        model, _, _, _, _ = model_sample()
        _, _, _, states, _, _ = cross_sample()
        valid = torch.tensor([[True, False, True], [True, True, True]])
        for index, block in enumerate(model.encoders):
            with self.subTest(layer=index):
                before = block(states, valid)
                self.assert_tensor(before, reference_encoder(block, states, valid))
                altered = states.clone()
                altered[:, 2] += torch.arange(8, dtype=states.dtype) - 3.
                after = block(altered, valid)
                self.assertGreater((after[:, 0] - before[:, 0]).abs().max().item(), 1e-4)
                altered = states.clone()
                altered[~valid] *= -17.
                self.assert_tensor(block(altered, valid)[valid], before[valid])

    def test_embedding_scaling_positions_start_at_zero_on_both_sides_and_input_dropout(self):
        class VisibleDrop(nn.Module):
            def forward(self, states):
                return states * .7 + torch.arange(states.shape[-1], dtype=states.dtype) * .03

        model, src_ids, tgt_ids, src_valid, tgt_valid = model_sample()
        model.src_drop, model.tgt_drop = VisibleDrop(), VisibleDrop()
        captured = {}
        handles = [model.encoders[0].register_forward_pre_hook(
                       lambda module, args: captured.update(source=args[0].detach().clone())),
                   model.decoders[0].register_forward_pre_hook(
                       lambda module, args: captured.update(target=args[0].detach().clone()))]
        try:
            model(src_ids, tgt_ids, src_valid, tgt_valid)
        finally:
            for handle in handles:
                handle.remove()
        expected_source = model.src_drop(math.sqrt(model.C) * model.src_table[src_ids] +
                                         reference_positions(src_ids.shape[1], model.C, model.src_table.dtype))
        expected_target = model.tgt_drop(math.sqrt(model.C) * model.tgt_table[tgt_ids] +
                                         reference_positions(tgt_ids.shape[1], model.C, model.tgt_table.dtype))
        self.assert_tensor(captured["source"], expected_source)
        self.assert_tensor(captured["target"], expected_target)

    def test_each_decoder_reads_final_encoder_memory_with_independent_projections(self):
        model, src_ids, tgt_ids, src_valid, tgt_valid = model_sample()
        captured, handles = [], []
        final_memory = []
        handles.append(model.encoders[-1].register_forward_hook(lambda module, args, output: final_memory.append(output)))
        for block in model.decoders:
            handles.append(block.cross_attn.register_forward_pre_hook(lambda module, args: captured.append(args[1])))
        try:
            model(src_ids, tgt_ids, src_valid, tgt_valid)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(captured), len(model.decoders))
        for memory in captured:
            self.assert_tensor(memory, final_memory[0])
        for name in ("Wq", "Wk", "Wv", "Wo"):
            pointers = [getattr(block.cross_attn, name).data_ptr() for block in model.decoders]
            self.assertEqual(len(set(pointers)), len(pointers))

    def test_target_loss_reaches_encoder_and_all_model_parameters(self):
        model, src_ids, tgt_ids, src_valid, tgt_valid = model_sample()
        reference = copy.deepcopy(model)
        actual = model(src_ids, tgt_ids, src_valid, tgt_valid)
        expected = reference_decode(reference, tgt_ids, reference_encode(reference, src_ids, src_valid), src_valid, tgt_valid)
        labels = torch.tensor([[6, 3, 2, 0], [5, 7, 12, 2]])
        loss_valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
        actual_loss = F.cross_entropy(actual[loss_valid], labels[loss_valid])
        expected_loss = F.cross_entropy(expected[loss_valid], labels[loss_valid])
        actual_grads = torch.autograd.grad(actual_loss, tuple(model.parameters()))
        expected_grads = torch.autograd.grad(expected_loss, tuple(reference.parameters()))
        for (name, _), actual_grad, expected_grad in zip(model.named_parameters(), actual_grads, expected_grads):
            with self.subTest(parameter=name):
                self.assert_tensor(actual_grad, expected_grad)
        gradients = dict(zip(dict(model.named_parameters()), actual_grads))
        self.assertGreater(gradients["src_table"][src_ids[src_valid]].abs().max().item(), 1e-5)
        self.assertGreater(gradients["encoders.0.self_attn.Wv"].abs().max().item(), 1e-5)

    def test_source_conditioning_pad_invariance_and_target_causality(self):
        model, src_ids, tgt_ids, src_valid, tgt_valid = model_sample()
        before = model(src_ids, tgt_ids, src_valid, tgt_valid)
        changed_source = src_ids.clone()
        changed_source[~src_valid] = 16
        self.assert_tensor(model(changed_source, tgt_ids, src_valid, tgt_valid), before)
        changed_source = src_ids.clone()
        changed_source[:, 0] = 14
        after = model(changed_source, tgt_ids, src_valid, tgt_valid)
        self.assertGreater((after[:, 0] - before[:, 0]).abs().max().item(), 1e-4)
        changed_target = tgt_ids.clone()
        changed_target[:, 2:] = 15
        self.assert_tensor(model(src_ids, changed_target, src_valid, tgt_valid)[:, :2], before[:, :2])
        changed_target = tgt_ids.clone()
        changed_target[~tgt_valid] = 16
        self.assert_tensor(model(src_ids, changed_target, src_valid, tgt_valid)[tgt_valid], before[tgt_valid])

    def test_forward_preserves_inputs_parameters_existing_gradients_and_modes(self):
        model, *inputs = model_sample()
        before_inputs = [value.clone() for value in inputs]
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, .123)
        saved = [(name, parameter, parameter.detach().clone(), parameter.grad, parameter.grad.clone())
                 for name, parameter in model.named_parameters()]
        model.train()
        model.decoders[1].eval()
        modules = tuple(model.modules())
        modes = tuple(module.training for module in modules)
        first = model(*inputs)
        self.assert_tensor(model(*inputs), first)
        self.assertTrue(first.requires_grad)
        for value, before in zip(inputs, before_inputs):
            self.assertTrue(torch.equal(value, before))
        self.assertEqual(tuple(model.modules()), modules)
        self.assertEqual(tuple(module.training for module in modules), modes)
        parameters = dict(model.named_parameters())
        self.assertEqual(set(parameters), {item[0] for item in saved})
        for name, parameter, value, grad_object, grad_value in saved:
            self.assertIs(parameters[name], parameter)
            self.assertTrue(torch.equal(parameter, value))
            self.assertIs(parameter.grad, grad_object)
            self.assertTrue(torch.equal(parameter.grad, grad_value))

    def test_all_invalid_source_is_rejected_through_encode(self):
        model, src_ids, _, src_valid, _ = model_sample()
        src_valid[1].fill_(False)
        with self.assertRaises(ValueError):
            model.encode(src_ids, src_valid)


if __name__ == "__main__":
    unittest.main()
