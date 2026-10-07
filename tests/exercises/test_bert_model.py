"""ex016 教师模型测试；数学参照仅供验收，不得从作业导入作为实现。

固定 CPU、种子 1603/1617；float64 rtol=1e-8/atol=1e-10，
float32 rtol=1e-5/atol=1e-6。逐头参照不调用学习者 MHA、LN 或 GELU。
空脚手架直接报 NotImplementedError；不跳过行为测试，不表示已完成。
"""

import copy
import math
import unittest

import torch
from torch.nn import functional as F

from exercises.ex016_mini_bert import model as learner


def setUpModule():
    global _old_threads
    _old_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_old_threads)


def reference_norm(x, norm):
    centered = x - x.sum(dim=-1, keepdim=True) / x.shape[-1]
    variance = centered.square().sum(dim=-1, keepdim=True) / x.shape[-1]
    return centered / torch.sqrt(variance + norm.eps) * norm.gamma + norm.beta


def reference_block(block, x, valid):
    """每个 head 独立投影和读取，刻意不共享待测拆头、mask 广播路径。"""
    width = x.shape[-1] // block.num_heads
    heads = []
    for head in range(block.num_heads):
        part = slice(head * width, (head + 1) * width)
        q = x @ block.Wq[:, part] + block.bq[part]
        k = x @ block.Wk[:, part] + block.bk[part]
        v = x @ block.Wv[:, part] + block.bv[part]
        samples = []
        for b in range(x.shape[0]):
            # 只抽取有效 key/value；所有 query（包括 PAD query）都保留。
            score = q[b] @ k[b, valid[b]].T / math.sqrt(width)
            samples.append(score.softmax(-1) @ v[b, valid[b]])
        heads.append(torch.stack(samples))
    attention = torch.cat(heads, dim=-1) @ block.Wo + block.bo
    u = reference_norm(x + attention, block.norm1)
    update = F.gelu(u @ block.W1 + block.b1, approximate="tanh") @ block.W2 + block.b2
    return reference_norm(u + update, block.norm2)


def reference_encode(model, ids, segments, valid):
    positions = torch.arange(ids.shape[1], device=ids.device)
    x = model.token_table[ids] + model.position_table[positions] + model.segment_table[segments]
    x = reference_norm(x, model.embedding_norm)
    for block in model.blocks:
        x = reference_block(block, x, valid)
    return x


def reference_mlm(model, z):
    r = reference_norm(F.gelu(z @ model.Wm + model.bm, approximate="tanh"), model.mlm_norm)
    return r @ model.token_table.T + model.bvocab


def reference_outputs(model, ids, segments, valid):
    z = reference_encode(model, ids, segments, valid)
    p = torch.tanh(z[:, 0, :] @ model.Wpool + model.bpool)
    return {"encoded": z, "mlm_logits": reference_mlm(model, z),
            "nsp_logits": p @ model.Wnsp + model.bnsp,
            "class_logits": p @ model.Wclass + model.bclass}


def fill_parameters(module, seed=1617):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator,
                                        dtype=parameter.dtype) * .27)
            if name.endswith("gamma"):
                parameter.add_(1.)


def block_sample(dtype=torch.float64, layout="contiguous", grad=False):
    with torch.random.fork_rng():
        torch.manual_seed(1603)
        block = learner.EncoderBlock(8, 2, 13, eps=.04, dtype=dtype)
    fill_parameters(block)
    generator = torch.Generator().manual_seed(1603)
    shape = (2, 8, 5) if layout == "transpose" else (2, 5, 16) if layout == "slice" else (2, 5, 8)
    base = torch.randn(shape, generator=generator, dtype=dtype).requires_grad_(grad)
    x = base.transpose(-2, -1) if layout == "transpose" else base[..., ::2] if layout == "slice" else base
    valid = torch.tensor([[True, True, False, True, False],
                          [True, False, True, True, True]])
    return block, base, x, valid


def model_sample(dtype=torch.float64):
    with torch.random.fork_rng():
        torch.manual_seed(1603)
        model = learner.MiniBert(17, 8, 2, 13, 2, 9, num_classes=3, eps=.04, dtype=dtype)
    fill_parameters(model)
    ids = torch.tensor([[1, 4, 2, 7, 2, 0], [1, 9, 10, 2, 8, 2]])
    segments = torch.tensor([[0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 1]])
    valid = torch.tensor([[True, True, True, True, True, False],
                          [True, True, True, True, True, True]])
    return model, ids, segments, valid


class TensorAssertions(unittest.TestCase):
    def assert_tensor(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        tolerance = dict(rtol=1e-5, atol=1e-6) if expected.dtype == torch.float32 else dict(rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(actual, expected, **tolerance)


class GeluTest(TensorAssertions):
    def test_tanh_approximation_values_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.tensor([[-4., -1.2, -.1, 0., .2, 1.7, 4.]], dtype=dtype, requires_grad=True)
                ref_x = x.detach().clone().requires_grad_(True)
                before = x.detach().clone()
                actual = learner.gelu(x)
                expected = F.gelu(ref_x, approximate="tanh")
                self.assert_tensor(actual, expected)
                self.assert_tensor(torch.autograd.grad(actual.sum(), x)[0],
                                   torch.autograd.grad(expected.sum(), ref_x)[0])
                self.assertTrue(torch.equal(x, before))


class EncoderBlockTest(TensorAssertions):
    def test_reference_dtypes_noncontiguous_layouts_and_biases(self):
        # B == H == 2，避免错误把 batch mask 当 head mask 而侥幸不报 shape 错。
        for dtype in (torch.float32, torch.float64):
            for layout in ("contiguous", "transpose", "slice"):
                with self.subTest(dtype=dtype, layout=layout):
                    block, base, x, valid = block_sample(dtype, layout)
                    before = base.clone()
                    self.assert_tensor(block(x, valid), reference_block(block, x, valid))
                    self.assertTrue(torch.equal(base, before))

    def test_input_and_all_parameter_gradients(self):
        block, base, x, valid = block_sample(layout="transpose", grad=True)
        ref = copy.deepcopy(block)
        ref_base = base.detach().clone().requires_grad_(True)
        actual = block(x, valid)
        expected = reference_block(ref, ref_base.transpose(-2, -1), valid)
        weight = torch.linspace(-.8, 1.1, actual.numel(), dtype=actual.dtype).reshape(actual.shape)
        actual_grads = torch.autograd.grad((actual * weight).sum(), (base, *block.parameters()))
        expected_grads = torch.autograd.grad((expected * weight).sum(), (ref_base, *ref.parameters()))
        for i, (actual_grad, expected_grad) in enumerate(zip(actual_grads, expected_grads)):
            with self.subTest(gradient=i):
                self.assert_tensor(actual_grad, expected_grad)

    def test_zero_updates_still_apply_two_post_layer_norms(self):
        block, _, x, valid = block_sample()
        with torch.no_grad():
            block.Wo.zero_()
            block.bo.zero_()
            block.W2.zero_()
            block.b2.zero_()
        expected = reference_norm(reference_norm(x, block.norm1), block.norm2)
        self.assertGreater((expected - x).abs().max().item(), .1)
        self.assert_tensor(block(x, valid), expected)

    def test_pad_keys_are_hidden_but_pad_queries_are_not_zeroed(self):
        block, _, x, valid = block_sample()
        baseline = block(x, valid)
        modified = x.clone()
        # 非均匀方向，不能被最后一轴 LN 当作纯平移抹掉。
        modified[~valid] += torch.tensor([3., -2., 5., .1, -4., 2., 0., 1.], dtype=x.dtype)
        changed = block(modified, valid)
        self.assert_tensor(changed[valid], baseline[valid])
        self.assertGreater(baseline[~valid].abs().max().item(), .1)
        self.assertGreater((changed[~valid] - baseline[~valid]).abs().max().item(), .01)

    def test_every_encoder_layer_reads_right_hand_valid_tokens(self):
        model, _, _, _ = model_sample()
        _, _, x, valid = block_sample()
        altered = x.clone()
        altered[:, 3, :] += torch.tensor([2., -3., .4, 5., 1., -2., 3., 0.], dtype=x.dtype)
        # 给每一层独立的输入：不会被“前层已经混入右侧信息”蒙混过关。
        for index, block in enumerate(model.blocks):
            with self.subTest(layer=index):
                baseline = block(x, valid)
                changed = block(altered, valid)
                self.assertGreater((changed[:, 0] - baseline[:, 0]).abs().max().item(), 1e-5,
                                   "本层 CLS 没读到右侧有效 token，检查是否误用了因果 wrapper")
                self.assert_tensor(baseline, reference_block(block, x, valid))

    def test_all_invalid_sample_is_rejected(self):
        block, _, x, valid = block_sample()
        valid[1].fill_(False)
        with self.assertRaises(ValueError):
            block(x, valid)


class MiniBertTest(TensorAssertions):
    def test_encode_three_tables_embedding_norm_and_all_layers(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                model, ids, segments, valid = model_sample(dtype)
                # transpose 两次不够制造非连续；从加宽容器每隔一项取样。
                ids_wide = torch.stack((ids, ids), dim=-1).reshape(2, -1)
                seg_wide = torch.stack((segments, segments), dim=-1).reshape(2, -1)
                valid_wide = torch.stack((valid, valid), dim=-1).reshape(2, -1)
                ids, segments, valid = ids_wide[:, ::2], seg_wide[:, ::2], valid_wide[:, ::2]
                self.assertFalse(ids.is_contiguous())
                self.assert_tensor(model.encode(ids, segments, valid), reference_encode(model, ids, segments, valid))

    def test_full_output_contract_and_independent_numerical_reference(self):
        model, ids, segments, valid = model_sample()
        actual = model(ids, segments, valid)
        expected = reference_outputs(model, ids, segments, valid)
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            with self.subTest(output=key):
                self.assert_tensor(actual[key], expected[key])

    def test_full_model_parameter_gradients_match_reference(self):
        model, ids, segments, valid = model_sample()
        ref = copy.deepcopy(model)
        actual = model(ids, segments, valid)
        expected = reference_outputs(ref, ids, segments, valid)
        actual_loss, expected_loss = 0., 0.
        for i, key in enumerate(expected):
            weight = torch.linspace(-.9 + .1 * i, 1.3, expected[key].numel(), dtype=torch.float64).reshape(expected[key].shape)
            actual_loss = actual_loss + (actual[key] * weight).sum()
            expected_loss = expected_loss + (expected[key] * weight).sum()
        actual_grads = torch.autograd.grad(actual_loss, tuple(model.parameters()))
        expected_grads = torch.autograd.grad(expected_loss, tuple(ref.parameters()))
        for (name, _), actual_grad, expected_grad in zip(model.named_parameters(), actual_grads, expected_grads):
            with self.subTest(parameter=name):
                self.assert_tensor(actual_grad, expected_grad)

    def test_mlm_uses_same_embedding_parameter_for_values_and_gradients(self):
        model, _, _, _ = model_sample()
        ref = copy.deepcopy(model)
        generator = torch.Generator().manual_seed(1603)
        z = torch.randn(2, 3, 8, generator=generator, dtype=torch.float64, requires_grad=True)
        ref_z = z.detach().clone().requires_grad_(True)
        actual, expected = model.mlm_logits(z), reference_mlm(ref, ref_z)
        self.assert_tensor(actual, expected)
        names = ("token_table", "Wm", "bm", "bvocab")
        params = (z, *(getattr(model, name) for name in names), *model.mlm_norm.parameters())
        ref_params = (ref_z, *(getattr(ref, name) for name in names), *ref.mlm_norm.parameters())
        weight = torch.linspace(-.7, 1.2, actual.numel(), dtype=actual.dtype).reshape(actual.shape)
        actual_grads = torch.autograd.grad((actual * weight).sum(), params)
        expected_grads = torch.autograd.grad((expected * weight).sum(), ref_params)
        self.assertGreater(expected_grads[1].abs().max().item(), .1)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assert_tensor(actual_grad, expected_grad)
        with torch.no_grad():
            model.token_table[5, 2].add_(.7)
        changed = model.mlm_logits(z)
        self.assertGreater((changed[..., 5] - actual[..., 5]).abs().max().item(), 1e-4,
                           "修改输入 embedding 后，MLM 输出必须立即使用同一份参数")

    def test_pool_uses_cls_not_mean_last_or_all_positions(self):
        model, _, _, _ = model_sample()
        z = torch.arange(2 * 4 * 8, dtype=torch.float64).reshape(2, 4, 8) / 37. - .6
        z.requires_grad_(True)
        expected = torch.tanh(z[:, 0, :] @ model.Wpool + model.bpool)
        self.assert_tensor(model.pool(z), expected)
        altered = z.detach().clone()
        altered[:, 1:, :] *= -7.
        self.assert_tensor(model.pool(altered), expected)
        grad = torch.autograd.grad(model.pool(z).square().sum(), z)[0]
        self.assertTrue(torch.equal(grad[:, 1:], torch.zeros_like(grad[:, 1:])))
        self.assertGreater(grad[:, 0].abs().max().item(), 1e-4)

    def test_pad_ids_and_segments_cannot_affect_valid_outputs_or_sentence_heads(self):
        model, ids, segments, valid = model_sample()
        before = model(ids, segments, valid)
        changed_ids, changed_segments = ids.clone(), segments.clone()
        changed_ids[~valid] = 12
        changed_segments[~valid] = 1
        after = model(changed_ids, changed_segments, valid)
        self.assert_tensor(after["encoded"][valid], before["encoded"][valid])
        self.assert_tensor(after["mlm_logits"][valid], before["mlm_logits"][valid])
        self.assert_tensor(after["nsp_logits"], before["nsp_logits"])
        self.assert_tensor(after["class_logits"], before["class_logits"])

    def test_segment_change_is_an_embedding_hint_not_an_attention_partition(self):
        model, ids, segments, valid = model_sample()
        original = model.encode(ids, segments, valid)
        changed_segments = segments.clone()
        changed_segments[0, 3] = 0
        changed = model.encode(ids, changed_segments, valid)
        self.assertGreater((changed[0, 0] - original[0, 0]).abs().max().item(), 1e-5)
        self.assert_tensor(changed, reference_encode(model, ids, changed_segments, valid))

    def test_forward_preserves_inputs_parameters_existing_grads_and_modes(self):
        model, ids, segments, valid = model_sample()
        before_inputs = [t.clone() for t in (ids, segments, valid)]
        for p in model.parameters():
            p.grad = torch.full_like(p, .123)
        saved = [(name, p, p.detach().clone(), p.grad, p.grad.clone()) for name, p in model.named_parameters()]
        model.train()
        model.blocks[1].eval()  # 混合模式也不能被 forward 偷偷改成统一状态。
        modules = tuple(model.modules())
        modes = tuple(m.training for m in modules)
        first = model(ids, segments, valid)
        second = model(ids, segments, valid)
        for key in first:
            self.assert_tensor(first[key], second[key])
            self.assertTrue(first[key].requires_grad)
        for current, old in zip((ids, segments, valid), before_inputs):
            self.assertTrue(torch.equal(current, old))
        self.assertEqual(tuple(model.modules()), modules)
        self.assertEqual(tuple(m.training for m in modules), modes)
        current_parameters = dict(model.named_parameters())
        self.assertEqual(set(current_parameters), {item[0] for item in saved})
        for name, p, value, grad_object, grad_value in saved:
            self.assertIs(current_parameters[name], p)
            self.assertTrue(torch.equal(p, value))
            self.assertIs(p.grad, grad_object)
            self.assertTrue(torch.equal(p.grad, grad_value))

    def test_length_limit_and_all_invalid_samples_are_rejected(self):
        model, ids, segments, valid = model_sample()
        too_long = torch.ones(1, 10, dtype=torch.long)
        with self.assertRaises(ValueError):
            model.encode(too_long, torch.zeros_like(too_long), torch.ones_like(too_long, dtype=torch.bool))
        valid[1].fill_(False)
        with self.assertRaises(ValueError):
            model.encode(ids, segments, valid)


if __name__ == "__main__":
    unittest.main()
