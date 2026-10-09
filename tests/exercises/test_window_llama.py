"""ex018 教师集成测试；独立逐头稠密窗口参照只用于验收。

CPU、固定种子 1817/1823；float64 rtol=1e-8/atol=1e-10，
float32 rtol=3e-5/atol=3e-6。检查数值、梯度和真实缓存存储，不测性能。
空脚手架有两个明确状态失败；对应行为组仅遇到 NotImplementedError 才跳过。
"""
import copy
import math
import unittest

import torch

from exercises.ex013_llama_style.model import LlamaLM
from exercises.ex018_efficient_attention import model as learner


def setUpModule():
    global _old_threads
    _old_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_old_threads)


def reference_norm(x, norm):
    return x / (x.square().sum(-1, keepdim=True) / x.shape[-1] + norm.eps).sqrt() * norm.gamma


def reference_rotated_head(x, positions, theta):
    # 每头、每对分别计算；不使用学习者拆头、RoPE 或本课辅助函数。
    pairs = []
    for pair in range(x.shape[-1] // 2):
        angle = positions.to(x.dtype) * theta ** (-2 * pair / x.shape[-1])
        even, odd = x[..., 2 * pair], x[..., 2 * pair + 1]
        pairs.extend((even * angle.cos() - odd * angle.sin(),
                      even * angle.sin() + odd * angle.cos()))
    return torch.stack(pairs, dim=-1)


def reference_block(block, x, positions, window):
    """返回全量块输出及完整紧凑 K/V；只用基础稠密数学。"""
    normalized = reference_norm(x, block.norm1)
    width, hq, hkv = block.head_dim, block.num_query_heads, block.num_kv_heads
    q, k, v = [], [], []
    for head in range(hq):
        projection = normalized @ block.Wq[:, head * width:(head + 1) * width]
        q.append(reference_rotated_head(projection, positions, block.rope_theta))
    for head in range(hkv):
        projection = normalized @ block.Wk[:, head * width:(head + 1) * width]
        k.append(reference_rotated_head(projection, positions, block.rope_theta))
        v.append(normalized @ block.Wv[:, head * width:(head + 1) * width])
    count = len(positions)
    allowed = torch.zeros(count, count, dtype=torch.bool)
    for row in range(count):
        for col in range(count):
            distance = int(positions[row]) - int(positions[col])
            allowed[row, col] = 0 <= distance < window
    outputs = []
    for head in range(hq):
        group = head // (hq // hkv)
        scores = q[head] @ k[group].transpose(-2, -1) / math.sqrt(width)
        outputs.append(scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ v[group])
    hidden = x + torch.cat(outputs, dim=-1) @ block.Wo
    normalized = reference_norm(hidden, block.norm2)
    gate = normalized @ block.ffn.Wgate
    update = ((gate * gate.sigmoid()) * (normalized @ block.ffn.Wup)) @ block.ffn.Wdown
    return hidden + update, torch.stack(k, dim=1), torch.stack(v, dim=1)


def reference_model(model, ids, window, start_position=0):
    positions = torch.arange(start_position, start_position + ids.shape[1])
    x, cached = model.token_table[ids], []
    for block in model.blocks:
        x, k, v = reference_block(block, x, positions, window)
        cached.append((k, v))
    return reference_norm(x, model.final_norm) @ model.vocab_proj, cached


def sample(dtype=torch.float64, hkv=2, length=128):
    with torch.random.fork_rng():
        torch.manual_seed(1817)
        model = LlamaLM(17, 24, 6, hkv, 31, 3, length,
                        eps=.03, rope_theta=79., dtype=dtype)
    generator = torch.Generator().manual_seed(1823)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=dtype) * .18)
            if name.endswith("gamma"):
                parameter.add_(1.)
    ids = torch.tensor([[1, 4, 6, 7, 8, 3, 12, 2, 15, 9, 10],
                        [2, 9, 5, 3, 11, 8, 6, 4, 13, 1, 16]])
    return model, ids


def block_probe():
    model, ids = sample()
    return learner.window_block(model.blocks[0], model.token_table[ids[:, :1]],
                                torch.tensor([3]), 2)


def step_probe():
    model, ids = sample()
    with torch.no_grad():
        return learner.window_step(model, ids[:, :1], learner.new_window_state(model, 2))


def skip_only_todo(probe):
    try:
        probe()
    except NotImplementedError as error:
        raise unittest.SkipTest(f"先完成该组依赖的核心函数：{error}") from error


class TensorAssertions(unittest.TestCase):
    def close(self, actual, expected, message=""):
        self.assertIsInstance(actual, torch.Tensor, message)
        self.assertEqual(actual.shape, expected.shape, message)
        self.assertEqual(actual.dtype, expected.dtype, message)
        self.assertEqual(actual.device, expected.device, message)
        tolerances = (3e-5, 3e-6) if expected.dtype == torch.float32 else (1e-8, 1e-10)
        torch.testing.assert_close(actual, expected, rtol=tolerances[0], atol=tolerances[1], msg=message)

    def snapshot(self, state):
        return [(cache.k, cache.v, cache.positions,
                 cache.k.clone(), cache.v.clone(), cache.positions.clone())
                for cache in state.layers]

    def assert_snapshot(self, state, snapshot):
        for cache, (k, v, positions, k_value, v_value, position_value) in zip(state.layers, snapshot):
            self.assertIs(cache.k, k)
            self.assertIs(cache.v, v)
            self.assertIs(cache.positions, positions)
            self.assertTrue(torch.equal(cache.k, k_value))
            self.assertTrue(torch.equal(cache.v, v_value))
            self.assertTrue(torch.equal(cache.positions, position_value))


class ImplementationStatusTest(unittest.TestCase):
    def test_window_block_is_implemented(self):
        try:
            block_probe()
        except NotImplementedError as error:
            self.fail(str(error))

    def test_window_step_is_implemented(self):
        try:
            step_probe()
        except NotImplementedError as error:
            self.fail(str(error))


class FullWindowModelTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        skip_only_todo(block_probe)

    def test_logits_match_independent_reference_for_windows_heads_dtypes_and_noncontiguous_ids(self):
        for dtype in (torch.float32, torch.float64):
            for hkv in (1, 2, 6):
                model, ids = sample(dtype, hkv)
                storage = torch.zeros(2, ids.shape[1] * 2, dtype=torch.long)
                storage[:, ::2] = ids
                ids = storage[:, ::2]
                self.assertFalse(ids.is_contiguous())
                for window in (1, 3, 20):
                    with self.subTest(dtype=dtype, hkv=hkv, window=window):
                        expected, _ = reference_model(model, ids, window, 7)
                        self.close(learner.window_forward(model, ids, window, start_position=7), expected)

    def test_noncontiguous_block_input_and_parameter_gradients_match(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                model, _ = sample(dtype)
                block, ref = model.blocks[0], copy.deepcopy(model.blocks[0])
                generator = torch.Generator().manual_seed(1841)
                base = torch.randn(2, 24, 8, generator=generator, dtype=dtype, requires_grad=True)
                ref_base = base.detach().clone().requires_grad_()
                x, positions = base.transpose(1, 2), torch.arange(9, 17)
                self.assertFalse(x.is_contiguous())
                actual = learner.window_block(block, x, positions, 3)
                expected, _, _ = reference_block(ref, ref_base.transpose(1, 2), positions, 3)
                self.close(actual, expected)
                coefficient = torch.linspace(-.8, .7, actual.numel(), dtype=dtype).reshape_as(actual)
                got = torch.autograd.grad((actual * coefficient).sum(), (base, *block.parameters()), allow_unused=True)
                want = torch.autograd.grad((expected * coefficient).sum(), (ref_base, *ref.parameters()))
                for name, ga, ge in zip(("x", *dict(block.named_parameters())), got, want):
                    self.assertIsNotNone(ga, f"梯度断开：{name}")
                    self.close(ga, ge, f"块梯度不符：{name}")

    def test_full_model_parameter_gradients_and_callers_state_are_preserved(self):
        model, ids = sample()
        model.train()
        model.blocks[1].norm1.eval()
        modes = [part.training for part in model.modules()]
        ref = copy.deepcopy(model)
        saved_ids = ids.clone()
        params = dict(model.named_parameters())
        saved = {name: parameter.detach().clone() for name, parameter in params.items()}
        for parameter in params.values():
            parameter.grad = torch.full_like(parameter, .13)
        actual = learner.window_forward(model, ids, 3, start_position=11)
        expected, _ = reference_model(ref, ids, 3, 11)
        coefficient = torch.linspace(-.9, .8, actual.numel(), dtype=actual.dtype).reshape_as(actual)
        got = torch.autograd.grad((actual * coefficient).sum(), tuple(params.values()), allow_unused=True)
        want = torch.autograd.grad((expected * coefficient).sum(), tuple(ref.parameters()))
        for (name, parameter), ga, ge in zip(params.items(), got, want):
            self.assertIsNotNone(ga, f"全量模型梯度断开：{name}")
            self.close(ga, ge, name)
            self.assertTrue(torch.equal(parameter, saved[name]))
            self.assertTrue(torch.equal(parameter.grad, torch.full_like(parameter, .13)))
        self.assertTrue(torch.equal(ids, saved_ids))
        self.assertEqual(modes, [part.training for part in model.modules()])
        self.assertEqual(set(saved), set(model.state_dict()), "不能把请求状态登记进模型")
        for name, parameter in model.named_parameters():
            self.assertIs(parameter, params[name])


class CachedWindowModelTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        skip_only_todo(step_probe)

    def test_all_new_logits_match_for_full_token_and_uneven_chunks(self):
        with torch.no_grad():
            for dtype in (torch.float32, torch.float64):
                for hkv in (1, 2, 6):
                    model, ids = sample(dtype, hkv)
                    for window in (1, 3, 20):
                        expected, _ = reference_model(model, ids, window, 7)
                        for sizes in ((11,), (1,) * 11, (6, 3, 2), (2, 5, 1, 3)):
                            with self.subTest(dtype=dtype, hkv=hkv, window=window, sizes=sizes):
                                state = learner.new_window_state(model, window, start_position=7)
                                offset = 0
                                for size in sizes:
                                    actual = learner.window_step(model, ids[:, offset:offset + size], state)
                                    self.close(actual, expected[:, offset:offset + size],
                                               "比较本次每一行 logits，不能只比较最后一行")
                                    offset += size
                                    self.assertEqual(state.next_position, 7 + offset)
                                    self.assertEqual([len(cache) for cache in state.layers],
                                                     [min(window, offset)] * model.n_layer)

    def test_compact_cache_contents_positions_storage_and_old_keys_after_eviction(self):
        with torch.no_grad():
            for dtype in (torch.float32, torch.float64):
                for hkv in (1, 2, 6):
                    model, ids = sample(dtype, hkv)
                    _, reference_kv = reference_model(model, ids, 3, 13)
                    state = learner.new_window_state(model, 3, start_position=13)
                    offset = 0
                    for size in (7, 1, 3):
                        old = [(cache.positions, cache.k) for cache in state.layers]
                        learner.window_step(model, ids[:, offset:offset + size], state)
                        offset += size
                        begin = max(0, offset - 3)
                        for layer, (cache, (expected_k, expected_v)) in enumerate(zip(state.layers, reference_kv)):
                            self.close(cache.k, expected_k[:, :, begin:offset], "K 必须仅旋转一次，按 Hkv 保存")
                            self.close(cache.v, expected_v[:, :, begin:offset], "V 不应用 RoPE")
                            self.assertTrue(torch.equal(cache.positions, torch.arange(13 + begin, 13 + offset)))
                            for tensor in (cache.k, cache.v, cache.positions):
                                self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size(),
                                                 "裁后 view 可能仍持有全部历史底层存储；需要独立紧凑存储")
                                self.assertIsNone(tensor.grad_fn)
                            old_positions, old_k = old[layer]
                            if old_positions is not None:
                                for old_index, position in enumerate(old_positions.tolist()):
                                    indices = (cache.positions == position).nonzero().flatten()
                                    if len(indices):
                                        self.assertTrue(torch.equal(cache.k[:, :, int(indices[0])], old_k[:, :, old_index]),
                                                        "尚留在窗口中的历史 K 不得被重复旋转或重算")
                        tensors = [tensor for cache in state.layers for tensor in (cache.k, cache.v)]
                        byte_count = 2 * model.n_layer * ids.shape[0] * hkv * min(3, offset) * 4 * ids.new_empty((), dtype=dtype).element_size()
                        self.assertEqual(sum(tensor.untyped_storage().nbytes() for tensor in tensors), byte_count)
                        pointers = [tensor.untyped_storage().data_ptr() for tensor in tensors]
                        self.assertEqual(len(pointers), len(set(pointers)), "层和 K/V 缓存应使用独立存储")

    def test_interleaved_requests_and_reset_restore_configured_origin(self):
        model, ids = sample()
        a = learner.new_window_state(model, 3, start_position=5)
        b = learner.new_window_state(model, 2, start_position=17)
        self.assertEqual(a.next_position, 5)
        self.assertTrue(all(cache.k is cache.v is cache.positions is None for cache in a.layers))
        self.assertEqual(len({id(cache) for cache in a.layers + b.layers}), 2 * model.n_layer)
        with torch.no_grad():
            a_first = learner.window_step(model, ids[:, :6], a)
            frozen = self.snapshot(a)
            b_first = learner.window_step(model, ids[:1, :4].flip(1), b)
            self.assert_snapshot(a, frozen)
            a_last = learner.window_step(model, ids[:, 6:], a)
            expected_a, _ = reference_model(model, ids, 3, 5)
            self.close(torch.cat((a_first, a_last), dim=1), expected_a)
            expected_b, _ = reference_model(model, ids[:1, :4].flip(1), 2, 17)
            self.close(b_first, expected_b)
            frozen_b = self.snapshot(b)
            a.reset()
            self.assertEqual((a.start_position, a.next_position, a.window), (5, 5, 3))
            self.assertTrue(all(cache.k is cache.v is cache.positions is None for cache in a.layers))
            self.assert_snapshot(b, frozen_b)
            # 重置后可换 batch，仍按本请求原起点开始，旧内容不得参与。
            restarted = learner.window_step(model, ids[:1, 3:9], a)
            expected, _ = reference_model(model, ids[:1, 3:9], 3, 5)
            self.close(restarted, expected)
            self.assertEqual(a.next_position, 11)

    def test_position_capacity_failure_keeps_every_layer_and_cursor_unchanged(self):
        model, ids = sample(length=13)
        state = learner.new_window_state(model, 3, start_position=5)
        with torch.no_grad():
            learner.window_step(model, ids[:, :7], state)
            frozen = self.snapshot(state)
            with self.assertRaises(ValueError):
                learner.window_step(model, ids[:, 7:9], state)
            self.assertEqual(state.next_position, 12)
            self.assert_snapshot(state, frozen)
            actual = learner.window_step(model, ids[:, 7:8], state)
            expected, _ = reference_model(model, ids[:, :8], 3, 5)
            self.close(actual, expected[:, -1:])
            self.assertEqual(state.next_position, 13)
            frozen = self.snapshot(state)
            with self.assertRaises(ValueError):
                learner.window_step(model, ids[:, 8:9], state)
            self.assert_snapshot(state, frozen)
            with self.assertRaises(ValueError):
                learner.window_forward(model, ids[:, :9], 3, start_position=5)

    def test_cached_calls_preserve_model_input_parameters_existing_grads_and_modes(self):
        model, ids = sample()
        model.eval()
        model.blocks[0].norm2.train()
        modes = [part.training for part in model.modules()]
        parameters = dict(model.named_parameters())
        saved = {name: p.detach().clone() for name, p in parameters.items()}
        saved_ids = ids.clone()
        for parameter in parameters.values():
            parameter.grad = torch.full_like(parameter, .19)
        state = learner.new_window_state(model, 3)
        with torch.no_grad():
            output = learner.window_step(model, ids[:, :8], state)
            self.assertFalse(output.requires_grad)
            learner.window_step(model, ids[:, 8:], state)
        self.assertEqual(modes, [part.training for part in model.modules()])
        self.assertTrue(torch.equal(ids, saved_ids))
        self.assertEqual(set(saved), set(model.state_dict()))
        for name, parameter in model.named_parameters():
            self.assertIs(parameter, parameters[name])
            self.assertTrue(torch.equal(parameter, saved[name]))
            self.assertTrue(torch.equal(parameter.grad, torch.full_like(parameter, .19)))


if __name__ == "__main__":
    unittest.main()
