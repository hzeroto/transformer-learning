"""ex013 教师集成测试：参照仅供验收，不能从作业导入。

CPU；float64 rtol=1e-8/atol=1e-10，float32 rtol=1e-5/atol=1e-6。
种子 1307/1313；独立逐头、逐特征对参照，不调用待测组件或已有 GQA。
未实现时三个状态测试明确失败，其余对应行为组仅捕获 NotImplementedError 后跳过。
"""
import copy
import math
import unittest

import torch
from torch.nn import functional as F

from exercises.ex011_kv_cache.cache import LayerKVCache
from exercises.ex013_llama_style import model as learner


def setUpModule():
    global _old_threads
    _old_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_old_threads)


def reference_rms(x, norm):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) / x.shape[-1] + norm.eps) * norm.gamma


def reference_rotated_head(x, positions, theta):
    """x=(B,n,D)：每一对单独旋转，避免共享待测拆头或广播路径。"""
    D = x.shape[-1]
    pairs = []
    for j in range(D // 2):
        angle = positions.to(dtype=x.dtype) * theta ** (-2 * j / D)
        a, b = x[..., 2 * j], x[..., 2 * j + 1]
        pairs.extend((a * angle.cos() - b * angle.sin(),
                      a * angle.sin() + b * angle.cos()))
    return torch.stack(pairs, dim=-1)


def reference_allowed(valid):
    B, T = valid.shape
    result = torch.zeros(B, T, T, dtype=torch.bool)
    for b in range(B):
        for i in range(T):
            result[b, i, :i + 1] = valid[b, :i + 1]
    return result


def reference_block(block, x, positions, allowed):
    """独立全量参照，返回块输出及本层应保存的紧凑旋转 K、未旋转 V。"""
    s = reference_rms(x, block.norm1)
    D, Hq, Hkv = block.head_dim, block.num_query_heads, block.num_kv_heads
    q_heads, k_heads, v_heads = [], [], []
    for h in range(Hq):
        q = s @ block.Wq[:, h * D:(h + 1) * D]
        q_heads.append(reference_rotated_head(q, positions, block.rope_theta))
    for g in range(Hkv):
        k = s @ block.Wk[:, g * D:(g + 1) * D]
        k_heads.append(reference_rotated_head(k, positions, block.rope_theta))
        v_heads.append(s @ block.Wv[:, g * D:(g + 1) * D])
    outputs = []
    for h in range(Hq):
        g = h // (Hq // Hkv)
        score = q_heads[h] @ k_heads[g].transpose(-2, -1) / math.sqrt(D)
        weight = score.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
        outputs.append(weight @ v_heads[g])
    u = x + torch.cat(outputs, dim=-1) @ block.Wo
    z = reference_rms(u, block.norm2)
    update = (F.silu(z @ block.ffn.Wgate) * (z @ block.ffn.Wup)) @ block.ffn.Wdown
    return u + update, torch.cat(k_heads, dim=-1), torch.cat(v_heads, dim=-1)


def reference_model(model, ids, valid):
    x = model.token_table[ids]
    positions = torch.arange(ids.shape[1])
    allowed = reference_allowed(valid)
    values = []
    for block in model.blocks:
        x, k, v = reference_block(block, x, positions, allowed)
        values.append((k, v))
    return reference_rms(x, model.final_norm) @ model.vocab_proj, values


def fill_parameters(module, seed=1313):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * .22)
            if name.endswith("gamma"):
                parameter.add_(1.)


def block_sample(dtype=torch.float64, Hkv=2, layout="contiguous", grad=False):
    with torch.random.fork_rng():
        torch.manual_seed(1307)
        block = learner.LlamaBlock(24, 6, Hkv, 31, eps=.03, rope_theta=100., dtype=dtype)
    fill_parameters(block)
    generator = torch.Generator().manual_seed(1307)
    shape = (2, 24, 6) if layout == "transpose" else (2, 6, 49) if layout == "slice" else (2, 6, 24)
    base = torch.randn(shape, generator=generator, dtype=dtype).requires_grad_(grad)
    x = base.transpose(-2, -1) if layout == "transpose" else base[..., 1:49:2] if layout == "slice" else base
    valid = torch.tensor([[True, True, False, True, True, False],
                          [True, False, True, True, False, True]])
    return block, base, x, valid


def model_sample(dtype=torch.float64, Hkv=2, L=128):
    with torch.random.fork_rng():
        torch.manual_seed(1307)
        model = learner.LlamaLM(11, 24, 6, Hkv, 64, 2, L,
                                eps=.03, rope_theta=100., dtype=dtype)
    fill_parameters(model)
    ids = torch.tensor([[1, 4, 6, 7, 8, 3], [1, 3, 9, 4, 2, 6]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    return model, ids, valid


def make_caches(model, limit=None):
    return [LayerKVCache(max_length=limit) for _ in range(model.n_layer)]


class TensorAssertions(unittest.TestCase):
    def assert_tensor(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        tolerance = dict(rtol=1e-5, atol=1e-6) if expected.dtype == torch.float32 else dict(rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(actual, expected, **tolerance)

    def assert_cache_unchanged(self, caches, saved):
        for cache, (old_k, old_v, k, v) in zip(caches, saved):
            self.assertIs(cache.k, old_k)
            self.assertIs(cache.v, old_v)
            if k is not None:
                self.assertTrue(torch.equal(cache.k, k))
                self.assertTrue(torch.equal(cache.v, v))

    def snapshot_caches(self, caches):
        return [(c.k, c.v, None if c.k is None else c.k.clone(),
                 None if c.v is None else c.v.clone()) for c in caches]


class ImplementationStatusTest(unittest.TestCase):
    def test_llama_block_is_implemented(self):
        block, _, x, valid = block_sample()
        try:
            block(x, torch.arange(x.shape[1]), reference_allowed(valid))
        except NotImplementedError as error:
            self.fail(str(error))

    def test_llama_lm_is_implemented(self):
        model, ids, valid = model_sample()
        try:
            model(ids, valid)
        except NotImplementedError as error:
            self.fail(str(error))

    def test_llama_model_step_is_implemented(self):
        model, ids, _ = model_sample()
        try:
            learner.llama_model_step(model, ids, make_caches(model))
        except NotImplementedError as error:
            self.fail(str(error))


class LlamaBlockTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        block, _, x, valid = block_sample()
        try:
            block(x, torch.arange(x.shape[1]), reference_allowed(valid))
        except NotImplementedError:
            raise unittest.SkipTest("先完成组件和 LlamaBlock.forward")

    def test_reference_dtypes_heads_layouts_and_nonzero_positions(self):
        for dtype in (torch.float32, torch.float64):
            for Hkv in (1, 2, 6):
                for layout in ("contiguous", "transpose", "slice"):
                    with self.subTest(dtype=dtype, Hkv=Hkv, layout=layout):
                        block, base, x, valid = block_sample(dtype, Hkv, layout)
                        saved = base.clone()
                        positions = torch.arange(7, 13)
                        allowed = reference_allowed(valid)
                        before_mask = allowed.clone()
                        expected, _, _ = reference_block(block, x, positions, allowed)
                        self.assert_tensor(block(x, positions, allowed), expected)
                        self.assertTrue(torch.equal(base, saved))
                        self.assertTrue(torch.equal(allowed, before_mask))

    def test_input_and_parameter_gradients_match_independent_reference(self):
        block, base, x, valid = block_sample(layout="transpose", grad=True)
        ref = copy.deepcopy(block)
        ref_base = base.detach().clone().requires_grad_(True)
        positions, allowed = torch.arange(5, 11), reference_allowed(valid)
        actual = block(x, positions, allowed)
        expected, _, _ = reference_block(ref, ref_base.transpose(-2, -1), positions, allowed)
        weight = torch.linspace(-.7, .9, actual.numel(), dtype=actual.dtype).reshape(actual.shape)
        actual_grad = torch.autograd.grad((actual * weight).sum(), (base, *block.parameters()))
        expected_grad = torch.autograd.grad((expected * weight).sum(), (ref_base, *ref.parameters()))
        for actual_value, expected_value in zip(actual_grad, expected_grad):
            self.assert_tensor(actual_value, expected_value)

    def test_causal_and_pad_key_perturbations(self):
        block, _, x, valid = block_sample()
        positions, allowed = torch.arange(6), reference_allowed(valid)
        baseline = block(x, positions, allowed)
        future = x.clone()
        future[:, 4:] += 3.
        self.assert_tensor(block(future, positions, allowed)[:, :4], baseline[:, :4])
        padded = x.clone()
        padded[~valid] += 7.
        changed = block(padded, positions, allowed)
        self.assert_tensor(changed[valid], baseline[valid])
        self.assertGreater((changed[~valid] - baseline[~valid]).abs().max().item(), 1.)

    def test_common_position_shift_preserves_output_but_not_rotated_keys(self):
        block, _, x, valid = block_sample()
        allowed = reference_allowed(valid)
        self.assert_tensor(block(x, torch.arange(6), allowed),
                           block(x, torch.arange(11, 17), allowed))
        _, k0, _ = reference_block(block, x, torch.arange(6), allowed)
        _, k1, _ = reference_block(block, x, torch.arange(11, 17), allowed)
        self.assertGreater((k0 - k1).abs().max().item(), .1)


class LlamaBlockCacheTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        block, _, x, _ = block_sample()
        try:
            block(x, torch.arange(6), reference_allowed(torch.ones(2, 6, dtype=torch.bool)), LayerKVCache())
        except NotImplementedError:
            raise unittest.SkipTest("先完成 LlamaBlock.forward 的缓存路径")

    def test_cache_contains_rotated_k_unrotated_v_and_old_keys_stay_fixed(self):
        block, _, x, _ = block_sample(layout="slice")
        allowed = reference_allowed(torch.ones(2, 6, dtype=torch.bool))
        expected, expected_k, expected_v = reference_block(block, x, torch.arange(6), allowed)
        cache, outputs, start = LayerKVCache(), [], 0
        for size in (3, 2, 1):
            end = start + size
            old_k = None if cache.k is None else cache.k.clone()
            outputs.append(block(x[:, start:end], torch.arange(start, end), allowed[:, start:end, :end], cache))
            self.assert_tensor(cache.k, expected_k[:, :end])
            self.assert_tensor(cache.v, expected_v[:, :end])
            if old_k is not None:
                self.assertTrue(torch.equal(cache.k[:, :start], old_k), "历史 K 不能被再次旋转")
            start = end
        self.assert_tensor(torch.cat(outputs, dim=1), expected)
        unrotated_k = reference_rms(x, block.norm1) @ block.Wk
        self.assertGreater((expected_k[:, 1:] - unrotated_k[:, 1:]).abs().max().item(), .1)

    def test_block_capacity_error_keeps_cache_unchanged(self):
        block, _, x, _ = block_sample()
        cache = LayerKVCache(max_length=4)
        allowed = reference_allowed(torch.ones(2, 6, dtype=torch.bool))
        block(x[:, :3], torch.arange(3), allowed[:, :3, :3], cache)
        before = self.snapshot_caches([cache])
        with self.assertRaises(ValueError):
            block(x[:, 3:5], torch.arange(3, 5), allowed[:, 3:5, :5], cache)
        self.assert_cache_unchanged([cache], before)


class LlamaModelTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        model, ids, valid = model_sample()
        try:
            model(ids, valid)
        except NotImplementedError:
            raise unittest.SkipTest("先完成 LlamaLM.forward 及其组件")

    def test_full_forward_reference_mha_mqa_gqa_dtypes_and_noncontiguous_ids(self):
        for dtype in (torch.float32, torch.float64):
            for Hkv in (1, 2, 6):
                with self.subTest(dtype=dtype, Hkv=Hkv):
                    model, ids, valid = model_sample(dtype, Hkv)
                    wide_ids = torch.zeros(2, 12, dtype=torch.long)
                    wide_ids[:, ::2] = ids
                    ids = wide_ids[:, ::2]
                    wide_valid = torch.ones(2, 12, dtype=torch.bool)
                    wide_valid[0, 4] = False
                    valid = wide_valid[:, ::2]
                    self.assertFalse(ids.is_contiguous())
                    expected, _ = reference_model(model, ids, valid)
                    self.assert_tensor(model(ids, valid), expected)

    def test_parameter_ledger_independent_embeddings_and_no_position_or_bias(self):
        model, _, _ = model_sample()
        shapes = {"token_table": (11, 24), "vocab_proj": (24, 11), "final_norm.gamma": (24,)}
        for layer in range(2):
            prefix = f"blocks.{layer}."
            shapes.update({prefix + "Wq": (24, 24), prefix + "Wk": (24, 8),
                           prefix + "Wv": (24, 8), prefix + "Wo": (24, 24),
                           prefix + "norm1.gamma": (24,), prefix + "norm2.gamma": (24,),
                           prefix + "ffn.Wgate": (24, 64), prefix + "ffn.Wup": (24, 64),
                           prefix + "ffn.Wdown": (64, 24)})
        actual = dict(model.named_parameters())
        self.assertEqual({name: tuple(p.shape) for name, p in actual.items()}, shapes)
        self.assertEqual(sum(p.numel() for p in actual.values()), 12936)
        self.assertNotEqual(model.token_table.untyped_storage().data_ptr(), model.vocab_proj.untyped_storage().data_ptr())
        self.assertEqual(len({p.untyped_storage().data_ptr() for p in actual.values()}), len(actual))
        self.assertFalse(hasattr(model, "position_table"))

    def test_every_parameter_gradient_matches_full_independent_reference(self):
        model, ids, valid = model_sample()
        ref = copy.deepcopy(model)
        actual, expected = model(ids, valid), reference_model(ref, ids, valid)[0]
        weight = torch.linspace(-.4, .8, actual.numel(), dtype=actual.dtype).reshape(actual.shape)
        (actual * weight).sum().backward()
        (expected * weight).sum().backward()
        for (name, parameter), (_, ref_parameter) in zip(model.named_parameters(), ref.named_parameters()):
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().max().item(), 1e-9)
                self.assert_tensor(parameter.grad, ref_parameter.grad)

    def test_model_causal_and_pad_semantics(self):
        model, ids, valid = model_sample()
        valid[0, 2], valid[1, 3] = False, False
        baseline = model(ids, valid)
        changed = ids.clone()
        changed[:, 4:] = (changed[:, 4:] + 2) % model.N
        self.assert_tensor(model(changed, valid)[:, :4], baseline[:, :4])
        changed = ids.clone()
        changed[~valid] = (changed[~valid] + 3) % model.N
        self.assert_tensor(model(changed, valid)[valid], baseline[valid])

    def test_full_forward_position_limit_works_without_position_table(self):
        model, ids, valid = model_sample(L=6)
        self.assert_tensor(model(ids, valid), reference_model(model, ids, valid)[0])
        with self.assertRaises(ValueError):
            model(torch.cat((ids, ids[:, :1]), dim=1), torch.ones(2, 7, dtype=torch.bool))


class LlamaModelStepTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        model, ids, _ = model_sample()
        try:
            learner.llama_model_step(model, ids, make_caches(model))
        except NotImplementedError:
            raise unittest.SkipTest("先完成 llama_model_step 及其组件")

    def test_all_logits_match_independent_reference_for_dtypes_heads_and_chunks(self):
        for dtype in (torch.float32, torch.float64):
            for Hkv in (1, 2, 6):
                for chunks in ((6,), (3, 2, 1), (1, 1, 1, 1, 1, 1)):
                    with self.subTest(dtype=dtype, Hkv=Hkv, chunks=chunks):
                        model, ids, valid = model_sample(dtype, Hkv)
                        expected, _ = reference_model(model, ids, valid)
                        caches, outputs, start = make_caches(model), [], 0
                        for size in chunks:
                            outputs.append(learner.llama_model_step(model, ids[:, start:start + size], caches))
                            start += size
                        self.assert_tensor(torch.cat(outputs, dim=1), expected)
                        self.assertEqual([len(cache) for cache in caches], [6, 6])

    def test_each_layer_cache_values_compact_shape_and_real_storage_bytes(self):
        model, ids, valid = model_sample()
        _, expected_caches = reference_model(model, ids, valid)
        caches = make_caches(model)
        learner.llama_model_step(model, ids[:, :3], caches)
        old = [cache.k.clone() for cache in caches]
        learner.llama_model_step(model, ids[:, 3:], caches)
        storages = {}
        for layer, (cache, (k, v)) in enumerate(zip(caches, expected_caches)):
            self.assert_tensor(cache.k, k)
            self.assert_tensor(cache.v, v)
            self.assertEqual(cache.k.shape, (2, 6, 8))
            self.assertTrue(torch.equal(cache.k[:, :3], old[layer]))
            for value in (cache.k, cache.v):
                storage = value.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
        expected_bytes = 2 * 2 * 2 * 6 * 2 * 4 * 8
        self.assertEqual(sum(c.k.numel() * c.k.element_size() + c.v.numel() * c.v.element_size() for c in caches), expected_bytes)
        self.assertEqual(sum(storages.values()), expected_bytes)
        self.assertEqual(len(storages), 4)
        self.assertGreater((caches[0].k - caches[1].k).abs().max().item(), .1)

    def test_capacity_overflow_is_atomic_for_all_layers(self):
        for limit in (None, 6):
            with self.subTest(cache_limit=limit):
                model, ids, _ = model_sample(L=6)
                caches = make_caches(model, limit)
                learner.llama_model_step(model, ids[:, :5], caches)
                before = self.snapshot_caches(caches)
                with self.assertRaises(ValueError):
                    learner.llama_model_step(model, ids[:, 4:6], caches)
                self.assert_cache_unchanged(caches, before)
                learner.llama_model_step(model, ids[:, 5:6], caches)
                self.assertEqual([len(c) for c in caches], [6, 6])
                before = self.snapshot_caches(caches)
                with self.assertRaises(ValueError):
                    learner.llama_model_step(model, ids[:, :1], caches)
                self.assert_cache_unchanged(caches, before)
                empty = make_caches(model, limit)
                with self.assertRaises(ValueError):
                    learner.llama_model_step(model, torch.cat((ids, ids[:, :1]), dim=1), empty)
                self.assertTrue(all(c.k is None and c.v is None for c in empty))

    def test_interleaved_requests_and_reset_do_not_cross_contaminate(self):
        model, ids, valid = model_sample()
        other = (ids + 4) % model.N
        caches_a, caches_b = make_caches(model), make_caches(model)
        a0 = learner.llama_model_step(model, ids[:, :3], caches_a)
        b0 = learner.llama_model_step(model, other[:, :2], caches_b)
        saved_b = self.snapshot_caches(caches_b)
        a1 = learner.llama_model_step(model, ids[:, 3:], caches_a)
        self.assert_cache_unchanged(caches_b, saved_b)
        b1 = learner.llama_model_step(model, other[:, 2:], caches_b)
        self.assert_tensor(torch.cat((a0, a1), dim=1), reference_model(model, ids, valid)[0])
        self.assert_tensor(torch.cat((b0, b1), dim=1), reference_model(model, other, valid)[0])
        for cache in caches_a:
            cache.reset()
        self.assert_tensor(learner.llama_model_step(model, other, caches_a), reference_model(model, other, valid)[0])
        self.assertEqual([len(c) for c in caches_b], [6, 6])

    def test_chunked_gradients_include_history_and_match_full_reference(self):
        model, ids, valid = model_sample()
        ref, caches = copy.deepcopy(model), make_caches(model)
        learner.llama_model_step(model, ids[:, :3], caches)
        actual = learner.llama_model_step(model, ids[:, 3:], caches)
        expected = reference_model(ref, ids, valid)[0][:, 3:]
        weight = torch.linspace(-.6, .9, actual.numel(), dtype=actual.dtype).reshape(actual.shape)
        (actual * weight).sum().backward()
        (expected * weight).sum().backward()
        for (name, parameter), (_, reference_parameter) in zip(model.named_parameters(), ref.named_parameters()):
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assert_tensor(parameter.grad, reference_parameter.grad)
        # token 1 只出现在历史前缀，后半段 loss 仍须沿缓存依赖到达其 embedding。
        self.assertGreater(model.token_table.grad[1].abs().max().item(), 1e-8)

    def test_forward_and_step_preserve_modes_values_inputs_and_existing_gradients(self):
        model, ids, valid = model_sample()
        model.train()
        model.blocks[1].eval()  # 子模块可能有不同模式，不能粗暴 eval() 后整体 train()。
        modes = {name: m.training for name, m in model.named_modules()}
        saved = {name: p.clone() for name, p in model.named_parameters()}
        grads = {}
        for name, p in model.named_parameters():
            p.grad = torch.full_like(p, .125)
            grads[name] = p.grad
        old_ids, old_valid = ids.clone(), valid.clone()
        for cached in (False, True):
            output = learner.llama_model_step(model, ids, make_caches(model)) if cached else model(ids, valid)
            self.assertTrue(output.requires_grad)
            self.assertEqual({name: m.training for name, m in model.named_modules()}, modes)
            for name, p in model.named_parameters():
                self.assertTrue(torch.equal(p, saved[name]))
                self.assertIs(p.grad, grads[name])
                self.assertTrue(torch.equal(p.grad, torch.full_like(p, .125)))
        self.assertTrue(torch.equal(ids, old_ids))
        self.assertTrue(torch.equal(valid, old_valid))
        with torch.no_grad():
            self.assertFalse(learner.llama_model_step(model, ids, make_caches(model)).requires_grad)


if __name__ == "__main__":
    unittest.main()
