"""ex012：GQA 的紧凑缓存与模型连接。

教师测试参照使用逐头索引和官方 LayerNorm/ReLU，不导入学习者 Attention
来产生预期值。这些参照仅用于验收，不应从作业中导入。
CPU float64：rtol=1e-8, atol=1e-10；float32：rtol=1e-5, atol=1e-6。
种子 401/409/419/421/431/433；不作性能计时。
"""
import copy
import unittest

import torch
import torch.nn.functional as F

from exercises.ex011_kv_cache.cache import LayerKVCache
from exercises.ex012_grouped_query_attention import gqa, model as learner


def setUpModule():
    global _threads
    _threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_threads)


def close(actual, expected, message=""):
    tol = dict(rtol=1e-8, atol=1e-10) if expected.dtype == torch.float64 else dict(rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual, expected, **tol, msg=message or None)


def make_block(hkv=2, dtype=torch.float64, seed=401):
    torch.manual_seed(seed)
    block = gqa.GQABlock(8, 4, hkv, 11, dtype=dtype)
    # 非平凡分支，避免小初始化令错误低于容差。
    with torch.no_grad():
        for name, param in block.named_parameters():
            if name.endswith("gamma"):
                param.copy_(0.8 + torch.rand_like(param) * 0.4)
            else:
                param.copy_(torch.randn_like(param) * 0.23)
    return block


def make_model(hkv=2, dtype=torch.float64, seed=409, length=9):
    torch.manual_seed(seed)
    result = learner.GQAMiniGPT(13, 8, 4, hkv, 11, 2, length, dtype=dtype)
    with torch.no_grad():
        for name, param in result.named_parameters():
            if name.endswith("gamma"):
                param.copy_(0.8 + torch.rand_like(param) * 0.4)
            else:
                param.copy_(torch.randn_like(param) * 0.23)
    return result


def norm_reference(x, norm):
    return F.layer_norm(x, (x.shape[-1],), norm.gamma, norm.beta, norm.eps)


def block_reference(block, x):
    """无 PAD 全量慢参照；逐头切片明确连续分组，不复用 split/group helper。"""
    normalized = norm_reference(x, block.norm1)
    q, k, v = (normalized @ weight for weight in (block.Wq, block.Wk, block.Wv))
    B, T, C = q.shape
    D = C // block.num_query_heads
    group_size = block.num_query_heads // block.num_kv_heads
    visible = torch.arange(T)[None, :] <= torch.arange(T)[:, None]
    pieces = []
    for h in range(block.num_query_heads):
        kv_head = h // group_size
        qh = q[:, :, h * D:(h + 1) * D]
        kh = k[:, :, kv_head * D:(kv_head + 1) * D]
        vh = v[:, :, kv_head * D:(kv_head + 1) * D]
        scores = (qh @ kh.transpose(-1, -2)) / D**0.5
        weights = scores.masked_fill(~visible, -torch.inf).softmax(dim=-1)
        pieces.append(weights @ vh)
    u = x + torch.cat(pieces, dim=-1) @ block.Wo
    ff = block.ffn
    y = u + F.relu(norm_reference(u, block.norm2) @ ff.W1 + ff.b1) @ ff.W2 + ff.b2
    return y


def model_reference(model, ids):
    x = model.token_table[ids] + model.position_table[:ids.shape[1]]
    for block in model.blocks:
        x = block_reference(block, x)
    return norm_reference(x, model.final_norm) @ model.vocab_proj


def block_probe():
    block = make_block()
    return learner.gqa_block_step(block, torch.ones(1, 1, 8, dtype=torch.float64), LayerKVCache())


def model_probe():
    model = make_model()
    return learner.gqa_model_step(model, torch.tensor([[1]]), [LayerKVCache() for _ in model.blocks])


def skip_only_todo(probe):
    # 其他异常必须作为 ERROR 显示，不能误吞为“尚未实现”。
    try:
        probe()
    except NotImplementedError as exc:
        raise unittest.SkipTest(f"先完成本组所依赖的核心函数：{exc}") from exc


class TestImplementationStatus(unittest.TestCase):
    def test_block_step_is_implemented(self):
        try:
            block_probe()
        except NotImplementedError as exc:
            self.fail(f"块缓存入口或其依赖仍未实现：{exc}")

    def test_model_step_is_implemented(self):
        try:
            model_probe()
        except NotImplementedError as exc:
            self.fail(f"模型缓存入口或其依赖仍未实现：{exc}")


class TestCompactBlockCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_only_todo(block_probe)

    def test_outputs_match_independent_reference_for_all_chunkings(self):
        for dtype in (torch.float64, torch.float32):
            for hkv in (1, 2, 4):
                block = make_block(hkv, dtype)
                torch.manual_seed(419)
                x = torch.randn(2, 8, 7, dtype=dtype).transpose(1, 2)
                self.assertFalse(x.is_contiguous())
                expected = block_reference(block, x)
                for sizes in ((7,), (2, 1, 4), (1,) * 7):
                    with self.subTest(dtype=dtype, hkv=hkv, sizes=sizes):
                        cache, outputs, offset = LayerKVCache(), [], 0
                        for size in sizes:
                            outputs.append(learner.gqa_block_step(block, x[:, offset:offset + size], cache))
                            offset += size
                            self.assertEqual(len(cache), offset)
                        close(torch.cat(outputs, dim=1), expected, "分块输出必须保留绝对位置因果权限")

    def test_cache_contains_only_compact_normalized_projections(self):
        for dtype in (torch.float64, torch.float32):
            for hkv in (1, 2, 4):
                block = make_block(hkv, dtype)
                x = torch.randn(3, 5, 8, dtype=dtype)
                cache = LayerKVCache(max_length=20)
                for start, end in ((0, 2), (2, 3), (3, 5)):
                    learner.gqa_block_step(block, x[:, start:end], cache)
                    normalized = norm_reference(x[:, :end], block.norm1)
                    for tensor, weight in ((cache.k, block.Wk), (cache.v, block.Wv)):
                        self.assertEqual(tuple(tensor.shape), (3, end, hkv * 2))
                        close(tensor, normalized @ weight, "缓存必须保存本层 norm1 输入投影")
                        expected_bytes = 3 * end * hkv * 2 * tensor.element_size()
                        self.assertEqual(tensor.numel() * tensor.element_size(), expected_bytes)
                        self.assertEqual(tensor.untyped_storage().nbytes(), expected_bytes,
                                         "不能持久缓存扩展副本或预分配额外容量")

    def test_later_output_gradients_reach_old_and_new_inputs_and_parameters(self):
        block = make_block()
        ref = copy.deepcopy(block)
        torch.manual_seed(421)
        old = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
        new = torch.randn(2, 2, 8, dtype=torch.float64, requires_grad=True)
        old_ref, new_ref = (x.detach().clone().requires_grad_() for x in (old, new))
        cache = LayerKVCache()
        learner.gqa_block_step(block, old, cache)
        got = learner.gqa_block_step(block, new, cache)
        expected = block_reference(ref, torch.cat([old_ref, new_ref], dim=1))[:, 3:]
        close(got, expected)
        coefficient = torch.randn_like(expected)
        actual_inputs = [old, new, *block.parameters()]
        expected_inputs = [old_ref, new_ref, *ref.parameters()]
        actual_grad = torch.autograd.grad((got * coefficient).sum(), actual_inputs, allow_unused=True)
        expected_grad = torch.autograd.grad((expected * coefficient).sum(), expected_inputs)
        self.assertGreater(expected_grad[0].abs().max().item(), 1e-5, "夹具必须激活历史 K/V 的梯度")
        for name, got_grad, want_grad in zip(["old_x", "new_x", *dict(block.named_parameters())], actual_grad, expected_grad):
            self.assertIsNotNone(got_grad, f"梯度断开：{name}，检查缓存是否 detach")
            close(got_grad, want_grad, f"梯度不符：{name}，检查缓存是否 detach")

    def test_capacity_failure_is_atomic_and_exact_capacity_succeeds(self):
        block = make_block()
        cache = LayerKVCache(max_length=4)
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        learner.gqa_block_step(block, x[:, :3], cache)
        k, v = cache.k, cache.v
        k_value, v_value = k.clone(), v.clone()
        with self.assertRaises(ValueError):
            learner.gqa_block_step(block, x[:, 3:], cache)
        self.assertEqual(len(cache), 3)
        self.assertIs(cache.k, k)
        self.assertIs(cache.v, v)
        close(cache.k, k_value)
        close(cache.v, v_value)
        got = learner.gqa_block_step(block, x[:, 3:4], cache)
        close(got, block_reference(block, x[:, :4])[:, 3:])
        self.assertEqual(len(cache), 4)

    def test_reset_starts_a_fresh_compact_request(self):
        block = make_block()
        cache = LayerKVCache(max_length=6)
        learner.gqa_block_step(block, torch.randn(2, 4, 8, dtype=torch.float64), cache)
        cache.reset()
        self.assertIsNone(cache.k)
        self.assertIsNone(cache.v)
        self.assertEqual(cache.max_length, 6)
        x = torch.randn(1, 3, 8, dtype=torch.float64)
        got = learner.gqa_block_step(block, x, cache)
        close(got, block_reference(block, x))
        self.assertEqual(tuple(cache.k.shape), (1, 3, 4))

    def test_forward_preserves_inputs_parameters_existing_grads_and_modes(self):
        block = make_block()
        for training in (False, True):
            block.train(training)
            block.norm2.eval()  # 子模块模式也不能被入口悄悄统一覆盖。
            modes = [part.training for part in block.modules()]
            x = torch.randn(2, 2, 8, dtype=torch.float64, requires_grad=True)
            before_x = x.detach().clone()
            params = list(block.parameters())
            before = [p.detach().clone() for p in params]
            for p in params:
                p.grad = torch.full_like(p, 0.17)
            rng_before = torch.random.get_rng_state().clone()
            got = learner.gqa_block_step(block, x, LayerKVCache())
            self.assertTrue(got.requires_grad)
            self.assertEqual(modes, [part.training for part in block.modules()])
            self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
            close(x, before_x)
            for p, value in zip(params, before):
                close(p, value)
                close(p.grad, torch.full_like(p, 0.17))


class TestCompactModelCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_only_todo(model_probe)

    def test_actual_logits_match_independent_and_full_model_for_chunkings(self):
        ids = torch.tensor([[1, 4, 2, 8, 3, 10, 6], [9, 2, 5, 3, 12, 1, 7]])
        for dtype in (torch.float64, torch.float32):
            for hkv in (1, 2, 4):
                model = make_model(hkv, dtype)
                expected = model_reference(model, ids)
                # 全量连接也必须通过独立参照；不能只让两条错误路径互相对齐。
                full = model(ids, torch.ones_like(ids, dtype=torch.bool))
                close(full, expected, "全量 GQA 模型偏离独立数学参照")
                for sizes in ((7,), (3, 1, 3), (1,) * 7):
                    with self.subTest(dtype=dtype, hkv=hkv, sizes=sizes):
                        caches = [LayerKVCache(max_length=model.L) for _ in model.blocks]
                        outputs, offset = [], 0
                        for size in sizes:
                            # 确实检查学习者 model_step 返回的每一个位置的 logits。
                            got = learner.gqa_model_step(model, ids[:, offset:offset + size], caches)
                            self.assertEqual(tuple(got.shape), (2, size, model.N))
                            outputs.append(got)
                            offset += size
                            self.assertTrue(all(len(c) == offset for c in caches))
                        close(torch.cat(outputs, dim=1), expected, "不能只对齐贪心 ID 或最后一行")

    def test_offset_comes_from_cache_length_not_number_of_calls(self):
        model = make_model(seed=431)
        ids = torch.tensor([[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]])
        caches_a = [LayerKVCache() for _ in model.blocks]
        caches_b = [LayerKVCache() for _ in model.blocks]
        learner.gqa_model_step(model, ids[:, :4], caches_a)
        for i in range(4):
            learner.gqa_model_step(model, ids[:, i:i + 1], caches_b)
        a = learner.gqa_model_step(model, ids[:, 4:], caches_a)
        b = learner.gqa_model_step(model, ids[:, 4:], caches_b)
        expected = model_reference(model, ids)[:, 4:]
        close(a, expected)
        close(b, expected)
        for ca, cb in zip(caches_a, caches_b):
            close(ca.k, cb.k)
            close(ca.v, cb.v)

    def test_registered_parameter_count_and_actual_cache_bytes(self):
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        for hkv in (1, 2, 4):
            model = make_model(hkv, dtype=torch.float32)
            C, N, L, F_, Hq, D = 8, 13, model.L, 11, 4, 2
            per_block = 2 * C * C + 2 * C * hkv * D + 5 * C + 2 * C * F_ + F_
            expected_parameters = 2 * N * C + L * C + 2 * C + model.n_layer * per_block
            named_before = dict(model.named_parameters())
            state_before = {name: value.clone() for name, value in model.state_dict().items()}
            self.assertEqual(sum(p.numel() for p in named_before.values()), expected_parameters)
            for i in range(model.n_layer):
                self.assertEqual(tuple(named_before[f"blocks.{i}.Wk"].shape), (C, hkv * D))
                self.assertEqual(tuple(named_before[f"blocks.{i}.Wv"].shape), (C, hkv * D))
            caches = [LayerKVCache(max_length=L) for _ in model.blocks]
            learner.gqa_model_step(model, ids, caches)
            expected_bytes = 2 * model.n_layer * ids.shape[0] * ids.shape[1] * hkv * D * 4
            tensors = [t for cache in caches for t in (cache.k, cache.v)]
            self.assertEqual(sum(t.numel() * t.element_size() for t in tensors), expected_bytes)
            self.assertEqual(sum(t.untyped_storage().nbytes() for t in tensors), expected_bytes)
            named_after = dict(model.named_parameters())
            self.assertEqual(set(named_before), set(named_after))
            for name, param in named_before.items():
                self.assertIs(named_after[name], param, "forward 不能创建或替换参数")
            self.assertEqual(set(state_before), set(model.state_dict()), "请求缓存不能进入 state_dict")
            for name, value in model.state_dict().items():
                close(value, state_before[name], name)

    def test_model_gradients_and_caller_modes_are_preserved(self):
        for training in (False, True):
            model = make_model(seed=433)
            model.train(training)
            model.blocks[0].norm1.eval()
            modes = [part.training for part in model.modules()]
            ref = copy.deepcopy(model)
            ids = torch.tensor([[1, 2, 3, 4, 5], [7, 6, 5, 4, 3]])
            ids_before = ids.clone()
            caches = [LayerKVCache() for _ in model.blocks]
            for p in model.parameters():
                p.grad = torch.full_like(p, 0.19)
            learner.gqa_model_step(model, ids[:, :3], caches)
            actual = learner.gqa_model_step(model, ids[:, 3:], caches)
            expected = model_reference(ref, ids)[:, 3:]
            close(actual, expected)
            self.assertTrue(actual.requires_grad, "缓存入口不能擅自关闭梯度")
            self.assertEqual(modes, [part.training for part in model.modules()])
            self.assertTrue(torch.equal(ids, ids_before))
            coefficient = torch.randn_like(actual)
            ga = torch.autograd.grad((actual * coefficient).sum(), list(model.parameters()), allow_unused=True)
            gr = torch.autograd.grad((expected * coefficient).sum(), list(ref.parameters()))
            for (name, param), actual_grad, expected_grad in zip(model.named_parameters(), ga, gr):
                self.assertIsNotNone(actual_grad, f"模型参数梯度断开：{name}")
                close(actual_grad, expected_grad, f"模型参数梯度不符：{name}")
                close(param.grad, torch.full_like(param, 0.19))
            # 调用方 no_grad 生效；入口不能反向打开梯度，也不能切换模式。
            with torch.no_grad():
                output = learner.gqa_model_step(model, ids[:, :1], [LayerKVCache() for _ in model.blocks])
            self.assertFalse(output.requires_grad)
            self.assertEqual(modes, [part.training for part in model.modules()])

    def test_interleaved_requests_and_layer_states_do_not_mix(self):
        model = make_model()
        a = torch.tensor([[1, 4, 6, 8, 10]])
        b = torch.tensor([[12, 9, 7, 5]])
        ca = [LayerKVCache() for _ in model.blocks]
        cb = [LayerKVCache() for _ in model.blocks]
        first_a = learner.gqa_model_step(model, a[:, :3], ca)
        frozen_a = [(c.k.clone(), c.v.clone()) for c in ca]
        first_b = learner.gqa_model_step(model, b[:, :1], cb)
        for cache, (k, v) in zip(ca, frozen_a):
            close(cache.k, k)
            close(cache.v, v)
        rest_a = learner.gqa_model_step(model, a[:, 3:], ca)
        rest_b = learner.gqa_model_step(model, b[:, 1:], cb)
        close(torch.cat([first_a, rest_a], dim=1), model_reference(model, a))
        close(torch.cat([first_b, rest_b], dim=1), model_reference(model, b))
        storages = [t.untyped_storage().data_ptr() for cache in ca + cb for t in (cache.k, cache.v)]
        self.assertEqual(len(set(storages)), len(storages), "层、请求和 K/V 不得共用同一份缓存数据")
        cb[0].reset()
        self.assertEqual([len(c) for c in ca], [5, 5])
        self.assertEqual(len(cb[1]), 4)

    def test_model_position_capacity_fails_before_any_layer_cache_changes(self):
        model = make_model(length=6)
        ids = torch.tensor([[1, 3, 5, 7, 9, 11, 12]])
        caches = [LayerKVCache() for _ in model.blocks]  # 容器无限长，必须由模型检查位置上限。
        learner.gqa_model_step(model, ids[:, :5], caches)
        snapshots = [(c.k, c.v, c.k.clone(), c.v.clone()) for c in caches]
        with self.assertRaises(ValueError):
            learner.gqa_model_step(model, ids[:, 5:], caches)
        for cache, (k, v, kval, vval) in zip(caches, snapshots):
            self.assertEqual(len(cache), 5)
            self.assertIs(cache.k, k)
            self.assertIs(cache.v, v)
            close(cache.k, kval)
            close(cache.v, vval)
        got = learner.gqa_model_step(model, ids[:, 5:6], caches)
        close(got, model_reference(model, ids[:, :6])[:, 5:])
        self.assertEqual([len(c) for c in caches], [6, 6])


if __name__ == "__main__":
    unittest.main()
