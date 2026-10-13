"""ex021 离线教师验收：无需模型下载、tokenizer、Transformers 或网络。

CPU float32，确定性解析夹具，rtol=atol=2e-5。完整前向采用教师独立
实现生成的固定数值，不把完整模型答案放进测试。局部逐元素数学参照只供
验收，不得由作业导入；通过教师测试不等于独立设计了测试。
"""

import math
import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from exercises.ex021_qwen3_inference import model as learner
from exercises.ex021_qwen3_inference import support
from exercises.ex021_qwen3_inference.support import LayerKV, QwenConfig


def setUpModule():
    global _old_threads
    _old_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_old_threads)


def fixture():
    """特意令 C=6，而 Hq*D=16；Q/K norm 参数不是全 1。"""
    config = QwenConfig(
        vocab_size=13, hidden_size=6, intermediate_size=11,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=4,
        max_position_embeddings=16, rms_norm_eps=1e-6,
        rope_theta=10000.0, tie_word_embeddings=True,
    )
    weights = {}
    phase = 0

    def matrix(shape, scale=0.24):
        nonlocal phase
        phase += 1
        return (torch.sin(torch.arange(math.prod(shape), dtype=torch.float32)
                          * 0.37 + phase * 0.71) * scale).reshape(shape)

    def norm(width):
        nonlocal phase
        phase += 1
        return 0.7 + torch.cos(torch.arange(width, dtype=torch.float32)
                               * 0.91 + phase * 0.51) * 0.35

    weights['model.embed_tokens.weight'] = matrix((13, 6), 0.8)
    for i in range(2):
        p = f'model.layers.{i}.'
        weights[p + 'input_layernorm.weight'] = norm(6)
        for name, shape in (
            ('q_proj', (16, 6)), ('k_proj', (8, 6)),
            ('v_proj', (8, 6)), ('o_proj', (6, 16)),
        ):
            weights[p + f'self_attn.{name}.weight'] = matrix(shape)
        weights[p + 'self_attn.q_norm.weight'] = norm(4)
        weights[p + 'self_attn.k_norm.weight'] = norm(4)
        weights[p + 'post_attention_layernorm.weight'] = norm(6)
        for name, shape in (
            ('gate_proj', (11, 6)), ('up_proj', (11, 6)),
            ('down_proj', (6, 11)),
        ):
            weights[p + f'mlp.{name}.weight'] = matrix(shape)
    weights['model.norm.weight'] = norm(6)
    weights['lm_head.weight'] = weights['model.embed_tokens.weight']
    return config, weights


def block_input():
    return torch.tensor([[
        [0.2, -0.7, 1.1, 0.5, -0.3, 0.8],
        [-0.4, 0.6, 0.9, -1.2, 0.2, 0.3],
        [1.0, 0.1, -0.8, 0.7, 0.4, -0.5],
    ]])


# 固定夹具由隔离教师实现生成；不是从 learner 计算后自我比较。
BLOCK_OUTPUT = [[
    [0.27061719, -0.50683779, 1.21915257, 0.62899786, -0.12573376, 0.83479047],
    [-0.33928436, 0.70675898, 0.99370915, -1.10526729, 0.29705128, 0.34404951],
    [1.05957651, 0.19232745, -0.72688168, 0.76512229, 0.46367988, -0.48485479],
]]
BLOCK_KEYS = [[
    [[-0.62670732, 0.40417546, 0.67306817, -1.30537939],
     [-0.72899967, 0.58554935, -0.30014488, -1.24985695],
     [-1.23223472, -1.18479621, -0.23283264, 0.02748646]],
    [[0.47434148, 0.16088299, -1.23600793, 1.25962007],
     [1.19052732, -0.05057082, -0.12077072, 1.30914557],
     [0.68269742, 1.09203064, -0.31552488, -0.75546229]],
]]
MODEL_LOGITS = [[
    [-1.99192190, 1.49017823, 0.19014059, -1.72007823, 1.88961339,
     -0.56466234, -1.20687771, 2.02390432, -1.24023271, -0.52433306,
     1.87420607, -1.74177790, 0.23178530],
    [-2.32152081, 1.36987460, 0.66519916, -2.17417002, 1.96359944,
     -0.20002741, -1.72174561, 2.28179812, -1.03718543, -1.02773237,
     2.27982116, -1.72880960, -0.18950993],
    [-2.22018456, -0.06172185, 2.29481292, -2.71294689, 0.98542345,
     1.52146637, -2.82503605, 1.89429569, 0.53463435, -2.54072499,
     2.53736782, -0.52721733, -1.89990711],
]]


def generation_stub(tokens):
    calls = []
    returned_caches = []

    def fake_forward(ids, weights, config, cache=None):
        step = len(calls)
        calls.append((ids.clone(), cache))
        logits = torch.zeros(ids.shape[0], ids.shape[1], config.vocab_size)
        logits[..., 12] = 9.0  # 非末行设不同赢家，防止取错位置。
        logits[:, -1] = 0.0
        logits[:, -1, tokens[step]] = 10.0
        old = 0 if cache is None else cache[0].k.shape[2]
        shape = (1, config.num_key_value_heads, old + ids.shape[1], config.head_dim)
        new = [LayerKV(torch.zeros(shape), torch.zeros(shape))
               for _ in range(config.num_hidden_layers)]
        returned_caches.append(new)
        return logits, new

    return fake_forward, calls, returned_caches


def probe(name):
    config, weights = fixture()
    x = torch.ones(1, 1, 4)
    if name == 'rms_norm':
        return learner.rms_norm(x, torch.ones(4), 1e-6)
    if name == 'apply_rope':
        return learner.apply_rope(x.unsqueeze(1), torch.tensor([0]), 10000.0)
    if name == 'grouped_attention':
        q = torch.ones(1, 4, 1, 4)
        k = torch.ones(1, 2, 1, 4)
        return learner.grouped_attention(q, k, k, torch.tensor([0]), torch.tensor([0]))
    if name == 'transformer_block':
        return learner.transformer_block(block_input(), weights, config, 0, torch.arange(3))
    if name == 'forward':
        return learner.forward(torch.tensor([[1, 4, 2]]), weights, config)
    if name == 'greedy_generate':
        fake, _, _ = generation_stub([2])
        with patch.object(learner, 'forward', fake):
            return learner.greedy_generate(torch.tensor([[1]]), weights, config,
                                           max_new_tokens=1, eos_token_ids=(12,))
    raise AssertionError(name)


def require(name):
    try:
        probe(name)
    except NotImplementedError as error:
        raise unittest.SkipTest(f'{name} 尚未实现：{error}') from None


class ImplementationStatusTest(unittest.TestCase):
    def test_all_required_core_functions_implemented(self):
        pending = []
        for name in ('rms_norm', 'apply_rope', 'grouped_attention',
                     'transformer_block', 'forward', 'greedy_generate'):
            try:
                probe(name)
            except NotImplementedError:
                pending.append(name)
        self.assertFalse(pending, '尚未完成，不是通过：' + ', '.join(pending))


class TensorAssertions(unittest.TestCase):
    def assert_tensor(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


class SupportTests(TensorAssertions):
    """教师适配层独立执行；即使六个核心 TODO 为空，也必须通过。"""

    def test_expected_shapes_keep_attention_width_independent_from_hidden_size(self):
        config, weights = fixture()
        expected = support.expected_shapes(config)
        self.assertEqual(expected['model.layers.0.self_attn.q_proj.weight'], (16, 6))
        self.assertEqual(expected['model.layers.0.self_attn.o_proj.weight'], (6, 16))
        self.assertEqual(expected['model.layers.1.self_attn.k_proj.weight'], (8, 6))
        self.assertEqual(expected['model.layers.1.self_attn.q_norm.weight'], (4,))
        self.assertEqual(expected, {name: tuple(value.shape) for name, value in weights.items()})

    def test_missing_tied_head_is_aliased_without_editing_original_dictionary(self):
        config, weights = fixture()
        supplied = {name: value.to(torch.float64) for name, value in weights.items()
                    if name != 'lm_head.weight'}
        before = {name: value.clone() for name, value in supplied.items()}
        loaded = support.validate_weights(supplied, config)
        self.assertNotIn('lm_head.weight', supplied)
        self.assertIs(loaded['lm_head.weight'], loaded['model.embed_tokens.weight'])
        for name, original in supplied.items():
            self.assertTrue(torch.equal(original, before[name]), name)
            self.assertEqual(original.dtype, torch.float64)
            self.assert_tensor(loaded[name], original.float())

    def test_disagreeing_duplicate_tied_head_is_rejected(self):
        config, weights = fixture()
        weights['lm_head.weight'] = weights['lm_head.weight'].clone()
        weights['lm_head.weight'][0, 0] += 0.5
        with self.assertRaises(ValueError):
            support.validate_weights(weights, config)

    def test_read_config_requires_explicit_head_dim_and_rejects_sliding_window(self):
        config, _ = fixture()
        payload = dict(asdict(config), model_type='qwen3', hidden_act='silu')
        with TemporaryDirectory(prefix='qwen-config-test-') as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(payload))
            self.assertEqual(support.read_config(path), config)
            variants = [dict(payload, use_sliding_window=True),
                        {name: value for name, value in payload.items() if name != 'head_dim'}]
            for invalid in variants:
                with self.subTest(keys=sorted(invalid)):
                    path.write_text(json.dumps(invalid))
                    with self.assertRaises(ValueError):
                        support.read_config(path)


class RMSNormTests(TensorAssertions):
    def test_no_centering_last_axis_nonuniform_weight_and_epsilon_inside_root(self):
        require('rms_norm')
        x = torch.tensor([[[2., 2., 2.], [1e-4, -2e-4, 3e-4]],
                          [[-1., 3., 7.], [0., 0., 0.]]])
        weight = torch.tensor([0.4, 1.7, -0.6])
        expected = torch.empty_like(x)
        for b in range(2):
            for t in range(2):
                denominator = math.sqrt(sum(float(a) ** 2 for a in x[b, t]) / 3 + 1e-6)
                expected[b, t] = x[b, t] / denominator * weight
        self.assert_tensor(learner.rms_norm(x, weight, 1e-6), expected)


class RoPETests(TensorAssertions):
    def setUp(self):
        require('apply_rope')

    def test_half_split_pairs_distinct_frequencies_and_nonzero_positions(self):
        x = torch.arange(2 * 3 * 2 * 16, dtype=torch.float32).reshape(2, 3, 2, 16)[..., ::2] / 31
        positions = torch.tensor([3, 11])
        expected = torch.empty_like(x)
        for t, position in enumerate(positions.tolist()):
            for j in range(4):
                angle = position / (10000.0 ** (2 * j / 8))
                a, b = x[:, :, t, j], x[:, :, t, j + 4]
                expected[:, :, t, j] = a * math.cos(angle) - b * math.sin(angle)
                expected[:, :, t, j + 4] = a * math.sin(angle) + b * math.cos(angle)
        self.assert_tensor(learner.apply_rope(x, positions, 10000.0), expected)

    def test_zero_position_identity_and_does_not_modify_input(self):
        x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 1, 8)
        before = x.clone()
        self.assert_tensor(learner.apply_rope(x, torch.tensor([0]), 10000.0), before)
        learner.apply_rope(x, torch.tensor([7]), 10000.0)
        self.assertTrue(torch.equal(x, before))


class GroupedAttentionTests(TensorAssertions):
    def setUp(self):
        require('grouped_attention')

    def test_contiguous_query_groups_share_the_correct_kv_head(self):
        q = torch.zeros(2, 6, 1, 2)
        k = torch.zeros(2, 2, 1, 2)
        v = torch.tensor([[[[1., 3.]], [[7., 11.]]],
                          [[[2., 5.]], [[13., 17.]]]])
        expected = torch.stack([v[:, 0], v[:, 0], v[:, 0],
                                v[:, 1], v[:, 1], v[:, 1]], 1)
        actual = learner.grouped_attention(q, k, v, torch.tensor([4]), torch.tensor([4]))
        self.assert_tensor(actual, expected)

    def test_scaled_scores_absolute_causality_and_rectangular_chunks(self):
        q = torch.sin(torch.arange(2 * 4 * 2 * 4).reshape(2, 4, 2, 4).float() * 0.6)
        k = torch.cos(torch.arange(2 * 2 * 5 * 4).reshape(2, 2, 5, 4).float() * 0.4)
        v = torch.sin(torch.arange(2 * 2 * 5 * 4).reshape(2, 2, 5, 4).float() * 0.8) * 3
        qp, kp = torch.tensor([7, 9]), torch.tensor([2, 4, 7, 9, 12])
        expected = torch.empty_like(q)
        for b in range(2):
            for h in range(4):
                for t in range(2):
                    indices = [i for i, pos in enumerate(kp.tolist()) if pos <= qp[t]]
                    scores = torch.stack([torch.dot(q[b, h, t], k[b, h // 2, i]) / 2 for i in indices])
                    probabilities = torch.softmax(scores, 0)
                    expected[b, h, t] = sum(p * v[b, h // 2, i] for p, i in zip(probabilities, indices))
        self.assert_tensor(learner.grouped_attention(q, k, v, qp, kp), expected)

    def test_future_values_cannot_change_earlier_queries(self):
        q = torch.ones(1, 4, 2, 4)
        k = torch.ones(1, 2, 5, 4)
        v = torch.arange(40, dtype=torch.float32).reshape(1, 2, 5, 4)
        qp, kp = torch.tensor([2, 3]), torch.arange(5)
        expected = learner.grouped_attention(q, k, v, qp, kp)
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[:, :, 4] = 100
        changed_v[:, :, 4] = -999
        self.assert_tensor(learner.grouped_attention(q, changed_k, changed_v, qp, kp), expected)


class TransformerBlockTests(TensorAssertions):
    def setUp(self):
        require('transformer_block')
        self.config, self.weights = fixture()

    def test_expanded_attention_width_qk_norm_residual_and_swiglu_fixed_values(self):
        actual, _ = learner.transformer_block(block_input(), self.weights, self.config, 0, torch.arange(3))
        self.assert_tensor(actual, torch.tensor(BLOCK_OUTPUT))

    def test_compact_cache_contains_normalized_rotated_keys_and_unrotated_values(self):
        _, kv = learner.transformer_block(block_input(), self.weights, self.config, 0, torch.arange(3))
        self.assertEqual(kv.k.shape, (1, 2, 3, 4))
        self.assertEqual(kv.v.shape, (1, 2, 3, 4))
        self.assert_tensor(kv.k, torch.tensor(BLOCK_KEYS))
        # 只检查 V 的独立投影；此处不含完整 Attention 或 Block 答案。
        x = block_input()
        p = 'model.layers.0.'
        z = x / torch.sqrt(x.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
        z = z * self.weights[p + 'input_layernorm.weight']
        projected = torch.nn.functional.linear(z, self.weights[p + 'self_attn.v_proj.weight'])
        for h in range(2):
            self.assert_tensor(kv.v[:, h], projected[:, :, h * 4:(h + 1) * 4])

    def test_block_chunk_append_keeps_old_key_rotation_and_supports_noncontiguous_x(self):
        x = torch.arange(1 * 5 * 12, dtype=torch.float32).reshape(1, 5, 12)[..., ::2] / 19
        whole, whole_kv = learner.transformer_block(x, self.weights, self.config, 1, torch.arange(5))
        first, kv = learner.transformer_block(x[:, :2], self.weights, self.config, 1, torch.arange(2))
        old_k, old_v = kv.k.clone(), kv.v.clone()
        second, final = learner.transformer_block(x[:, 2:], self.weights, self.config, 1, torch.arange(2, 5), kv)
        self.assert_tensor(torch.cat((first, second), 1), whole)
        self.assert_tensor(final.k, whole_kv.k)
        self.assert_tensor(final.v, whole_kv.v)
        self.assertTrue(torch.equal(kv.k, old_k))
        self.assertTrue(torch.equal(kv.v, old_v))
        self.assertTrue(torch.equal(final.k[:, :, :2], old_k))


class ForwardTests(TensorAssertions):
    def setUp(self):
        require('forward')
        self.config, self.weights = fixture()

    def test_embedding_layers_final_norm_and_tied_lm_head_fixed_logits(self):
        actual, cache = learner.forward(torch.tensor([[1, 4, 2]]), self.weights, self.config)
        self.assert_tensor(actual, torch.tensor(MODEL_LOGITS))
        self.assertEqual(len(cache), 2)

    def test_full_and_multiple_chunk_partitions_align_at_every_position(self):
        ids = torch.tensor([[1, 4, 2, 5, 3, 8], [6, 1, 7, 2, 0, 4]])
        whole, whole_cache = learner.forward(ids, self.weights, self.config)
        for partition in ((1, 1, 1, 1, 1, 1), (2, 3, 1), (4, 2)):
            with self.subTest(partition=partition):
                cache, start, parts = None, 0, []
                for size in partition:
                    result, cache = learner.forward(ids[:, start:start + size], self.weights, self.config, cache)
                    self.assertEqual(result.shape, (2, size, 13), '必须返回全部新增位置，而不只是最后一行')
                    parts.append(result)
                    start += size
                self.assert_tensor(torch.cat(parts, 1), whole)
                for actual, expected in zip(cache, whole_cache):
                    self.assert_tensor(actual.k, expected.k)
                    self.assert_tensor(actual.v, expected.v)
                    self.assertEqual(actual.k.shape, (2, 2, 6, 4))

    def test_future_token_perturbation_does_not_change_prefix_logits(self):
        ids = torch.tensor([[1, 4, 2, 5, 3]])
        expected, _ = learner.forward(ids, self.weights, self.config)
        changed = ids.clone()
        changed[:, 3:] = torch.tensor([[9, 10]])
        actual, _ = learner.forward(changed, self.weights, self.config)
        self.assert_tensor(actual[:, :3], expected[:, :3])
        self.assertGreater(float((actual[:, 3:] - expected[:, 3:]).abs().max()), 1e-3)

    def test_batch_independence_and_cache_none_starts_an_isolated_request(self):
        ids = torch.tensor([[1, 4, 2], [8, 5, 6]])
        batch, _ = learner.forward(ids, self.weights, self.config)
        for b in range(2):
            single, _ = learner.forward(ids[b:b + 1], self.weights, self.config, None)
            self.assert_tensor(single, batch[b:b + 1])
        _, old = learner.forward(ids[:1], self.weights, self.config)
        learner.forward(torch.tensor([[3, 7]]), self.weights, self.config, old)
        fresh, fresh_cache = learner.forward(ids[:1], self.weights, self.config, None)
        self.assert_tensor(fresh, batch[:1])
        self.assertEqual(fresh_cache[0].k.shape[2], 3)

    def test_forward_does_not_mutate_ids_weights_or_previous_cache(self):
        ids = torch.tensor([[1, 4, 2]])
        original_ids = ids.clone()
        original_weights = {name: weight.clone() for name, weight in self.weights.items()}
        _, cache = learner.forward(ids[:, :2], self.weights, self.config)
        original_cache = [(kv.k.clone(), kv.v.clone()) for kv in cache]
        _, fresh = learner.forward(ids[:, 2:], self.weights, self.config, cache)
        self.assertTrue(torch.equal(ids, original_ids))
        for name, weight in self.weights.items():
            self.assertTrue(torch.equal(weight, original_weights[name]), name)
        self.assertEqual(cache[0].k.shape[2], 2)
        self.assertEqual(fresh[0].k.shape[2], 3)
        for kv, (k, v) in zip(cache, original_cache):
            self.assertTrue(torch.equal(kv.k, k))
            self.assertTrue(torch.equal(kv.v, v))

    def test_total_position_capacity_is_checked_before_modifying_cache(self):
        config = replace(self.config, max_position_embeddings=4)
        learner.forward(torch.tensor([[1, 2, 3, 4]]), self.weights, config)
        with self.assertRaises(ValueError):
            learner.forward(torch.tensor([[1, 2, 3, 4, 5]]), self.weights, config)
        _, cache = learner.forward(torch.tensor([[1, 2, 3]]), self.weights, config)
        old = [(kv.k.clone(), kv.v.clone()) for kv in cache]
        with self.assertRaises(ValueError):
            learner.forward(torch.tensor([[4, 5]]), self.weights, config, cache)
        for kv, (k, v) in zip(cache, old):
            self.assertTrue(torch.equal(kv.k, k))
            self.assertTrue(torch.equal(kv.v, v))


class GenerationTests(TensorAssertions):
    def setUp(self):
        require('greedy_generate')
        self.config, self.weights = fixture()

    def test_prefill_once_then_feed_only_new_ids_and_reuse_returned_cache(self):
        fake, calls, caches = generation_stub([7, 2, 9])
        prompt = torch.tensor([[1, 4, 6]])
        with patch.object(learner, 'forward', fake):
            result = learner.greedy_generate(prompt, self.weights, self.config,
                                            max_new_tokens=3, eos_token_ids=(12,))
        self.assertTrue(torch.equal(result, torch.tensor([[7, 2, 9]])))
        self.assertEqual(len(calls), 3)
        for (actual, _), expected in zip(calls, (prompt, torch.tensor([[7]]), torch.tensor([[2]]))):
            self.assertTrue(torch.equal(actual, expected))
        self.assertIsNone(calls[0][1])
        self.assertIs(calls[1][1], caches[0])
        self.assertIs(calls[2][1], caches[1])
        self.assertTrue(torch.equal(prompt, torch.tensor([[1, 4, 6]])))

    def test_any_eos_is_returned_and_stops_before_another_forward(self):
        for sequence, expected in (([2, 9, 4], [2, 9]), ([3, 7], [3])):
            with self.subTest(expected=expected):
                fake, calls, _ = generation_stub(sequence)
                with patch.object(learner, 'forward', fake):
                    result = learner.greedy_generate(torch.tensor([[1, 4]]), self.weights, self.config,
                                                    max_new_tokens=5, eos_token_ids=(3, 9))
                self.assertTrue(torch.equal(result, torch.tensor([expected])))
                self.assertEqual(len(calls), len(expected))

    def test_zero_budget_has_empty_generated_only_result_and_no_forward(self):
        with patch.object(learner, 'forward', side_effect=AssertionError('零预算不应调用 forward')):
            result = learner.greedy_generate(torch.tensor([[1, 4]]), self.weights, self.config,
                                            max_new_tokens=0, eos_token_ids=(3, 9))
        self.assertEqual(result.shape, (1, 0))
        self.assertEqual(result.dtype, torch.long)
        self.assertEqual(result.device.type, 'cpu')

    def test_conservative_capacity_rejects_prompt_plus_budget_before_forward(self):
        config = replace(self.config, max_position_embeddings=4)
        with patch.object(learner, 'forward', side_effect=AssertionError('容量检查应先发生')):
            with self.assertRaises(ValueError):
                learner.greedy_generate(torch.tensor([[1, 2, 3]]), self.weights, config,
                                        max_new_tokens=2, eos_token_ids=(9,))
        fake, calls, _ = generation_stub([2])
        with patch.object(learner, 'forward', fake):
            result = learner.greedy_generate(torch.tensor([[1, 2, 3]]), self.weights, config,
                                            max_new_tokens=1, eos_token_ids=(9,))
        self.assertTrue(torch.equal(result, torch.tensor([[2]])))
        self.assertEqual(len(calls), 1)

    def test_negative_budget_and_overlong_prompt_are_rejected_even_without_generation(self):
        config = replace(self.config, max_position_embeddings=4)
        for prompt, budget in (([[1, 2]], -1), ([[1, 2, 3, 4, 5]], 0)):
            with self.subTest(prompt=prompt, budget=budget):
                with patch.object(learner, 'forward', side_effect=AssertionError('非法长度不能进入 forward')):
                    with self.assertRaises(ValueError):
                        learner.greedy_generate(torch.tensor(prompt), self.weights, config,
                                                max_new_tokens=budget, eos_token_ids=(9,))

    def test_real_generation_matches_uncached_full_prefix_recomputation(self):
        require('forward')
        prompt = torch.tensor([[1, 4, 2]])
        prefix = prompt.clone()
        for _ in range(3):
            logits, _ = learner.forward(prefix, self.weights, self.config, None)
            predicted = logits[:, -1].argmax(-1, keepdim=True)
            prefix = torch.cat((prefix, predicted), 1)
        actual = learner.greedy_generate(prompt, self.weights, self.config,
                                         max_new_tokens=3, eos_token_ids=())
        self.assertTrue(torch.equal(actual, prefix[:, prompt.shape[1]:]))


if __name__ == '__main__':
    unittest.main()
