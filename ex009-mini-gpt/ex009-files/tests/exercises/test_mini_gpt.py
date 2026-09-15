"""ex009 的行为测试：教师提供，覆盖输入管线、模型组装与两种归一化排列。

约定：CPU、float64、随机种子固定；浮点比较使用显式容差。
精确索引与"不变"类断言要求严格相等，数值路径不同的比较用 allclose。
"""

import unittest

import torch
from torch import nn

from exercises.ex008_transformer_block.block import LayerNorm, TransformerBlock
from exercises.ex009_mini_gpt.model import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    VOCAB,
    MiniGPT,
    build_two_styles,
    text_to_ids,
)

DTYPE = torch.float64
ATOL = 1e-10


def make_model(norm_style="pre", n_layer=2, seed=0, p=0.0, max_positions=8):
    torch.manual_seed(seed)
    return MiniGPT(
        vocab_size=len(VOCAB),
        C=4,
        num_heads=2,
        ffn_hidden=8,
        n_layer=n_layer,
        max_positions=max_positions,
        norm_style=norm_style,
        p=p,
        dtype=DTYPE,
    )


class TestTextToIds(unittest.TestCase):
    def test_basic_sequence(self):
        self.assertEqual(text_to_ids("A X U"), [BOS_ID, 3, 5, 6, EOS_ID])

    def test_returns_plain_list_of_int(self):
        out = text_to_ids("B V")
        self.assertIsInstance(out, list)
        self.assertTrue(all(isinstance(v, int) for v in out))
        self.assertEqual(out, [BOS_ID, 4, 7, EOS_ID])

    def test_empty_text_keeps_both_markers(self):
        self.assertEqual(text_to_ids(""), [BOS_ID, EOS_ID])
        self.assertEqual(text_to_ids("   "), [BOS_ID, EOS_ID])

    def test_extra_whitespace_is_ignored(self):
        self.assertEqual(text_to_ids("  A   X  "), [BOS_ID, 3, 5, EOS_ID])

    def test_repeated_tokens_are_not_deduplicated(self):
        self.assertEqual(text_to_ids("A A A"), [BOS_ID, 3, 3, 3, EOS_ID])

    def test_unknown_token_raises_keyerror(self):
        with self.assertRaises(KeyError):
            text_to_ids("A ZZZ")

    def test_does_not_mutate_vocab(self):
        before = dict(VOCAB)
        try:
            text_to_ids("A QQQ")
        except KeyError:
            pass
        self.assertEqual(VOCAB, before)


class TestForwardShapeAndContract(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.ids = torch.tensor([[BOS_ID, 3, 5], [BOS_ID, 4, 7]])
        self.valid = torch.ones(2, 3, dtype=torch.bool)

    def test_output_shape_and_dtype(self):
        out = self.model(self.ids, self.valid)
        self.assertEqual(out.shape, (2, 3, len(VOCAB)))
        self.assertEqual(out.dtype, DTYPE)

    def test_single_position_is_allowed(self):
        out = self.model(torch.tensor([[BOS_ID]]), torch.ones(1, 1, dtype=torch.bool))
        self.assertEqual(out.shape, (1, 1, len(VOCAB)))

    def test_too_long_input_raises(self):
        long_ids = torch.full((1, self.model.L + 1), 3, dtype=torch.long)
        long_valid = torch.ones(1, self.model.L + 1, dtype=torch.bool)
        with self.assertRaises(ValueError):
            self.model(long_ids, long_valid)

    def test_max_length_is_accepted(self):
        """恰好等于 L 必须正常返回，不能把边界一并拒绝。"""
        ids = torch.full((1, self.model.L), 3, dtype=torch.long)
        valid = torch.ones(1, self.model.L, dtype=torch.bool)
        out = self.model(ids, valid)
        self.assertEqual(out.shape, (1, self.model.L, len(VOCAB)))

    def test_inputs_are_not_modified(self):
        ids_copy = self.ids.clone()
        valid_copy = self.valid.clone()
        self.model(self.ids, self.valid)
        self.assertTrue(torch.equal(self.ids, ids_copy))
        self.assertTrue(torch.equal(self.valid, valid_copy))

    def test_output_keeps_autograd_path(self):
        out = self.model(self.ids, self.valid)
        self.assertTrue(out.requires_grad)
        out.sum().backward()
        self.assertIsNotNone(self.model.token_table.grad)
        self.assertIsNotNone(self.model.vocab_proj.grad)

    def test_logits_are_raw_scores_not_probabilities(self):
        out = self.model(self.ids, self.valid)
        sums = out.exp().sum(dim=-1)
        self.assertFalse(torch.allclose(sums, torch.ones_like(sums), atol=1e-3))


class TestPositionTableIsUsed(unittest.TestCase):
    def test_same_token_at_different_positions_differs(self):
        model = make_model()
        ids = torch.tensor([[3, 3]])
        valid = torch.ones(1, 2, dtype=torch.bool)
        out = model(ids, valid)
        self.assertFalse(torch.allclose(out[0, 0], out[0, 1], atol=ATOL))

    def test_changing_position_table_changes_output(self):
        """只扰动某一行位置向量，对应位置的输出应改变。

        注意不能给整张表加同一个常数：Pre-LN 块的入口 LayerNorm 会精确
        消去逐 token 的共同偏移，那样的改动不会传到输出。
        """
        model = make_model()
        ids = torch.tensor([[BOS_ID, 3, 5]])
        valid = torch.ones(1, 3, dtype=torch.bool)
        before = model(ids, valid).detach().clone()
        with torch.no_grad():
            model.position_table[1, 0] += 5.0
        after = model(ids, valid)
        self.assertFalse(torch.allclose(before[:, 1], after[:, 1], atol=ATOL))

    def test_position_table_applied_once_not_per_layer(self):
        """两层与一层模型共享同一份块参数时，位置表不应被累加两次。"""
        one = make_model(n_layer=1, seed=3)
        ids = torch.tensor([[BOS_ID, 3, 5]])
        valid = torch.ones(1, 3, dtype=torch.bool)
        from exercises.ex003_position_embedding.position_embedding import (
            embed_with_positions,
        )

        expected_first = embed_with_positions(
            ids, one.token_table, one.position_table
        )
        got = one.blocks[0](expected_first, valid)
        if one.final_norm is not None:
            got = one.final_norm(got)
        got = got @ one.vocab_proj
        self.assertTrue(torch.allclose(one(ids, valid), got, atol=ATOL))


class TestCausality(unittest.TestCase):
    def test_future_token_does_not_change_earlier_logits(self):
        model = make_model(n_layer=2, seed=1)
        valid = torch.ones(1, 4, dtype=torch.bool)
        a = torch.tensor([[BOS_ID, 3, 5, 6]])
        b = torch.tensor([[BOS_ID, 3, 5, 7]])
        out_a = model(a, valid)
        out_b = model(b, valid)
        self.assertTrue(torch.allclose(out_a[:, :3], out_b[:, :3], atol=ATOL))
        self.assertFalse(torch.allclose(out_a[:, 3], out_b[:, 3], atol=ATOL))

    def test_prefix_consistency(self):
        """喂前 3 个 token 与喂 4 个 token，位置 0..2 的 logits 应一致。"""
        model = make_model(n_layer=2, seed=2)
        short = torch.tensor([[BOS_ID, 3, 5]])
        long = torch.tensor([[BOS_ID, 3, 5, 6]])
        out_s = model(short, torch.ones(1, 3, dtype=torch.bool))
        out_l = model(long, torch.ones(1, 4, dtype=torch.bool))
        self.assertTrue(torch.allclose(out_s, out_l[:, :3], atol=ATOL))

    def test_pad_key_does_not_leak_into_valid_positions(self):
        model = make_model(n_layer=2, seed=4)
        ids = torch.tensor([[BOS_ID, 3, PAD_ID, PAD_ID]])
        valid = torch.tensor([[True, True, False, False]])
        base = model(ids, valid)
        other = ids.clone()
        other[0, 2] = 7
        other[0, 3] = 4
        changed = model(other, valid)
        self.assertTrue(torch.allclose(base[:, :2], changed[:, :2], atol=ATOL))


class TestParameterRegistration(unittest.TestCase):
    def test_all_blocks_are_registered(self):
        model = make_model(n_layer=3)
        names = [n for n, _ in model.named_parameters()]
        for i in range(3):
            self.assertTrue(
                any(n.startswith(f"blocks.{i}.") for n in names),
                f"blocks.{i} 的参数没有出现在 named_parameters 中",
            )

    def test_blocks_have_independent_parameters(self):
        model = make_model(n_layer=3)
        self.assertIsNot(model.blocks[0], model.blocks[1])
        self.assertIsNot(
            model.blocks[0].norm1.gamma, model.blocks[1].norm1.gamma
        )
        unique = {id(p) for p in model.parameters()}
        self.assertEqual(len(unique), len(list(model.parameters())))

    def test_state_dict_covers_every_parameter(self):
        model = make_model(n_layer=2)
        self.assertEqual(len(model.state_dict()), len(list(model.named_parameters())))

    def test_every_parameter_receives_gradient(self):
        model = make_model(n_layer=2)
        ids = torch.tensor([[BOS_ID, 3, 5]])
        valid = torch.ones(1, 3, dtype=torch.bool)
        model(ids, valid).pow(2).sum().backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} 没有收到梯度")


class TestNormStyles(unittest.TestCase):
    def test_pre_has_final_norm_and_post_does_not(self):
        pre, post = build_two_styles(len(VOCAB), 4, 2, 8, 2, 8, seed=0, dtype=DTYPE)
        self.assertIsInstance(pre.final_norm, LayerNorm)
        self.assertIsNone(post.final_norm)

    def test_two_models_are_distinct_objects(self):
        pre, post = build_two_styles(len(VOCAB), 4, 2, 8, 2, 8, seed=0, dtype=DTYPE)
        self.assertIsNot(pre, post)
        self.assertIsNot(pre.blocks[0], post.blocks[0])
        self.assertIsNot(pre.token_table, post.token_table)

    def test_shared_parameters_are_numerically_aligned(self):
        pre, post = build_two_styles(len(VOCAB), 4, 2, 8, 2, 8, seed=0, dtype=DTYPE)
        post_params = dict(post.named_parameters())
        for name, param in pre.named_parameters():
            if name.startswith("final_norm"):
                continue
            self.assertIn(name, post_params, f"{name} 在 post 模型中缺失")
            self.assertTrue(
                torch.equal(param, post_params[name]),
                f"{name} 在两个模型间数值不同",
            )

    def test_pre_output_is_normalized_but_post_path_differs(self):
        """pre 模型末尾过 LN，故进入词表投影前的表示逐 token 均值近似为零。"""
        pre, post = build_two_styles(len(VOCAB), 4, 2, 8, 3, 8, seed=5, dtype=DTYPE)
        ids = torch.tensor([[BOS_ID, 3, 5, 6]])
        valid = torch.ones(1, 4, dtype=torch.bool)
        out_pre = pre(ids, valid)
        out_post = post(ids, valid)
        self.assertEqual(out_pre.shape, out_post.shape)
        self.assertFalse(torch.allclose(out_pre, out_post, atol=1e-6))

    def test_post_model_skips_final_norm(self):
        """post 模型的输出应等于最后一块输出直接乘 vocab_proj。"""
        _, post = build_two_styles(len(VOCAB), 4, 2, 8, 2, 8, seed=7, dtype=DTYPE)
        from exercises.ex003_position_embedding.position_embedding import (
            embed_with_positions,
        )

        ids = torch.tensor([[BOS_ID, 3, 5]])
        valid = torch.ones(1, 3, dtype=torch.bool)
        x = embed_with_positions(ids, post.token_table, post.position_table)
        for block in post.blocks:
            x = block(x, valid)
        expected = x @ post.vocab_proj
        self.assertTrue(torch.allclose(post(ids, valid), expected, atol=ATOL))


class TestTrainingIntegration(unittest.TestCase):
    def test_loss_decreases_and_block_params_change(self):
        from exercises.ex005_training_loop.loss import masked_cross_entropy
        from exercises.ex005_training_loop.training import prepare_next_token_batch

        model = make_model(n_layer=2, seed=11)
        ids = torch.tensor([[BOS_ID, 3, 5, 6, EOS_ID]])
        valid = torch.ones(1, 5, dtype=torch.bool)
        input_ids, targets, target_valid = prepare_next_token_batch(ids, valid)
        input_valid = valid[:, :-1]

        before = model.blocks[0].norm1.gamma.detach().clone()
        opt = torch.optim.SGD(model.parameters(), lr=0.3)
        first = None
        for _ in range(40):
            logits = model(input_ids, input_valid)
            loss = masked_cross_entropy(logits, targets, target_valid)
            if first is None:
                first = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), first)
        self.assertFalse(
            torch.equal(model.blocks[0].norm1.gamma, before),
            "块内参数没有被更新，检查是否全部注册",
        )


if __name__ == "__main__":
    unittest.main()
