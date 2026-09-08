"""教师提供的生成行为测试；固定转移表便于隔离生成循环的错误。"""

import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from exercises.ex005_training_loop import generation as exercise
    from exercises.ex005_training_loop.training import forward_logits


def transition_parameters(dtype=None):
    if dtype is None:
        dtype = torch.float64
    # eye(5) 是单位矩阵。因此 E[id] @ W 恰好读取 W 的对应行。
    # 每行只给指定后继高分：0->0、1->2、2->3、3->4、4->1。
    # 这些是教师构造的可预期参数，并不冒充训练结果。
    E = torch.eye(5, dtype=dtype).requires_grad_(True)
    W = torch.full((5, 5), -3.0, dtype=dtype)
    for token, successor in enumerate((0, 2, 3, 4, 1)):
        W[token, successor] = 3.0
    return E, W.requires_grad_(True)


def probe():
    E, W = transition_parameters()
    return exercise.greedy_generate(torch.tensor([[1]]), E, W, 4, 1)


class GenerationImplementationStatusTest(unittest.TestCase):
    def test_function_is_implemented(self):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境。")
        try:
            probe()
        except NotImplementedError as error:
            self.fail(str(error))


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class GreedyGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            probe()
        except NotImplementedError as error:
            raise unittest.SkipTest(str(error))

    def assert_ids(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.dtype, torch.long)
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(tuple(actual.shape), (1, len(expected)))
        self.assertEqual(actual.tolist(), [expected])

    def test_feeds_predictions_back_and_keeps_eos(self):
        E, W = transition_parameters()
        result = exercise.greedy_generate(torch.tensor([[1]]), E, W, 4, 8)
        self.assert_ids(result, [1, 2, 3, 4])

    def test_uses_last_position_and_counts_only_new_tokens(self):
        E, W = transition_parameters()
        result = exercise.greedy_generate(torch.tensor([[1, 2]]), E, W, 4, 1)
        self.assert_ids(result, [1, 2, 3])

    def test_stops_at_budget_even_without_eos(self):
        E, W = transition_parameters()
        # EOS=0 永远不会被 1、2、3、4 的循环生成。
        result = exercise.greedy_generate(torch.tensor([[1]]), E, W, 0, 5)
        self.assert_ids(result, [1, 2, 3, 4, 1, 2])

    def test_zero_budget_and_finished_prefix_need_no_forward(self):
        E, W = transition_parameters()
        for tokens, budget in (([1, 2], 0), ([1, 2, 4], 5), ([4], 1)):
            with self.subTest(tokens=tokens, budget=budget):
                prefix = torch.tensor([tokens])
                with patch.object(exercise, "forward_logits", side_effect=AssertionError("无需再预测")):
                    result = exercise.greedy_generate(prefix, E, W, 4, budget)
                self.assert_ids(result, tokens)
                result[0, 0] = 0
                self.assertEqual(prefix.tolist(), [tokens], "返回值需要独立存储")

    def test_eos_id_is_configurable_and_zero_is_not_implicitly_eos(self):
        E, W = transition_parameters()
        result = exercise.greedy_generate(torch.tensor([[1]]), E, W, 3, 8)
        self.assert_ids(result, [1, 2, 3])
        result = exercise.greedy_generate(torch.tensor([[0]]), E, W, 4, 2)
        self.assert_ids(result, [0, 0, 0])

    def test_ties_choose_smallest_candidate_id(self):
        E, W = transition_parameters()
        with torch.no_grad():
            W[1] = -3.0
            W[1, 0] = 3.0
            W[1, 2] = 3.0
        result = exercise.greedy_generate(torch.tensor([[1]]), E, W, 4, 1)
        self.assert_ids(result, [1, 0])

    def test_preserves_parameters_prefix_gradients_and_grad_mode(self):
        for has_grad in (False, True):
            with self.subTest(existing_grad=has_grad):
                E, W = transition_parameters()
                prefix = torch.tensor([[1, 2]])
                if has_grad:
                    E.grad = torch.full_like(E, 12.0)
                    W.grad = torch.full_like(W, -7.0)
                values = (prefix.clone(), E.detach().clone(), W.detach().clone())
                grad_objects = (E.grad, W.grad)
                grad_values = tuple(None if g is None else g.clone() for g in grad_objects)
                modes = []

                def checked_forward(*args, **kwargs):
                    modes.append(torch.is_grad_enabled())
                    return forward_logits(*args, **kwargs)

                with torch.enable_grad():
                    with (
                        patch.object(exercise, "forward_logits", side_effect=checked_forward),
                        patch.object(torch.autograd, "backward", side_effect=AssertionError("生成不能反向")),
                    ):
                        result = exercise.greedy_generate(prefix, E, W, 4, 5)
                    self.assertTrue(torch.is_grad_enabled(), "不能永久关闭调用方求导")
                self.assert_ids(result, [1, 2, 3, 4])
                self.assertTrue(modes, "请复用已有 forward_logits")
                self.assertFalse(any(modes), "生成前向应在 no_grad 范围内")
                for actual, before in zip((prefix, E, W), values):
                    self.assertTrue(torch.equal(actual, before))
                for parameter, old_grad, old_value in zip((E, W), grad_objects, grad_values):
                    self.assertTrue(parameter.requires_grad)
                    self.assertIs(parameter.grad, old_grad, "生成不能清理或替换旧梯度")
                    if old_value is not None:
                        self.assertTrue(torch.equal(parameter.grad, old_value))

    def test_other_shapes_dtypes_and_parameters_without_grad(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                # N=3、C=2，不依赖测试中的 5x5 单位矩阵或固定词表。
                E = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=dtype)
                W = torch.tensor([[-3.0, 4.0, 0.0], [-3.0, 0.0, 4.0]], dtype=dtype)
                with torch.no_grad():
                    result = exercise.greedy_generate(torch.tensor([[0]]), E, W, 2, 6)
                    self.assertFalse(torch.is_grad_enabled())
                self.assert_ids(result, [0, 1, 2])
                self.assertFalse(E.requires_grad)
                self.assertFalse(W.requires_grad)


if __name__ == "__main__":
    unittest.main()
