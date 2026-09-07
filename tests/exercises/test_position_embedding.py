"""作业 003 的行为测试；不包含待实现函数的参考实现。"""

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from exercises.ex003_position_embedding.position_embedding import (
        embed_with_positions,
    )


class PositionEmbeddingImplementationStatusTest(unittest.TestCase):
    def test_required_function_is_implemented(self):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境运行测试。")
        try:
            embed_with_positions(
                torch.tensor([[0]], dtype=torch.long),
                torch.tensor([[1.0]]),
                torch.tensor([[10.0]]),
            )
        except NotImplementedError as error:
            self.fail(str(error))


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class PositionEmbeddingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            embed_with_positions(
                torch.tensor([[0]], dtype=torch.long),
                torch.tensor([[1.0]]),
                torch.tensor([[10.0]]),
            )
        except NotImplementedError:
            raise unittest.SkipTest("先完成 position_embedding.py 中的函数")

    def setUp(self):
        self.token_table = torch.tensor(
            [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32
        )
        self.position_table = torch.tensor(
            [[10, 100], [20, 200], [30, 300], [40, 400]],
            dtype=torch.float32,
        )

    def assert_output(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(actual, expected)

    def test_required_example_and_shared_positions(self):
        ids = torch.tensor([[2, 0, 2], [1, 3, 0]], dtype=torch.long)
        actual = embed_with_positions(ids, self.token_table, self.position_table)
        expected = torch.tensor(
            [
                [[15, 106], [21, 202], [35, 306]],
                [[13, 104], [27, 208], [31, 302]],
            ],
            dtype=torch.float32,
        )
        self.assert_output(actual, expected)

    def test_repeated_token_uses_current_position(self):
        ids = torch.tensor([[2, 2, 2]], dtype=torch.long)
        actual = embed_with_positions(ids, self.token_table, self.position_table)
        expected = torch.tensor(
            [[[15, 106], [25, 206], [35, 306]]], dtype=torch.float32
        )
        self.assert_output(actual, expected)

    def test_different_shape_and_float64(self):
        ids = torch.tensor([[0], [1], [0]], dtype=torch.long)
        token_table = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float64
        )
        position_table = torch.tensor(
            [[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.float64
        )
        actual = embed_with_positions(ids, token_table, position_table)
        expected = torch.tensor(
            [[[11, 22, 33, 44]], [[15, 26, 37, 48]], [[11, 22, 33, 44]]],
            dtype=torch.float64,
        )
        self.assert_output(actual, expected)

    def test_sequence_can_use_entire_position_table(self):
        ids = torch.tensor([[3, 2, 1, 0]], dtype=torch.long)
        actual = embed_with_positions(ids, self.token_table, self.position_table)
        expected = torch.tensor(
            [[[17, 108], [25, 206], [33, 304], [41, 402]]],
            dtype=torch.float32,
        )
        self.assert_output(actual, expected)

    def test_rejects_sequence_longer_than_position_table(self):
        ids = torch.tensor([[0, 1, 2, 3, 0]], dtype=torch.long)
        with self.assertRaises(ValueError):
            embed_with_positions(ids, self.token_table, self.position_table)

    def test_token_renumbering_does_not_change_position_assignment(self):
        ids = torch.tensor([[2, 0, 2], [1, 3, 0]], dtype=torch.long)
        original = embed_with_positions(ids, self.token_table, self.position_table)
        # 只交换 token 的编号与内容表行；位置表不能跟着交换。
        remap = torch.tensor([2, 1, 0, 3], dtype=torch.long)
        remapped = embed_with_positions(
            remap[ids], self.token_table[remap], self.position_table
        )
        self.assert_output(remapped, original)

    def test_does_not_modify_any_input(self):
        ids = torch.tensor([[2, 0, 2], [1, 3, 0]], dtype=torch.long)
        original_ids = ids.clone()
        original_tokens = self.token_table.clone()
        original_positions = self.position_table.clone()
        embed_with_positions(ids, self.token_table, self.position_table)
        self.assertTrue(torch.equal(ids, original_ids))
        self.assertTrue(torch.equal(self.token_table, original_tokens))
        self.assertTrue(torch.equal(self.position_table, original_positions))


if __name__ == "__main__":
    unittest.main()
