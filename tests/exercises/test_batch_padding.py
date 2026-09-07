"""作业 002 的行为测试；不包含 make_batch 的参考实现。"""

import copy
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from exercises.ex002_batch_padding.batch_padding import make_batch


class BatchPaddingImplementationStatusTest(unittest.TestCase):
    def test_required_function_is_implemented(self):
        if torch is None:
            self.fail(
                "当前 Python 环境未安装 PyTorch。请先安装依赖，"
                "详见 exercises/ex002_batch_padding/README.md。"
            )
        try:
            make_batch([[4]], pad_id=0)
        except NotImplementedError as error:
            self.fail(str(error))


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class BatchPaddingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            make_batch([[4]], pad_id=0)
        except NotImplementedError:
            raise unittest.SkipTest("先完成 batch_padding.py 中的 make_batch")

    def assert_batch(self, sequences, pad_id, expected_ids, expected_valid):
        ids, valid = make_batch(sequences, pad_id=pad_id)

        self.assertIsInstance(ids, torch.Tensor)
        self.assertIsInstance(valid, torch.Tensor)
        expected_shape = (len(sequences), max(len(seq) for seq in sequences))
        self.assertEqual(tuple(ids.shape), expected_shape)
        self.assertEqual(tuple(valid.shape), expected_shape)
        self.assertEqual(ids.dtype, torch.long)
        self.assertEqual(valid.dtype, torch.bool)
        self.assertTrue(
            torch.equal(ids, torch.tensor(expected_ids, dtype=torch.long)),
            f"补齐结果不正确：{ids}",
        )
        self.assertTrue(
            torch.equal(valid, torch.tensor(expected_valid, dtype=torch.bool)),
            f"有效位置标记不正确：{valid}",
        )
        self.assertEqual(valid.sum(dim=1).tolist(), [len(seq) for seq in sequences])

    def test_mixed_lengths_preserve_unknown_token(self):
        self.assert_batch(
            sequences=[[4, 1, 5], [7], [6, 8]],
            pad_id=0,
            expected_ids=[[4, 1, 5], [7, 0, 0], [6, 8, 0]],
            expected_valid=[
                [True, True, True],
                [True, False, False],
                [True, True, False],
            ],
        )

    def test_single_sequence_needs_no_padding(self):
        self.assert_batch(
            sequences=[[9, 1, 4]],
            pad_id=0,
            expected_ids=[[9, 1, 4]],
            expected_valid=[[True, True, True]],
        )

    def test_single_token_sequence(self):
        self.assert_batch(
            sequences=[[1]],
            pad_id=0,
            expected_ids=[[1]],
            expected_valid=[[True]],
        )

    def test_equal_lengths_add_no_extra_column(self):
        self.assert_batch(
            sequences=[[4, 1], [6, 7]],
            pad_id=0,
            expected_ids=[[4, 1], [6, 7]],
            expected_valid=[[True, True], [True, True]],
        )

    def test_longest_sequence_is_last(self):
        self.assert_batch(
            sequences=[[4], [5, 6], [7, 8, 1]],
            pad_id=0,
            expected_ids=[[4, 0, 0], [5, 6, 0], [7, 8, 1]],
            expected_valid=[
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ],
        )

    def test_nonzero_pad_id_keeps_zero_and_one_valid(self):
        self.assert_batch(
            sequences=[[0, 1, 5], [7]],
            pad_id=9,
            expected_ids=[[0, 1, 5], [7, 9, 9]],
            expected_valid=[[True, True, True], [True, False, False]],
        )

    def test_does_not_modify_input(self):
        sequences = [[4, 1, 5], [7], [6, 8]]
        before = copy.deepcopy(sequences)
        make_batch(sequences, pad_id=0)
        self.assertEqual(sequences, before)


if __name__ == "__main__":
    unittest.main()
