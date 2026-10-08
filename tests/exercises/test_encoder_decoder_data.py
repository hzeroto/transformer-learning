"""ex017 数据与位置编码教师测试；不提供可供作业导入的实现。

CPU，纯确定性夹具，无随机种子依赖。索引、mask 精确比较；浮点默认
float64，rtol=1e-10 / atol=1e-12；float32 为 1e-5 / 1e-6。
测试实际执行教师导读版的数据与位置实现，不跳过行为测试。
"""

import copy
import math
import unittest

import torch

from exercises.ex017_original_transformer import data, positions


class TensorAssertions(unittest.TestCase):
    def assert_tensor(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, torch.device("cpu"))
        if expected.dtype == torch.float32:
            tolerance = dict(rtol=1e-5, atol=1e-6)
        elif expected.dtype == torch.float64:
            tolerance = dict(rtol=1e-10, atol=1e-12)
        else:
            tolerance = dict(rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, **tolerance)


class TestTeacherForcingBatch(TensorAssertions):
    def test_source_and_target_are_padded_independently_and_shifted(self):
        batch = data.make_teacher_forcing_batch(
            [[3, 4, 5, 6, 7], [8, 9]], [[12, 13], [7, 8, 9]])
        expected = {
            "src_ids": torch.tensor([[3, 4, 5, 6, 7], [8, 9, 0, 0, 0]]),
            "src_valid": torch.tensor([[True] * 5, [True, True, False, False, False]]),
            "tgt_input_ids": torch.tensor([[1, 12, 13, 2], [1, 7, 8, 9]]),
            "tgt_input_valid": torch.tensor([[True] * 4, [True] * 4]),
            "labels": torch.tensor([[12, 13, 2, 0], [7, 8, 9, 2]]),
            "target_valid": torch.tensor([[True, True, True, False], [True] * 4]),
        }
        self.assertEqual(set(batch), set(expected))
        for name, value in expected.items():
            with self.subTest(field=name):
                self.assert_tensor(batch[name], value)
        self.assertNotEqual(batch["src_ids"].shape[1], batch["labels"].shape[1])

    def test_short_target_eos_input_is_valid_while_pad_label_is_not(self):
        batch = data.make_teacher_forcing_batch([[3], [4]], [[5], [6, 7, 8, 9]])
        self.assert_tensor(batch["tgt_input_ids"][0], torch.tensor([1, 5, 2, 0, 0]))
        self.assert_tensor(batch["labels"][0], torch.tensor([5, 2, 0, 0, 0]))
        self.assert_tensor(batch["tgt_input_valid"][0],
                           torch.tensor([True, True, True, False, False]))
        self.assert_tensor(batch["target_valid"][0],
                           torch.tensor([True, True, False, False, False]))
        self.assertTrue(batch["target_valid"][0, 1].item(), "EOS 标签也必须受监督")
        self.assertTrue(batch["tgt_input_valid"][0, 2].item(), "输入 EOS 是可读取的真实 token")
        self.assertFalse(batch["target_valid"][0, 2].item(), "EOS 后面的 PAD 标签不计分")

    def test_single_token_source_and_target_keep_bos_and_eos_boundary(self):
        batch = data.make_teacher_forcing_batch([[9]], [[10]])
        self.assert_tensor(batch["src_ids"], torch.tensor([[9]]))
        self.assert_tensor(batch["src_valid"], torch.tensor([[True]]))
        self.assert_tensor(batch["tgt_input_ids"], torch.tensor([[1, 10]]))
        self.assert_tensor(batch["labels"], torch.tensor([[10, 2]]))
        self.assert_tensor(batch["tgt_input_valid"], torch.tensor([[True, True]]))
        self.assert_tensor(batch["target_valid"], torch.tensor([[True, True]]))

    def test_inputs_are_not_modified_by_padding_or_adding_special_tokens(self):
        sources, targets = [[3, 4], [5]], [[7], [8, 9, 10]]
        before = copy.deepcopy((sources, targets))
        batch = data.make_teacher_forcing_batch(sources, targets)
        self.assertEqual((sources, targets), before)
        batch["src_ids"][0, 0] = 99
        batch["tgt_input_ids"][0, 0] = 99
        self.assertEqual((sources, targets), before)


class TestSinusoidalPositions(TensorAssertions):
    def test_width_four_has_frequencies_one_and_one_hundredth(self):
        actual = positions.sinusoidal_positions(torch.tensor([0, 1, 100]), 4)
        expected = torch.tensor([
            [0., 1., 0., 1.],
            [math.sin(1), math.cos(1), math.sin(.01), math.cos(.01)],
            [math.sin(100), math.cos(100), math.sin(1), math.cos(1)],
        ], dtype=torch.float64)
        self.assert_tensor(actual, expected)

    def test_odd_width_keeps_last_unpaired_sine(self):
        for width in (1, 3, 5):
            with self.subTest(width=width):
                actual = positions.sinusoidal_positions(torch.tensor([0, 7]), width)
                expected_rows = []
                for position in (0, 7):
                    row = []
                    for feature in range(width):
                        angle = position / (10000 ** (2 * (feature // 2) / width))
                        row.append(math.sin(angle) if feature % 2 == 0 else math.cos(angle))
                    expected_rows.append(row)
                self.assert_tensor(actual, torch.tensor(expected_rows, dtype=torch.float64))

    def test_actual_positions_and_noncontiguous_input_preserve_order_and_offsets(self):
        backing = torch.tensor([9, 77, 3, 77, 9, 77, 5, 77])
        selected = backing[::2]
        before = backing.clone()
        self.assertFalse(selected.is_contiguous())
        actual = positions.sinusoidal_positions(selected, 6)
        complete = positions.sinusoidal_positions(torch.arange(10), 6)
        self.assert_tensor(actual, complete[torch.tensor([9, 3, 9, 5])])
        self.assert_tensor(backing, before)
        self.assertGreater((actual[0] - actual[1]).abs().sum().item(), .1,
                           "不同实际位置必须产生不同的向量，不能只返回常数")

    def test_explicit_dtype_and_empty_positions(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                actual = positions.sinusoidal_positions(torch.tensor([0, 2]), 2, dtype=dtype)
                expected = torch.tensor([[0., 1.], [math.sin(2), math.cos(2)]], dtype=dtype)
                self.assert_tensor(actual, expected)
                empty = positions.sinusoidal_positions(torch.empty(0, dtype=torch.long), 5,
                                                       dtype=dtype)
                self.assert_tensor(empty, torch.empty(0, 5, dtype=dtype))

    def test_nonpositive_width_raises_value_error(self):
        for width in (0, -1, -8):
            with self.subTest(width=width):
                with self.assertRaises(ValueError):
                    positions.sinusoidal_positions(torch.tensor([0, 1]), width)


if __name__ == "__main__":
    unittest.main()
