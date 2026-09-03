import itertools
import math
import unittest

from exercises.ex001_tensor_shape_index.tensor_index import indices, offset

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _implementation_pending() -> bool:
    try:
        offset((1,), (0,))
        indices((1,), 0)
    except NotImplementedError:
        return True
    return False


IMPLEMENTATION_PENDING = _implementation_pending()


class ImplementationStatusTest(unittest.TestCase):
    def test_required_functions_are_implemented(self) -> None:
        try:
            offset((1,), (0,))
            indices((1,), 0)
        except NotImplementedError as error:
            self.fail(str(error))


@unittest.skipIf(IMPLEMENTATION_PENDING, "先完成 tensor_index.py 中的两个函数")
class TensorShapeIndexTest(unittest.TestCase):
    def test_required_offset_example(self) -> None:
        self.assertEqual(offset((2, 3, 4, 5), (1, 0, 2, 1)), 71)

    def test_required_indices_example(self) -> None:
        self.assertEqual(indices((2, 3, 4, 5), 71), (1, 0, 2, 1))

    def test_round_trip_every_index(self) -> None:
        shape = (2, 3, 4, 5)
        ranges = (range(axis_length) for axis_length in shape)
        for original_indices in itertools.product(*ranges):
            with self.subTest(indices=original_indices):
                flat_offset = offset(shape, original_indices)
                self.assertEqual(indices(shape, flat_offset), original_indices)

    def test_round_trip_every_offset(self) -> None:
        shape = (2, 3, 4, 5)
        for original_offset in range(math.prod(shape)):
            with self.subTest(offset=original_offset):
                multi_axis_indices = indices(shape, original_offset)
                self.assertEqual(offset(shape, multi_axis_indices), original_offset)

    def test_rejects_invalid_shapes(self) -> None:
        for invalid_shape in ((), (2, 0, 4), (2, -1, 4)):
            with self.subTest(shape=invalid_shape):
                with self.assertRaises(ValueError):
                    offset(invalid_shape, tuple(0 for _ in invalid_shape))
                with self.assertRaises(ValueError):
                    indices(invalid_shape, 0)

    def test_rejects_wrong_index_count(self) -> None:
        with self.assertRaises(ValueError):
            offset((2, 3, 4), (1, 2))
        with self.assertRaises(ValueError):
            offset((2, 3, 4), (1, 2, 3, 0))

    def test_rejects_out_of_bounds_indices(self) -> None:
        shape = (2, 3, 4)
        for invalid_indices in ((-1, 0, 0), (2, 0, 0), (0, 3, 0), (0, 0, 4)):
            with self.subTest(indices=invalid_indices):
                with self.assertRaises(IndexError):
                    offset(shape, invalid_indices)

    def test_rejects_out_of_bounds_offsets(self) -> None:
        shape = (2, 3, 4)
        for invalid_offset in (-1, math.prod(shape)):
            with self.subTest(offset=invalid_offset):
                with self.assertRaises(IndexError):
                    indices(shape, invalid_offset)

    @unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
    def test_matches_pytorch_flatten_order(self) -> None:
        shape = (2, 3, 4, 5)
        tensor = torch.arange(math.prod(shape)).reshape(shape)
        flattened = tensor.flatten()

        ranges = (range(axis_length) for axis_length in shape)
        for multi_axis_indices in itertools.product(*ranges):
            with self.subTest(indices=multi_axis_indices):
                flat_offset = offset(shape, multi_axis_indices)
                self.assertEqual(
                    tensor[multi_axis_indices].item(),
                    flattened[flat_offset].item(),
                )


if __name__ == "__main__":
    unittest.main()
