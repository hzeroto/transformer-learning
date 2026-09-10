"""作业 007 第一部分：分别核对拆合头、不同存储布局和梯度。"""

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from exercises.ex007_multi_head_attention import heads as learner


def split_reference(x, h):
    # 按每个 token 的特征区间取各头，用 stack 新建 head 轴，避免重复待测变形。
    dh = x.shape[-1] // h
    return torch.stack([x[:, :, head * dh:(head + 1) * dh] for head in range(h)], dim=1)


def merge_reference(heads):
    # 每次取一个 head 的全部 token，再沿已有特征轴拼接。
    return torch.cat([heads[:, head] for head in range(heads.shape[1])], dim=-1)


def split_input(shape, layout, dtype, requires_grad=False):
    b, t, c = shape
    raw_shape = {
        "contiguous": (b, t, c),
        "transposed": (b, c, t),
        "sliced": (b, t, c * 2 + 3),
    }[layout]
    generator = torch.Generator().manual_seed(71)
    base = torch.randn(raw_shape, dtype=dtype, generator=generator, requires_grad=requires_grad)
    if layout == "transposed":
        x = base.transpose(1, 2)
    elif layout == "sliced":
        x = base[:, :, 1:1 + c * 2:2]
    else:
        x = base
    return base, x


def merge_input(shape, layout, dtype, requires_grad=False):
    b, h, t, dh = shape
    raw_shape = {
        "contiguous": (b, h, t, dh),
        "transposed": (b, t, h, dh),
        "sliced": (b, h, t, dh * 2 + 3),
    }[layout]
    generator = torch.Generator().manual_seed(73)
    base = torch.randn(raw_shape, dtype=dtype, generator=generator, requires_grad=requires_grad)
    if layout == "transposed":
        heads = base.transpose(1, 2)
    elif layout == "sliced":
        heads = base[:, :, :, 1:1 + dh * 2:2]
    else:
        heads = base
    return base, heads


class ImplementationStatusTest(unittest.TestCase):
    def test_split_is_implemented(self):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境。")
        try:
            learner.split_heads(torch.ones(1, 3, 4), 2)
        except NotImplementedError as error:
            self.fail(str(error))

    def test_merge_is_implemented(self):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境。")
        try:
            learner.merge_heads(torch.ones(1, 2, 3, 2))
        except NotImplementedError as error:
            self.fail(str(error))


class LayoutAssertions(unittest.TestCase):
    def assert_tensor(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        # 纯布局操作不改变数值，因此这里要求精确相等，不涉及浮点加法误差。
        self.assertTrue(torch.equal(actual, expected), "shape 相同，但元素对应关系不正确")


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class SplitHeadsTest(LayoutAssertions):
    @classmethod
    def setUpClass(cls):
        try:
            learner.split_heads(torch.ones(1, 3, 4), 2)
        except NotImplementedError:
            raise unittest.SkipTest("先完成 split_heads")

    def test_numbered_elements_have_exact_head_token_mapping(self):
        x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        actual = learner.split_heads(x, 2)
        self.assert_tensor(actual, split_reference(x, 2))
        self.assert_tensor(actual[0, 0], torch.tensor([[0., 1.], [4., 5.], [8., 9.]], dtype=x.dtype))
        self.assertEqual(actual[0, 0, 1, 0].item(), x[0, 1, 0].item())

    def test_shapes_dtypes_layouts_and_no_input_mutation(self):
        cases = ((2, 3, 8, 2), (1, 1, 1, 1), (2, 1, 6, 3), (1, 4, 4, 4), (2, 5, 6, 1))
        for dtype in (torch.float32, torch.float64):
            for b, t, c, h in cases:
                for layout in ("contiguous", "transposed", "sliced"):
                    with self.subTest(dtype=dtype, shape=(b, t, c), h=h, layout=layout):
                        base, x = split_input((b, t, c), layout, dtype)
                        before = base.clone()
                        actual = learner.split_heads(x, h)
                        self.assert_tensor(actual, split_reference(x, h))
                        self.assert_tensor(base, before)

    def test_one_changed_feature_stays_in_its_token_and_head(self):
        _, x = split_input((2, 3, 8), "contiguous", torch.float64)
        baseline = learner.split_heads(x, 2)
        changed_x = x.clone()
        changed_x[1, 2, 5] += 16
        expected = baseline.clone()
        expected[1, 1, 2, 1] += 16  # Dh=4；特征 5 属于 head 1 的分量 1。
        self.assert_tensor(learner.split_heads(changed_x, 2), expected)

    def test_nonpositive_or_nondividing_head_counts_raise_value_error(self):
        _, x = split_input((2, 3, 8), "sliced", torch.float64)
        before = x.clone()
        for h in (0, -1, -4, 3, 5, 9):
            with self.subTest(h=h):
                with self.assertRaises(ValueError):
                    learner.split_heads(x, h)
                self.assert_tensor(x, before)


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class MergeHeadsTest(LayoutAssertions):
    @classmethod
    def setUpClass(cls):
        try:
            learner.merge_heads(torch.ones(1, 2, 3, 2))
        except NotImplementedError:
            raise unittest.SkipTest("先完成 merge_heads")

    def test_independent_head_output_is_merged_by_token(self):
        # 输入独立构造且连续，不依赖 split_heads，防止错误往返互相抵消。
        heads = torch.arange(12, dtype=torch.float64).reshape(1, 2, 3, 2)
        expected = torch.tensor([[[0., 1., 6., 7.], [2., 3., 8., 9.], [4., 5., 10., 11.]]], dtype=heads.dtype)
        self.assert_tensor(learner.merge_heads(heads), expected)

    def test_shapes_dtypes_layouts_and_no_input_mutation(self):
        shapes = ((2, 2, 3, 4), (1, 1, 1, 1), (2, 3, 1, 2), (1, 4, 5, 1), (2, 1, 3, 6))
        for dtype in (torch.float32, torch.float64):
            for shape in shapes:
                for layout in ("contiguous", "transposed", "sliced"):
                    with self.subTest(dtype=dtype, shape=shape, layout=layout):
                        base, heads = merge_input(shape, layout, dtype)
                        before = base.clone()
                        self.assert_tensor(learner.merge_heads(heads), merge_reference(heads))
                        self.assert_tensor(base, before)


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class RoundTripAndGradientTest(LayoutAssertions):
    @classmethod
    def setUpClass(cls):
        try:
            learner.split_heads(torch.ones(1, 3, 4), 2)
            learner.merge_heads(torch.ones(1, 2, 3, 2))
        except NotImplementedError:
            raise unittest.SkipTest("先完成两个布局函数")

    def test_round_trips_in_both_directions(self):
        for layout in ("contiguous", "transposed", "sliced"):
            _, x = split_input((2, 3, 8), layout, torch.float64)
            self.assert_tensor(learner.merge_heads(learner.split_heads(x, 2)), x)
            _, heads = merge_input((2, 2, 3, 4), layout, torch.float64)
            self.assert_tensor(learner.split_heads(learner.merge_heads(heads), 2), heads)

    def check_gradient(self, operation, reference, factory, shape, layout):
        base, value = factory(shape, layout, torch.float64, requires_grad=True)
        ref_base, ref_value = factory(shape, layout, torch.float64, requires_grad=True)
        before = base.detach().clone()
        base.grad = torch.full_like(base, 0.125)
        ref_base.grad = torch.full_like(ref_base, 0.125)
        actual = operation(value)
        expected = reference(ref_value)
        self.assert_tensor(actual, expected)
        self.assertTrue(actual.requires_grad, "不能用 detach/no_grad 切断布局转换的梯度")
        self.assert_tensor(base.grad, torch.full_like(base, 0.125))
        generator = torch.Generator().manual_seed(79)
        upstream = torch.randn(actual.shape, dtype=actual.dtype, generator=generator)
        (actual * upstream).sum().backward()
        (expected * upstream).sum().backward()
        self.assertIsNotNone(base.grad)
        self.assertIsNotNone(ref_base.grad)
        torch.testing.assert_close(base.grad, ref_base.grad, rtol=1e-10, atol=1e-12)
        self.assert_tensor(base.detach(), before)

    def test_split_gradients_reach_underlying_input(self):
        for layout in ("contiguous", "transposed", "sliced"):
            with self.subTest(layout=layout):
                self.check_gradient(lambda x: learner.split_heads(x, 2), lambda x: split_reference(x, 2),
                                    split_input, (2, 3, 8), layout)

    def test_merge_gradients_reach_underlying_input(self):
        for layout in ("contiguous", "transposed", "sliced"):
            with self.subTest(layout=layout):
                self.check_gradient(learner.merge_heads, merge_reference,
                                    merge_input, (2, 2, 3, 4), layout)


if __name__ == "__main__":
    unittest.main()
