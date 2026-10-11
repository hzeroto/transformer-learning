"""ViT 教师测试；参照只用于验算，不可从待实现代码导入。

独立卷积检查图块投影；集成检查直接经过学习者 forward。
CPU、固定随机种子，float64 容差 1e-9/1e-11，float32 为 1e-5/1e-6。
"""

import unittest

import torch
from torch.nn import functional as F

from exercises.ex020_vit import model as learner


def setUpModule():
    global _previous_threads
    _previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_previous_threads)


def probe_patch():
    return learner.patch_embed(
        torch.ones(1, 1, 2, 2, dtype=torch.float64),
        torch.ones(4, 3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64), 2,
    )


def probe_model():
    return learner.MiniViT()(torch.ones(1, 3, 4, 6, dtype=torch.float64))


def require_implementation(probe):
    try:
        probe()
    except NotImplementedError as error:
        raise unittest.SkipTest(str(error)) from None


def reference_patch(images, weight, bias, patch_size):
    channels, dim = images.shape[1], weight.shape[1]
    kernel = weight.T.reshape(dim, channels, patch_size, patch_size)
    features = F.conv2d(images, kernel, bias, stride=patch_size)
    return features.flatten(2).transpose(1, 2)


def reference_tokens(model, images):
    patches = reference_patch(images, model.patch_weight, model.patch_bias, model.patch_size)
    return torch.cat((model.cls_token.expand(images.shape[0], -1, -1), patches), dim=1) + model.position


def reference_logits(model, images):
    # 核心待测入口不复用；只共用已提供并回归验证的 Encoder 计算。
    tokens = reference_tokens(model, images)
    for block in model.blocks:
        tokens = block(tokens)
    cls = F.layer_norm(
        tokens[:, 0], (model.dim,), model.final_norm.gamma,
        model.final_norm.beta, model.final_norm.eps,
    )
    return cls @ model.class_weight + model.class_bias


def swap_corner_patches(images):
    changed = images.clone()
    changed[:, :, :2, :2] = images[:, :, -2:, -2:]
    changed[:, :, -2:, -2:] = images[:, :, :2, :2]
    return changed


class ImplementationStatusTest(unittest.TestCase):
    def test_both_todos_are_implemented(self):
        pending = []
        for probe in (probe_patch, probe_model):
            try:
                probe()
            except NotImplementedError as error:
                pending.append(str(error))
        self.assertFalse(pending, "尚未完成，不是通过：\n" + "\n".join(pending))


class TensorAssertions(unittest.TestCase):
    def assert_tensor(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        tolerance = dict(rtol=1e-5, atol=1e-6) if expected.dtype == torch.float32 else dict(rtol=1e-9, atol=1e-11)
        torch.testing.assert_close(actual, expected, **tolerance)


class PatchEmbeddingTests(TensorAssertions):
    def setUp(self):
        require_implementation(probe_patch)
        torch.manual_seed(2026)

    def test_exact_patch_and_channel_order(self):
        images = torch.arange(2 * 3 * 4 * 6, dtype=torch.float64).reshape(2, 3, 4, 6)
        actual = learner.patch_embed(images, torch.eye(12, dtype=images.dtype), torch.zeros(12, dtype=images.dtype), 2)
        # 用独立的像素坐标枚举，不用 unfold/permute 来生成期望。
        expected = torch.tensor([
            [[images[b, c, 2 * row + r, 2 * col + s].item()
              for c in range(3) for r in range(2) for s in range(2)]
             for row in range(2) for col in range(3)]
            for b in range(2)
        ], dtype=images.dtype)
        self.assertTrue(torch.equal(actual, expected), "图块顺序或块内通道/行/列顺序错误")

    def test_matches_convolution_across_shapes_and_layouts(self):
        for dtype in (torch.float32, torch.float64):
            for batch, channels, height, width, p, dim in ((2, 3, 4, 6, 2, 5), (1, 2, 3, 5, 1, 4), (2, 3, 4, 4, 4, 7)):
                for sliced in (False, True):
                    with self.subTest(dtype=dtype, shape=(batch, channels, height, width), p=p, sliced=sliced):
                        images = torch.randn(batch, channels, height, width * 2 if sliced else width, dtype=dtype)
                        if sliced:
                            images = images[..., ::2]
                            self.assertFalse(images.is_contiguous())
                        weight = torch.randn(channels * p * p, dim, dtype=dtype)
                        bias = torch.randn(dim, dtype=dtype)
                        self.assert_tensor(learner.patch_embed(images, weight, bias, p), reference_patch(images, weight, bias, p))

    def test_single_pixel_changes_only_its_patch(self):
        images = torch.zeros(2, 3, 4, 6, dtype=torch.float64)
        weight = torch.arange(12 * 5, dtype=images.dtype).reshape(12, 5) / 10
        bias = torch.randn(5, dtype=images.dtype)
        before = learner.patch_embed(images, weight, bias, 2)
        changed = images.clone()
        changed[1, 2, 3, 4] = 7
        delta = learner.patch_embed(changed, weight, bias, 2) - before
        expected = torch.zeros_like(before)
        expected[1, 5] = 7 * weight[10]
        self.assert_tensor(delta, expected)

    def test_gradients_match_independent_convolution(self):
        values = [torch.randn(2, 3, 4, 6, dtype=torch.float64), torch.randn(12, 5, dtype=torch.float64), torch.randn(5, dtype=torch.float64)]
        actual_inputs = [v.clone().requires_grad_() for v in values]
        ref_inputs = [v.clone().requires_grad_() for v in values]
        actual = learner.patch_embed(*actual_inputs, 2)
        expected = reference_patch(*ref_inputs, 2)
        self.assert_tensor(actual, expected)
        upstream = torch.randn_like(actual)
        grads = torch.autograd.grad(actual, actual_inputs, upstream, allow_unused=True)
        ref_grads = torch.autograd.grad(expected, ref_inputs, upstream)
        for name, actual_grad, expected_grad in zip(("images", "weight", "bias"), grads, ref_grads):
            with self.subTest(parameter=name):
                self.assertIsNotNone(actual_grad, f"{name} 的梯度路径断开")
                self.assert_tensor(actual_grad, expected_grad)

    def test_does_not_mutate_values_or_existing_gradients(self):
        inputs = [torch.randn(2, 3, 4, 6, dtype=torch.float64, requires_grad=True), torch.randn(12, 5, dtype=torch.float64, requires_grad=True), torch.randn(5, dtype=torch.float64, requires_grad=True)]
        before = [v.detach().clone() for v in inputs]
        for value in inputs:
            value.grad = torch.full_like(value, 0.25)
        learner.patch_embed(*inputs, 2)
        for value, original in zip(inputs, before):
            self.assertTrue(torch.equal(value, original))
            self.assertTrue(torch.equal(value.grad, torch.full_like(value, 0.25)))


class ViTConnectionTests(TensorAssertions):
    def setUp(self):
        require_implementation(probe_model)
        torch.manual_seed(2031)

    def test_first_block_receives_cls_patches_and_positions_once(self):
        model = learner.MiniViT()
        images = torch.randn(2, 3, 4, 6, dtype=torch.float64)
        seen = []
        handle = model.blocks[0].register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
        try:
            model(images)
        finally:
            handle.remove()
        self.assertEqual(len(seen), 1)
        self.assert_tensor(seen[0], reference_tokens(model, images))

    def test_class_logits_match_reference_with_all_layers(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                model = learner.MiniViT(dtype=dtype)
                images = torch.randn(2, 3, 4, 12, dtype=dtype)[..., ::2]
                actual = model(images)
                self.assertEqual(tuple(actual.shape), (2, 3), "返回的必须是每张图的类别分数")
                self.assert_tensor(actual, reference_logits(model, images))

    def test_all_parameter_and_image_gradients_and_no_state_mutation(self):
        model = learner.MiniViT()
        images = torch.randn(2, 3, 4, 6, dtype=torch.float64, requires_grad=True)
        ref_images = images.detach().clone().requires_grad_()
        params = dict(model.named_parameters())
        values = (images, *params.values())
        before = [v.detach().clone() for v in values]
        for value in values:
            value.grad = torch.full_like(value, 0.125)
        actual = model(images)
        expected = reference_logits(model, ref_images)
        self.assert_tensor(actual, expected)
        upstream = torch.randn_like(actual)
        grads = torch.autograd.grad(actual, values, upstream, allow_unused=True)
        ref_grads = torch.autograd.grad(expected, (ref_images, *params.values()), upstream)
        for name, value, original, grad, ref_grad in zip(("images", *params), values, before, grads, ref_grads):
            with self.subTest(parameter=name):
                self.assertIsNotNone(grad, f"{name} 的计算图断开")
                self.assert_tensor(grad, ref_grad)
                self.assertTrue(torch.equal(value, original))
                self.assertTrue(torch.equal(value.grad, torch.full_like(value, 0.125)))
        self.assertGreater(grads[0].abs().max().item(), 1e-9, "分类输出必须能依赖图片")
        self.assertTrue(model.training)

    def test_cls_classification_depends_on_image_content(self):
        model = learner.MiniViT()
        images = torch.randn(2, 3, 4, 6, dtype=torch.float64)
        changed = images.clone()
        changed[:, :, -2:, -2:] += torch.tensor([2., -1., 3.], dtype=images.dtype).reshape(1, 3, 1, 1)
        self.assertGreater((model(changed) - model(images)).abs().max().item(), 1e-8)

    def test_patch_permutation_with_and_without_position(self):
        model = learner.MiniViT()
        images = torch.randn(2, 3, 4, 6, dtype=torch.float64)
        swapped = swap_corner_patches(images)
        with torch.no_grad():
            model.position.copy_(torch.randn_like(model.position) * 0.3)
        # 仅这个固定夹具应当有差异，不声称有位置编码必然使任意输入不同。
        self.assertGreater((model(images) - model(swapped)).abs().max().item(), 1e-8)
        with torch.no_grad():
            model.position.zero_()
        self.assert_tensor(model(images), model(swapped))

    def test_small_classification_batch_can_train(self):
        model = learner.MiniViT(depth=1)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        images = torch.eye(3, dtype=torch.float64)[labels, :, None, None].expand(-1, -1, 4, 6).clone()
        images += torch.randn_like(images) * 0.03
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        initial = F.cross_entropy(model(images), labels).item()
        for _ in range(45):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
        logits = model(images)
        self.assertLess(F.cross_entropy(logits, labels).item(), initial * 0.2)
        self.assertTrue(torch.equal(logits.argmax(-1), labels))


if __name__ == "__main__":
    unittest.main()
