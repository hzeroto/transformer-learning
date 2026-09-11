"""教师测试：参照仅用于验算，不可从学习者实现导入。

用官方 LayerNorm 验算归一化，用独立逐头参照验算 Attention。
Dropout 检查分布、数值、梯度和模式，不要求复现某个库的采样序列。
"""

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from torch.nn import functional as F
    from exercises.ex008_transformer_block import block as learner
    from tests.exercises.test_multi_head_attention import self_reference


def make_x(shape=(2, 5, 4), dtype=None, layout="contiguous", grad=False):
    dtype = torch.float64 if dtype is None else dtype
    b, t, c = shape
    if layout == "sliced":
        x = torch.randn(b, t, 2 * c + 1, dtype=dtype)[..., 1::2]
    elif layout == "transposed":
        x = torch.randn(b, c, t, dtype=dtype).transpose(-2, -1)
    else:
        x = torch.randn(shape, dtype=dtype)
    return x.detach().requires_grad_(grad)


def valid_for(x):
    valid = torch.ones(x.shape[:2], dtype=torch.bool)
    if x.shape[1] > 2:
        valid[-1, -2:] = False
    return valid


def varied_parameters(module):
    # 不只检查默认 gamma=1 / beta=0 / bias=0 的特例。
    with torch.no_grad():
        for name, p in module.named_parameters():
            p.copy_(torch.randn_like(p) * 0.3)
            if name.endswith("gamma"):
                p.add_(1.0)
    return module


def norm_reference(x, params, eps, prefix=""):
    return F.layer_norm(x, (x.shape[-1],), params[prefix + "gamma"], params[prefix + "beta"], eps)


def ffn_reference(x, params, prefix=""):
    hidden = F.relu(x @ params[prefix + "W1"] + params[prefix + "b1"])
    return hidden @ params[prefix + "W2"] + params[prefix + "b2"]


def block_reference(x, params, valid, heads, eps):
    # 此参照只用于 p=0 或 eval。Attention 逐 head 独立索引，不调用学习者 MHA。
    normalized = norm_reference(x, params, eps, "norm1.")
    a, _ = self_reference(normalized, params["Wq"], params["Wk"], params["Wv"], params["Wo"], valid, heads)
    u = x + a
    return u + ffn_reference(norm_reference(u, params, eps, "norm2."), params, "ffn.")


def probe(cls):
    x = torch.ones(1, 2, 4, dtype=torch.float64)
    if cls == "LayerNorm":
        learner.LayerNorm(4)(x)
    elif cls == "FeedForward":
        learner.FeedForward(4, 7)(x)
    elif cls == "Dropout":
        learner.Dropout(0.5)(x)
    else:
        learner.TransformerBlock(4, 2, 7)(x, valid_for(x))


def require_implementation(name):
    try:
        probe(name)
    except NotImplementedError as error:
        raise unittest.SkipTest(str(error)) from None


class ImplementationStatusTest(unittest.TestCase):
    def test_four_forward_methods_are_implemented(self):
        if torch is None:
            self.fail("请使用仓库中已安装 PyTorch 的 Python 环境。")
        pending = []
        for name in ("LayerNorm", "FeedForward", "Dropout", "TransformerBlock"):
            try:
                probe(name)
            except NotImplementedError as error:
                pending.append(f"{name}: {error}")
        self.assertFalse(pending, "尚未完成（不是测试通过）：\n" + "\n".join(pending))


class TensorAssertions(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(108)

    def assert_tensor(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        tolerance = dict(rtol=1e-5, atol=1e-6) if expected.dtype == torch.float32 else dict(rtol=1e-9, atol=1e-11)
        torch.testing.assert_close(actual, expected, **tolerance)

    def assert_forward_and_grad(self, module, x, reference, *args):
        params = dict(module.named_parameters())
        before = {name: p.detach().clone() for name, p in params.items()}
        old_grads = {}
        for name, p in params.items():
            p.grad = torch.full_like(p, 0.125)
            old_grads[name] = p.grad.clone()
        x.grad = torch.full_like(x, 0.25)
        x_before, x_grad_before = x.detach().clone(), x.grad.clone()
        ref_x = x.detach().clone().requires_grad_()
        ref_params = {name: p.detach().clone().requires_grad_() for name, p in params.items()}
        actual = module(x, *args)
        expected = reference(ref_x, ref_params)
        self.assert_tensor(actual, expected)
        self.assertTrue(torch.equal(x, x_before), "forward 原地修改了输入")
        self.assertTrue(torch.equal(x.grad, x_grad_before), "forward 修改了已有输入梯度")
        for name, p in params.items():
            self.assertTrue(torch.equal(p, before[name]), f"forward 修改了参数 {name}")
            self.assertTrue(torch.equal(p.grad, old_grads[name]), f"forward 修改了已有梯度 {name}")
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (x, *params.values()), upstream, allow_unused=True)
        expected_grads = torch.autograd.grad(expected, (ref_x, *ref_params.values()), upstream)
        for name, a, e in zip(("X", *params.keys()), actual_grads, expected_grads):
            with self.subTest(gradient=name):
                self.assertIsNotNone(a, f"{name} 的计算图断开了")
                self.assert_tensor(a, e)


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class ProvidedStructureTest(TensorAssertions):
    def test_registered_parameter_shapes_and_independence(self):
        model = learner.TransformerBlock(4, 2, 8, p=0.2)
        expected = {name: (4, 4) for name in ("Wq", "Wk", "Wv", "Wo")}
        expected.update({name: (4,) for name in ("norm1.gamma", "norm1.beta", "norm2.gamma", "norm2.beta")})
        expected.update({"ffn.W1": (4, 8), "ffn.b1": (8,), "ffn.W2": (8, 4), "ffn.b2": (4,)})
        self.assertEqual({n: tuple(p.shape) for n, p in model.named_parameters()}, expected)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 156)
        self.assertEqual(len({id(p) for p in model.parameters()}), 12)
        self.assertEqual(list(model.drop1.parameters()), [])
        model.eval()
        self.assertTrue(all(not child.training for child in model.modules()))
        model.train()
        self.assertTrue(all(child.training for child in model.modules()))

    def test_scope_configuration_errors(self):
        for p in (-0.1, 1.0, 1.1):
            with self.subTest(p=p), self.assertRaises(ValueError):
                learner.Dropout(p)
        for c, h, f in ((4, 3, 8), (0, 1, 8), (4, 0, 8), (4, 2, 0)):
            with self.subTest(C=c, H=h, F=f), self.assertRaises(ValueError):
                learner.TransformerBlock(c, h, f)
        with self.assertRaises(ValueError):
            learner.LayerNorm(4, eps=0)


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class LayerNormTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        require_implementation("LayerNorm")

    def test_forward_and_all_gradients_across_layouts_and_dtypes(self):
        for dtype in (torch.float32, torch.float64):
            for layout in ("contiguous", "sliced", "transposed"):
                with self.subTest(dtype=dtype, layout=layout):
                    m = varied_parameters(learner.LayerNorm(4, dtype=dtype))
                    x = make_x(dtype=dtype, layout=layout, grad=True)
                    self.assert_forward_and_grad(m, x, lambda a, p: norm_reference(a, p, m.eps))

    def test_population_variance_and_epsilon_inside_sqrt(self):
        m = varied_parameters(learner.LayerNorm(4, eps=0.2))
        for values in ([1., 3., 5., 7.], [1., 1.001, 0.999, 1.002]):
            x = torch.tensor([[values]], dtype=torch.float64, requires_grad=True)
            self.assert_forward_and_grad(m, x, lambda a, p: norm_reference(a, p, m.eps))

    def test_constant_vectors_and_one_feature_are_finite_with_gradients(self):
        for c in (1, 4):
            with self.subTest(C=c):
                m = varied_parameters(learner.LayerNorm(c))
                x = torch.full((2, 3, c), 7.0, dtype=torch.float64, requires_grad=True)
                self.assert_tensor(m(x), m.beta.expand_as(x))
                self.assert_forward_and_grad(m, x, lambda a, p: norm_reference(a, p, m.eps))

    def test_other_tokens_and_batches_do_not_change_this_token(self):
        m = varied_parameters(learner.LayerNorm(4))
        x = make_x()
        original = m(x)[0, 0].clone()
        changed = torch.randn_like(x) * 30
        changed[0, 0] = x[0, 0]
        self.assert_tensor(m(changed)[0, 0], original)


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class FeedForwardTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        require_implementation("FeedForward")

    def test_forward_and_all_gradients_across_layouts_and_dtypes(self):
        for dtype in (torch.float32, torch.float64):
            for layout in ("contiguous", "sliced", "transposed"):
                with self.subTest(dtype=dtype, layout=layout):
                    m = varied_parameters(learner.FeedForward(3, 5, dtype=dtype))
                    x = make_x((2, 4, 3), dtype, layout, grad=True)
                    self.assert_forward_and_grad(m, x, ffn_reference)

    def test_relu_negative_zero_and_positive_values(self):
        m = learner.FeedForward(2, 2)
        with torch.no_grad():
            m.W1.copy_(torch.eye(2, dtype=torch.float64))
            m.W2.copy_(torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64))
        x = torch.tensor([[[-2., 0.], [1., 3.]]], dtype=torch.float64, requires_grad=True)
        self.assert_forward_and_grad(m, x, ffn_reference)

    def test_positionwise_not_sequence_mixing(self):
        m = varied_parameters(learner.FeedForward(4, 7))
        x = make_x()
        before = m(x)[0, 2].clone()
        changed = torch.randn_like(x) * 10
        changed[0, 2] = x[0, 2]
        self.assert_tensor(m(changed)[0, 2], before)


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class DropoutTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        require_implementation("Dropout")

    def test_eval_and_zero_probability_are_identity_without_random_draws(self):
        for training, p in ((False, 0.6), (True, 0.0), (False, 0.0)):
            with self.subTest(training=training, p=p):
                m = learner.Dropout(p).train(training)
                x = make_x(grad=True, layout="sliced")
                rng_before = torch.get_rng_state().clone()
                y = m(x)
                self.assert_tensor(y, x)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng_before), "恒等分支不应抽随机数")
                upstream = torch.randn_like(y)
                (grad,) = torch.autograd.grad(y, x, upstream)
                self.assert_tensor(grad, upstream)

    def test_elementwise_sampling_scale_and_expected_mean(self):
        for dtype in (torch.float32, torch.float64):
            for p in (0.25, 0.7):
                with self.subTest(dtype=dtype, p=p):
                    # 每个输入都非零，才能由输出 0 辨认本次丢弃位置。
                    x = torch.full((5000, 3, 4), 2.0, dtype=dtype)
                    x[:, 1, :] = -2.0
                    m = learner.Dropout(p).train()
                    before = x.clone()
                    y = m(x)
                    kept = y != 0
                    self.assert_tensor(y, torch.where(kept, x / (1 - p), torch.zeros_like(x)))
                    self.assertTrue(torch.equal(x, before))
                    # 同时检查每个 (t,c) 切片，避免共享整条特征/整批 mask 蒙混过关。
                    fractions = kept.to(torch.float64).mean(dim=0)
                    self.assertTrue(bool(((fractions - (1 - p)).abs() < 0.035).all()))
                    self.assertAlmostEqual(float(kept.to(torch.float64).mean()), 1 - p, delta=0.015)
                    self.assertLess(abs(float(y.mean() - x.mean())), 0.06)
                    for axis, left, right in (
                        ("C", kept[:, 0, 0], kept[:, 0, 1]),
                        ("T", kept[:, 0, 0], kept[:, 1, 0]),
                        ("B", kept[:-1, 0, 0], kept[1:, 0, 0]),
                    ):
                        agreement = (left == right).to(torch.float64).mean()
                        self.assertAlmostEqual(float(agreement), p * p + (1 - p) ** 2, delta=0.04, msg=f"检查是否沿 {axis} 轴共享了 mask")

    def test_backward_uses_forward_mask_and_preserves_input(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m = learner.Dropout(0.4).train()
                x = (make_x((3, 5, 4), dtype, "transposed").abs() + 1).detach().requires_grad_()
                before = x.clone()
                y = m(x)
                kept = y != 0
                upstream = torch.randn_like(y)
                torch.rand(113)  # 之后的抽样不能改变已经建立的反向路径。
                (grad,) = torch.autograd.grad(y, x, upstream)
                self.assert_tensor(grad, upstream * kept / (1 - m.p))
                self.assertTrue(torch.equal(x, before))

    def test_no_grad_does_not_disable_training_dropout_and_seed_is_external(self):
        m = learner.Dropout(0.5).train()
        x = torch.ones(20, 3, 4, dtype=torch.float64)
        torch.manual_seed(711)
        with torch.no_grad():
            first = m(x)
        self.assertTrue(bool((first == 0).any()) and bool((first == 2).any()))
        torch.manual_seed(711)
        self.assert_tensor(m(x), first)
        rng_before = torch.get_rng_state().clone()
        second = m(x)
        self.assertFalse(torch.equal(torch.get_rng_state(), rng_before), "训练前向必须重新采样")
        # 此固定种子和足够多元素下结果确实不同；不是宣称任意两次抽样必然不同。
        self.assertFalse(torch.equal(first, second), "检查是否在 forward 重设种子或复用 mask")


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class BlockTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        require_implementation("TransformerBlock")

    def test_forward_matches_independent_reference(self):
        for dtype in (torch.float32, torch.float64):
            for b, t, c, h, f in ((2, 5, 4, 2, 7), (3, 4, 6, 3, 5), (1, 1, 1, 1, 3)):
                for layout in ("contiguous", "transposed", "sliced"):
                    with self.subTest(dtype=dtype, B=b, T=t, C=c, H=h, F=f, layout=layout):
                        m = varied_parameters(learner.TransformerBlock(c, h, f, p=0.4, dtype=dtype)).eval()
                        x = make_x((b, t, c), dtype, layout)
                        valid = valid_for(x)
                        self.assert_tensor(m(x, valid), block_reference(x, dict(m.named_parameters()), valid, h, m.norm1.eps))

    def test_input_and_all_parameter_gradients_and_no_forward_mutation(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                m = varied_parameters(learner.TransformerBlock(4, 2, 7, p=0.0, dtype=dtype)).train()
                x = make_x(dtype=dtype, layout="sliced", grad=True)
                valid = valid_for(x)
                before = valid.clone()
                self.assert_forward_and_grad(m, x, lambda a, p: block_reference(a, p, valid, 2, m.norm1.eps), valid)
                self.assertTrue(torch.equal(valid, before))

    def test_future_change_does_not_change_prefix(self):
        m = varied_parameters(learner.TransformerBlock(4, 2, 7)).eval()
        x = make_x()
        valid = torch.ones(2, 5, dtype=torch.bool)
        original = m(x, valid)
        changed = x.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 9
        self.assert_tensor(m(changed, valid)[:, :3], original[:, :3])
        self.assert_tensor(m(x[:, :3], valid[:, :3]), original[:, :3])

    def test_pad_key_change_does_not_change_valid_outputs(self):
        m = varied_parameters(learner.TransformerBlock(4, 2, 7)).eval()
        x = make_x()
        # 有空洞的有效性，检查后面的真实 query 不能读取中间无效 key。
        valid = torch.tensor([[True, False, True, True, False], [True, True, False, True, True]])
        original = m(x, valid)
        changed = x.clone()
        changed[~valid] = torch.randn_like(changed[~valid]) * 30
        self.assert_tensor(m(changed, valid)[valid], original[valid])

    def test_zero_updates_keep_raw_residual_and_its_gradient_including_pad_queries(self):
        m = learner.TransformerBlock(4, 2, 7, p=0.6).train()
        with torch.no_grad():
            m.Wo.zero_()
            m.ffn.W2.zero_()
            m.ffn.b2.zero_()
        x = make_x(grad=True)
        y = m(x, valid_for(x))
        self.assert_tensor(y, x)
        (grad,) = torch.autograd.grad(y.sum(), x)
        self.assert_tensor(grad, torch.ones_like(x))

    def assert_unit_branch_dropout(self, attention_branch):
        m = learner.TransformerBlock(4, 2, 7, p=0.5).train()
        with torch.no_grad():
            m.Wq.zero_()
            m.Wk.zero_()
            m.norm1.gamma.zero_()
            m.norm1.beta.fill_(1.0)
            m.Wv.copy_(torch.eye(4, dtype=torch.float64))
            m.Wo.copy_(torch.eye(4, dtype=torch.float64) if attention_branch else torch.zeros_like(m.Wo))
            m.ffn.W2.zero_()
            m.ffn.b2.fill_(0.0 if attention_branch else 1.0)
        # 被选分支恒为 1，另一分支恒为 0；这样可以单独定位两处 Dropout。
        x = torch.full((20, 3, 4), 3.0, dtype=torch.float64, requires_grad=True)
        y = m(x, valid_for(x))
        delta = y - x
        self.assertTrue(bool(((delta == 0) | (delta == 2)).all()), "Dropout 位置或保留值缩放错误")
        self.assertTrue(bool((delta == 0).any()) and bool((delta == 2).any()), "此分支没有随机失活")
        (grad,) = torch.autograd.grad(y.sum(), x)
        self.assert_tensor(grad, torch.ones_like(x))

    def test_dropout_on_attention_update_not_residual(self):
        self.assert_unit_branch_dropout(attention_branch=True)

    def test_dropout_on_ffn_update_not_residual(self):
        self.assert_unit_branch_dropout(attention_branch=False)

    def test_fully_blocked_query_still_raises(self):
        m = learner.TransformerBlock(4, 2, 7)
        x = make_x()
        valid = valid_for(x)
        valid[0, 0] = False
        with self.assertRaises(ValueError):
            m(x, valid)

    def test_mode_autograd_and_persistent_objects(self):
        m = learner.TransformerBlock(4, 2, 7, p=0.4)
        params = {n: id(p) for n, p in m.named_parameters()}
        modules = {n: id(v) for n, v in m.named_modules()}
        x = make_x(grad=True)
        valid = valid_for(x)
        m.eval()
        state = torch.get_rng_state().clone()
        a = m(x, valid)
        self.assertTrue(a.requires_grad, "eval 不应关闭 autograd")
        with torch.no_grad():
            b = m(x, valid)
        self.assertFalse(b.requires_grad)
        self.assert_tensor(a, b)
        self.assertTrue(torch.equal(state, torch.get_rng_state()), "eval 不应重新创建随机参数或执行 Dropout 抽样")
        self.assertTrue(all(not child.training for child in m.modules()))
        m.train()
        with torch.no_grad():
            m(x, valid)
        self.assertFalse(torch.equal(state, torch.get_rng_state()), "no_grad 没有关闭训练模式下的 Dropout")
        self.assertTrue(all(child.training for child in m.modules()))
        self.assertEqual({n: id(p) for n, p in m.named_parameters()}, params)
        self.assertEqual({n: id(v) for n, v in m.named_modules()}, modules)

    def test_state_dict_roundtrip_and_two_independent_stacked_blocks(self):
        blocks = torch.nn.ModuleList([learner.TransformerBlock(4, 2, 7, p=0.2) for _ in range(2)]).eval()
        self.assertEqual(len(list(blocks.parameters())), 24)
        self.assertEqual(len({id(p) for p in blocks.parameters()}), 24)
        restored = torch.nn.ModuleList([learner.TransformerBlock(4, 2, 7, p=0.2) for _ in range(2)]).eval()
        restored.load_state_dict(blocks.state_dict())
        x = make_x(grad=True)
        valid = valid_for(x)
        y, expected, loaded = x, x, x
        for model, copy in zip(blocks, restored):
            y = model(y, valid)
            expected = block_reference(expected, dict(model.named_parameters()), valid, 2, model.norm1.eps)
            loaded = copy(loaded, valid)
        self.assert_tensor(y, expected)
        self.assert_tensor(loaded, y)
        y.square().mean().backward()
        for name, p in blocks.named_parameters():
            self.assertIsNotNone(p.grad, f"堆叠后参数 {name} 没有梯度")
            self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
        self.assertTrue(bool(torch.isfinite(x.grad).all()))


if __name__ == "__main__":
    unittest.main()
