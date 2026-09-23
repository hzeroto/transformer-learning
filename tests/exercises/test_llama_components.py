"""ex013 组件教师测试：独立数学参照只用于验收，不应从作业导入。

CPU float32/64；种子 1301/1307/1313；float64 rtol=1e-8、atol=1e-10，
float32 rtol=1e-5、atol=1e-6。性质测试之外，RoPE 必须通过逐对数值参照。
"""

import copy
import math
import unittest

import torch

from exercises.ex013_llama_style import components as learner


def setUpModule():
    global _old_threads
    _old_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_old_threads)


def random_input(shape, dtype=torch.float64, layout="contiguous", grad=False):
    generator = torch.Generator().manual_seed(1307)
    raw_shape = list(shape)
    if layout == "transpose":
        raw_shape[-1], raw_shape[-2] = raw_shape[-2], raw_shape[-1]
    elif layout == "slice":
        raw_shape[-1] = 2 * raw_shape[-1] + 1
    base = torch.randn(raw_shape, generator=generator, dtype=dtype).requires_grad_(grad)
    if layout == "transpose":
        value = base.transpose(-1, -2)
    elif layout == "slice":
        value = base[..., 1::2]
    else:
        value = base
    return base, value


def make_norm(C=4, eps=.03, dtype=torch.float64):
    norm = learner.RMSNorm(C, eps=eps, dtype=dtype)
    with torch.no_grad():
        norm.gamma.copy_(torch.linspace(.6, 1.4, C, dtype=dtype))
    return norm


def make_swiglu(C=4, hidden_dim=7, dtype=torch.float64):
    with torch.random.fork_rng():
        torch.manual_seed(1301)
        module = learner.SwiGLU(C, hidden_dim, dtype=dtype)
    generator = torch.Generator().manual_seed(1313)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=dtype) * .4)
    return module


def rms_reference(x, gamma, eps):
    """逐 batch、逐 token 计算，避免与学习者共用归一化轴 helper。"""
    batches = []
    for batch in x:
        tokens = []
        for token in batch:
            mean_square = sum(value * value for value in token) / token.numel()
            tokens.append(token * torch.rsqrt(mean_square + eps) * gamma)
        batches.append(torch.stack(tokens))
    return torch.stack(batches)


def swiglu_reference(x, module):
    gate_input = x @ module.Wgate
    content = x @ module.Wup
    return ((gate_input * torch.sigmoid(gate_input)) * content) @ module.Wdown


def rope_reference(x, positions, theta=10000.):
    """固定 b/h/t 后逐相邻特征对旋转；角度用 Python 标量计算。"""
    B, H, n, D = x.shape
    batches = []
    for b in range(B):
        heads = []
        for h in range(H):
            tokens = []
            for t in range(n):
                features = []
                for j in range(D // 2):
                    angle = positions[t].item() * theta ** (-2 * j / D)
                    cosine, sine = math.cos(angle), math.sin(angle)
                    even, odd = x[b, h, t, 2 * j], x[b, h, t, 2 * j + 1]
                    features.extend((even * cosine - odd * sine,
                                     even * sine + odd * cosine))
                tokens.append(torch.stack(features))
            heads.append(torch.stack(tokens))
        batches.append(torch.stack(heads))
    return torch.stack(batches)


class TensorAssertions(unittest.TestCase):
    def assert_tensor(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        tolerance = (dict(rtol=1e-5, atol=1e-6) if expected.dtype == torch.float32
                     else dict(rtol=1e-8, atol=1e-10))
        torch.testing.assert_close(actual, expected, **tolerance)


class TestImplementationStatus(unittest.TestCase):
    def test_rmsnorm_is_implemented(self):
        try:
            learner.RMSNorm(4)(torch.ones(1, 2, 4, dtype=torch.float64))
        except NotImplementedError as error:
            self.fail(str(error))

    def test_swiglu_is_implemented(self):
        try:
            learner.SwiGLU(4, 7)(torch.ones(1, 2, 4, dtype=torch.float64))
        except NotImplementedError as error:
            self.fail(str(error))

    def test_rope_is_implemented(self):
        try:
            learner.apply_rope(torch.ones(1, 2, 3, 4, dtype=torch.float64),
                               torch.arange(3, dtype=torch.long))
        except NotImplementedError as error:
            self.fail(str(error))


class TestRMSNorm(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        try:
            learner.RMSNorm(4)(torch.ones(1, 2, 4, dtype=torch.float64))
        except NotImplementedError:
            raise unittest.SkipTest("先完成 RMSNorm.forward")

    def test_parameters_numerics_layouts_and_no_mutation(self):
        for dtype in (torch.float32, torch.float64):
            fresh = learner.RMSNorm(4, eps=.03, dtype=dtype)
            self.assertEqual(set(dict(fresh.named_parameters())), {"gamma"})
            self.assertEqual(fresh.C, 4)
            self.assertEqual(fresh.eps, .03)
            self.assert_tensor(fresh.gamma, torch.ones(4, dtype=dtype))
            for layout in ("contiguous", "transpose", "slice"):
                with self.subTest(dtype=dtype, layout=layout):
                    norm = make_norm(dtype=dtype)
                    base, x = random_input((2, 3, 4), dtype, layout)
                    before, gamma = base.clone(), norm.gamma.detach().clone()
                    parameter_id = id(norm.gamma)
                    self.assert_tensor(norm(x), rms_reference(x, norm.gamma, norm.eps))
                    self.assertTrue(torch.equal(base, before))
                    self.assertTrue(torch.equal(norm.gamma, gamma))
                    self.assertEqual(id(norm.gamma), parameter_id)

    def test_zero_constant_single_feature_and_epsilon_inside_sqrt(self):
        for dtype in (torch.float32, torch.float64):
            for C in (1, 4):
                with self.subTest(dtype=dtype, C=C):
                    norm = learner.RMSNorm(C, eps=.25, dtype=dtype)
                    x = torch.tensor([0., 2., -2.], dtype=dtype).reshape(1, 3, 1).expand(1, 3, C)
                    denominator = torch.tensor([.5, math.sqrt(4.25), math.sqrt(4.25)], dtype=dtype)
                    expected = x / denominator.reshape(1, 3, 1)
                    self.assert_tensor(norm(x), expected)
                    self.assertTrue(torch.isfinite(norm(x)).all().item())
                    # 小数输入让 sqrt(mean(x²))+eps 与 sqrt(mean(x²)+eps) 明显分离。
                    tiny = torch.full((1, 1, C), .001, dtype=dtype)
                    self.assert_tensor(norm(tiny), tiny / math.sqrt(.000001 + .25))

    def test_does_not_remove_mean_or_mix_tokens(self):
        norm = learner.RMSNorm(4, eps=.01)
        x = torch.tensor([[[1., 3., 5., 7.], [101., 103., 105., 107.]]], dtype=torch.float64)
        result = norm(x)
        self.assert_tensor(result, rms_reference(x, norm.gamma, norm.eps))
        self.assertGreater(result[0, 0].mean().item(), .5)
        self.assertGreater((result[0, 0] - result[0, 1]).abs().max().item(), .3)
        changed = x.clone()
        changed[:, 1] *= -30.
        self.assert_tensor(norm(changed)[:, 0], result[:, 0])

    def test_input_and_gamma_gradients_with_noncontiguous_input(self):
        norm = make_norm()
        ref_norm = copy.deepcopy(norm)
        base, x = random_input((2, 3, 4), layout="transpose", grad=True)
        ref_base, ref_x = random_input((2, 3, 4), layout="transpose", grad=True)
        values, ref_values = [base, norm.gamma], [ref_base, ref_norm.gamma]
        for value in (*values, *ref_values):
            value.grad = torch.full_like(value, .125)
        output = norm(x)
        expected = rms_reference(ref_x, ref_norm.gamma, ref_norm.eps)
        self.assert_tensor(output, expected)
        seed = torch.linspace(-.8, 1.3, output.numel(), dtype=output.dtype).reshape_as(output)
        for value in values:
            self.assert_tensor(value.grad, torch.full_like(value, .125))
        (output * seed).sum().backward()
        (expected * seed).sum().backward()
        for name, value, ref_value in zip(("input", "gamma"), values, ref_values):
            with self.subTest(name=name):
                self.assertIsNotNone(value.grad)
                self.assert_tensor(value.grad, ref_value.grad)


class TestSwiGLU(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        try:
            learner.SwiGLU(4, 7)(torch.ones(1, 2, 4, dtype=torch.float64))
        except NotImplementedError:
            raise unittest.SkipTest("先完成 SwiGLU.forward")

    def test_parameter_contract_numerics_layouts_and_no_mutation(self):
        for dtype in (torch.float32, torch.float64):
            for layout in ("contiguous", "transpose", "slice"):
                with self.subTest(dtype=dtype, layout=layout):
                    module = make_swiglu(dtype=dtype)
                    parameters = dict(module.named_parameters())
                    self.assertEqual(set(parameters), {"Wgate", "Wup", "Wdown"})
                    self.assertEqual(module.C, 4)
                    self.assertEqual(module.hidden_dim, 7)
                    self.assertEqual(module.Wgate.shape, (4, 7))
                    self.assertEqual(module.Wup.shape, (4, 7))
                    self.assertEqual(module.Wdown.shape, (7, 4))
                    self.assertEqual(sum(p.numel() for p in parameters.values()), 3 * 4 * 7)
                    self.assertIsNot(module.Wgate, module.Wup)
                    base, x = random_input((2, 3, 4), dtype, layout)
                    saved = [value.detach().clone() for value in (base, *parameters.values())]
                    ids = {name: id(value) for name, value in parameters.items()}
                    self.assert_tensor(module(x), swiglu_reference(x, module))
                    for value, before in zip((base, *parameters.values()), saved):
                        self.assertTrue(torch.equal(value, before))
                    self.assertEqual(ids, {name: id(value) for name, value in module.named_parameters()})

    def test_independent_content_branch_and_signed_unbounded_gate(self):
        module = make_swiglu(C=2, hidden_dim=2)
        with torch.no_grad():
            module.Wgate.copy_(torch.eye(2, dtype=torch.float64))
            module.Wup.copy_(torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64))
            module.Wdown.copy_(torch.eye(2, dtype=torch.float64))
        x = torch.tensor([[[-2., 2.]]], dtype=torch.float64)
        expected = torch.tensor([[[-4. / (1. + math.exp(2.)),
                                   4. / (1. + math.exp(-2.))]]], dtype=x.dtype)
        self.assert_tensor(module(x), expected)
        self.assertLess(module(x)[0, 0, 0].item(), 0.)
        self.assertGreater(module(x)[0, 0, 1].item(), 2.)
        with torch.no_grad():
            module.Wup.mul_(3.)
        self.assert_tensor(module(x), 3 * expected)
        with torch.no_grad():
            module.Wgate.neg_()
        self.assert_tensor(module(x), swiglu_reference(x, module))
        self.assertGreater(module(x)[0, 0, 0].item(), 0.)
        with torch.no_grad():
            module.Wup.zero_()
        self.assert_tensor(module(x), torch.zeros_like(x))

    def test_input_and_all_three_parameter_gradients(self):
        module = make_swiglu()
        ref_module = copy.deepcopy(module)
        base, x = random_input((2, 3, 4), layout="slice", grad=True)
        ref_base, ref_x = random_input((2, 3, 4), layout="slice", grad=True)
        values, refs = [base, *module.parameters()], [ref_base, *ref_module.parameters()]
        for value in (*values, *refs):
            value.grad = torch.full_like(value, .125)
        result, expected = module(x), swiglu_reference(ref_x, ref_module)
        self.assert_tensor(result, expected)
        seed = torch.linspace(-1.1, .9, result.numel(), dtype=result.dtype).reshape_as(result)
        for value in values:
            self.assert_tensor(value.grad, torch.full_like(value, .125))
        (result * seed).sum().backward()
        (expected * seed).sum().backward()
        for name, value, reference in zip(("input", *dict(module.named_parameters())), values, refs):
            with self.subTest(name=name):
                self.assertIsNotNone(value.grad)
                self.assert_tensor(value.grad, reference.grad)


class TestRoPE(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        try:
            learner.apply_rope(torch.ones(1, 2, 3, 4, dtype=torch.float64),
                               torch.arange(3, dtype=torch.long))
        except NotImplementedError:
            raise unittest.SkipTest("先完成 apply_rope")

    def test_scalar_reference_distinct_frequencies_offsets_and_layouts(self):
        positions = torch.tensor([2, 5, 9], dtype=torch.long)
        for dtype in (torch.float32, torch.float64):
            for D in (4, 8):
                for theta in (10000., 256.):
                    for layout in ("contiguous", "transpose", "slice"):
                        with self.subTest(dtype=dtype, D=D, theta=theta, layout=layout):
                            base, x = random_input((2, 3, 3, D), dtype, layout)
                            before, old_positions = base.clone(), positions.clone()
                            self.assert_tensor(learner.apply_rope(x, positions, theta),
                                               rope_reference(x, positions, theta))
                            self.assertTrue(torch.equal(base, before))
                            self.assertTrue(torch.equal(positions, old_positions))

    def test_independent_known_answer_excludes_identity_rotation(self):
        x = torch.tensor([[[[1., 0., 1., 0.]]]], dtype=torch.float64)
        expected = torch.tensor([[[[math.cos(1.), math.sin(1.),
                                   math.cos(.01), math.sin(.01)]]]], dtype=x.dtype)
        self.assert_tensor(learner.apply_rope(x, torch.tensor([1])), expected)

    def test_position_zero_pair_norm_and_common_position_shift_dot(self):
        _, q = random_input((2, 3, 4, 8))
        _, k = random_input((2, 3, 5, 8))
        zeros = torch.zeros(4, dtype=torch.long)
        self.assert_tensor(learner.apply_rope(q, zeros), q)
        qp = torch.tensor([1, 3, 4, 8], dtype=torch.long)
        kp = torch.tensor([0, 2, 5, 6, 10], dtype=torch.long)
        qr, kr = learner.apply_rope(q, qp), learner.apply_rope(k, kp)
        self.assert_tensor(qr.reshape(2, 3, 4, 4, 2).square().sum(-1),
                           q.reshape(2, 3, 4, 4, 2).square().sum(-1))
        shifted_q, shifted_k = learner.apply_rope(q, qp + 11), learner.apply_rope(k, kp + 11)
        self.assert_tensor(shifted_q @ shifted_k.transpose(-1, -2),
                           qr @ kr.transpose(-1, -2))

    def test_gradients_noncontiguous_and_existing_gradients_preserved(self):
        positions = torch.tensor([3, 6, 8], dtype=torch.long)
        for layout in ("contiguous", "transpose", "slice"):
            with self.subTest(layout=layout):
                base, x = random_input((2, 2, 3, 8), layout=layout, grad=True)
                ref_base, ref_x = random_input((2, 2, 3, 8), layout=layout, grad=True)
                before = base.detach().clone()
                for value in (base, ref_base):
                    value.grad = torch.full_like(value, .125)
                result = learner.apply_rope(x, positions)
                expected = rope_reference(ref_x, positions)
                self.assert_tensor(result, expected)
                self.assertTrue(torch.equal(base, before))
                self.assert_tensor(base.grad, torch.full_like(base, .125))
                seed = torch.linspace(-.7, 1.2, result.numel(), dtype=result.dtype).reshape_as(result)
                (result * seed).sum().backward()
                (expected * seed).sum().backward()
                self.assertIsNotNone(base.grad)
                self.assert_tensor(base.grad, ref_base.grad)

    def test_odd_head_dimension_raises_value_error(self):
        for D in (1, 3, 5):
            with self.subTest(D=D), self.assertRaises(ValueError):
                learner.apply_rope(torch.ones(1, 2, 3, D, dtype=torch.float64),
                                   torch.arange(3, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
