"""ex018 底层 Attention 教师测试；参照仅供验收，不得从作业导入。

CPU，随机种子 1801/1802。通常 float64 rtol=1e-8/atol=1e-10，
float32 rtol=2e-5/atol=2e-6；极端分数的 float32 Q 梯度 atol=3e-4。
运算观察只检查逻辑配对/单块大小，不测运行速度或 autograd 峰值内存。
"""

import math
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from exercises.ex018_efficient_attention import attention as learner


def setUpModule():
    global _old_threads
    _old_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_old_threads)


def reference(q, k, v, q_positions, k_positions, key_valid, window=None):
    """独立逐 batch/head/query 参照：显式索引，避免共用广播/分块路径。"""
    batches = []
    for b in range(q.shape[0]):
        heads = []
        for h in range(q.shape[1]):
            rows = []
            for i, position in enumerate(q_positions.tolist()):
                indices = [j for j, kp in enumerate(k_positions.tolist())
                           if bool(key_valid[b, j]) and kp <= position
                           and (window is None or position - kp < window)]
                if not indices:
                    raise ValueError("存在没有合法 key 的 query")
                scores = torch.stack([
                    torch.dot(q[b, h, i], k[b, h, j]) / math.sqrt(q.shape[-1])
                    for j in indices
                ])
                weights = torch.softmax(scores, dim=0)
                rows.append(sum(weights[jj] * v[b, h, j]
                                for jj, j in enumerate(indices)))
            heads.append(torch.stack(rows))
        batches.append(torch.stack(heads))
    return torch.stack(batches)


def sample(dtype=torch.float64, layout="contiguous", grad=False):
    generator = torch.Generator().manual_seed(1801)
    values = []
    for shape in ((2, 2, 5, 5), (2, 2, 8, 5), (2, 2, 8, 3)):
        if layout == "slice":
            value = torch.randn((*shape[:-1], shape[-1] * 2),
                                generator=generator, dtype=dtype)[..., ::2]
        elif layout == "transpose":
            value = torch.randn((*shape[:-2], shape[-1], shape[-2]),
                                generator=generator, dtype=dtype).transpose(-1, -2)
        else:
            value = torch.randn(shape, generator=generator, dtype=dtype)
        values.append(value.detach().requires_grad_(grad))
    qp = torch.tensor([11, 14, 18, 23, 29])
    kp = torch.tensor([7, 10, 11, 14, 17, 18, 23, 29])
    valid = torch.tensor([[True, False, True, True, False, True, True, True],
                          [False, True, True, True, True, True, True, True]])
    return (*values, qp, kp, valid)


class OperationAudit(TorchDispatchMode):
    """观察实际 Tensor 运算，不读取学习者源码或相信其自报计数。

    结构夹具中 Dk=13，Dv=7，而窗口/块宽小于 13，所以可区分 QK 与 AV。
    einsum、@、matmul 都会分派到底层 mm/bmm；也接受 mv/dot。
    全矩阵形状检查用于发现预建完整 mask 或拼回完整 score/weight。
    """

    def __init__(self, *, dk=None, tq=None, tk=None, online=False):
        super().__init__()
        self.dk, self.tq, self.tk, self.online = dk, tq, tk, online
        self.pairs = 0
        self.largest_score = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if "scaled_dot_product" in name or "flash_attention" in name:
            raise AssertionError("本题不能由融合 Attention 接口代做")
        if self.online and "softmax" in name:
            raise AssertionError("online_attention 需自行合并在线指数状态，不能调用 Softmax")
        result = func(*args, **(kwargs or {}))
        if self.tq is not None:
            outputs = result if isinstance(result, (tuple, list)) else (result,)
            for output in outputs:
                if isinstance(output, torch.Tensor) and output.ndim >= 2:
                    if tuple(output.shape[-2:]) in ((self.tq, self.tk), (self.tk, self.tq)):
                        raise AssertionError("不能生成完整 Tq×Tk 的 score、weight 或 mask")
        if self.dk is not None and isinstance(result, torch.Tensor):
            is_score = False
            if name in ("aten.mm.default", "aten.bmm.default"):
                is_score = args[0].shape[-1] == self.dk and args[1].shape[-2] == self.dk
            elif name == "aten.mv.default":
                is_score = args[0].shape[-1] == self.dk and args[1].numel() == self.dk
            elif name == "aten.dot.default":
                is_score = args[0].numel() == self.dk and args[1].numel() == self.dk
            if is_score:
                self.pairs += result.numel()
                self.largest_score = max(self.largest_score, result.numel())
        return result


def call(kind, args, window, **blocks):
    with OperationAudit(online=kind == "online"):
        return getattr(learner, f"{kind}_attention")(*args, window, **blocks)


class TensorAssertions(unittest.TestCase):
    def assert_tensor(self, actual, expected, *, atol=None):
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertTrue(torch.isfinite(actual).all().item(), "输出或梯度出现 NaN/Inf")
        tolerance = dict(rtol=2e-5, atol=2e-6) if expected.dtype == torch.float32 else dict(rtol=1e-8, atol=1e-10)
        if atol is not None:
            tolerance["atol"] = atol
        torch.testing.assert_close(actual, expected, **tolerance)

    def assert_gradients(self, kind, args, window, *, extreme=False, **blocks):
        inputs = tuple(t.detach().requires_grad_() for t in args[:3])
        ref_inputs = tuple(t.detach().clone().requires_grad_() for t in args[:3])
        for t in inputs:
            t.grad = torch.full_like(t, .375)
        saved = [t.detach().clone() for t in (*inputs, *args[3:])]
        output = call(kind, (*inputs, *args[3:]), window, **blocks)
        expected = reference(*ref_inputs, *args[3:], window)
        self.assert_tensor(output, expected)
        self.assertTrue(output.requires_grad, "输出不得 detach")
        for t, before in zip((*inputs, *args[3:]), saved):
            self.assertTrue(torch.equal(t, before), "前向不得修改输入或位置/mask")
        for t in inputs:
            self.assertTrue(torch.equal(t.grad, torch.full_like(t, .375)), "前向不得改已有 .grad")
        probe = torch.linspace(-.7, 1.1, output.numel(), dtype=output.dtype).reshape_as(output)
        actual_grads = torch.autograd.grad((output * probe).sum(), inputs)
        expected_grads = torch.autograd.grad((expected * probe).sum(), ref_inputs)
        for index, (actual, target) in enumerate(zip(actual_grads, expected_grads)):
            with self.subTest(gradient=("Q", "K", "V")[index]):
                atol = 3e-4 if extreme and output.dtype == torch.float32 else None
                self.assert_tensor(actual, target, atol=atol)

    def assert_empty_rows_rejected(self, kind):
        args = sample()
        # 一种是只有 batch 1 的单个 query 全屏蔽；另一种是位置窗口里没有 key。
        valid = args[-1].clone()
        valid[1, 0:3] = False
        cases = [(*args[:-1], valid), (*args[:3], args[3] + 100, *args[4:])]
        for inputs in cases:
            with self.subTest(kind=kind, positions=inputs[3].tolist()):
                with self.assertRaises(ValueError):
                    call(kind, inputs, 3)


class ImplementationStatusTest(unittest.TestCase):
    def test_attention_functions_are_implemented(self):
        missing = []
        for kind in ("local", "online"):
            try:
                call(kind, sample(), 4)
            except NotImplementedError:
                missing.append(f"{kind}_attention")
        self.assertFalse(missing, "请实现 attention.py：" + ", ".join(missing))


class LocalAttentionTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        try:
            call("local", sample(), 4)
        except NotImplementedError:
            raise unittest.SkipTest("先完成 local_attention；最终验收必须无跳过")

    def test_dtypes_layouts_absolute_positions_and_window_boundaries(self):
        for dtype in (torch.float32, torch.float64):
            for layout in ("contiguous", "slice", "transpose"):
                args = sample(dtype, layout)
                for window in (1, 3, 7, 100):
                    with self.subTest(dtype=dtype, layout=layout, window=window):
                        self.assert_tensor(call("local", args, window), reference(*args, window))

    def test_gradients_inputs_and_existing_grad_are_preserved(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                self.assert_gradients("local", sample(dtype, "slice"), 7)

    def test_future_pad_and_old_values_cannot_influence_protected_output(self):
        generator = torch.Generator().manual_seed(1802)
        q = torch.randn(2, 2, 1, 5, generator=generator, dtype=torch.float64)
        k = torch.randn(2, 2, 8, 5, generator=generator, dtype=q.dtype)
        v = torch.randn(2, 2, 8, 3, generator=generator, dtype=q.dtype)
        qp, kp = torch.tensor([15]), torch.arange(10, 18)
        valid = torch.ones(2, 8, dtype=torch.bool)
        valid[0, 4] = False
        valid[1, 3] = False
        args = q, k, v, qp, kp, valid
        output = call("local", args, 3)
        self.assert_tensor(output, reference(*args, 3))
        altered_k, altered_v = k.clone(), v.clone()
        for b in range(2):
            forbidden = [j for j, p in enumerate(kp.tolist())
                         if p < 13 or p > 15 or not valid[b, j]]
            altered_k[b, :, forbidden, :] += 1000
            altered_v[b, :, forbidden, :] -= 500
        self.assert_tensor(call("local", (q, altered_k, altered_v, qp, kp, valid), 3), output)

    def test_empty_query_row_is_rejected(self):
        self.assert_empty_rows_rejected("local")

    def test_actual_dot_product_count_is_local(self):
        generator = torch.Generator().manual_seed(1802)
        b, h, t, dk, dv, window = 2, 2, 29, 13, 7, 4
        q = torch.randn(b, h, t, dk, generator=generator, dtype=torch.float64)
        k = torch.randn(q.shape, generator=generator, dtype=q.dtype)
        v = torch.randn(b, h, t, dv, generator=generator, dtype=q.dtype)
        pos = torch.arange(100, 100 + t)
        args = q, k, v, pos, pos, torch.ones(b, t, dtype=torch.bool)
        audit = OperationAudit(dk=dk, tq=t, tk=t)
        with audit:
            output = call("local", args, window)
        self.assert_tensor(output, reference(*args, window))
        self.assertGreater(audit.pairs, 0, "QK 点积请使用题面允许的矩阵乘法 API")
        self.assertLessEqual(audit.pairs, b * h * sum(min(i + 1, window) for i in range(t)),
                             "mask 前已计算过多 QK 配对，尚未做到真实局部读取")


class OnlineAttentionTest(TensorAssertions):
    @classmethod
    def setUpClass(cls):
        try:
            call("online", sample(), 4)
        except NotImplementedError:
            raise unittest.SkipTest("先完成 online_attention；最终验收必须无跳过")

    def test_dtypes_layouts_mask_broadcast_and_uneven_blocks(self):
        for dtype in (torch.float32, torch.float64):
            for layout in ("contiguous", "slice", "transpose"):
                args = sample(dtype, layout)
                for window, qb, kb in ((None, 3, 4), (7, 2, 3), (1, 1, 1), (100, 9, 11)):
                    with self.subTest(dtype=dtype, layout=layout, window=window, blocks=(qb, kb)):
                        output = call("online", args, window, query_block=qb, key_block=kb)
                        self.assert_tensor(output, reference(*args, window))

    def test_gradients_with_initial_middle_and_final_empty_blocks(self):
        for dtype in (torch.float32, torch.float64):
            args = sample(dtype, "transpose")
            for window in (3, None):
                with self.subTest(dtype=dtype, window=window):
                    self.assert_gradients("online", args, window, query_block=3, key_block=2)

    def test_extreme_scores_rescale_old_state_and_keep_gradients_finite(self):
        for dtype in (torch.float32, torch.float64):
            q = torch.ones(1, 1, 2, 1, dtype=dtype)
            k = torch.tensor([-1000, -999, 1000, 1001, 1002], dtype=dtype).reshape(1, 1, 5, 1)
            v = torch.tensor([1., -2., 3., -5., 8., 2., 4., -3., 7., 1.], dtype=dtype).reshape(1, 1, 5, 2)
            args = q, k, v, torch.tensor([3, 4]), torch.arange(5), torch.ones(1, 5, dtype=torch.bool)
            with self.subTest(dtype=dtype):
                self.assert_gradients("online", args, None, extreme=True, query_block=2, key_block=2)

    def test_block_outputs_cannot_be_averaged_and_score_shift_is_harmless(self):
        q = torch.ones(1, 1, 1, 1, dtype=torch.float64)
        values = torch.tensor([0., 0., 10., 10.], dtype=q.dtype).reshape(1, 1, 4, 1)
        scores = torch.tensor([0., 0., math.log(3), math.log(3)], dtype=q.dtype)
        for shift in (0., 1000., -1000.):
            args = q, (scores + shift).reshape(1, 1, 4, 1), values, torch.tensor([3]), torch.arange(4), torch.ones(1, 4, dtype=torch.bool)
            with self.subTest(shift=shift):
                self.assert_tensor(call("online", args, None, query_block=1, key_block=2),
                                   torch.full_like(values[:, :, :1], 7.5))

    def test_full_causal_and_local_window_are_distinct_read_sets(self):
        q = torch.zeros(1, 1, 8, 2, dtype=torch.float64)
        k = torch.zeros_like(q)
        v = torch.arange(8, dtype=q.dtype).reshape(1, 1, 8, 1)
        args = q, k, v, torch.arange(8), torch.arange(8), torch.ones(1, 8, dtype=torch.bool)
        full = call("online", args, None)
        local = call("online", args, 3)
        self.assert_tensor(full, reference(*args))
        self.assert_tensor(local, reference(*args, 3))
        self.assert_tensor(full[..., -1:, :], torch.full_like(full[..., -1:, :], 3.5))
        self.assert_tensor(local[..., -1:, :], torch.full_like(local[..., -1:, :], 6.))
        self.assert_tensor(call("online", args, 1), v)

    def test_empty_query_row_is_rejected(self):
        self.assert_empty_rows_rejected("online")

    def test_tiles_stay_small_and_window_skips_disjoint_tiles_before_qk(self):
        generator = torch.Generator().manual_seed(1802)
        b, h, tq, tk, dk, dv, qb, kb = 2, 2, 29, 31, 13, 7, 3, 4
        q = torch.randn(b, h, tq, dk, generator=generator, dtype=torch.float64)
        k = torch.randn(b, h, tk, dk, generator=generator, dtype=q.dtype)
        v = torch.randn(b, h, tk, dv, generator=generator, dtype=q.dtype)
        qp, kp = torch.arange(100, 100 + tq), torch.arange(100, 100 + tk)
        args = q, k, v, qp, kp, torch.ones(b, tk, dtype=torch.bool)
        for window in (None, 4):
            with self.subTest(window=window):
                audit = OperationAudit(dk=dk, tq=tq, tk=tk, online=True)
                with audit:
                    output = call("online", args, window, query_block=qb, key_block=kb)
                self.assert_tensor(output, reference(*args, window))
                self.assertGreater(audit.pairs, 0, "QK 点积请使用题面允许的矩阵乘法 API")
                self.assertLessEqual(audit.largest_score, b * h * qb * kb,
                                     "单次分数计算超过设定 tile 的元素预算")
                budget = 0
                for qi in range(0, tq, qb):
                    for ki in range(0, tk, kb):
                        qs, ks = qp[qi:qi + qb].tolist(), kp[ki:ki + kb].tolist()
                        if any(0 <= x - y and (window is None or x - y < window) for x in qs for y in ks):
                            budget += b * h * len(qs) * len(ks)
                self.assertLessEqual(audit.pairs, budget,
                                     "完全不可见的 KV tile 必须在 QK 乘法之前跳过")


if __name__ == "__main__":
    unittest.main()
