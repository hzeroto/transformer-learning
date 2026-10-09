"""滑窗与在线 Softmax 的 CPU 教师探针，不是学习者作业或 GPU FlashAttention。

运行：.venv/bin/python -B -m examples.efficient_attention_probe

三种实现接受同一接口：Q/K 是 (B,H,Tq/Tk,Dk)，V 是 (B,H,Tk,Dv)，
q_positions/k_positions 是严格递增的绝对位置，key_valid 是 (B,Tk)。
window=W 表示包含自身的因果窗口：0 <= q_position-k_position < W。
window=None 表示完整因果读取。每个 query 至少要有一个有效 key。
Q/K/V 的输入元素以及缩放点积分数必须有限；本探针不处理输入 NaN/Inf
或点积本身已经溢出的情形。减最大值只防止指数溢出，不能修复此前的溢出。

本文件故意保留普通 autograd，以比较 Q/K/V 梯度；反向会保留多个分块的
中间结果，不能由“最大 score tile 很小”推断训练峰值内存是线性的。
统计只记录逻辑点积数量和最大单块分数元素，不测延迟、显存或硬件带宽。
"""

import math
from dataclasses import dataclass

import torch


# @dataclass 根据下面的字段自动生成 __init__ 等方法；这里只是装两个计数器。
@dataclass
class Work:
    """逻辑账本；score_pairs 包含计算后被 mask 掉的 pair。"""

    score_pairs: int = 0
    max_score_elements: int = 0

    def record(self, scores):
        self.score_pairs += scores.numel()
        self.max_score_elements = max(self.max_score_elements, scores.numel())


def _validate(q, k, v, q_positions, k_positions, key_valid, window):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q/K/V 必须有四个轴")
    b, h, tq, dk = q.shape
    tk = k.shape[2]
    if k.shape != (b, h, tk, dk) or v.shape[:3] != (b, h, tk):
        raise ValueError("Q/K/V 形状不匹配")
    if tq == 0 or tk == 0 or dk == 0 or v.shape[-1] == 0:
        raise ValueError("不接受空维度")
    if q_positions.shape != (tq,) or k_positions.shape != (tk,):
        raise ValueError("绝对位置长度不匹配")
    if key_valid.shape != (b, tk) or key_valid.dtype != torch.bool:
        raise ValueError("key_valid 必须为 (B,Tk) 的 bool 张量")
    if any(x.device.type != "cpu" for x in (q, k, v, q_positions, k_positions, key_valid)):
        raise ValueError("该教师探针仅用于 CPU")
    if not (q.dtype == k.dtype == v.dtype) or q.dtype not in (torch.float32, torch.float64):
        raise ValueError("Q/K/V 使用相同 float32 或 float64")
    if q_positions.dtype != torch.long or k_positions.dtype != torch.long:
        raise ValueError("绝对位置使用 torch.long")
    if not (torch.all(q_positions[1:] > q_positions[:-1])
            and torch.all(k_positions[1:] > k_positions[:-1])):
        raise ValueError("绝对位置必须严格递增，不要求从 0 开始或连续")
    if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window < 1):
        raise ValueError("window 必须为 None 或正整数")


def _allowed(q_positions, k_positions, key_valid, window):
    distance = q_positions[:, None] - k_positions[None, :]
    allowed = distance >= 0
    if window is not None:
        allowed = allowed & (distance < window)
    return allowed[None, None, :, :] & key_valid[:, None, None, :]


def dense_attention(q, k, v, q_positions, k_positions, key_valid, window=None):
    """完整 score 矩阵 + mask：正确性参照，不是局部优化。"""
    _validate(q, k, v, q_positions, k_positions, key_valid, window)
    allowed = _allowed(q_positions, k_positions, key_valid, window)
    if not allowed.any(dim=-1).all():
        raise ValueError("存在没有任何合法 key 的 query")
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    work = Work()
    work.record(scores)
    weights = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1)
    return weights @ v, work


def local_attention(q, k, v, q_positions, k_positions, key_valid, window=None):
    """每个 query 只切出位置合法的 K/V 区间，不创建完整 Tq×Tk mask。"""
    _validate(q, k, v, q_positions, k_positions, key_valid, window)
    work, outputs = Work(), []
    for i, position in enumerate(q_positions):
        # searchsorted 返回该绝对位置在有序 k_positions 中的插入下标。
        start = 0 if window is None else int(torch.searchsorted(k_positions, position - window + 1))
        stop = int(torch.searchsorted(k_positions, position, right=True))
        valid = key_valid[:, start:stop]
        if not valid.any(dim=-1).all():
            raise ValueError("存在没有任何合法 key 的 query")
        scores = (q[:, :, i:i + 1, :] @ k[:, :, start:stop, :].transpose(-2, -1)) / math.sqrt(q.shape[-1])
        work.record(scores)
        weights = torch.softmax(scores.masked_fill(~valid[:, None, None, :], -torch.inf), dim=-1)
        outputs.append(weights @ v[:, :, start:stop, :])
    return torch.cat(outputs, dim=2), work


def tiled_attention(q, k, v, q_positions, k_positions, key_valid, window=None,
                    *, query_block=3, key_block=4):
    """精确在线归一化的教学算法，不调用任何融合 attention 接口。

    每行只跨块携带 m（历史最大值）、l（分母）、u（未归一化加权和）。
    主体不分配全序列 mask；允许行在前几个块没有合法 key。
    """
    _validate(q, k, v, q_positions, k_positions, key_valid, window)
    if query_block < 1 or key_block < 1:
        raise ValueError("块大小必须为正数")
    b, h, tq, dk = q.shape
    work, outputs = Work(), []
    for qi in range(0, tq, query_block):
        qb = q[:, :, qi:qi + query_block, :]
        qr = q_positions[qi:qi + query_block]
        state_shape = (b, h, qb.shape[2], 1)
        m = torch.full(state_shape, -torch.inf, dtype=q.dtype)
        l = torch.zeros(state_shape, dtype=q.dtype)
        u = torch.zeros((*state_shape[:-1], v.shape[-1]), dtype=q.dtype)
        for ki in range(0, k.shape[2], key_block):
            kr = k_positions[ki:ki + key_block]
            allowed = _allowed(qr, kr, key_valid[:, ki:ki + key_block], window)
            if not allowed.any():
                continue  # 整个 tile 都无效，可连点积一起跳过。
            kb, vb = k[:, :, ki:ki + key_block, :], v[:, :, ki:ki + key_block, :]
            scores = (qb @ kb.transpose(-2, -1)) / math.sqrt(dk)
            work.record(scores)
            scores = scores.masked_fill(~allowed, -torch.inf)
            new_m = torch.maximum(m, scores.amax(dim=-1, keepdim=True))
            # 某行可能历史和当前块均为空。用 0 作临时基准，避免 -inf-(-inf)。
            safe_m = torch.where(torch.isfinite(new_m), new_m, torch.zeros_like(new_m))
            alpha = torch.exp(m - safe_m)  # 空历史为 exp(-inf)=0。
            p = torch.exp(scores - safe_m)  # 当前块无效位置也恰为 0。
            u = alpha * u + p @ vb
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            m = new_m
        if (l == 0).any():
            raise ValueError("存在没有任何合法 key 的 query")
        outputs.append(u / l)
    return torch.cat(outputs, dim=2), work


def _compare_case(name, q, k, v, qp, kp, valid, window, *, gradient_atol=None,
                  query_block=3, key_block=4):
    """以同一个线性标量目标比较输出与 Q/K/V 梯度。"""
    source = tuple(x.detach().requires_grad_(True) for x in (q, k, v))
    expected, _ = dense_attention(*source, qp, kp, valid, window)
    target_weights = torch.linspace(-0.8, 1.1, expected.numel(), dtype=q.dtype).reshape(expected.shape)
    expected_grads = torch.autograd.grad((expected * target_weights).sum(), source)
    tol = 2e-5 if q.dtype == torch.float32 else 2e-11
    gradient_atol = tol if gradient_atol is None else gradient_atol
    max_output_error = max_grad_error = 0.0
    for implementation in (local_attention, tiled_attention):
        source = tuple(x.detach().requires_grad_(True) for x in (q, k, v))
        options = {"query_block": query_block, "key_block": key_block} if implementation is tiled_attention else {}
        actual, _ = implementation(*source, qp, kp, valid, window, **options)
        actual_grads = torch.autograd.grad((actual * target_weights).sum(), source)
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
        max_output_error = max(max_output_error, float((actual - expected).abs().max().detach()))
        for got, want in zip(actual_grads, expected_grads):
            assert torch.isfinite(got).all()
            torch.testing.assert_close(got, want, atol=gradient_atol, rtol=tol)
            max_grad_error = max(max_grad_error, float((got - want).abs().max()))
    print(f"{name:24s} {str(q.dtype):13s} output max err={max_output_error:.2e}, grad max err={max_grad_error:.2e}")


def verify_equivalence():
    generator = torch.Generator().manual_seed(1701)
    for dtype in (torch.float64, torch.float32):
        def rand(*shape):
            return torch.randn(shape, generator=generator, dtype=dtype)

        # Tq=7、Tk=10；query 从绝对位置 23 开始。块大小 3/4 均不能整除。
        q, k, v = rand(2, 2, 7, 4), rand(2, 2, 10, 4), rand(2, 2, 10, 3)
        kp, qp = torch.arange(20, 30), torch.arange(23, 30)
        valid = torch.ones(2, 10, dtype=torch.bool)
        valid[0, 0] = False
        valid[1, 1] = False
        for window, label in ((None, "full causal"), (4, "local W=4 + PAD"), (1, "self-only W=1"), (20, "W covers prefix")):
            _compare_case(label, q, k, v, qp, kp, valid, window)

        # 块大小是执行选择，不改变数学结果；覆盖单元素、非整除及大于序列的块。
        for query_block, key_block in ((1, 1), (2, 3), (16, 16)):
            _compare_case(f"block sizes {query_block}/{key_block}", q, k, v, qp, kp, valid, 4,
                          query_block=query_block, key_block=key_block)

        # 一个 batch 的首个 key tile 完全无效，另一 batch 非空；空行状态必须安全。
        prefix_pad = valid.clone()
        prefix_pad[0, :4] = False
        _compare_case("first tile empty row", q[:, :, 1:], k, v, qp[1:], kp, prefix_pad, None)

        # 所有 batch 的第一个 tile 均无效，验证整 tile 跳过分支。
        prefix_pad[:, :4] = False
        _compare_case("first tile all empty", q[:, :, 1:], k, v, qp[1:], kp, prefix_pad, None)

        # 不同绝对位置之间允许存在间隔；window 按位置差，不按存储槽下标计算。
        sparse_kp = torch.tensor([40, 41, 43, 46, 47, 50, 51, 54, 55, 58])
        _compare_case("gapped absolute positions", q, k, v, sparse_kp[-7:], sparse_kp, valid, 5)

        # 输入切片是非连续视图，算法不能依赖 view 强行重排。
        _compare_case("noncontiguous inputs", rand(2, 2, 7, 8)[..., ::2],
                      rand(2, 2, 10, 8)[..., ::2], rand(2, 2, 10, 6)[..., ::2], qp, kp, valid, 4)

        # 极端但有限的 logits：第一批低分、后续块高分，直接 exp 会溢出。
        # float32 的 Q 梯度需将约 1000 倍的项相减，绝对误差被放大；
        # 此例单独使用 2e-4 梯度绝对容差，前向及常规案例仍用原有容差。
        extreme_q = torch.ones(1, 1, 2, 1, dtype=dtype)
        extreme_k = torch.tensor([-1000, -999, 1000, 1001, 1002], dtype=dtype).reshape(1, 1, 5, 1)
        _compare_case("extreme logits", extreme_q, extreme_k, rand(1, 1, 5, 2),
                      torch.tensor([3, 4]), torch.arange(5), torch.ones(1, 5, dtype=torch.bool), None,
                      gradient_atol=2e-4 if dtype == torch.float32 else 2e-11)

        # 全无合法 key 是接口错误，三种实现都拒绝；不输出 NaN 或静默全零。
        for implementation in (dense_attention, local_attention, tiled_attention):
            for mask in (torch.zeros_like(valid), valid):
                positions = qp if not mask.any() else qp + 100
                try:
                    implementation(q, k, v, positions, kp, mask, 2)
                except ValueError:
                    pass
                else:
                    raise AssertionError("全屏蔽 query 应当报错")
    print("all-masked rows: rejected by all 3 implementations")


def demonstrate_wrong_merges():
    # 固定一个 query/head。两个 key 块各有两个元素，局部 softmax 都是 [1/2,1/2]。
    scores = torch.tensor([0.0, 0.0, math.log(3.0), math.log(3.0)], dtype=torch.float64)
    values = torch.tensor([0.0, 0.0, 10.0, 10.0], dtype=torch.float64)
    correct = (scores.softmax(dim=0) * values).sum()
    block_outputs = [(scores[i:i + 2].softmax(dim=0) * values[i:i + 2]).sum() for i in (0, 2)]
    average_blocks = torch.stack(block_outputs).mean()
    # 第二块 max 从 0 变成 ln(3)；忘记旧 l/u 乘 exp(-ln(3)) 会混合不同基准。
    old_l, old_u = 2.0, 0.0
    new_p = torch.exp(scores[2:] - scores[2:].max())
    forgot_rescale = (old_u + float((new_p * values[2:]).sum())) / (old_l + float(new_p.sum()))
    torch.testing.assert_close(correct, torch.tensor(7.5, dtype=torch.float64))
    assert average_blocks.item() == forgot_rescale == 5.0

    q = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    v = values.reshape(1, 1, 4, 1)
    qp, kp, valid = torch.tensor([3]), torch.arange(4), torch.ones(1, 4, dtype=torch.bool)
    for shift in (0.0, 1000.0):
        k = (scores + shift).reshape(1, 1, 4, 1)
        tiled, _ = tiled_attention(q, k, v, qp, kp, valid, key_block=2)
        dense, _ = dense_attention(q, k, v, qp, kp, valid)
        torch.testing.assert_close(tiled.squeeze(), correct, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(dense, tiled, atol=1e-11, rtol=1e-11)
    print(f"wrong merge: exact={correct.item():.6f}, average={average_blocks.item():.6f}, missing-rescale={forgot_rescale:.6f}")
    print("adding 1000 to every score: dense/tiled output remains 7.5")


def demonstrate_work():
    # 所有 score 都相等，故区别完全来自可见集合，避免随机数掩盖差异。
    t, window = 8, 3
    q = torch.zeros(1, 1, t, 2, dtype=torch.float64)
    k = torch.zeros_like(q)
    v = torch.arange(t, dtype=torch.float64).reshape(1, 1, t, 1)
    positions = torch.arange(t)
    valid = torch.ones(1, t, dtype=torch.bool)
    with torch.no_grad():
        full, _ = dense_attention(q, k, v, positions, positions, valid)
        masked, dense_work = dense_attention(q, k, v, positions, positions, valid, window)
        local, local_work = local_attention(q, k, v, positions, positions, valid, window)
        tiled_full, tiled_full_work = tiled_attention(q, k, v, positions, positions, valid)
        tiled_local, tiled_local_work = tiled_attention(q, k, v, positions, positions, valid, window)
    torch.testing.assert_close(masked, local)
    torch.testing.assert_close(masked, tiled_local)
    torch.testing.assert_close(full, tiled_full)
    covers_all, _ = local_attention(q, k, v, positions, positions, valid, t)
    self_only, _ = tiled_attention(q, k, v, positions, positions, valid, 1)
    torch.testing.assert_close(covers_all, full)
    torch.testing.assert_close(self_only, v)
    assert dense_work.score_pairs == t * t
    assert local_work.score_pairs == sum(min(i + 1, window) for i in range(t))
    assert local_work.max_score_elements == window
    assert full[0, 0, -1, 0].item() == 3.5
    assert local[0, 0, -1, 0].item() == 6.0
    print("same Q/K/V, last query: full=3.5, local W=3=6.0 (different semantics)")
    for label, work in (("dense+window mask", dense_work), ("actual local slices", local_work),
                        ("tiled full causal", tiled_full_work), ("tiled local", tiled_local_work)):
        print(f"{label:24s} score pairs={work.score_pairs:3d}, largest score tensor={work.max_score_elements:3d} elements")
    print("These are logical counters, NOT peak memory measurements; autograd may retain all tiles.")


def demonstrate_cache():
    """只验证已给定 Q/K/V 的窗口缓存，不冒充完整模型/RoPE/logits 验收。"""
    q = torch.zeros(1, 1, 9, 2, dtype=torch.float64)
    k = torch.zeros_like(q)
    v = torch.arange(9, dtype=torch.float64).reshape(1, 1, 9, 1)
    positions = torch.arange(9)  # 即使缓存切片，仍保留原始绝对位置。
    valid = torch.ones(1, 9, dtype=torch.bool)
    window = 3
    reference, _ = dense_attention(q, k, v, positions, positions, valid, window)

    # 已有缓存位置 3、4、5；追加 6、7。第一个新 query 仍需要 4。
    # 先附加整个新 chunk，再计算各 query；工作集允许大于 W。
    work_k, work_v, work_pos = k[:, :, 3:8], v[:, :, 3:8], positions[3:8]
    output, _ = local_attention(q[:, :, 6:8], work_k, work_v, positions[6:8],
                                work_pos, valid[:, 3:8], window)
    torch.testing.assert_close(output, reference[:, :, 6:8])
    bad_output, _ = local_attention(q[:, :, 6:8], work_k[:, :, -window:], work_v[:, :, -window:],
                                    positions[6:8], work_pos[-window:], valid[:, -window:], window)
    assert not torch.allclose(bad_output[:, :, :1], reference[:, :, 6:7])
    assert output[0, 0, 0, 0].item() == 5.0
    assert bad_output[0, 0, 0, 0].item() == 5.5

    # 当前 chunk 全部算完后再选择最后 W 个。下次追加 8，旧 key=5 过期。
    # 这里只验证逻辑内容，切片仍共享 storage；实际释放旧存储需复制或固定容量缓冲区。
    kept_k, kept_v, kept_pos = work_k[:, :, -window:], work_v[:, :, -window:], work_pos[-window:]
    next_k = torch.cat((kept_k, k[:, :, 8:9]), dim=2)
    next_v = torch.cat((kept_v, v[:, :, 8:9]), dim=2)
    next_pos = torch.cat((kept_pos, positions[8:9]))
    next_out, _ = local_attention(q[:, :, 8:9], next_k, next_v, positions[8:9],
                                  next_pos, torch.ones(1, 4, dtype=torch.bool), window)
    torch.testing.assert_close(next_out, reference[:, :, 8:9])
    assert torch.equal(next_pos[-window:], torch.tensor([6, 7, 8]))
    print("QKV-only cache: append [6,7] correct first output=5.0; premature last-W trim gives 5.5")
    print("after next append, persisted absolute positions=[6,7,8]; no index reset")


def main():
    torch.set_num_threads(1)
    verify_equivalence()
    demonstrate_wrong_merges()
    demonstrate_work()
    demonstrate_cache()
    print("PASS: CPU forward/gradient/edge-case probes; GPU fusion, speed and peak memory unmeasured.")


if __name__ == "__main__":
    main()
