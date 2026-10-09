"""两种 Attention 执行方式的带注释参考实现；接口约定见 README.md。

Q/K/V 已经由上游投影、拆头，若使用 RoPE，Q/K 也已旋转。
这里只返回每头输出，不返回完整 Attention 权重，不做 Wo 投影。
"""

import torch


def local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    key_valid: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """真正局部读取的因果窗口 Attention。

    输入（所有 shape、dtype、device 和参数合法性由调用方保证）：
      q: CPU float32/64 (B,H,Tq,Dk)。
      k: 同 dtype (B,H,Tk,Dk)；v: 同 dtype (B,H,Tk,Dv)。
      B/H/Tq/Tk/Dk/Dv 均为正数；Tq/Tk 和 Dk/Dv 不一定相同。
      q_positions: CPU long (Tq,)；k_positions: CPU long (Tk,)。
          严格递增的非负绝对位置，可有间隔；整个 batch 共用这些位置。
      key_valid: CPU bool (B,Tk)，True 表示该 key 有效，沿 head 轴广播。
          不屏蔽 query；不同 batch 可以有不同的有效 key。
      window: 正整数，包含 query 自身位置的因果窗口宽度。

    key j 对 query i 可读，当且仅当 key_valid[b,j] 为 True，且
      0 <= q_positions[i] - k_positions[j] < window。
    窗口按绝对位置差定义；不是筛完 PAD 后再数 window 个有效 key。
    打分使用 q·k / sqrt(Dk)，Softmax 沿允许的 key 归一化，然后加权 V。

    返回 (B,H,Tq,Dv)，保持输入 dtype/device。
    任意 query 在某个 batch 中整行没有可读 key：抛 ValueError。
    支持非连续 q/k/v；不修改任何输入或已有 .grad，保留 Q/K/V 的求导路径。
    不在内部 backward、detach 或关闭求导。输入及未屏蔽的点积分数均有限。

    执行约束：实际 QK 乘法只包含因果与窗口范围内的位置；PAD key 可以
    在局部打分后屏蔽。不先算完整 Tq×Tk 分数、权重或权限矩阵。
    点积使用 @ / matmul / mm / bmm / einsum，供教师测试观察实际计算范围；
    其余允许基础 Tensor 运算，局部归一化可用 torch.softmax。
    不调用 online_attention、教师探针、测试参照或高级/融合 Attention。
    """
    # 1. 先确定每个 query 的可读位置区间，再做点积。
    # 对绝对位置 p，合法 key 位置是闭区间 [p-window+1, p]。
    # k_positions 有序，searchsorted 通过二分查找返回插入下标：
    # 左边界取第一个 >= p-window+1 的下标，右边界取第一个 > p 的下标。
    # 得到的两个向量都是 (Tq,)，不会构造完整的 (Tq,Tk) 权限表。
    left_edges = torch.searchsorted(k_positions, q_positions - window + 1)
    right_edges = torch.searchsorted(k_positions, q_positions, right=True)
    scale = q.shape[-1] ** -0.5  # 按匹配宽度 Dk 缩放，而不是按 Dv。
    rows = []

    for row in range(q.shape[-2]):
        left, right = left_edges[row].item(), right_edges[row].item()
        local_valid = key_valid[:, left:right]  # (B,w)，w 是本行候选 key 数。

        # 2. 每个 batch 的这一行都必须有至少一个有效 key。
        # w=0 时 any 也返回 False；同一检查覆盖位置区间为空和全是 PAD。
        # 必须在 softmax 前检查，避免一整行 -inf 产生 NaN。
        if not local_valid.any(dim=-1).all().item():
            raise ValueError("存在没有合法 key 的 query")

        # 3. 只取窗口内的 K/V；窗口外和未来位置完全不参与 QK 点积。
        # 保留 query 轴长度 1：(B,H,1,Dk) @ (B,H,Dk,w) -> (B,H,1,w)。
        query = q[:, :, row:row + 1, :]
        keys = k[:, :, left:right, :]
        values = v[:, :, left:right, :]
        scores = (query @ keys.transpose(-2, -1)) * scale
        allowed = local_valid[:, None, None, :]  # (B,1,1,w)，广播到所有头。
        scores = scores.masked_fill(~allowed, -torch.inf)

        # 4. 在本行全部合法 key 上归一化，再加权读取 V。
        # 切片、softmax、矩阵乘法和 cat 都保留自动求导关系。
        weights = torch.softmax(scores, dim=-1)
        rows.append(weights @ values)  # (B,H,1,Dv)

    return torch.cat(rows, dim=-2)  # (B,H,Tq,Dv)


def online_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    key_valid: torch.Tensor,
    window: int | None = None,
    *,
    query_block: int = 3,
    key_block: int = 4,
) -> torch.Tensor:
    """支持全历史或滑窗的分块在线归一化 Attention。

    输入/输出、位置、PAD、缩放、dtype、布局、求导和只读契约同 local_attention。
    window=None 时保留全部因果历史；正整数时沿用包含自身的窗口规则。
    query_block/key_block 是正整数块尺寸，不是 batch 大小；可以大于序列长度。
    更换块尺寸不能改变数学结果；浮点求和顺序的差异按测试容差判断。

    自行维护每个 query 的在线归一化状态，用基础 Tensor 的指数、归约、
    矩阵乘法完成，不用 torch.softmax/log_softmax 或调用 local_attention 代做。
    每个 score/权限块的 query/key 轴不得超过指定块尺寸（尾块可更小）。
    不预先创建完整 Tq×Tk score、权重或 mask；若整个输入恰好装进指定单块，
    允许用一个块处理。QK 点积的可用 API 同 local_attention。
    整块对所有 batch/query 都无可读 key 时，在 QK 乘法之前跳过；
    边界块可计算后逐元素屏蔽。允许遍历候选块，不考直接定位块的调度优化。

    某一行在当前块暂时没有可读项时，保留该行已有状态；其他有效行继续更新。
    未读到有效项的空状态也要安全处理，不能让 NaN 污染输出或梯度。
    扫完所有块后，任意 batch/query 整行仍无可读 key：抛 ValueError。

    这是 CPU 数学与执行范围考核，不要求自定义反向或 GPU kernel，
    也不要求 Python 循环更快或 autograd 训练总内存线性增长。
    """
    batch, heads, query_count, key_width = q.shape
    scale = key_width ** -0.5
    output_blocks = []

    # 1. 外层遍历 query 块；每块独立维护它的逐行状态。
    # Qb/Kb 表示当前块的实际长度，尾块可以小于配置的块尺寸。
    for query_start in range(0, query_count, query_block):
        query_end = min(query_start + query_block, query_count)
        queries = q[:, :, query_start:query_end, :]  # (B,H,Qb,Dk)
        query_positions = q_positions[query_start:query_end]
        state_shape = (batch, heads, query_end - query_start, 1)

        # m = 已读合法分数的最大值；l = sum(exp(score-m))；
        # u = sum(exp(score-m)*value)。最后的输出才是 u/l。
        # m/l 保留末轴 1，方便广播；u 的末轴是内容宽度 Dv。
        # new_* 继承 q 的 dtype/device，不会误用默认的 float32。
        running_max = q.new_full(state_shape, -torch.inf)  # m；-inf 表示尚未读到 key。
        normalizer = q.new_zeros(state_shape)  # l
        weighted_sum = q.new_zeros((*state_shape[:-1], v.shape[-1]))  # u

        for key_start in range(0, k.shape[-2], key_block):
            key_end = min(key_start + key_block, k.shape[-2])
            key_positions = k_positions[key_start:key_end]

            # 2. 只为当前块建立权限：(Qb,Kb) -> (B,1,Qb,Kb)。
            # 用绝对位置之差判断因果和窗口；块内下标不能代表真实距离。
            distance = query_positions[:, None] - key_positions[None, :]
            position_allowed = distance >= 0
            if window is not None:
                position_allowed = position_allowed & (distance < window)
            allowed = (
                position_allowed[None, None, :, :]
                & key_valid[:, None, None, key_start:key_end]
            )
            # 全块无合法配对就不做 QK。某些行为空、其他行有效时仍继续。
            if not allowed.any().item():
                continue

            # 3. 分数仅为 (B,H,Qb,Kb)；不拼回完整分数表或权重表。
            keys = k[:, :, key_start:key_end, :]
            values = v[:, :, key_start:key_end, :]
            scores = (queries @ keys.transpose(-2, -1)) * scale
            scores = scores.masked_fill(~allowed, -torch.inf)
            block_max = scores.amax(dim=-1, keepdim=True)
            new_max = torch.maximum(running_max, block_max)

            # 4. 为尚未读到合法 key 的行构造安全的减法基准。
            # 这些行的 new_max=-inf，直接算 -inf-(-inf) 会得到 NaN。
            # 仅在做减法时用 0 占位：exp(-inf-0)=0；真实 m 仍保留 -inf。
            # where 的两个候选值会先计算，所以应先选择安全基准，
            # 再做减法/exp，不能先产生 NaN 再试图用 where 遮住。
            safe_max = torch.where(
                torch.isfinite(new_max), new_max, torch.zeros_like(new_max),
            )
            old_scale = (running_max - safe_max).exp()  # alpha = exp(m-m')
            unnormalized = (scores - safe_max).exp()  # p_j = exp(score_j-m')

            # 5. 旧分子、分母都换到新基准，再加上本块贡献。
            # 不能平均各块的 softmax 输出；不同块的指数总量通常不相等。
            # 已有历史而本块为空的行：alpha=1、p=0，状态自然保持。
            # 历史和本块都为空的行：alpha=0、p=0，l/u 继续为 0。
            normalizer = old_scale * normalizer + unnormalized.sum(dim=-1, keepdim=True)
            weighted_sum = old_scale * weighted_sum + unnormalized @ values
            running_max = new_max

        # 6. 只有扫完全部 key 块后，才能判断某行是否真的没有合法 key。
        # 有合法 key 的行至少有一项 exp(max-max)=1，因此分母必定为正。
        if (normalizer == 0).any().item():
            raise ValueError("存在没有合法 key 的 query")
        output_blocks.append(weighted_sum / normalizer)  # (B,H,Qb,Dv)

    # 拼接的是每头的输出，不是分数或权重；只需要 (B,H,Tq,Dv) 空间。
    return torch.cat(output_blocks, dim=-2)
