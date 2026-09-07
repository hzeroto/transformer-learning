"""训练闭环的第一部分：按有效标签平均的稳定交叉熵。"""

import torch


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """返回所有有效标签位置的平均交叉熵，shape 为 ()。

    输入约定：
        logits: (B, T, N)，CPU 上的 float32 或 float64 Tensor。
        targets: (B, T)，torch.long，所有 ID 均在 [0, N) 内。
        target_valid: (B, T)，torch.bool，True 表示该标签参与损失。
        B、T、N 均为正数，logits 元素有限；不要求额外的输入校验。

    要求：
        返回与 logits 同 dtype、device 的标量 Tensor，保留自动求导路径。
        仅依据 target_valid 筛选，不根据某个特定 token ID 推断有效性。
        每个有效标签权重相同，不先逐序列平均，不把 PAD 计入分母。
        使用已学的基础张量操作，不修改输入，不使用现成交叉熵或归一化封装。

    Raises:
        ValueError: 整个 batch 没有有效标签。
        某一条序列没有有效标签、但其他序列有时，仍应正常计算。
    """
    # TODO 1：逐位置平移分数，计算稳定公式中的对数求和项。
    shifted =  logits - logits.amax(dim=-1, keepdim=True)
    log_denomistor = torch.log(shifted.exp().sum(dim=-1, keepdim=True)) # (B, T, 1)
    # 交叉熵：Loss = ln(sigma(exp(logits))) - logits[targets]
    # TODO 2：按 targets 取得对应位置的正确答案分数，得到逐位置损失。
    index = targets.unsqueeze(-1) # (B, T, 1)
    loss = log_denomistor - shifted.gather(dim=-1, index=index) # (B, T, 1)
    # TODO 3：只选有效标签，处理全部无效的情况，返回平均损失。
    if target_valid.sum() == 0:
        raise ValueError("整个 batch 没有有效标签。")
    valid_loss = loss[target_valid].mean()
    return valid_loss
