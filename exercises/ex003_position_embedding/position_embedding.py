"""从已有内容表和位置表生成带位置信息的 token 表示。"""

import torch


def embed_with_positions(
    ids: torch.Tensor,
    token_table: torch.Tensor,
    position_table: torch.Tensor,
) -> torch.Tensor:
    """返回 shape 为 (B, T, C) 的带位置表示。

    Args:
        ids: (B, T) 的 torch.long 编号张量，B 和 T 均为正数。
        token_table: (N, C) 的浮点内容表，所有输入 ID 均在有效范围内。
        position_table: (L, C) 的浮点位置表，与内容表 dtype 相同。

    约定：
        输入均在 CPU 上，N、L、C 为正数，两张表的 C 相同。
        每条序列的位置从 0 开始，所有 batch 共享同一张位置表。
        不修改任何输入，不在函数内部重新初始化参数表。
        本函数不屏蔽 PAD；有效位置标记由调用方保留供后续使用。

    Raises:
        ValueError: T 超过位置表支持的长度 L。
        其他非法输入不属于本练习要求。
    """
    # TODO 1：取得 T，检查它是否超出位置表的长度。
    B, T = ids.shape
    N, C = token_table.shape
    L, _ = position_table.shape
    if T > L:
        raise ValueError(f"T={T} 超过位置表支持的长度 L={L}")
    # TODO 2：分别按 token ID 和当前位置取得两类向量。
    token_vectors = token_table[ids]  # shape (B, T, C)
    position_vectors = position_table[:T]  # shape (T, C)
    # TODO 3：用广播完成相加，返回 (B, T, C) 的结果。
    return token_vectors + position_vectors  # shape (B, T, C)
