"""单头 Attention 的第一部分：学习者实现投影与原始匹配分数。"""

import torch


def project_qkv(
    X: torch.Tensor,
    Wq: torch.Tensor,
    Wk: torch.Tensor,
    Wv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从同一份输入分别计算 Q、K、V，按此顺序返回。

    X: (B,T,C)；Wq/Wk: (C,Dk)；Wv: (C,Dv)。
    返回 Q/K: (B,T,Dk)，V: (B,T,Dv)。
    各维度为正整数，Dk 和 Dv 可以不同。
    输入均为 CPU 上相同 dtype 的 float32 或 float64，数值有限。
    参数在所有 batch 和位置间共享。不加偏置，不修改任何输入。
    保持 dtype/device；输入或参数需要求导时，保留相应计算图。
    不要求额外输入校验；使用基础张量运算，不用 nn.Linear 等封装。
    """
    return X @ Wq, X @ Wk, X @ Wv


def raw_attention_scores(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """为每个查询位置与每个候选位置计算原始点积分数。

    Q: (B,Tq,Dk)，K: (B,Tk,Dk)，返回 (B,Tq,Tk)。
    scores[b,i,j] 是同一 batch 中 Q[b,i,:] 与 K[b,j,:] 的点积。
    此接口允许 Tq != Tk；当前同序列投影调用时二者都等于 T。
    输入均为 CPU 上相同 dtype 的 float32 或 float64，数值有限。
    不除以向量长度，不缩放、不加 mask、不做 Softmax。
    保持 dtype/device 和计算图，不修改输入，不要求额外输入校验。
    使用基础张量运算，不用现成 Attention 接口。
    """
    return Q @ K.transpose(-2, -1)
