"""学习者实现：多头 Attention 的布局转换，不包含 Attention 数学计算。"""

import torch


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """把每个 token 的投影特征分配给各个 head。

    x: CPU float32/float64 Tensor，shape (B,T,C)，各维度为正。
    x 可以是连续张量，也可以是转置/切片得到的非连续张量。
    num_heads 是 Python 整数 H；要求 H>0 且 C 能被 H 整除，
    否则抛出 ValueError。不要求额外的类型或维度数量校验。
    令 Dh=C//H，返回 shape 为 (B,H,T,Dh) 的 Tensor，满足：
        result[b,h,t,d] = x[b,t,h*Dh+d]
    x 可代表已投影的 Q/K/V；本函数不做投影，不增加参数，不改数值。
    保持 dtype/device 和自动求导路径，不修改输入及调用者已有梯度。
    使用基础 Tensor 布局操作即可；允许 view/reshape/transpose/contiguous。
    返回值可以共享存储也可以是副本，不要求特定 stride 或连续性。
    """
    B, T, C = x.shape
    if num_heads <= 0 or C % num_heads != 0:
        raise ValueError("num_heads 必须为正整数，且能整除 C。")
    return x.reshape(B, T, num_heads, C // num_heads).transpose(1, 2)



def merge_heads(heads: torch.Tensor) -> torch.Tensor:
    """按 token 合并各个 head 的内容。

    heads: CPU float32/float64 Tensor，shape (B,H,T,Dh)，各维度为正。
    支持连续和非连续输入；输入不一定来自 split_heads，
    也可能是每个 head 的 Attention 计算新产生的张量。
    返回 shape 为 (B,T,H*Dh) 的 Tensor，满足：
        result[b,t,h*Dh+d] = heads[b,h,t,d]
    保持 dtype/device 和自动求导路径，不修改输入及调用者已有梯度。
    不求平均/求和，不加残差，不做输出投影，不要求额外输入校验。
    返回值可以共享存储也可以是副本，不要求特定 stride 或连续性。
    """
    B, H, T, Dh = heads.shape
    return heads.transpose(1, 2).reshape(B, T, H * Dh)
