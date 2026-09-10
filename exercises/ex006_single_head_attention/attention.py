"""单头 Attention：保留已完成的投影/分数，继续填写下方三个接口。"""

import math

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


def make_causal_allowed(input_valid: torch.Tensor) -> torch.Tensor:
    """生成同序列、位置对齐的因果读取权限，不计算分数。

    input_valid: CPU bool Tensor，shape (B,T)，各维度为正。
    True 表示真实输入位置，False 表示 PAD；不是 target_valid。
    返回 CPU bool Tensor，shape (B,T,T)。
    allowed[b,i,j] 为 True 当且仅当 j<=i 且 input_valid[b,j] 为 True。
    不用 input_valid[b,i] 清空 query 行，不修改输入。
    此函数仅生成权限，即使产生全 False 行也返回，由下一接口拒绝该行。
    不要求额外校验 shape/dtype 或有效位置是否连续。
    """
    B, T = input_valid.shape
    causal_mask = torch.tril(torch.ones((T, T), dtype=torch.bool))
    allowed = input_valid.unsqueeze(1) & causal_mask.unsqueeze(0)
    return allowed


    


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    allowed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从给定 Q/K/V 完成缩放、屏蔽、归一化和内容读取。

    Q: (B,Tq,Dk)，K: (B,Tk,Dk)，V: (B,Tk,Dv)。各维度为正。
    Tq/Tk、Dk/Dv 均可不同。返回顺序为 (output, weights)：
    output: (B,Tq,Dv)，weights: (B,Tq,Tk)。不额外加 Q/X/V，不加输出投影。
    Q/K/V 均为 CPU 上相同 dtype 的 float32/float64；输入及原始点积分数有限。
    allowed 为 None 时允许全部候选，且不隐式添加因果规则。
    否则为同设备 bool Tensor，shape 恰好 (B,Tq,Tk)，True=允许。
    只要求一个错误检查：若有任意全 False 行，Softmax 前抛出 ValueError。
    不要求其他输入校验，不支持额外 mask 广播形式。
    使用已有 raw_attention_scores；除以 sqrt(Dk)，在 Softmax 前屏蔽，
    沿 Tk 归一化，返回权重及加权读取结果。允许使用 torch.softmax。
    不修改任何输入或已有 .grad；两个返回值都保留自动求导路径。
    不调用 backward，不用 no_grad/detach 切断计算，不用高级 Attention 封装。
    """
    scores = raw_attention_scores(Q, K)  # (B,Tq,Tk)
    Dk  = Q.shape[-1]
    scaled_scores = scores / math.sqrt(Dk)
    if allowed is not None:
        scaled_scores = scaled_scores.masked_fill(~allowed, float('-inf'))
        if torch.any(torch.all(~allowed, dim=-1)):
            raise ValueError("存在全 False 行，无法计算 Softmax。")
    weights = torch.softmax(scaled_scores, dim=-1)  # (B,Tq,Tk)
    output = weights @ V  # (B,Tq,Dv)
    return output, weights



def single_head_self_attention(
    X: torch.Tensor,
    Wq: torch.Tensor,
    Wk: torch.Tensor,
    Wv: torch.Tensor,
    input_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """组合已写的函数，完成因果单头自注意力。

    X: (B,T,C)，Wq/Wk: (C,Dk)，Wv: (C,Dv)，input_valid: (B,T)。
    其余 dtype/device/有限性约定与上述接口相同。
    调用 project_qkv、make_causal_allowed 和 scaled_dot_product_attention。
    返回 (output, weights)，shape 分别为 (B,T,Dv)、(B,T,T)。
    这是 Attention 输出，不是词表 logits；不额外添加 X、位置编码或输出投影。
    不按 query 有效性把输出置零；全屏蔽行的 ValueError 由内层传出。
    保持计算图、不修改任何输入或已有梯度。
    """
    Q, K, V = project_qkv(X, Wq, Wk, Wv)
    allowed = make_causal_allowed(input_valid)
    output, weights = scaled_dot_product_attention(Q, K, V, allowed)
    return output, weights
