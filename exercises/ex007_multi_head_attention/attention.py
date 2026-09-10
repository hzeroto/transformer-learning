"""学习者实现：复用拆合头，完成多头读取及因果自注意力。"""

import math

import torch

from exercises.ex006_single_head_attention.attention import (
    make_causal_allowed,
    project_qkv,
)
from exercises.ex007_multi_head_attention.heads import merge_heads, split_heads


def multi_head_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Wo: torch.Tensor,
    num_heads: int,
    allowed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """接收已经投影的 Q/K/V，完成拆头、逐头读取、合头和输出投影。

    Q: (B,Tq,C)，K/V: (B,Tk,C)，Wo: (C,C)。
    本题采用等宽头，H=num_heads，Dh=C//H；Tq/Tk 可以不同。
    H 必须为正整数且整除 C，否则 ValueError，可复用 split_heads 的校验。
    allowed 为 None 时所有 key 可读，不隐式添加因果规则。
    否则为 CPU bool (B,Tq,Tk)，True=允许，各头共享同一份位置权限；
    内部补出 head 轴进行广播，不复制成 H 份，不接受其他外部 mask 形状。
    任意 query 没有可读 key 时，在 Softmax 前抛出 ValueError。

    返回 (output, weights)：(B,Tq,C)、(B,H,Tq,Tk)。
    weights 沿 key 轴归一化，每个 head 使用自己的分数和 sqrt(Dh) 缩放。
    output 是合头结果乘 Wo；Wo 不参与权重计算。不对 head 平均或求和。
    调用已写的 split_heads、merge_heads；逐头计算用批量 Tensor 运算，
    不写遍历 batch/head/query/key 的 Python 循环。

    Q/K/V/Wo 为 CPU 相同 dtype 的 float32/float64；维度均正，
    输入与原始点积分数有限，支持连续/转置/步长切片输入。
    保留 dtype/device、两个输出的计算图，不修改输入及已有 .grad。
    不重新投影 Q/K/V，不加偏置、残差、位置编码或词表投影；
    不调用 backward/no_grad/detach、高级 Attention 或融合算子。
    不要求其余输入校验、GPU 支持或特定输出 stride。
    """
    # 将 ex006 的读取数学推广到 (B,H) 批次前缀，再接 Wo。
    # 不直接向 ex006 的三维接口传入四维张量；该接口合同未扩展。


    # error check
    C = Q.shape[2]
    if num_heads <= 0 or C % num_heads != 0:
        raise ValueError("xx")
    Dh = C // num_heads
   
    # 1. 拆head
    mQ = split_heads(Q, num_heads) # (B, H, Tq, Dh)
    mK = split_heads(K, num_heads) # (B, H, Tk, Dh)
    mV = split_heads(V, num_heads) # (B, H, Tk, Dh)
    # 1. 计算分数
    grades = mQ @ mK.transpose(-2, -1) # (B, H, Tq, Tk)
    # 2. 变形处理
    grades_sq = grades/math.sqrt(Dh)
    # 3. mask
    if allowed is not None:
        # 4. softmax
        if torch.any(torch.all(~allowed, dim=-1)):
            raise ValueError("None of the keys are allowed for some queries.")
        grades_sq = grades_sq.masked_fill(~allowed.unsqueeze(1), float("-inf"))

    
    grades_sm = torch.softmax(grades_sq, dim=-1)
    
    outputs = grades_sm @ mV # (B, H, Tq, Dh)
    m_outputs =  merge_heads(outputs)
    return m_outputs @ Wo, grades_sm

def multi_head_self_attention(
    X: torch.Tensor,
    Wq: torch.Tensor,
    Wk: torch.Tensor,
    Wv: torch.Tensor,
    Wo: torch.Tensor,
    input_valid: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从同一份输入得到 Q/K/V，组合成完整因果多头自注意力。

    X: (B,T,C)，Wq/Wk/Wv/Wo: (C,C)，input_valid: CPU bool (B,T)。
    复用 project_qkv、make_causal_allowed 和上述 multi_head_attention。
    allowed[b,i,j] = (j<=i) 且 input_valid[b,j]，不能改成 target_valid，
    不用 query 有效性清空整行；全屏蔽行错误由内层传出。
    返回 (output, weights)，shape 为 (B,T,C)、(B,H,T,T)。
    其余合同同上；参数共享于所有 batch/token，不在函数内创建或更新参数。
    输出是含上下文的特征，不是词表 logits，也不是整个 Transformer Block。
    """
    Q, K, V = project_qkv(X, Wq, Wk, Wv)  
    allowed = make_causal_allowed(input_valid)
    return multi_head_attention(Q, K, V, Wo, num_heads, allowed)
