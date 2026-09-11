"""构造函数由教师提供；请完成四个 forward，不导入教师测试参照。

统一契约见同目录 README.md。基础 Tensor/autograd 可用，核心数学手写。
"""

import torch
from torch import nn

from exercises.ex007_multi_head_attention.attention import multi_head_self_attention


def _matrix(rows, cols, dtype):
    """教师辅助：给矩阵一个小的随机起点，不是本次待实现内容。"""
    return nn.Parameter(torch.randn(rows, cols, dtype=dtype) * 0.05)


class LayerNorm(nn.Module):
    def __init__(self, C, eps=1e-5, *, dtype=torch.float64):
        super().__init__()
        if not isinstance(C, int) or isinstance(C, bool) or C <= 0:
            raise ValueError("C 必须是正整数")
        if not eps > 0:
            raise ValueError("eps 必须大于 0")
        self.C = C
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(C, dtype=dtype))
        self.beta = nn.Parameter(torch.zeros(C, dtype=dtype))

    def forward(self, X):
        """X: (B,T,C) 浮点激活 → 同 shape 输出；每个 token 独立归一化。

        使用最后一轴的总体方差（除以 C），eps 在平方根内。
        gamma/beta: (C,) 是共享的可训练参数；保留 X 和参数的梯度路径。
        输入可以不连续，不能原地修改。完整边界见 README。
        """
        # with torch.no_grad():
        mean = X.mean(dim=-1, keepdim=True)
        X_centered = X - mean # (B, T, C)
        var = (X_centered ** 2).mean(dim=-1, keepdim=True) # (B, T, 1)
        X_normalized = X_centered / torch.sqrt(var + self.eps) # (B, T, C)

        return self.gamma * X_normalized + self.beta


class FeedForward(nn.Module):
    def __init__(self, C, F, *, dtype=torch.float64):
        super().__init__()
        for name, size in (("C", C), ("F", F)):
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError(f"{name} 必须是正整数")
        self.W1 = _matrix(C, F, dtype)
        self.b1 = nn.Parameter(torch.zeros(F, dtype=dtype))
        self.W2 = _matrix(F, C, dtype)
        self.b2 = nn.Parameter(torch.zeros(C, dtype=dtype))

    def forward(self, X):
        """X: (B,T,C) → 中间表示 (B,T,F) → 输出 (B,T,C)。

        两次仿射变换之间使用 ReLU；每个位置共享参数，不混合不同 token。
        ReLU 在 0 点采用梯度 0。输入及四个参数的梯度都要保留。
        """
        # TODO：实现两次仿射变换和中间的 ReLU。
        X0 = X @ self.W1 + self.b1 # (B, T, F)
        X1 =  torch.where(X0 > 0, X0, torch.zeros_like(X0)) # (B, T, F)
        X2 = X1 @ self.W2 + self.b2 # (B, T, C)
        return X2

class Dropout(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("本练习要求 0 <= p < 1，不处理 p=1")
        self.p = p

    def forward(self, X):
        """X: (B,T,C) → 同 shape 输出；本模块没有可训练参数。

        self.training 为 True 且 p>0：每个元素独立以 p 概率置零，
        保留值按 1/(1-p) 缩放；反向使用这一次前向对应的 mask。
        eval 或 p=0：恒等映射，不抽随机数，不截断梯度。
        不原地修改 X，不在 forward 中设随机种子或缓存跨调用的 mask。
        """
        if self.training and self.p > 0:
            mask = (torch.rand_like(X) >= self.p).to(X.dtype)
            return mask * X / (1 - self.p)
        return X


class TransformerBlock(nn.Module):
    def __init__(self, C, num_heads, ffn_hidden, p=0.0, eps=1e-5, *, dtype=torch.float64):
        super().__init__()
        for name, size in (("C", C), ("num_heads", num_heads), ("ffn_hidden", ffn_hidden)):
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if C % num_heads != 0:
            raise ValueError("C 必须能被 num_heads 整除")
        self.C = C
        self.num_heads = num_heads
        self.Wq = _matrix(C, C, dtype)
        self.Wk = _matrix(C, C, dtype)
        self.Wv = _matrix(C, C, dtype)
        self.Wo = _matrix(C, C, dtype)
        self.norm1 = LayerNorm(C, eps, dtype=dtype)
        self.norm2 = LayerNorm(C, eps, dtype=dtype)
        self.ffn = FeedForward(C, ffn_hidden, dtype=dtype)
        self.drop1 = Dropout(p)
        self.drop2 = Dropout(p)

    def forward(self, X, input_valid):
        """X: (B,T,C), input_valid: bool (B,T) → Y: (B,T,C)。

        实现两条残差分支组成的 Pre-LN Block；Dropout 只放在各分支输出上。
        复用已完成的 multi_head_self_attention，不重写拆头/因果 mask。
        input_valid=True 表示该位置可作为真实 key；不是 loss mask。
        保留 PAD query 的输出，不人为清零；无任何可读 key 的行继续抛 ValueError。
        只返回 Y，不返回注意力权重；本层不加 embedding、位置编码或词表投影。
        不创建或替换参数/子模块，不在前向切换 train/eval 或启用 no_grad。
        """
        N1 = self.norm1(X) # (B, T, C)
        multih_head_output, _ = multi_head_self_attention(
            N1, self.Wq, self.Wk, self.Wv, self.Wo, input_valid, self.num_heads
        ) # (B, T, C)

        U = X + self.drop1(multih_head_output) # (B, T, C)

        N2 = self.norm2(U) # (B, T, C)
        ffn_output = self.ffn(N2) # (B, T, C)
        Y = U + self.drop2(ffn_output) # (B, T, C)

        return Y
