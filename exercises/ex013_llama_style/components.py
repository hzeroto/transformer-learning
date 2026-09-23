"""学习者填写三个核心运算；参数构造由教师提供。

CPU float32/float64，使用基础 Tensor/autograd。统一题面见 README.md。
不从 tests 导入参照，不修改已验收练习。
"""
import torch
from torch import nn


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


def _weight(rows, cols, dtype):
    return nn.Parameter(torch.randn(rows, cols, dtype=dtype) * 0.05)


class RMSNorm(nn.Module):
    """每个 token 的均方根归一化；只有 gamma，无 beta。"""

    def __init__(self, C, eps=1e-5, *, dtype=torch.float64):
        super().__init__()
        _positive_int("C", C)
        if not eps > 0:
            raise ValueError("eps 必须大于 0")
        self.C, self.eps = C, eps
        self.gamma = nn.Parameter(torch.ones(C, dtype=dtype))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: CPU 浮点 (B,T,C) -> 同 shape、dtype 的输出。

        只沿最后一轴测量原始 X 的平均平方，不减均值；eps 在根号内。
        每个 token 独立缩放，再使用共享的 gamma(C,)。参考讲义第 2 节。
        支持非连续 X、零向量及 C=1；保留 X/gamma 的求导路径。
        不修改输入、参数或已有 .grad，不调用 backward/no_grad/detach。
        shape/dtype 合法由调用方保证，数值只要求本课 float32/64 范围。
        """
        # 1. 计算每个 token 的均方根
        avg = torch.abs(X).pow(2).mean(dim=-1, keepdim=True)  # (B,T,1)
        rms = torch.sqrt(avg + self.eps)  # (B,T,1)
        # 2. 归一化并缩放
        X_normed = X / rms  # (B,T,C)
        return X_normed * self.gamma  # (B,T,C)


class SwiGLU(nn.Module):
    """无偏置三矩阵 FFN；中间宽 hidden_dim 对应讲义的 G。"""

    def __init__(self, C, hidden_dim, *, dtype=torch.float64):
        super().__init__()
        _positive_int("C", C)
        _positive_int("hidden_dim", hidden_dim)
        self.C, self.hidden_dim = C, hidden_dim
        self.Wgate = _weight(C, hidden_dim, dtype)
        self.Wup = _weight(C, hidden_dim, dtype)
        self.Wdown = _weight(hidden_dim, C, dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: CPU 浮点 (B,T,C) -> 同 shape、dtype 的 FFN 更新量。

        Wgate/Wup 从同一个 X 产生两条独立投影，均为 (B,T,hidden_dim)。
        门控采用 SiLU（可用 torch.sigmoid 表达），两路逐元素相乘后由
        Wdown 映射回 C。此函数不做归一化或残差相加，不混合不同 token。
        不用现成 F.silu/MLP 替代本次激活数学；参考讲义第 4 节。
        支持非连续输入，保留输入和三份权重梯度，不改任何输入或已有 .grad。
        """
        a = X @ self.Wgate  # (B,T,hidden_dim)
        gate = a * torch.sigmoid(a)  # (B,T,hidden_dim)
        up = X @ self.Wup  # (B,T,hidden_dim)
        return (gate * up) @ self.Wdown  # (B,T,C)


def apply_rope(
    x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0,
) -> torch.Tensor:
    """为已拆头的一份 Q 或 K 施加相邻特征对旋转。

    x: CPU float32/64 (B,H,n,D)，允许非连续，D 为正整数。
    positions: CPU long (n,)，非负整数，按位置轴对应 x 的 n 个 token；
        不一定从 0 开始；所有 batch/head 共享本次位置。
    theta: 固定正数，默认 10000；不是训练参数。
    返回与 x 同 shape、dtype/device 的旋转结果，保留 x 的梯度。

    固定配对 (0,1),(2,3),...；第 j 对使用 theta**(-2*j/D)，
    角度来自调用方 positions（弧度），旋转方向见讲义第 3.1 节。
    必须对不同特征对使用各自频率，不在这里重新生成从零开始的位置。
    Raises: ValueError，当 D 是奇数；其他 shape、dtype、值范围由调用方保证。
    不旋转 V、不修改 x/positions/已有 .grad、不创建可训练参数。
    使用批量 Tensor 运算，不写逐 batch/head/token/特征对的 Python 循环。
    """
    D = x.shape[-1]
    if D % 2 != 0:
        raise ValueError("D 必须是偶数")
    # 1. 计算每个特征对的频率
    half_D = D // 2
    j = torch.arange(half_D, dtype=x.dtype, device=x.device)  #(half_D,)
    freqs = theta ** (-2 * j / D)  # (half_D,)
    # 2. 计算每个 token 的旋转角度
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (n, half_D)
    # 3. 计算旋转矩阵的 cos/sin
    cos = torch.cos(angles)  # (n, half_D)
    sin = torch.sin(angles)  # (n, half_D)
    # 4. 将 x 拆分为偶数/奇数特征对
    x_even = x[..., 0::2]  # (B,H,n,half_D)
    x_odd = x[..., 1::2]   # (B,H,n,half_D)
    # 5. 应用旋转 逆时针旋转 a = r * cos(x+theta); b = r * sin(x+theta)
    x_rotated_even = x_even * cos[None, None, ...] - x_odd * sin[None, None, ...]  # (B,H,n,half_D)
    x_rotated_odd = x_even * sin[None, None, ...] + x_odd * cos[None, None, ...]  # (B,H,n,half_D)
    # 6. 将旋转后的偶数/奇数特征对重新组合
    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = x_rotated_even
    x_rotated[..., 1::2] = x_rotated_odd
    return x_rotated
