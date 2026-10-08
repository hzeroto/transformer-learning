"""原版 Transformer 的固定正弦位置编码。"""

import torch


def sinusoidal_positions(
    positions: torch.Tensor,
    C: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """步骤 2：给实际位置编号生成固定位置向量，返回 CPU (T,C)。

    positions 是 CPU long 一维 Tensor，长度 T，可非连续、可为空；
    元素为非负位置编号，不保证连续，也不保证从 0 开始。C 是特征宽度，
    dtype 为 torch.float32 或 torch.float64。输出采用指定 dtype，
    无训练参数，不改变 positions，也不查可学习位置表。

    对每个位置 p、从 0 开始的特征对编号 i：
      frequency_i = 10000 ** (-2*i/C)
      PE[p, 2*i]   = sin(p * frequency_i)
      PE[p, 2*i+1] = cos(p * frequency_i)
    输出行的顺序与 positions 一致；公式中的 p 是实际位置编号，
    不是输出行号。C 为奇数时，最后一个未配对维度只保留 sin。
    本函数只返回 PE；乘 sqrt(C) 的 embedding 与相加由模型入口处理。

    C<=0 时抛 ValueError，其余输入按上述合法约定。可以使用
    arange、广播、sin、cos 等基础 Tensor 运算；不调用现成位置编码。
    float64 测试 rtol=1e-10、atol=1e-12；float32 为 1e-5、1e-6。
    """
    if C <= 0:
        raise ValueError("C 必须是正整数")

    # 0、2、4……就是公式中的 2*i；每两个特征共用一个频率。
    # 这里创建普通 Tensor，没有 nn.Parameter，因此位置编码不会被训练更新。
    pair_starts = torch.arange(0, C, 2, dtype=dtype, device=positions.device)
    frequencies = 10000.0 ** (-pair_starts / C)  # (ceil(C/2),)

    # 实际位置 (T,1) 乘频率 (1,ceil(C/2))，得到每个位置在每个频率下的角度。
    # positions=[3,4] 就用 3、4；不能因为只有两行而替换成位置 0、1。
    angles = positions.to(dtype=dtype).unsqueeze(1) * frequencies.unsqueeze(0)
    encoded = torch.empty((positions.numel(), C), dtype=dtype, device=positions.device)
    encoded[:, 0::2] = torch.sin(angles)
    # 奇数 C 时，cos 的槽位少一个；最后一个未配对的维度保留 sin。
    encoded[:, 1::2] = torch.cos(angles[:, :C // 2])
    return encoded
