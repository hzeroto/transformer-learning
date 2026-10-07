"""教师演示：跨序列读取的权限，以及固定正弦位置的偏移关系。

运行：.venv/bin/python -B -m examples.encoder_decoder_probe
复用学习者已有的 ex007 MHA；这里提供的是手选特征对照，不是完整模型、
训练结果或学习者作业答案。CPU float64，断言容差 1e-12。
"""

import torch

from exercises.ex007_multi_head_attention.attention import (
    multi_head_attention,
    multi_head_self_attention,
)


def sinusoidal_positions(positions: torch.Tensor, width: int) -> torch.Tensor:
    """位置下标 (T,) → 编码 (T,C)；奇数 C 时最后一维仅保留 sin。

    这是讲义公式的教师计算工具，不是可训练参数或模型完整输入层。
    本演示限定 CPU、非空非负整数 positions、正整数 width。
    """
    exponents = torch.arange(0, width, 2, dtype=torch.float64) / width
    angles = positions.to(torch.float64).unsqueeze(1) / (10000.0 ** exponents)
    result = torch.empty((positions.numel(), width), dtype=torch.float64)
    result[:, 0::2] = torch.sin(angles)
    result[:, 1::2] = torch.cos(angles[:, :width // 2])
    return result


def close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def cross_attention_probe():
    # E 是人为指定的“Encoder 最终表示”，不是 token ID 或训练出的语义。
    # B=1, S=5, T=4, C=2, H=1。最后两个源位置视为 PAD。
    memory = torch.tensor(
        [[[1., 0.], [0., 1.], [2., 3.], [99., 99.], [-99., -99.]]],
        dtype=torch.float64,
    )
    target_state = torch.zeros(1, 4, 2, dtype=torch.float64)
    source_valid = torch.tensor([[True, True, True, False, False]])
    allowed = source_valid.unsqueeze(1).expand(1, 4, 5)
    identity = torch.eye(2, dtype=torch.float64)

    def read(source, permissions):
        # 本对照 Wq/Wk/Wv/Wo 都是单位矩阵；Q 为零使有效分数相等。
        return multi_head_attention(
            target_state @ identity, source @ identity, source @ identity,
            identity, 1, permissions,
        )

    output, weights = read(memory, allowed)
    expected = memory[:, :3].mean(dim=1, keepdim=True).expand(1, 4, 2)
    close(output, expected)
    assert torch.count_nonzero(weights[..., 3:]).item() == 0

    changed_pad = memory.clone()
    changed_pad[:, 3:] += 1000.
    pad_output, _ = read(changed_pad, allowed)
    close(output, pad_output)

    changed_source = memory.clone()
    changed_source[:, 2, 0] += 3.
    source_output, _ = read(changed_source, allowed)
    close(source_output - output, torch.tensor([1., 0.]).to(output).expand_as(output))

    # 错误对照：目标位置 i 与源位置 j 不属于同一条时间轴。
    wrong_triangle = torch.arange(5).unsqueeze(0) <= torch.arange(4).unsqueeze(1)
    wrong_output, _ = read(memory, allowed & wrong_triangle.unsqueeze(0))
    close(wrong_output[:, 0], memory[:, 0])
    assert not torch.allclose(wrong_output, output)

    print('跨序列输出 shape:', tuple(output.shape))
    print('正确 cross mask，首位置读取:', output[0, 0].tolist())
    print('错误三角 mask，首位置读取:', wrong_output[0, 0].tolist())
    print('源 PAD 特征扰动不影响结果；有效源特征扰动能影响目标首位置：通过')


def target_causal_probe():
    x = torch.tensor([[[.1, .2], [.3, -.1], [.2, .4], [-.2, .5]]], dtype=torch.float64)
    valid = torch.ones(1, 4, dtype=torch.bool)
    identity = torch.eye(2, dtype=torch.float64)
    baseline, _ = multi_head_self_attention(x, identity, identity, identity, identity, valid, 1)
    changed = x.clone()
    changed[:, 2:] += torch.tensor([2., -1.], dtype=x.dtype)
    observed, _ = multi_head_self_attention(changed, identity, identity, identity, identity, valid, 1)
    close(baseline[:, :2], observed[:, :2])
    assert not torch.allclose(baseline[:, 2:], observed[:, 2:])
    print('目标 self-attention：未来输入扰动不影响前两个位置，通过')


def position_probe():
    positions = torch.arange(4)
    pe = sinusoidal_positions(positions, 4)
    print('C=4，位置 0..3 的正弦编码:')
    print(pe)
    close(pe[0], torch.tensor([0., 1., 0., 1.], dtype=pe.dtype))

    # 每对 [sin(p*w), cos(p*w)] 的平方和恒为 1。
    close(pe.reshape(4, 2, 2).square().sum(-1), torch.ones(4, 2, dtype=pe.dtype))
    delta = 3
    frequencies = 10000.0 ** (-torch.arange(0, 4, 2, dtype=pe.dtype) / 4)
    sin_delta = torch.sin(delta * frequencies)
    cos_delta = torch.cos(delta * frequencies)
    shifted_from_pairs = torch.empty_like(pe)
    shifted_from_pairs[:, 0::2] = pe[:, 0::2] * cos_delta + pe[:, 1::2] * sin_delta
    shifted_from_pairs[:, 1::2] = pe[:, 1::2] * cos_delta - pe[:, 0::2] * sin_delta
    close(shifted_from_pairs, sinusoidal_positions(positions + delta, 4))

    # 位置切片必须取原位置编号，不能把新切片错误地重新从 0 开始。
    full = sinusoidal_positions(torch.arange(7), 4)
    close(full[3:7], sinusoidal_positions(torch.arange(3, 7), 4))
    odd = sinusoidal_positions(positions, 5)
    assert odd.shape == (4, 5)
    close(odd[:, 4], torch.sin(positions.to(pe.dtype) / (10000.0 ** (4 / 5))))
    print('固定偏移关系、位置切片、奇数宽度约定：通过')


def main():
    torch.set_num_threads(1)
    torch.set_printoptions(precision=6, sci_mode=False)
    with torch.no_grad():
        cross_attention_probe()
        target_causal_probe()
        position_probe()
    print('教师对照全部通过；未训练完整 Encoder–Decoder。')


if __name__ == '__main__':
    main()
