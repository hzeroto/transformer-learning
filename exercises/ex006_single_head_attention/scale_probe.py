"""教师尺度实验：独立采样的理想模型，不是学习者 Attention 实现。"""

import math

import torch

from exercises.ex006_single_head_attention.attention import raw_attention_scores


def moments(x):
    # x 为一组分数样本；无 dim 的 mean 在这里平均所有样本。
    mean = x.mean()
    variance = ((x - mean) ** 2).mean()
    return mean.item(), variance.sqrt().item()


def probability_sensitivity():
    # 观察一个权重对原始分数的导数，不将这个权重当成语言模型训练目标。
    # 假设 Dk=64，因此缩放分支除以 8。每个分支新建叶子张量，无历史梯度累积。
    for divisor in (1.0, 8.0):
        raw_scores = torch.tensor([0.0, 8.0], dtype=torch.float64, requires_grad=True)
        weights = torch.softmax(raw_scores / divisor, dim=-1)
        weights[1].backward()
        print("divisor:", divisor, "second weight:", weights[1].item())
        print("d(second weight)/d(raw scores):", raw_scores.grad.tolist())


def main():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(41)
    samples = 8192
    # 每个 batch 只有一个 query 和一个 key，batch 之间独立采样。
    # randn 给出均值 0、方差 1 的标准正态样本；不代表训练后的真实 Q/K。
    with torch.no_grad():
        for d in (4, 64, 256):
            q = torch.randn(samples, 1, d, generator=generator, dtype=torch.float64)
            k = torch.randn(samples, 1, d, generator=generator, dtype=torch.float64)
            raw = raw_attention_scores(q, k)  # (samples,1,1)
            scaled = raw / math.sqrt(d)  # math.sqrt 对一个 Python 数字开平方。
            raw_mean, raw_std = moments(raw)
            scaled_mean, scaled_std = moments(scaled)
            print("Dk:", d, "raw mean/std:", raw_mean, raw_std)
            print("scaled mean/std:", scaled_mean, scaled_std)

        row = torch.tensor([0.0, 8.0], dtype=torch.float64)
        print("softmax([0,8]):", torch.softmax(row, dim=-1).tolist())
        print("softmax([0,8] - max):", torch.softmax(row - row.max(), dim=-1).tolist())
        print("softmax([0,8] / 8):", torch.softmax(row / 8, dim=-1).tolist())

    # 在 no_grad 之外执行，才能观察经缩放与 Softmax 回到原始分数的完整梯度。
    probability_sensitivity()


if __name__ == "__main__":
    main()
