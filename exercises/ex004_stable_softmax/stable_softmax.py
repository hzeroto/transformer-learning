"""只用基础张量操作实现沿最后一个轴的稳定 Softmax。"""

import torch


def stable_softmax(scores: torch.Tensor) -> torch.Tensor:
    """返回与 scores 同 shape、dtype、device 的归一化权重。

    输入约定：至少一个轴，各轴长度为正，CPU 上的 float32/float64 Tensor，
    所有输入元素有限。本题不处理 NaN、正负无穷、空输入或 Attention mask。

    沿最后一个轴独立归一化，不修改输入，保留自动求导计算图。
    不能调用 torch.softmax、Tensor.softmax 或其他现成 Softmax 封装。
    """
    # TODO 1：为每组分数取得自己的最大值，保留末尾长度为 1 的轴。
    max_scores = scores.amax(dim=-1, keepdim=True)
    # TODO 2：平移后逐元素取指数。
    # scores -= max_scores
    shifted_scores = scores - max_scores

    # TODO 3：计算每组自己的分母，归一化并返回。
    escore = torch.exp(shifted_scores)
    fm = escore.sum(dim=-1, keepdim=True)
    return escore / fm
