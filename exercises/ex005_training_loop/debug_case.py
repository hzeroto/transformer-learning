"""第四部分的候选实现：刻意保留一处逻辑故障，仅用于独立排错练习。"""

import torch

from exercises.ex005_training_loop.loss import masked_cross_entropy
from exercises.ex005_training_loop.training import forward_logits


def candidate_train_step(
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    target_valid: torch.Tensor,
    E: torch.Tensor,
    W: torch.Tensor,
    lr: float,
) -> float:
    """目标契约与已完成的 train_step 相同，但本候选版本尚未满足契约。

    应只使用本批数据的梯度，原地对 E、W 各执行一次基础 SGD 更新，
    返回更新前的平均有效标签损失，且不修改输入数据。

    请用实验和回归测试定位后作最小修复；不要直接调用已通过的
    train_step 来绕过此候选函数，也不要修改原来的正确实现。
    """
    E.grad = None
    W.grad = None
    logits = forward_logits(input_ids, E, W)
    loss = masked_cross_entropy(logits, targets, target_valid)
    loss.backward()
    with torch.no_grad():
        E -= lr * E.grad
        W -= lr * W.grad
    return loss.item()
