"""训练闭环的第二部分：学习者实现三个函数，复用已有交叉熵。"""

import torch

from exercises.ex005_training_loop.loss import masked_cross_entropy


def prepare_next_token_batch(
    ids: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按下一 token 目标，返回 input_ids、targets、target_valid。

    输入：
        ids: (B,S)，CPU torch.long，B >= 1，S >= 2，所有 ID 合法。
        valid: (B,S)，CPU torch.bool，每行至少一个真实 token。
        每行的 True 连续位于左侧，False 仅用于右侧补齐。
        valid 是有效性的唯一依据，不根据固定 PAD ID 推断。

    输出：
        三个张量均为 (B,S-1)，前两个 long，第三个 bool。
        输入位置 t 预测原序列 t+1 的 token；标签标记与该目标对齐。
        EOS 是有效目标；EOS 后的 PAD 不是。
        允许输出全部无效的 target_valid，由损失函数决定如何处理。
        不修改输入；允许返回切片视图，不要求复制。

    不要求额外的输入校验，也不负责添加 BOS/EOS 或再次补齐。
    """
    inputs = ids[:, :-1]  # (B, S-1)
    targets = ids[:, 1:]  # (B, S-1)
    target_valid = valid[:, 1:]  # (B, S-1)
    return inputs, targets, target_valid


def forward_logits(
    input_ids: torch.Tensor,
    E: torch.Tensor,
    W: torch.Tensor,
) -> torch.Tensor:
    """只根据当前位置 token，计算每个词表候选的原始分数。

    input_ids: (B,T)，CPU long，所有 ID 在 [0,N) 内。
    E: (N,C)，W: (C,N)，CPU 上相同的 float32 或 float64 dtype。
    B、T、N、C 均为正数，不要求额外的输入校验。

    返回 (B,T,N)，dtype/device 与 E 相同，不修改任何输入。
    查表后使用共享 W 做矩阵乘法，不加 bias、位置、Softmax 或 Attention。
    参数需要求导时保留计算图；参数不需要求导时也应能正常计算。
    """
    real_E = E[input_ids]  # (B, T, C)
    logits = real_E @ W  # (B, T, N)
    return logits


def train_step(
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    target_valid: torch.Tensor,
    E: torch.Tensor,
    W: torch.Tensor,
    lr: float,
) -> float:
    """执行一次基础 SGD 更新，返回更新前 loss 的 Python float。

    输入张量沿用前两个函数和 masked_cross_entropy 的约定。
    E、W 是直接创建后开启 requires_grad 的独立参数 Tensor，
    并非其他需要求导的 Tensor 的计算结果。调用方持有它们。
    lr 为有限正数，本题不要求额外校验。

    要求：
        复用 forward_logits 与已经实现的 masked_cross_entropy。
        本次更新只使用本批数据的梯度，不混入旧的 .grad。
        对 E、W 各做一次：新参数 = 旧参数 - lr * 本次梯度。
        在 no_grad 范围内原地更新，调用方持有的同一张量必须改变。
        保留参数之后继续参与求导的能力；不修改输入数据或标签。
        .grad 在函数返回时可以保留本次梯度，也可以已清理。
        返回更新前 loss 的 float 数值；不要把 Tensor/计算图交给日志。
        整个 batch 无有效标签时传播 ValueError，不修改 E、W 的数值。

    禁用 torch.optim、nn.Linear、nn.Embedding 及通过 .data 绕过 autograd。
    """
    # TODO 1：清理旧梯度。
    E.grad = None
    W.grad = None
    # TODO 2：使用当前 E、W 计算 logits 和 loss。
    logits = forward_logits(input_ids, E, W)
    loss = masked_cross_entropy(logits, targets, target_valid)
    # TODO 3：反向传播，并在不记录计算图的范围内更新两个参数。
    loss.backward()
    with torch.no_grad():
        E -= lr * E.grad
        W -= lr * W.grad
    # TODO 4：返回更新前损失的 Python 数字。
    return loss.item()
