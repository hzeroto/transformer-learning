"""第三部分：用中心差分检查一个参数元素，由学习者实现核心计算。"""

import torch

from exercises.ex005_training_loop.loss import masked_cross_entropy
from exercises.ex005_training_loop.training import forward_logits


def finite_difference_one(
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    target_valid: torch.Tensor,
    E: torch.Tensor,
    W: torch.Tensor,
    parameter_name: str,
    index: tuple[int, int],
    h: float = 1e-5,
) -> float:
    """返回指定参数元素的中心差分近似，结果为 Python float。

    输入沿用现有前向和损失函数的约定，但 E、W 固定为 CPU float64。
    parameter_name 为 "E" 或 "W"，index 是该矩阵中的合法 (行,列)。
    h 是有限正数。上述约定不要求额外校验。
    E、W 可以需要求导，也可以不需要；已有 .grad 可以是 None 或任意数值。

    语义：
        只把选定参数元素在副本中分别加 h、减 h，其余参数和数据固定。
        各自复用 forward_logits 与 masked_cross_entropy 重算完整前向。
        返回 (loss_plus - loss_minus) / (2 * h)。
        当 parameter_name 为 "E" 时，修改的是 embedding 参数，
        不是某个序列位置的中间结果；重复 ID 的所有使用位置应一起受影响。

    无副作用约束：
        数值探测在 torch.no_grad() 范围内执行，只改参数副本。
        不改变原参数、数据、requires_grad 状态或原有 .grad。
        损失计算抛出 ValueError 时，原参数及梯度仍须保持不变。

    禁止读取 .grad 当作答案，禁止 backward、autograd.grad、gradcheck
    或调用 train_step；这是独立的数值检查，不执行 SGD 更新。
    不需要计算整个梯度矩阵，不负责判断近似值是否通过容差。
    """
    with torch.no_grad():
        E_copy_add = E.clone()
        E_copy_min = E.clone()
        W_copy_add = W.clone()
        W_copy_min = W.clone()
        if parameter_name == "E":
            E_copy_add[index] += h
            E_copy_min[index] -= h
        elif parameter_name == "W":
            W_copy_add[index] += h
            W_copy_min[index] -= h
        else:
            raise ValueError("parameter_name must be 'E' or 'W'")
        logits_add = forward_logits(input_ids, E_copy_add, W_copy_add)
        logits_min = forward_logits(input_ids, E_copy_min, W_copy_min)
        loss_add = masked_cross_entropy(logits_add, targets, target_valid)
        loss_min = masked_cross_entropy(logits_min, targets, target_valid)
        finite_diff = (loss_add - loss_min) / (2 * h)
    return finite_diff.item()
