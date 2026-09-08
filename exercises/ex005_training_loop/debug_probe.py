"""教师提供的诊断入口：展示观测值，不代替学习者编写回归测试。"""

import math

import torch

from exercises.ex002_batch_padding.batch_padding import make_batch
from exercises.ex005_training_loop.debug_case import candidate_train_step
from exercises.ex005_training_loop.gradient_check import finite_difference_one
from exercises.ex005_training_loop.loss import masked_cross_entropy
from exercises.ex005_training_loop.training import forward_logits, prepare_next_token_batch


def make_debug_case():
    """返回新参数 E、W 和两份不同的 batch；可在学习者测试中复用。"""
    torch.manual_seed(17)
    E = (0.15 * torch.randn(5, 4, dtype=torch.float64)).requires_grad_(True)
    W = (0.15 * torch.randn(4, 5, dtype=torch.float64)).requires_grad_(True)
    first_ids, first_valid = make_batch([[1, 2, 3, 4], [2, 3, 4]], pad_id=0)
    second_ids, second_valid = make_batch([[2, 3, 4], [3, 4]], pad_id=0)
    batches = (
        prepare_next_token_batch(first_ids, first_valid),
        prepare_next_token_batch(second_ids, second_valid),
    )
    return E, W, batches


def main():
    torch.set_num_threads(1)
    E, W, batches = make_debug_case()
    lr = 0.2
    print("诊断输出，不是自动验收结果：")

    for step, data in enumerate(batches, start=1):
        inputs, targets, valid = data
        # 检查点和差分值都对应本次更新之前的同一组参数。
        before_E = E[2, 0].item()
        before_W = W[0, 2].item()
        numeric_E = finite_difference_one(inputs, targets, valid, E, W, "E", (2, 0))
        numeric_W = finite_difference_one(inputs, targets, valid, E, W, "W", (0, 2))

        reported_loss = candidate_train_step(inputs, targets, valid, E, W, lr)
        with torch.no_grad():
            after_loss = masked_cross_entropy(
                forward_logits(inputs, E, W), targets, valid
            ).item()

        # 由 SGD 公式反推本次实际更新所使用的方向和幅度。
        observed_E = (before_E - E[2, 0].item()) / lr
        observed_W = (before_W - W[0, 2].item()) / lr
        print("step:", step, "loss before:", reported_loss, "loss after:", after_loss)
        print(
            "E[2,0] numeric:", numeric_E,
            "from update:", observed_E,
            "close:", math.isclose(numeric_E, observed_E, rel_tol=1e-5, abs_tol=1e-8),
        )
        print(
            "W[0,2] numeric:", numeric_W,
            "from update:", observed_W,
            "close:", math.isclose(numeric_W, observed_W, rel_tol=1e-5, abs_tol=1e-8),
        )

    print("请根据证据定位根因，自行编写回归测试，再最小修复候选版本。")


if __name__ == "__main__":
    main()
