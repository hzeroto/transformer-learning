"""教师提供的运行入口；核心计算由 training.py 中的学习者函数完成。"""

import torch

from exercises.ex002_batch_padding.batch_padding import make_batch
from exercises.ex005_training_loop.loss import masked_cross_entropy
from exercises.ex005_training_loop.training import (
    forward_logits,
    prepare_next_token_batch,
    train_step,
)


def main():
    # 固定随机种子，让每次运行从相同参数开始。
    torch.manual_seed(17)
    torch.set_num_threads(1)

    # PAD=0、BOS=1、A=2、B=3、EOS=4。
    # 第二条是从 A 开始的片段。所有受监督的后继均一致：
    # BOS -> A，A -> B，B -> EOS，不要求模型学习 EOS -> PAD。
    ids, valid = make_batch([[1, 2, 3, 4], [2, 3, 4]], pad_id=0)
    input_ids, targets, target_valid = prepare_next_token_batch(ids, valid)

    # randn 创建随机初始数值；先缩放，再声明它们是待求导的参数。
    E = (0.15 * torch.randn(5, 4, dtype=torch.float64)).requires_grad_(True)
    W = (0.15 * torch.randn(4, 5, dtype=torch.float64)).requires_grad_(True)

    with torch.no_grad():
        initial = masked_cross_entropy(
            forward_logits(input_ids, E, W), targets, target_valid
        ).item()
    print("initial loss:", initial)

    # range(160) 依次产生 0 到 159；这些是参数更新次数，不是课程安排。
    for step in range(160):
        before_update = train_step(input_ids, targets, target_valid, E, W, lr=0.5)
        if step in (0, 19, 79, 159):
            print("update", step + 1, "loss before this update:", before_update)

    # 这里只读取更新后的结果，不进行训练，因此不需要记录计算图。
    with torch.no_grad():
        logits = forward_logits(input_ids, E, W)
        final = masked_cross_entropy(logits, targets, target_valid).item()
        # argmax 返回末轴最高分的候选下标，也就是这里的预测 token ID。
        predictions = logits.argmax(dim=-1)

    print("final loss:", final)
    print("valid targets:", targets[target_valid].tolist())
    print("valid predictions:", predictions[target_valid].tolist())
    print("This checks learned token transitions, not general language ability.")


# 作为 python -m 的运行入口时执行；被其他文件导入时不自动训练。
if __name__ == "__main__":
    main()
