"""教师提供训练后生成的入口；生成循环由学习者实现。"""

import torch

from exercises.ex002_batch_padding.batch_padding import make_batch
from exercises.ex005_training_loop.generation import greedy_generate
from exercises.ex005_training_loop.training import (
    forward_logits,
    prepare_next_token_batch,
    train_step,
)


def main():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    # 沿用已经通过的训练演示：PAD=0、BOS=1、A=2、B=3、EOS=4。
    ids, valid = make_batch([[1, 2, 3, 4], [2, 3, 4]], pad_id=0)
    inputs, targets, target_valid = prepare_next_token_batch(ids, valid)
    E = (0.15 * torch.randn(5, 4, dtype=torch.float64)).requires_grad_(True)
    W = (0.15 * torch.randn(4, 5, dtype=torch.float64)).requires_grad_(True)
    for step in range(160):
        train_step(inputs, targets, target_valid, E, W, lr=0.5)

    # 生成函数只接收前缀和已训练参数；不传标签，也不访问训练序列。
    prefix = torch.tensor([[1]], dtype=torch.long)
    output = greedy_generate(prefix, E, W, eos_id=4, max_new_tokens=8)
    print("prefix:", prefix.tolist())
    print("generated (including prefix):", output.tolist())
    print("expected for this toy training set:", [[1, 2, 3, 4]])

    # 前面的 token 不同，但末 token 相同。当前模型能否区分？
    # 刻意选择等长前缀，避免把区别误归因于当前位置。
    context_a = torch.tensor([[1, 2]], dtype=torch.long)
    context_b = torch.tensor([[3, 2]], dtype=torch.long)
    with torch.no_grad():
        last_a = forward_logits(context_a, E, W)[:, -1, :]
        last_b = forward_logits(context_b, E, W)[:, -1, :]
    # abs 是逐元素绝对值；max 把最大差值取出；这里只打印观察量。
    print("different prefixes, same final token:", context_a.tolist(), context_b.tolist())
    print("maximum last-logit difference:", (last_a - last_b).abs().max().item())
    print("This probes the current baseline, not a general property of Transformers.")


if __name__ == "__main__":
    main()
