"""教师演示：在 FP16 训练循环中使用 GradScaler，不需要学习者补写。

在仓库根目录运行：
    .venv/bin/python -B -m examples.gradient_scaling_demo

复用 ex015 的 FP32 模型和固定 batch，在 CPU 上执行三次参数更新。
当前 Mac / PyTorch 2.14.0 已验证此路径；用于观察调用顺序，不是性能基准，
也不能用同一个 batch 的三步 loss 下降证明长期训练收敛。
"""

import torch

from exercises.ex005_training_loop.loss import masked_cross_entropy
from exercises.ex015_mixed_precision.fixtures import make_case


def main():
    torch.set_num_threads(1)
    model, batch = make_case()  # 参数本体为 FP32；前向的部分运算使用 FP16。
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # 在循环外创建一次，保留跨训练步的缩放倍数和调整状态。
    scaler = torch.amp.GradScaler("cpu")

    print(f"device=cpu, torch={torch.__version__}, 同一 batch 更新三次")
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        scale_before = scaler.get_scale()

        with torch.autocast("cpu", dtype=torch.float16):
            logits = model(batch["input_ids"], batch["input_valid"])
            loss = masked_cross_entropy(
                logits.float(), batch["targets"], batch["target_valid"]
            )

        # ① 替换 loss.backward()：对 loss × S 求导，梯度也被放大 S 倍。
        # 反向放在 autocast 上下文之外；中间梯度仍会经过低精度路径。
        scaler.scale(loss).backward()

        # 可选：如果要在更新前比较、保存梯度或做梯度裁剪，先还原尺度。
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # 裁剪限制梯度的整体大小；必须对恢复尺度后的梯度操作。

        # ② 替换 optimizer.step()：还原梯度尺度，检查 NaN/inf。
        # 梯度有限才调用 optimizer.step()，否则跳过本次参数更新。
        # 若已显式 unscale_，这里会记住，不会再次除以 S。
        scaler.step(optimizer)

        # ③ 调整后续训练步的 S，不是再次更新模型参数。
        # 遇到非有限梯度通常减小 S；连续正常一定步数后可以增大 S。
        scaler.update()

        print(
            f"step={step + 1} loss={loss.item():.6f} "
            f"scale={scale_before:g}->{scaler.get_scale():g} "
            f"logits={logits.dtype} loss_dtype={loss.dtype}"
        )


if __name__ == "__main__":
    main()
