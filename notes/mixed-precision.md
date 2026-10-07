# 数值精度与混合精度

沿用已有模型，观察把部分计算换成 16 位之后，存储、输出和梯度怎样变化。
目标是解释误差、选择计算类型并完成对照；以 Mac CPU 验证数值机制，不规定加速倍数。

## 1. 同样是 16 位，为什么数值表现不同

浮点表示类似科学记数法：有效数字决定细节，指数决定缩放范围；二进制浮点使用 2 的幂。
同样的总位数，可以分配更多位给指数，也可以分配更多位保存有效数字。
因此“能表示很大或很小的数”和“能区分两个很接近的数”是两件事。

下面的范围和间距用 `torch.finfo` 与 `torch.nextafter` 在 CPU 上核对。
`finfo` 返回浮点类型的数值信息；`nextafter(1, 2)` 返回从 1 朝 2 方向的下一个可表示值。

| 类型 | 每元素字节 | 最大有限正数（约） | 1 上方相邻数与 1 的距离 |
|---|---:|---:|---:|
| FP32 | 4 | 3.40 × 10^38 | 0.0000001192 |
| FP16 | 2 | 65504 | 0.0009765625 |
| BF16 | 2 | 3.39 × 10^38 | 0.0078125 |

这些间距只描述 1 附近；浮点数之间的间距会随数值所在区间改变。
BF16 的范围接近 FP32，但相近数的区分能力弱于 FP16，不能把范围大理解为更精确。

当前 CPU / PyTorch 2.14.0 的标量张量对照：

| 操作 | FP32 | FP16 | BF16 |
|---|---:|---:|---:|
| 把 100000 转成该类型 | 100000 | inf | 99840 |
| 把 10^-8 转成该类型 | 约 10^-8 | 0 | 约 10^-8 |
| 该类型中计算 1 + 1/1024 | 1.0009765625 | 1.0009765625 | 1 |

`inf` 是无穷大，此处由超出有限表示范围产生；很小的数也可能小到最终舍入成 0。
舍入是将无法精确表示的结果取成某个可表示值，因此即使没有 `inf` 或 0，也可能有误差。
最后一行的 BF16 结果为 1，是因为这一增量小于它在 1 附近所能分辨的间隔。

## 2. 为什么相同的加法，换个顺序会不同

取三个 BF16 标量：`a=256`、`b=1`、`c=-256`。
数学上两种括号都得到 1，但每次张量加法的结果都会存回对应类型：

```text
(a + b) + c：256 + 1 舍入成 256，随后 256 - 256 = 0
a + (b + c)：1 - 256 得到 -255，随后 256 - 255 = 1
```

BF16 在 256 及其上方的这一段中相邻数相隔 2，257 位于 256 和 258 中间，
按这里的舍入规则得到 256；-255 则可以精确表示。
这说明有限精度加法不能任意改变结合顺序而保证逐位一致。

把多个数合成一个数的求和、均值等操作称为归约；矩阵乘法中的点积也包含求和。
实际算子可能采用比输出类型更高的累加精度，因此不能把上面逐步 BF16 加法的结果
直接当成所有 BF16 矩阵乘法或 `sum` 的行为。应核对实际算子与执行后端。

复现以上现象：

```python
import torch

for dtype in (torch.float32, torch.float16, torch.bfloat16):
    one = torch.tensor(1.0, dtype=dtype)
    next_one = torch.nextafter(one, torch.tensor(2.0, dtype=dtype))
    a, b, c = [torch.tensor(v, dtype=dtype) for v in (256.0, 1.0, -256.0)]
    print(dtype, torch.finfo(dtype).max, float(next_one - one))
    print(float((a + b) + c), float(a + (b + c)))
```

## 3. 从这些现象到混合精度

矩阵乘法可以尝试低精度输入和输出，而数值敏感的计算可保留或显式转成 FP32。
`autocast` 按后端与算子规则选择计算类型；它不会把所有参数永久改成 16 位，
也不能保证自写模块的每一步都自动获得所需精度。已经在低精度中丢失的细节，
事后转回 FP32 不会自动恢复。

下面用 X 的 shape=(2,24)、W 的 shape=(24,64) 演示。两行 X 分别是一份输入特征，
W 是需要学习的投影参数，Y 的 shape=(2,64)。损失暂取输出平方的均值，只用于观察反向路径，
不是模型任务中的交叉熵。`.float()` 转为 FP32；`.square()` 逐元素平方。

```python
import torch

x = torch.randn(2, 24, dtype=torch.float32)
w = torch.randn(24, 64, dtype=torch.float32, requires_grad=True)

with torch.autocast("cpu", dtype=torch.bfloat16):
    y = x @ w
    loss = y.float().square().mean()

loss.backward()
```

`with` 限定 autocast 规则生效的范围。调用矩阵乘法时，输入按规则使用 BF16 临时表示；
原来的 x、w 对象仍为 FP32。反向在 autocast 上下文之外调用，其相关运算沿前向选定的
类型传播，不能把最终参数梯度为 FP32 理解为所有中间梯度都用 FP32。

| 对象 | 本例实际 dtype |
|---|---|
| 原始 x、参数 w | FP32 |
| 矩阵乘法输出 y | BF16 |
| 显式转为 FP32 后计算的 loss | FP32 |
| 最终 w.grad | FP32 |

使用 seed=1403、单线程，固定 x 和初始化为 randn(24,64)/sqrt(24) 的同一份 w，
分别执行 FP32 与 CPU BF16 autocast，当前 PyTorch 2.14.0 上观察到：

| 指标 | 结果 |
|---|---:|
| FP32 loss | 1.17079997 |
| BF16 混合精度 loss | 1.17109430 |
| 输出的最大绝对误差 | 0.01185393 |
| 参数梯度的最大绝对误差 | 0.00078013 |

两次输出和梯度均为有限值。这些只是该小例子的观察，不自动说明某个误差阈值适合全部模型。
完整模型仍需固定权重和输入，对照 logits、loss 及梯度的误差、有限性和实际 dtype。
一次数值对照不代表训练收敛或性能收益。

## 4. loss 正常，梯度也可能已经变成 0

取 X=W=[[1]]，W 的存储为 FP32，令 `Y=X@W`、`loss=Y*10^-8`。
这里 loss 的数值计算用 FP32，数学上的 dloss/dW 为 10^-8。
当乘法使用 FP16 时，反向传向 FP16 中间张量的梯度也需要用对应类型表示；
10^-8 太小，可能舍入为 0。即使最终 w.grad 是 FP32，也无法恢复已经丢失的梯度。

梯度缩放让这个梯度在低精度路径中先保持较大数值。令 S 为正缩放因子：

```text
loss_scaled = loss × S
dloss_scaled/dW = S × dloss/dW
反向完成后：w.grad /= S
随后才用恢复尺度后的梯度更新参数
```

当前 CPU 的 FP16 autocast 也能执行本例。保持输入、参数和损失相同，实测：

| 计算方式 | 最终恢复尺度后的 w.grad |
|---|---:|
| FP32 | 约 10^-8 |
| FP16 autocast，不缩放 | 0 |
| FP16 autocast，S=1024 | 1.00117177 × 10^-8 |

最后一行反向后尚未除以 S 时，梯度约为 1.02519989 × 10^-5。
三种方式未缩放的 loss 均约为 10^-8，只看 loss 是否正常会漏掉这个问题。

以下是隔离机制的手工演示：

```python
import torch

for scale in (1.0, 1024.0):
    w = torch.tensor([[1.0]], dtype=torch.float32, requires_grad=True)
    x = torch.tensor([[1.0]], dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.float16):
        y = x @ w
        loss = y.float().sum() * 1e-8

    (loss * scale).backward()
    w.grad.div_(scale)  # 最终梯度为 FP32，在这里恢复原尺度
    print(scale, loss.item(), w.grad.item())
```

完整 FP16 训练通常让 `GradScaler` 管理缩放、非有限梯度检查及缩放因子的调整。
缩放过大也会导致溢出；它不能恢复前向已经丢失的信息，也不能替代数值检查。
BF16 的范围较宽，通常不需要为同样的小梯度使用缩放，但它仍有舍入误差。

把这些机制放到原模型上验证，见[综合练习：FP32 / BF16 数值对照](../exercises/ex015_mixed_precision/README.md)。
完成一次指定精度的前向和反向，再比较 logits、loss、全部参数梯度；模型与交叉熵继续复用。

## 5. GradScaler 加在训练循环的哪里

完整教师示例见 [gradient_scaling_demo.py](../examples/gradient_scaling_demo.py)，
复用 ex015 模型和数据，在 CPU 上使用 FP16 autocast，对同一个 batch 更新三次。
这是可直接阅读、运行的参考代码，不增加练习任务；ex015 本身仍做 FP32 / BF16 对照。

在仓库根目录运行：

```bash
.venv/bin/python -B -m examples.gradient_scaling_demo
```

`scaler = torch.amp.GradScaler("cpu")` 在训练循环外创建一次。每步先清梯度，
再在 autocast 内完成前向和 FP32 loss，随后执行：

```python
scaler.scale(loss).backward()  # 替换 loss.backward()，对放大后的 loss 求导
scaler.step(optimizer)        # 还原梯度尺度，检查后决定是否调用 optimizer.step()
scaler.update()               # 调整下一轮缩放倍数，不是更新模型参数
```

`step` 检测到梯度中有 NaN/inf 时会跳过参数更新，`update` 随后通常减小缩放倍数；
连续正常一定步数后，缩放倍数可以增大。它不能保证解决所有数值异常。

若要在参数更新前检查、保存梯度或做梯度裁剪，在 `backward()` 之后、`step()` 之前
调用 `scaler.unscale_(optimizer)`。此后 `.grad` 才是恢复尺度后的梯度；
`scaler.step()` 会记住已经还原过，不会重复除以缩放倍数。示例中已标出插入位置。

Mac CPU / PyTorch 2.14.0 的三步 loss 约为 `2.8501 → 2.7272 → 2.6191`，
缩放倍数保持 `65536`。这些数值仅说明本例的运行结果，不是长期收敛或性能结论。

参考：[PyTorch 数值精度](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)、
[自动混合精度与梯度缩放](https://docs.pytorch.org/docs/2.14/amp.html)、
[GradScaler 训练与梯度裁剪示例](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)。
