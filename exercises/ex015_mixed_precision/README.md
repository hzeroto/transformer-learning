# 015：同一个模型的 FP32 / BF16 数值对照

复用已经写好的 LLaMA 风格模型，在同一份权重和下一 token 输入上执行两次前向、
loss 和反向，解释降低精度改变了哪些结果。[precision.py](precision.py) 的两个函数
已按学习者请求提供正确版，供对照理解；原接口契约保留。
[模型与数据](fixtures.py)、[打印入口](demo.py) 和测试也由教师提供。

先修机制与例子见[混合精度讲义](../../notes/mixed-precision.md)。
本练习使用当前 Mac CPU 的 BF16 autocast；FP16 梯度缩放的机制已单独讲解，
不要求在 BF16 实验里添加缩放，不以加速倍数作为目标。

需要回看 FP16 训练中梯度缩放的调用位置，可直接阅读和运行
[GradScaler 教师演示](../../examples/gradient_scaling_demo.py)；运行命令及解释见
[讲义第 5 节](../../notes/mixed-precision.md#5-gradscaler-加在训练循环的哪里)，无需补写。

## 两个函数的分工

1. **run_precision_pass**：清除旧梯度，选择 FP32 或 CPU BF16 autocast，调用原模型；
   把 logits 转为 FP32 后调用已有交叉熵，反向一次，返回 logits、loss、全部参数梯度快照。
2. **compare_results**：报告 logits 和梯度的最大绝对误差、loss 的绝对差；
   检查两份结果是否都只有有限数值。

最大绝对误差就是先按元素做差、取绝对值，再取最大值。先把结果转成 FP32 再比较。
NaN 表示无效数值，inf 表示无穷；两者都不属于有限数。
若某一组含非有限数，该组误差报告为 inf，有限性报告为 False，
不通过忽略异常元素来把实验说成成功。

这次不更新参数。两次运行共用原模型，前一轮留下的梯度必须清掉；
返回的梯度应独立存储，避免下一次清梯度或修改梯度影响已经保存的结果。
不重新写模型、交叉熵或梯度缩放器，不把整个模型强制转换成 BF16。

## 本次对象与约束

默认配置：B=2、T=6、C=24、Hq=6、Hkv=2、G=64、两层、词表 N=17，
位置容量 32，seed=1503。参数为 CPU FP32，模型无 Dropout，使用全量前向。

| 对象 | 来源与 shape |
|---|---|
| input_ids | 两条序列去掉最后一个编号，CPU long (B,T) |
| targets | 同两条序列去掉第一个编号，CPU long (B,T)，已经与输入错位对齐 |
| input_valid | 模型可读取的 key，CPU bool (B,T)，True 表示可读取 |
| target_valid | 参与交叉熵的标签，CPU bool (B,T)，True 表示参与 |
| logits | 原模型输出 (B,T,N)，保留原输出 dtype 保存快照 |
| loss | 复用 ex005 masked_cross_entropy，FP32 标量 |
| grads | 参数名到同 shape 的 FP32 梯度快照，覆盖全部可训练参数 |

函数签名及精确返回键见 precision.py。输入保证合法，不考通用校验。
输入可能非连续；不得修改输入、模型参数、参数 dtype 或 train/eval 模式。
允许修改模型的 .grad。eval 并不关闭求导，此实验仍需要反向。

## API 备忘

- torch.autocast("cpu", dtype=torch.bfloat16, enabled=...)：开启或关闭本段的自动混合精度。
- model.zero_grad(set_to_none=True)：清掉上次梯度；不改变参数。
- logits.float()：转成 FP32，保留计算图；不是恢复已丢失的精度。
- model.named_parameters()：逐个得到参数名和参数，参数梯度位于 .grad。
- detach().clone()：得到无计算图、独立存储的快照，应在所需反向完成后使用。
- torch.isfinite(tensor)：逐元素检查是否既不是 NaN 也不是 inf。
- tensor.item()：把单元素 Tensor 转成 Python 标量。

原 ex013 的 float32/64 正确性契约继续有效。这里在它外层增加 autocast 对照，
不据此宣称内部所有步骤都转为 BF16，或认为自写归一化、RoPE 自动采用最安全的精度。
打印入口会额外观察第一层 FFN 的输出 dtype，不要求你编写观察代码。

## 运行与完成标准

在仓库根目录：

~~~bash
.venv/bin/python -B -m unittest tests.exercises.test_mixed_precision -v
.venv/bin/python -B -m exercises.ex015_mixed_precision.demo
~~~

需要保存报告时：

~~~bash
.venv/bin/python -B -m exercises.ex015_mixed_precision.demo --output /tmp/mixed-precision.json
~~~

当前正确版的 13 项测试已全部实际执行通过、无跳过，demo 也已完成两种精度的对照。
这是教师实现的验证结果，不自动等同于学习者独立实现或掌握。
教师测试通过比较同一种精度策略的独立 loss/梯度参照检查实现；
不会要求 BF16 与 FP32 逐位相同。默认比较同策略浮点结果使用 rtol=1e-5、atol=1e-6。

最后结合自己的报告说明：哪些对象保持 FP32、哪些输出变为 BF16；
两次 logits、loss、梯度是否有限、差异多大；为什么仅看 loss 正常还不够。
本次只有固定 batch 的数值证据，不代表长期训练收敛、所有 shape 安全或 CUDA 性能。
老师提供的数值对照、测试和实验入口不计作学习者独立设计。

脚手架交付自检：隔离的教师参照通过 13 项测试、无跳过；禁用 autocast、忘记清梯度、
低精度 loss、梯度快照共享存储、只比较最后一个参数梯度、忽略非有限数这 6 类错误均被识别。
当时已有交叉熵、LLaMA 组件与模型的 47 项回归通过。
当前两个核心函数已由教师按请求补全；阅读时重点比较精度上下文、反向顺序、
参数与梯度的区别，以及 Tensor 归约与 Python 标量之间的转换。
