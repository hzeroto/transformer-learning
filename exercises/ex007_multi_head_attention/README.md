# 作业 007：逐步构造多头 Attention

## 当前验收状态

学习者已完成 `heads.py` 中的 `split_heads` 和 `merge_heads`，教师未修改实现。
实际实现通过全部 11 项布局测试及全仓 125 项回归测试，无跳过。
已验证独立元素映射、连续/转置/步长切片输入、头数校验、输入不被修改和梯度回传。
这表示拆头与合头的实现通过，不代表完整多头 Attention 已完成，也不代表学习者独立设计了全部测试。

教师另以同一组元素检查实际存储：拆头后立即合头可以共享原存储；
先生成连续的 `(B,H,T,Dh)` 输入再合头，该样例中的 `reshape` 会复制，元素仍然正确。
这只是教师诊断，不作为学习者已独立解释复制条件的证据；该能力留待后续多头集成中验证。

当前任务已推进到第二部分：完整 MHA。新增 attention.py 的两个函数待学习者填写，
新增 15 项综合测试；教师临时参照通过新 15 项及全仓 140 项，不是学习者实现证据。
实际空框架的两个未实现失败、两个行为类跳过均为预期；完成后应 15 项通过、无跳过。

## 当前任务在整体计算中的位置

ex006 已完成单头 Attention 的投影、匹配、缩放、mask 和读取。
多头将同一份输入投影出各头的 Q/K/V，分别读取后再合并；已完成的布局转换位于：

| 位置 | 本次接口 | 数据语义 |
|---|---|---|
| Q/K/V 投影之后 | `split_heads` | 把每个 token 的投影特征按 head 分组 |
| 每个 head 的读取之后 | `merge_heads` | 把同一个 token 的各头输出按 head 顺序拼接 |

当前将完整多头计算、mask 的 head 轴、输出投影 Wo 合成一份实现任务，见下方第二部分。
这两个布局函数不创建参数、不修改特征数值、不做 Attention，也不把已有基线改成完整语言模型。
ex005 的逐位置训练模型与 ex006 的单头组件目前仍未端到端集成。

## 两个函数的任务约定（实现已通过）

### split_heads(x, num_heads)

```text
输入 x:     (B,T,C)
头数 H:     num_heads
每头宽度:   Dh=C//H
输出:       (B,H,T,Dh)

result[b,h,t,d] = x[b,t,h*Dh+d]
```

x 是通用参数名，代表已投影的 Q/K/V 等特征；不要在这个函数里再乘 Wq/Wk/Wv。
检查 H>0 且 C 能被 H 整除，否则抛出 ValueError。
这里 C 是传入张量的总特征宽度，不强行绑定到原模型输入的宽度。

### merge_heads(heads)

```text
输入 heads: (B,H,T,Dh)
输出:       (B,T,H*Dh)

result[b,t,h*Dh+d] = heads[b,h,t,d]
```

输入可能来自各头计算的新结果，不保证是 split_heads 刚返回的那个视图。
先按“同一个 token 的各个 head”组织，再合并特征，不对 head 求和或平均。

两函数都只用基础 Tensor 操作完成，保留 dtype/device 和自动求导路径，不修改输入或已有梯度。
支持 CPU float32/float64、各轴长度为正、连续与非连续输入。
除此之外不要求通用输入类型/轴数校验，也不要求 GPU 支持。
输出可以共享存储也可以复制，不要求特定 stride 或强制连续；调用者不能假定输出是独立副本。
可以查看[布局讲义](../../notes/tensor-layout.md)，不用背 API，也不需要重写 ex006。

## 用一组数核对含义，不要求再次手算

输入 `x[0]=[[0,1,2,3],[4,5,6,7],[8,9,10,11]]`，H=2：

| token 位置 | head 0 应取的特征 | head 1 应取的特征 |
|---|---|---|
| 0 | `[0,1]` | `[2,3]` |
| 1 | `[4,5]` | `[6,7]` |
| 2 | `[8,9]` | `[10,11]` |

每个 head 保留全部 token，区别是取哪组投影特征。
测试会独立检查两函数的元素对应关系，不只检查 shape 或往返恢复。

## 非连续输入与本次语法提示

已讲过的 transpose 可以生成非连续视图；切片也可能产生非连续视图：

```python
x = base[:, :, 1:9:2]
```

`start:stop:step` 是 Python 的步长切片语法，包含 start、不包含 stop；
这里取最后一轴的下标 1、3、5、7。x 的最后一轴长度是 4，
但相邻逻辑元素在底层并不相邻，而且视图不是从底层第 0 个元素开始。
函数应处理 x 的逻辑元素，不去猜底层数据如何排列。

- `C // H` 是整数除法；本题先保证整除，结果就是每头宽度。
- `C % H` 是余数，可用于检查能否整除；先处理 H<=0，避免除零。
- `reshape` 可能复制但不会因此切断 autograd；`view` 要求目标形状与现有 stride 兼容。
- `contiguous()` 已经连续时返回自身，否则按当前逻辑顺序复制；不要为了通过反向测试而禁止必要复制。
- 教师测试中的 `torch.stack` 沿新轴堆叠，`torch.cat` 沿已有轴拼接；它们只用于独立构造参照，不要求你按参照实现。

## 运行与验收

在当前设备的项目仓库根目录运行：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_head_layout -v
```

本题 11 项测试，包括：精确索引、单元素扰动、不同 B/T/H/Dh、H=1 和 Dh=1、
连续/转置/步长切片输入、非法 head 数、两方向往返，以及回到原始输入存储对应 Tensor 的梯度。
合头测试使用独立生成的每头输出，不能依赖错误拆头与错误合头互相抵消。
固定随机种子 71/73/79。纯布局的前向数值要求精确相等；
梯度测试使用 CPU float64，rtol=1e-10、atol=1e-12。

空框架会出现两个明确的未实现失败，三个行为测试类暂时跳过。
完成标准是 11 项全部通过、无跳过，不能把跳过视为完成。
教师自检的临时答案只在内存使用，不写入你的函数，也不作为学习者证据。

全仓回归入口：

```bash
.venv/bin/python -B -m unittest discover -s tests -t .
```

实现 review 已通过。后续结合多头集成继续验证实际使用的 view/reshape/contiguous
在哪些布局下需要复制，不重复上次的错误 reshape 口头题。
如果只跑通了测试，也不能据此声称已独立完成所有测试设计或完整多头模型。

## 第二部分：完整多头 Attention（当前任务）

先看[完整数据流与 Wo 讲义](../../notes/multi-head-attention.md)。
本次不重做拆轴口头题，只补 attention.py：

| 函数 | 输入与职责 | 输出 |
|---|---|---|
| multi_head_attention | 已投影 Q(B,Tq,C)、K/V(B,Tk,C)，Wo(C,C)，H；按头读取并输出投影 | output(B,Tq,C)，weights(B,H,Tq,Tk) |
| multi_head_self_attention | X(B,T,C)，四个 (C,C) 参数矩阵，input_valid(B,T)，H；投影、生成因果权限、调用上一函数 | output(B,T,C)，weights(B,H,T,T) |

采用 C=H*Dh，缩放用 sqrt(Dh)，按 key 轴 Softmax。
通用函数的 allowed 为 (B,Tq,Tk) 或 None，各头共享；None 不添加任何隐式因果限制。
在内部用 unsqueeze(1) 补 head 轴；不用 repeat 复制权限。
保留既有全屏蔽行 ValueError 规则，不在 query PAD 处强制输出零。
复用 heads.py 与 ex006 的 project_qkv/make_causal_allowed，既有答案无需修改。
第一层用批量 Tensor 操作，不写 batch/head/token 循环；不调用高级或融合 Attention。
具体输入、dtype、非连续支持和梯度约定见函数 docstring。

```bash
.venv/bin/python -B -m unittest tests.exercises.test_multi_head_attention -v
```

15 项综合测试覆盖不同长度/头数/布局、每头缩放和分布、B==H 时静默 mask 轴错配、
Wo 只改变内容、单头参数子空间隔离、未来/PAD/前缀可见性、全屏蔽行拒绝，
以及 Q/K/V/Wo 和 X/Wq/Wk/Wv/Wo 两条接口的梯度。
CPU float32 前向容限 rtol=1e-5、atol=1e-6；
float64 前向及梯度 rtol=1e-10、atol=1e-12，固定种子 83/89/97/101。
教师参照逐头切片、逐 query 选择可读 key，不依赖待测拆合头或四维广播路径。
独立参照只用于测试，不填进学习者函数；完成时要求无跳过。

实现完成后一起 review 组合语义和边界，不再为每个计算步骤各交一次作业。
