# 作业 005：最小训练闭环

这份作业随教学逐步扩展，最终把输入、输出分数、损失、反向传播和更新串起来。
第一部分 `loss.py` 中的 `masked_cross_entropy` 已通过验收，不需要重写。
当前完成第二部分：`training.py` 中的三个函数，把标签构造、前向计算和一次 SGD 更新接起来。
教师提供测试和 `demo.py` 运行入口，核心函数仍由学习者实现。

## 第一部分：批量稳定交叉熵

```text
logits:       (B, T, N)  float32 或 float64
targets:      (B, T)     torch.long，正确候选 ID
target_valid: (B, T)     torch.bool，标签是否参与损失

返回 loss:    ()         与 logits 同 dtype、device 的标量 Tensor
```

数学与 API 均已在讲解中出现。本题将它们组合起来，不要求手写梯度。
只修改待实现函数，不修改测试来迎合实现。

## 输入约定和行为要求

- 输入均在 CPU 上，B、T、N 为正数，分数均为有限值。
- 所有 targets 均为合法词表 ID，包括无效标签位置；本题不使用 -100 等越界占位值。
- `target_valid` 已由调用方与标签对齐，本函数不生成或移动标签，也不推断 PAD ID。
  ID 0 可以是有效答案，也可以出现在无效位置，最终以布尔标记为准。
- 所有有效 token 权重相同：有效损失之和除以有效标签数量。
  不先逐序列求平均；补更多 PAD 不得改变损失。
- 某一条序列全部无效可以正常处理；整个 batch 全部无效时抛出 `ValueError`。
- 不修改任何输入，保留自动求导路径，返回 Tensor 而不是 Python 数字。
- 使用基础张量操作手写稳定公式。不可调用 `cross_entropy`、`nll_loss`、
  `softmax`、`log_softmax`、`logsumexp` 或对应的现成封装；不可用截断概率代替原目标。
  官方交叉熵只在测试中作为参考。
- 可以查下方 API 提示；命名熟练度不是本题验收目标。

## 计算语义

对每个固定的 `(b, t)`，将这一组的每个分数减去该组最大值，得到 z：

```text
该位置的损失 = log(sum_j exp(z[j])) - z[targets[b, t]]

correct_scores[b, t] = z[b, t, targets[b, t]]
```

先得到逐位置损失，再根据 `target_valid` 选出需要平均的元素。
也允许先筛选有效位置再计算，只要满足相同数值、归约和梯度要求。

## 已学 API 速查

```python
x.amax(dim=-1, keepdim=True)  # 沿末轴取最大值，保留长度为 1 的轴
torch.exp(x)                 # 逐元素计算指数
x.sum(dim=-1)                # 沿末轴求和，去掉末轴
torch.log(x)                 # 自然对数
targets.unsqueeze(-1)       # (B,T) → (B,T,1)
x.gather(dim=-1, index=idx)  # 沿候选轴按 idx 取值，输出形状与 idx 相同
x.squeeze(-1)               # 去掉末尾长度为 1 的轴
x[target_valid]             # 同形状布尔索引，选出 True 对应的元素
x.numel()                   # 元素总数
x.mean()                    # 全部元素的算术平均
raise ValueError("说明")     # 抛出明确错误
```

## 验收重点

测试覆盖不同位置的不同标签、正确答案不是最高分、极端分数、逐组平移不变、
不同有效长度、增加或修改无效位置、全无效输入、不修改输入，以及梯度对齐。
其中 `logits=[1000,-1000]`、正确标签为 1 的损失应约为 2000，而不是无穷或截断常数。

在仓库根目录运行：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_training_loss -v
```

本部分已通过原理、实现与测试验收，`learning-mechanics.loss` 已上报掌握。
它不代表尚未验证的训练与生成能力也已经掌握。

## 第二部分：可运行的参数更新闭环

本次只修改 `training.py` 的三个函数；保留已完成的 `loss.py`。
接口已经限定数据含义和异常行为，不需要自行设计额外的参数校验。

### 1. prepare_next_token_batch

输入 `ids`、`valid` 均为 `(B,S)`，S 是补齐后原始序列长度，且 S 至少为 2。
返回 `input_ids`、`targets`、`target_valid`，均为 `(B,S-1)`。
这相当于前面讲解中的 `T = S-1`。

输入与目标错开一位，目标有效标记必须跟随目标移动。每行至少一个真实 token，
真实位置连续在左侧，只在右侧补齐；有效性只依据 `valid`，不猜测某个 PAD ID。
只剩一个真实 token 的行没有有效预测目标，但仍保留该行和形状。

对短序列，EOS 可能出现在输入中，但它后面的 PAD 不参与损失；预测 EOS 本身则参与。
允许返回全 False 的 `target_valid`，由损失函数拒绝没有有效目标的 batch。

### 2. forward_logits

输入 `input_ids (B,T)`、`E (N,C)`、`W (C,N)`，输出原始分数 `(B,T,N)`。
只用当前 token 查表后的特征与共享 W 做矩阵乘法，不加入位置、bias、Softmax 或 Attention。
保持计算图，不修改输入或参数。

输出的最后一轴对应词表候选，W 的第 j 列产生候选 ID j 的分数。
因此，重新编号 token 时，输入编号、E 的对应行和 W 的对应候选列必须一致地重排；
测试会验证新旧输出按相同词表映射对应，不能只交换 E 的行而忽略输出列。

### 3. train_step

复用前向函数和你的交叉熵，完成一次 SGD 更新：清旧梯度、前向、反向、原地更新 E/W。
参数和梯度的 shape 相同；在 `torch.no_grad()` 范围内修改调用方持有的参数，
离开该范围后，参数还应能够继续用于自动求导。

本函数返回**更新前 loss 的 Python float**，便于记录。已算出的 loss 不会随参数变化自动刷新。
允许在结束时保留本次梯度或清理梯度，但下一次调用不能混入旧梯度。
全无效 batch 必须抛出 `ValueError`，并保持 E、W 的数值不变。

不要使用 `torch.optim`、`nn.Linear`、`nn.Embedding` 或通过 `.data` 绕开自动求导机制。
本题的参数是直接创建并开启 `requires_grad` 的 Tensor；输入由测试和演示入口准备好。

## 第二部分 API 提示

```python
ids[:, :-1]           # 所有行，去掉最后一列
ids[:, 1:]            # 所有行，去掉第一列
E[input_ids]          # 按整数 ID 查 E 的行
parameter.grad = None # 丢弃此前保存的梯度，不修改参数数值
loss.backward()      # 计算并累加梯度，不更新参数
loss.item()          # 单元素 Tensor 转成 Python 数字

with torch.no_grad():
    ...              # 只有这块缩进范围不记录新的求导计算图
```

`.item()` 用在日志返回值上；反向传播仍需要原来的 loss Tensor。
`return a, b, c` 返回三个元素组成的 tuple，调用方可用 `a, b, c = function(...)` 接收。
`-> float` 是函数返回类型提示，不会自动帮你把 Tensor 转成 float。

演示入口中另有一些辅助 API，不需要你实现：

- `torch.manual_seed(17)`：固定随机种子，便于复现相同的初始参数。
- `torch.randn(N, C)`：创建给定形状的随机初始数值，初始化分布后续再展开。
- `.requires_grad_(True)`：让这个 Tensor 开始参与自动求导；本题先创建和缩放初值，再调用它。
- `torch.set_num_threads(1)`：指定 CPU 计算线程数，避免小样例的线程调度开销。
- `range(160)`：依次产生 0 到 159；`for` 每次调用一次你写的更新函数。
- `argmax(dim=-1)`：取末轴最高分的候选下标，得到 `(B,T)` 的预测 ID。
- `.tolist()`：把 Tensor 数值转为 Python 列表，方便打印。
- `if __name__ == "__main__":`：仅在作为程序入口运行时执行，不在导入时自动训练。

## 运行和验收第二部分

在仓库根目录运行新测试：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_training_loop -v
```

三个函数未实现时，对应实现检查会失败，相关功能测试暂时跳过。
可以按函数顺序逐个实现，不用等全部写好才运行测试；完成后应全部通过且无跳过。

测试检查标签/EOS/PAD 对齐、原始分数和词表映射、一次精确更新、两次连续更新、
预先存在的梯度、学习率缩放、参数身份与后续求导、补 PAD 不改变更新，以及可拟合数据。
参考计算和测试辅助代码由教师提供，不要求先学完它们的全部 Python 语法。

通过后运行演示：

```bash
.venv/bin/python -B -m exercises.ex005_training_loop.demo
```

运行入口复用你之前写的 `make_batch`，使用如下词表和两条序列：

```text
PAD=0, BOS=1, A=2, B=3, EOS=4

[BOS, A, B, EOS]
[A, B, EOS]       # 从 A 开始的片段
```

它们的有效转移一致：BOS 后是 A，A 后是 B，B 后是 EOS。
因此只看当前 token 的模型就能学会，不需要尚未实现的上下文读取能力。
演示打印初始损失、若干更新前损失、最终损失，以及有效位置的目标和预测 ID。
预期损失明显下降，有效位置预测正确；不要求任意语料可拟合，也不承诺每一步损失都下降。

全量练习回归命令：

```bash
.venv/bin/python -B -m unittest discover -s tests/exercises -v
```

这次验收的是数据、前向和参数更新，不自动宣告整个训练闭环关卡毕业。
有限差分验证、独立排错测试、逐步生成及基线表达能力限制，还会在本作业上继续学习和验证。
