# 作业 005：最小训练闭环

这份作业随教学逐步扩展，最终把输入、输出分数、损失、反向传播和更新串起来。
第一部分 `loss.py` 中的 `masked_cross_entropy` 已通过验收，不需要重写。
第二部分 `training.py` 已通过 20 项测试及训练演示，标签构造、前向和一次 SGD 更新已接通。
第三部分 `gradient_check.py` 已通过 9 项专项测试及全部 77 项练习回归。
第四部分已通过：定位并修复 E 的旧梯度累积，所写回归测试能捕获原故障，全部 78 项练习通过。
当前进入第五部分：固定已训练参数，手写单条序列的逐步生成；不重写此前实现。
教师提供测试和运行入口，核心函数仍由学习者实现。

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

本部分的三个函数已经完成；以下保留原接口约定，后续不需要重写。
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

## 第三部分：单个参数元素的有限差分

只实现 `gradient_check.py` 中的 `finite_difference_one`，不用改 `training.py` 或 `loss.py`。
本次先验证数值检查工具；不会因为表示理解或教师测试通过，就自动认定独立排错能力也已验收。

### 输入、输出和检查对象

前五个输入与现有模型一致：`input_ids`、`targets`、`target_valid`、E、W。
本题将参数精度固定为 CPU `float64`，参数可以需要求导，也可以不需要。

其余输入是：

- `parameter_name`：字符串 `"E"` 或 `"W"`，选择要检查的参数矩阵。
- `index`：两个整数的 tuple，选择该参数矩阵的一个元素。
- `h`：正的有限扰动大小，默认 `1e-5`，不是 SGD 学习率。

例如 `parameter_name="E", index=(2,0)` 检查 E 的 token ID 2、第 0 个特征。
Python 中 `index = (2, 0)` 是一个二元素 tuple；`E[index]` 等价于 `E[2, 0]`。
W 的索引则是 `(特征下标, 输出候选 ID)`，不是序列位置。

返回指定元素的中心差分近似，类型为 Python `float`：

```text
(正向扰动后的完整 loss - 负向扰动后的完整 loss) / (2 * h)
```

每次只改变选定的一个参数元素，其余参数、数据、标签、有效标记全部固定。
两个 loss 都需要重新调用完整前向，再调用已经实现的交叉熵。

### 必须通过的语义边界

`x[b,t,c] = E[input_ids[b,t],c]`。
因此 E 的行号是 token ID，x 的中间轴是序列位置，不能把两者当成同一个索引。
修改 E 的某个元素后必须重新查表，让这个 token 的所有使用位置都受到影响。
只修改某个位置的 x，测到的是中间结果的局部影响，不是共享参数的总影响。

例如 `input_ids=[[1,2,3,2]]` 时，检查 E[2,0] 应影响位置 1 和 3，而非固定位置 2。
这个例子会进入测试，不再单独安排索引手算题。

### 无副作用和求导边界

- 在 `torch.no_grad()` 范围内计算探测值，用 `.clone()` 创建独立副本后再扰动。
- 不修改原参数、输入数据或原有 `.grad`，也不改变原参数是否参与求导的状态。
- 数值检查不执行训练，不调用 `train_step`，不先清梯度。
- 不读取 `.grad` 当答案，不调用 `backward`、`autograd.grad` 或 `gradcheck`。
- 整个 batch 无有效标签时，传播已有交叉熵的 `ValueError`；原参数和梯度也不能残留变化。

`.clone()` 复制独立存储；单纯写 `copy = E` 只是多了一个指向同一对象的名字。
本题约定参数名、索引和 h 均合法，不要求额外输入校验。

### 运行方式

在仓库根目录运行：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_gradient_check -v
```

未实现时，一个实现检查失败，功能测试暂时跳过。完成后共 9 项测试应全部通过且无跳过。
测试包含 E/W 的多个坐标、重复 token、无需求导的参数、原参数及旧梯度不变、
仅在无效位置使用的 embedding、异常安全、指定扰动大小和不同形状。

小扰动的梯度参考比较采用 `rel_tol=1e-5, abs_tol=1e-8`，即按相对和绝对误差容限比较，
不要求浮点数逐位相等。另有一个较大 h 的测试只检查是否正确执行指定中心差分公式，
不是说该 h 应该得到最精确的导数。h 太小也可能因浮点舍入而失效。

通过后仍用原来的全量练习回归命令。有限差分通过只说明当前数值函数和梯度相符，
不能替代对标签任务本身的正确性检查；独立排错与生成仍在本作业后续完成。

## 第四部分：独立排错和回归测试

本次不增加新的模型计算。教师在 `debug_case.py` 中放置了一份带有一处核心逻辑故障的
`candidate_train_step`，目标契约与已通过的 `train_step` 相同。
它可以正常返回 loss，单次运行看起来可能正常，但不能满足完整更新契约。

这份候选代码仅用于练习，不会被原来的训练演示调用。不要修改已经通过的
`training.py`、`loss.py` 或 `gradient_check.py`。

### 先观察，再构造证据

在仓库根目录运行诊断入口：

```bash
.venv/bin/python -B -m exercises.ex005_training_loop.debug_probe
```

它使用同一组参数，连续处理两份不同的 batch，并展示两种检查量：

- `numeric`：在本次更新前，用有限差分计算的参数梯度近似。
- `from update`：用 `(更新前参数值 - 更新后参数值) / lr` 反推本次实际更新使用的量。

正确的基础 SGD 应让这两者在误差容限内一致，因为更新公式是
`新参数 = 旧参数 - lr * 本次梯度`。
诊断入口对齐了同一组更新前参数，并且有限差分不会改动现有状态。
`close` 是按相对容差 `1e-5`、绝对容差 `1e-8` 判断两者是否接近。

这个入口只打印证据，不作自动验收：正常退出并不表示候选实现正确。
请判断异常出现在哪次调用、涉及哪个参数，解释数值证据与代码的对应关系。
不要求重复手推模型导数，也不要仅根据 loss 是否下降下结论。

### 由你完成的一项回归测试

编辑 `tests/exercises/test_training_regression.py`。
框架已留好测试类和方法，但数据/调用顺序、观测量和断言由你选择。
可以复用 `make_debug_case()` 生成相同的初始参数和数据，也可以自己构造更小的例子。

需要做到：

1. 在候选函数尚未修复时，测试因真实的数值或状态错误失败。
2. 对 `candidate_train_step` 作最小修复后，同一测试通过。
3. 能解释测试捕获什么行为错误，以及为什么只有 loss 下降或首次调用成功不足以验收。

不要只删除占位的 `self.fail` 让测试空跑；不要把错误结果硬编码为正确答案，
也不要仅调用整个诊断入口再断言它没有抛异常。
允许在测试中使用已有正确实现作参考，但需要结合有限差分的诊断证据解释根因。
修复候选代码时，不允许直接代理到正确的 `train_step` 以绕过定位。

### Python 测试语法提示

```python
class TrainingRegressionTest(unittest.TestCase):
    def test_candidate_step_regression(self):
        ...
```

- `class ... (unittest.TestCase)` 声明一个继承测试基类的类，框架负责创建对象和运行测试。
- `self` 是当前测试对象，作用类似 C++ 的 `this`，由 Python 在调用方法时自动传入。
- 方法名以 `test_` 开头，会被 unittest 自动发现。用缩进表示方法体。
- `self.fail("说明")` 无条件使测试失败；当前这行只是提醒测试尚未完成，不是有效复现证据。
- `self.assertTrue(condition)` 检查条件为真。
- `self.assertAlmostEqual(actual, expected, delta=1e-7)` 检查两个标量的绝对误差不超过给定容差。
- 对 Tensor 中的单个元素可先 `.item()` 再比较；对整张 Tensor 可用 `torch.testing.assert_close`。

固定输入、初始参数和随机种子，并声明使用的精度、h 与容差，保证失败可以复现。
`make_debug_case()` 已固定种子 17，参数为 CPU float64；默认差分 h 为 `1e-5`。

### 运行与提交结果

运行自己的测试：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_training_regression -v
```

修复后运行全部练习：

```bash
.venv/bin/python -B -m unittest discover -s tests/exercises
```

本部分现已通过：原故障在内存中重现时，学习者回归测试于第二次 E 更新失败；
最小修复后全部 78 项练习测试通过。以下提交要求保留作回顾。
完成后把根因、关键数值证据以及修复前后测试结果发来，我会 review 最小修改，
并检查测试能否识别重新引入的原故障，而不只检查最终代码能跑。

这一轮验收的是独立排错和测试设计。逐步生成及上下文表达能力仍会在现有模型上继续。

## 第五部分：固定参数，逐步生成

只新增一个核心函数：`generation.py` 中的 `greedy_generate`。教师提供行为测试和
`generation_demo.py`，不提供生成循环答案。先读本节，再实现函数即可，不重复定义题。

### 1. 训练与生成沿用同一个前向函数

训练时，真实序列已经存在，多个输入位置可以同时预测它们各自的下一 token，
再用已知标签计算损失和更新参数。生成时，续写 token 尚不存在；每次选出一个，
把它追加到已有前缀，再预测后一个。将自己的输出接回下一次输入，称为自回归生成。

以演示中学到的 BOS=1、A=2、B=3、EOS=4 为例：

```text
已有 [1]       ：选出 2，追加后为 [1,2]
已有 [1,2]     ：选出 3，追加后为 [1,2,3]
已有 [1,2,3]   ：选出 4，追加后为 [1,2,3,4]，遇 EOS 停止
```

这里只展示期望行为，不把期望答案交给生成函数。函数既不接收 targets，也不计算 loss。
本题始终固定 E/W，没有 backward、参数更新或清梯度；生成导致的是输入序列变长。

每步选当前最高分候选叫贪心选择（greedy）。它不保证整段序列概率最大。
Softmax 在数学上不改变候选的大小顺序，所以只选最高分时直接对 logits 取 argmax；
先 Softmax 多做计算，在有限精度下还可能损失可区分的分数差异。按概率随机选择留到后续。

### 2. 最后一个位置，才是在预测尚未生成的下一 token

本题先固定 batch 大小 B=1。设当前前缀长度为 L：

```python
logits = forward_logits(ids, E, W)              # (1,L,N)
last_logits = logits[:, -1, :]                  # (1,N)
next_id = last_logits.argmax(dim=-1, keepdim=True)  # (1,1)，long
```

`logits[0,t,j]` 是“输入位置 t 对下一 token 为词表 ID j 的分数”，
不是“位置 t 本身属于哪个 ID”。前缀末位置 L-1 的输出才预测前缀后第 L 个 token。
在当前模型中，输出候选 j 对应 `W[:, j]`，因此 argmax 返回的候选下标就能作为
token ID，再次用于 E 查表。E 的行、W 的列和词表编号必须使用同一套对应关系。

Python/PyTorch 提示：

- `[:, -1, :]` 的整数索引 `-1` 选最后位置并去掉该轴，所以得到 `(1,N)`。
- `argmax` 返回最大值的下标，不是最大值；`keepdim=True` 将候选轴保留为长度 1。
- `torch.cat((ids, next_id), dim=1)` 沿 token 轴拼接，得到新的 `(1,L+1)` 张量。
  内层 `(ids, next_id)` 是包含两个张量的元组；cat 不修改这两个输入。
- `.clone()` 创建独立副本；这里只给同一 Tensor 多取一个变量名不等于复制。
- `next_id.item()` 将只有一个元素的 Tensor 变成 Python 整数，便于与 `eos_id` 比较。
- `for step in range(max_new_tokens)` 最多执行指定次数；`break` 立即退出循环，与 Go/C++ 类似。

推荐让整个生成前向处于 `with torch.no_grad():` 中，避免保存不需要的求导图。
这个上下文不永久关闭参数的求导能力，也不会自动清理 `.grad`，更不会阻止你手动写入参数；
参数保持不变来自“生成函数不执行更新”，不是 no_grad 提供了只读保护。
`nn.Module` 与 `eval()` 尚未引入，本题不需要它们，不能把 eval 当成 no_grad 的别名。

### 3. 函数契约和运行

- 只处理 `(1,T)` 的非空、无补齐前缀，所有 ID 合法；不添加额外输入校验。
- 返回“原前缀 + 新 token”；`max_new_tokens` 只限制新增数量，不是总长度。
- 新生成 EOS 时先保留它，再停止；若前缀已经以 EOS 结束，不再调用模型。
- 预算为 0 或前缀已结束时，也返回独立副本。一般情况下也不能改动调用方的输入。
- EOS 由参数指定，不能写死为 4；ID 0 不自动代表结束。本题不额外屏蔽任何候选。
- 不修改 E/W、它们已有的梯度和求导状态。复用现有 `forward_logits`。
- 首先用完整当前前缀作输入，便于以后接上下文模型。对本基线而言只传末 token 也等价，
  但这种等价来自当前结构，不是一般 Transformer 的性质。

在仓库根目录运行：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_generation -v
.venv/bin/python -B -m exercises.ex005_training_loop.generation_demo
```

未实现时，1 项实现检查会明确失败，行为测试类暂时跳过；实现后本题 9 项测试须全部通过、
无跳过。此前 78 项仍应通过；加上本题，全量将是 87 项。
测试使用可预期的转移参数隔离循环错误，不把生成测试通过冒充训练成功。
演示才会调用你已实现的训练函数，再从只有 BOS 的前缀生成。

### 4. 接到 Attention 的动机：输入存在，不等于计算使用了它

当前前向是 `E[input_ids] @ W`，所以最后位置实际只计算：

```python
last_logits = E[ids[0, -1]] @ W  # (C,) @ (C,N) -> (N,)
```

即使传入整个前缀，这个位置也没有读取更早 token 的计算路径。
例如一项任务要求 `[BOS,A,X]` 后接 C、`[BOS,B,X]` 后接 D：两个前缀都以 X 结束，
当前模型都会给出同一份 `E[X] @ W`。增加训练次数或仅增大 C，不会凭空建立对 A/B 的读取。
这里假设 A/B/X/C/D 各是一个 token，两个前缀等长。

演示会打印两个“更早 token 不同、末 token 相同”的前缀，其最后分数最大差值应为 0。
这是结构检查，不代表模型已经学会理解上下文。后续 Attention 要新增的正是跨位置读取。

完成后发来测试和演示结果，并顺带回答一个边界判断：如果只给当前查表结果加上已经学过的
绝对位置向量 P，这两个等长前缀能否在最后位置给出不同结果？用该位置实际读取的数据解释，
不要求手算分数。该解释与本次实现合并验收，然后继续单头 Attention。
