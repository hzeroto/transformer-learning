# Mini-BERT 代码导读：跟着一个样本走完训练

本版由教师按学习者要求补全。阅读时抓住一条主线：**改写输入，让模型从上下文还原原词；同一份上下文表示还可以判断句对关系和下游类别。**

符号沿用代码：B 为样本数，T 为输入长度，C 为特征宽度，N 为词表大小，K 为下游类别数。
演示使用 B=8、T=8、C=16、N=12、K=2，两层 Encoder、每层四个 head。

![Mini-BERT 整体架构：输入、完整 Encoder、MLM 与 CLS 任务头、预训练和分类损失](../../notes/assets/encoder-and-bert/bert-architecture.png)

[打开可放大的 SVG 原图](../../notes/assets/encoder-and-bert/bert-architecture.svg)。蓝色 `encode` 已经包含全部 Attention 和 FFN，输出的是特征；紫色 MLM 头和橙色分类头随后才产生预测分数。`pool` 只准备 CLS 特征，不直接输出分类分数。

## 1. `make_mlm_batch`：改变题目，不改变答案

入口在 [data.py](data.py)。以固定实验的第一条样本为例：

```text
位置：          0      1      2     3     4   5    6      7
clean_ids：    CLS     A0     A0   SEP    B0  B0  SEP    PAD
input_ids：    CLS    MASK    A0   SEP    B0  B0  SEP    PAD
targets：      CLS     A0     A0   SEP    B0  B0  SEP    PAD
target_valid：  F      T      F     F     F   F    F      F
input_valid：   T      T      T     T     T   T    T      F
segment_ids：   0      0      0     0     1   1    1      0
```

模型收到 `input_ids`，正确答案来自 `targets`。位置 1 要还原 A0，右侧位置 2 的 A0 提供上下文线索。
`input_valid` 决定能读取哪些位置；`target_valid` 决定哪些输出要评分。MASK 仍是有效输入，PAD 不能当 key 被读取。

函数先排除非法目标，再分别 `clone()` 出输入和标签，只在输入副本上执行替换。
三种策略为换 MASK、换随机词、保持原词；三种选中位置都参与 loss，不能看到原词没变就取消监督。
`targets` 从始至终保留原文同位置 ID，不像 next-token 任务那样错开一个位置。

`sample_mlm_plan` 负责随机选位置和策略，`make_mlm_batch` 只执行已有计划。
演示为方便复现，固定只遮盖位置 1；80/10/10 三种分支另有测试覆盖。

## 2. `gelu`：给线性变换加入逐元素非线性

实现在 [model.py](model.py)，用于 FFN 中间和 MLM 头：

```python
0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3)))
```

这一操作按每个元素的值施加平滑系数，shape 不变，也不会在位置之间传递信息。
如果 FFN 两次仿射变换之间没有非线性，它们可以合并为一次仿射变换；加入 GELU 后不再受这个限制。

## 3. `EncoderBlock.forward`：读取上下文，再处理每个位置的特征

先从同一份 `x(B,T,C)` 分别得到 Q、K、V，每份都是 `(B,T,C)`。
读取权限由下面这一行决定：

```python
allowed = input_valid.unsqueeze(1).expand(B, T, T)
# allowed[b, query, key] = input_valid[b, key]
```

所有 query 都可以读取有效 key，包括右侧和另一句段。这里没有下三角因果限制。
现有 `multi_head_attention` 负责拆 head、计算权重、读取 V、合 head 和乘 Wo，返回的 Attention 输出仍为 `(B,T,C)`。

然后执行两个子层：

```python
y = norm1(x + attention + bo)
hidden = gelu(y @ W1 + b1)
update = hidden @ W2 + b2
out = norm2(y + update)
```

Attention 负责不同位置之间的信息交流；FFN 对每个位置的特征先扩宽再压回 C。
残差保留子层输入到输出的直接路径；LayerNorm 在每个位置的 C 维特征上归一化。
本版是 **Post-LN**：先相加，再归一化。PAD query 可以有输出，但 PAD key 在每一层都不可读。

## 4. `encode`：把 ID 变成带上下文的表示

内容表 `token_table(N,C)`、位置表 `position_table(max_positions,C)`、句段表 `segment_table(2,C)` 都在构造器注册为可学习参数。
查内容和句段得到 `(B,T,C)`；查位置得到 `(T,C)`，相加时沿 B 广播。

```python
x = token_vectors + position_vectors + segment_vectors
x = embedding_norm(x)
for block in blocks:
    x = block(x, input_valid)
```

三种向量分别告诉模型“是什么词、在哪个位置、属于哪一段”。相加不改变特征宽度。
位置从 0 连续编号，句段切换时不归零；句段编号也不限制 Attention 的读取范围。
最终 `z(B,T,C)` 的每个位置都已经经过全部 Encoder 层，含有上下文信息，此时还不是词表分数。

## 5. `mlm_logits`：把每个位置的表示变成还原原词的分数

```python
x = z @ Wm + bm             # (B,T,C)
x = gelu(x)                 # (B,T,C)
x = mlm_norm(x)             # (B,T,C)
logits = x @ token_table.T + bvocab  # (B,T,N)
```

Wm 将主干表示变换为用于 MLM 的特征；GELU 与 LN 都作用于 C 维，最后才转成 N 个候选词分数。
位置 1 的这 N 个分数会拿去与答案 A0 比较。这个函数对所有位置算分数，选择哪些位置计 loss 留给训练目标。

输出投影直接使用输入查表时的同一个 `token_table`，没有复制新参数。
一次反向传播时，查表路径与输出投影路径对它的梯度自动相加；`.T` 不会切断这条计算图。

## 6. `pool`：为整句任务取出 CLS 位置

```python
cls = z[:, 0, :]             # (B,C)
pooled = tanh(cls @ Wpool + bpool)
```

第 0 个位置是 CLS，它在双向 Encoder 中可以读取所有有效位置。
这里选 CLS 后做一次变换，不是对全部 token 求平均，也不重新运行 Encoder。
CLS 能为任务提供有用信息，要靠句级目标的训练信号塑造，不是因为这个 token 天然就懂整句话。

## 7. `forward`：一次主干，连接三个任务头

`forward` 只执行一次 `encode` 和一次 `pool`，然后返回：

| 输出 | shape | 用途 |
|---|---|---|
| `encoded` | `(B,T,C)` | 供观察或后续使用的上下文表示 |
| `mlm_logits` | `(B,T,N)` | 每个位置的原词预测分数 |
| `nsp_logits` | `(B,2)` | 判断两段是否为语料中的实际相邻片段 |
| `class_logits` | `(B,K)` | 下游任务分类分数 |

两个整句头共用 `pooled`，但分别用 Wnsp/bnsp 和 Wclass/bclass。
NSP 的类别与下游类别是两个不同任务，不能因为都是二分类就混用标签。
前向负责计算分数，loss 和参数更新各自留在外层。

## 8. `pretraining_loss`：让两个预训练任务一起更新主干

入口在 [training.py](training.py)，模型实际收到受损输入，原文只交给损失：

```python
outputs = model(batch["input_ids"], batch["segment_ids"], batch["input_valid"])
mlm = masked_cross_entropy(outputs["mlm_logits"], batch["targets"], batch["target_valid"])
nsp = classification_loss(outputs["nsp_logits"], nsp_labels)
total = mlm + nsp
```

MLM 按整个 batch 的选中位置数平均；NSP 按样本数平均。`classification_loss` 只是给句级分数补一个长度为 1 的位置轴，复用原有交叉熵。
`total` 保持为可求导 Tensor。两项 loss 的梯度在共享主干上相加，下游分类头尚未参与这个目标。

未选中位置的 MLM logits 对该 loss 的直接梯度为零，但这些位置仍能作为上下文影响别处，所以它们的输入向量可能得到梯度。

## 9. `demo`：从前向计算走到参数真正发生变化

[demo.py](demo.py) 把上述函数串成三个阶段：

1. **预训练**：用受损输入计算 MLM + NSP，依次 `zero_grad → backward → optimizer.step`。前者教模型还原原词，后者教模型判断片段关系。
2. **保存与恢复检查**：复制参数存档，用同一配置创建新模型并加载参数，逐项比较四种输出。这一步不继续训练。
3. **分类微调**：恢复后的模型读取完整原文，只用 `class_logits` 和下游标签计算 loss，同时更新主干与分类头。当前标签是第二段编号的奇偶性，与 NSP 不同。

固定小语料共 8 条句对，参数和所有数据都在本地 CPU 上处理。这个实验用于验证数据、模型、目标、梯度和参数更新能组成可学习的整体；训练样本拟合不代表自然语言泛化能力。

在仓库根目录运行完整演示：

```bash
.venv/bin/python -B -m exercises.ex016_mini_bert.demo
```

## 本次 CPU 实跑结果

| 检查 | 结果 |
|---|---|
| Mini-BERT 数据、模型、梯度与目标专项 | 29 项通过，无跳过 |
| 全仓 Python 回归 | 399 项通过，无跳过 |
| 初始联合预训练 loss | 3.272149 |
| 300 步后的联合预训练 loss | 0.001226（MLM 0.000940，NSP 0.000286） |
| 训练集 MLM / NSP 准确率 | 100% / 100% |
| 参数与配置恢复 | 四项输出逐元素完全一致 |
| 160 步分类微调后的 loss / 准确率 | 0.001506 / 100% |
| 微调是否更新主干 | 内容表参数确实发生变化 |

验证命令与完整接口见 [README](README.md)。这些结果验证的是当前教师补全实现；本课按完整代码导读收尾。
