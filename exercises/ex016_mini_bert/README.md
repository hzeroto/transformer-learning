# 016：Mini-BERT——把双向表示接成可训练的模型

对应已学习的[Encoder 与 BERT 合并讲义](../../notes/encoder-and-bert.md)。
本次不追加口头题；直接用一份综合实现贯通 **受损输入 → 双向 Encoder → MLM/NSP 预训练 → 完整文本分类微调**。

当前为按学习者要求补全的**教师导读版**：八处核心计算已实现，代码按步骤编号并解释数据形状和作用。
请沿[代码导读](WALKTHROUGH.md)走完一次训练的数据流；下文保留各部分的接口与验证范围。
实现与测试结果用于说明这份代码的行为，不记作学习者独立实现的证据。
已有 MHA、LayerNorm 和稳定交叉熵继续复用，不重写已经验收的数学。

## 三个核心文件的阅读顺序

| 顺序 | 文件 / 步骤 | 输入 → 输出或职责 |
|---|---|---|
| 1 | [data.py](data.py)：`make_mlm_batch` | 原文与指定替换计划 → 受损输入、原位标签和监督范围 |
| 2 | [model.py](model.py)：`gelu` | 讲义中的 tanh 近似激活，保持 shape |
| 3 | 同文件：`EncoderBlock.forward` | `(B,T,C)` → 双向 Post-LN 块 → `(B,T,C)` |
| 4 | 同文件：`MiniBert.encode` | 内容/位置/句段表示 → 多层 Encoder → `Z(B,T,C)` |
| 5 | 同文件：`MiniBert.mlm_logits` | `Z(B,T,C)` → 共享内容表的词表分数 `(B,T,N)` |
| 6 | 同文件：`MiniBert.pool` | CLS 所在位置的表示 → pooler 输出 `(B,C)` |
| 7 | 同文件：`MiniBert.forward` | 共用一次主干和 pooler，连接 MLM、NSP、分类三种输出 |
| 8 | [training.py](training.py)：`pretraining_loss` | 调用真正的模型入口，返回总 loss、MLM loss、NSP loss |

每处步骤的 docstring 保留完整契约，函数体用编号注释对应实际运算。
`fixtures.py` 提供数据，`demo.py` 串起训练与恢复，`sample_mlm_plan` 生成随机替换计划。

## 先把数据接口分清

B 是 batch，T 是包含特殊 token/PAD 的输入长度；C 是主干特征宽度，N 是词表大小，K 是下游类别数。
`make_mlm_batch` 的六个输入都为 `(B,T)`：

| 对象 | 来源与含义 |
|---|---|
| `clean_ids` | 原始 token ID；标签的来源，不应直接作为 MLM 模型输入 |
| `input_valid` | 真实输入位置为 True，PAD 为 False；`[MASK]` 位置仍为 True |
| `segment_ids` | 已构造好的句段 0/1；不是读取权限，不重置 position |
| `selected` | 被选为还原目标的位置；决定 loss 的计分范围 |
| `replacement_kind` | 对选中位置，0 换 MASK、1 换随机词、2 保留原词 |
| `random_ids` | 已抽样的随机词 ID；只有 kind=1 的选中位置使用 |

返回五项 dict：`input_ids`、`targets`、`input_valid`、`segment_ids`、`target_valid`。
五个对象仍为 `(B,T)`，彼此以及与入参都不共享存储。

**替换策略由调用方给定，你只执行策略。** 本函数不需要实现随机采样器。
教师 `sample_mlm_plan` 提供近似 15% 选中和条件 80/10/10 的计划；测试使用固定计划，
同时覆盖 MASK、随机替换、保持原样，不靠短样本恰好出现三种随机分支。

随机替换可碰巧抽到原词；保持原样的选中位置也要计 loss。不能根据输入是否等于 MASK 推断 `target_valid`。
所有 `targets` 都保留合法原 ID，包括不计分位置；现有 CE 先 gather 再筛选，不接受 `-100` 作为忽略标签。

只要求三个数据异常：选中位置落在特殊 token/PAD/无效输入上；整个 batch 没有目标；选中位置的 kind 非 0/1/2。
某一条样本无目标但其他样本有目标是合法情况。未选中位置的 kind 不校验。

## 模型结构已经固定，不用猜配置

- CPU float32/float64，输入激活与参数 dtype 一致；Tensor 可非连续。
- 内容表、可学习绝对位置表、两行句段表相加，再做 embedding LayerNorm。
- 多层双向 MHA + GELU FFN，每个子层均为 Post-LN；保留 Q/K/V/Wo 和 FFN 的偏置。
- 逐层读取权限只由 `input_valid` 决定，不额外加 causal 规则，不按句段隔离。
- PAD key 不可读；PAD query 输出保留，不清空整行。全无有效 key 的样本抛 `ValueError`。
- MLM 头使用讲义的仿射 → GELU → LN → 内容表转置 → 词表偏置。
- 内容查表与 MLM 输出使用**同一份** `token_table` 参数；不是初始化时复制一份相同的表。
- pooler 取 `Z[:,0,:]`，经仿射与 tanh；NSP 和下游分类头分别使用自己的参数。
- 所有 Dropout 固定为 0，LayerNorm eps 默认为 `1e-5`；不添加 RoPE、GQA、final norm 或 KV Cache。

每层的调用关系已在讲义解释；本题没有新算法。注意旧 `multi_head_self_attention`、
`PostLNBlock` 仍内置因果权限。可直接复用的是 `multi_head_attention`：它接收已经投影的 Q/K/V，
要求 `allowed` 恰为 `(B,T,T)`，返回 `(output, weights)`；它做了 Wo 矩阵乘法，但没有输出偏置 bo。

`MiniBert.forward` 一次返回四项：

| key | shape | 对象 |
|---|---|---|
| `encoded` | `(B,T,C)` | 主干上下文表示 Z |
| `mlm_logits` | `(B,T,N)` | 原位置的词表分数 |
| `nsp_logits` | `(B,2)` | 是否为实际后续片段的二分类分数 |
| `class_logits` | `(B,K)` | 下游任务分数，本 demo 的 K=2 |

所有前向只计算输出，不抽新的遮盖计划，不更新参数，不清 `.grad`，不切换模式，不截断计算图。
层之间使用循环，MHA 复用既有批量 Tensor 实现。模型使用基础 Tensor/autograd、`nn.Parameter`/`Module`，
没有用 `nn.Transformer`、`nn.MultiheadAttention`、融合 Attention、现成 GELU 或测试中的参考函数代替核心计算。
测试中的独立参照用于核对前向、梯度和行为边界。

这是 **BERT 风格小模型**：不用真实 WordPiece 语料，关闭全部 Dropout，使用自定初始化和训练配方，
不提供官方权重兼容。并未省略 NSP 目标，但使用下面的合成相邻片段来验证它。

## 一份小语料，完整走过两个训练阶段

[fixtures.py](fixtures.py) 提供四个文档。第 i 个文档有两段：`[Ai Ai]` 后接 `[Bi Bi]`，i=0..3。
每个文档产生一对实际相邻片段与一对不相邻片段，共 8 个样本：

```text
相邻：   [CLS] Ai Ai [SEP] Bi     Bi     [SEP] [PAD]   NSP label=0
不相邻： [CLS] Ai Ai [SEP] B(i+1) B(i+1) [SEP] [PAD]   NSP label=1
```

`i+1` 按模 4 回绕。句段编号前四项为 0，接下来三项为 1；位置始终是 0..7，不重新计数。

拟合 demo **固定只将位置 1 的 Ai 换为 MASK**，位置 2 的 Ai 仍可读。
这样标签可由上下文恢复，而且正确线索确实在被预测位置的右侧；不设计多个同样受损输入却要求不同答案的矛盾样本。
固定全 MASK 是这个确定性拟合实验的约定，不声称它复刻了 80/10/10 随机预训练；三种替换由专项测试验证。

阶段一：`pretraining_loss` 将受损 `input_ids` 送入模型，分别计算 MLM 与 NSP，再相加。
两项平均方式不同：MLM 按整个 batch 中的选中位置数平均，NSP 按样本数平均。
已有 `classification_loss` 只是把整句 logits 加一个长度为 1 的位置轴，复用你写过的 CE，不要求另写损失数学。

阶段二：教师调用器从内存存档恢复相同主干，输入**未改写原文**，改用下游分类头。
类别为第二段编号 j 的奇偶性；这不是 NSP 的相邻/不相邻标签。分类 loss 同时更新主干与分类头，不是冻结主干的实验。

在仅做预训练时，下游分类头没计入 loss，它的 `.grad` 可以为 None；分类微调时 MLM 头不计入 loss，也一样。
不能把“没有参与当前目标的头没有梯度”误判成计算图断了。共享内容表和 Encoder 则服务当前目标。

demo 检查固定训练样本的拟合、参数/配置恢复后的四项输出一致、分类微调实际更新主干。
它**不证明自然语言能力、未见组合泛化或预训练一定比从头训练更好**；不重复把这类结论当成本课通关要求。
保存恢复只核对模型参数与配置，优化器状态恢复已经在前课验证，不在这里另做一次工程题。

## 运行与通过标准

在仓库根目录，可以单独检查数据、模型，也可以直接运行完整训练演示：

```bash
# 检查数据构造
.venv/bin/python -B -m unittest tests.exercises.test_bert_data_objectives.TestMLMBatch -v

# 检查模型六处计算
.venv/bin/python -B -m unittest tests.exercises.test_bert_model -v

# 运行全部 29 项专项测试
.venv/bin/python -B -m unittest tests.exercises.test_bert_data_objectives tests.exercises.test_bert_model -v

# 调用本目录完整实现：预训练、参数恢复、分类微调
.venv/bin/python -B -m exercises.ex016_mini_bert.demo

# 最终全仓回归，不省略 -t .
.venv/bin/python -B -m unittest discover -s tests -t .
```

代码验证要求：八处步骤都实现；29 项全部执行通过且无跳过；默认 demo 的三项准确率为 100%，联合预训练 loss<0.08、分类 loss<0.05；
已有回归保持通过。测试声明 CPU 容差：float64 `rtol=1e-8, atol=1e-10`，float32 `rtol=1e-5, atol=1e-6`。
不要要求不同数学路径的最终训练 loss 逐位相同。

测试中特别覆盖了刚讨论的边界：

- 检查真正传入模型的 input_ids，不能仅以 loss 下降证明输入被正确遮盖。
- 给每一层独立输入，检查右侧有效 token 的读取；不让前层的信息混合掩盖后层的因果错误。
- 三类选中位置都被监督、标签不 shift，未选中位置的 logits 梯度为零。
- 双向读取仍屏蔽 PAD；句段表不是隔离 mask。
- MLM 的共享内容表既影响输出，也接收输出路径的梯度；整句头取 CLS 而非平均/末位置。

### 必要的 Python/PyTorch 提示

- `dict[str, torch.Tensor]` 是返回值类型注解，不会自动替你创建字段或校验数据。
- `clone()` 建立独立存储；布尔索引可选中位置，`torch.where(condition,a,b)` 按条件逐元素选择。
  必须在副本上赋值，不能顺手把 `clean_ids` 改掉。
- `unsqueeze(1)` 给 `(B,T)` 增加一个 query 广播轴；`expand` 可以把权限扩展成 `(B,T,T)` 视图，
  无需复制到每个 head。不要原地修改 expand 的共享视图。
- `z[:,0,:]` 取每条样本的第 0 个位置，去掉位置轴，得到 `(B,C)`；这不是求平均。
- `a**3` 为逐元素三次方，`torch.tanh` 为逐元素非线性；GELU 公式可直接查讲义 §7，不考背公式。
- demo 中 `MiniBert(**model.config, dtype=...)` 的 `**` 把 dict 展开成命名参数；它不是矩阵运算。
- 构造器和模块注册已经完成，不需要你新学 dataclass、写优化器或重新设计参数初始化。

### 测试来源与证据边界

发布前在临时目录用独立教师实现运行这 29 项测试，并核对默认 demo；没有把答案临时写进你的文件。
还注入了第二层因果、训练入口误喂原文、只监督 MASK、共享表断梯度、漏句段表示五类错误，均被对应测试拦截。
这些发布前结果用于确认题面和测试可完成。本次教师补全后的实跑结果见代码导读；两者都不冒充学习者独立实现或测试设计的证据。
