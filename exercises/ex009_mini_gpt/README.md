# 009：mini-GPT 的输入管线与模型组装

把已验收的 `TransformerBlock` 接成一个完整的 decoder-only 语言模型：
给一段前缀，输出每个位置对下一 token 的词表分数。

讲义见[从 Block 到 mini-GPT](../../notes/block-to-mini-gpt.md)，
参数注册与模式切换见[Dropout 与模块组织](../../notes/dropout-and-module-organization.md)第 3、4 节。

## 待完成

只修改 [model.py](model.py) 中的四项学习者实现，不改测试；已经填写的部分继续保留：

| 位置 | 要实现的内容 |
|---|---|
| `text_to_ids` | 空格分隔的文本 → 带 BOS/EOS 的 ID 列表 |
| `MiniGPT.forward` | 输入表示 → 多层块 → （Pre-LN 时）末尾归一化 → 词表投影 |
| `build_two_styles` | 构造一对只在 `norm_style` 上不同、其余参数对齐的模型 |
| `PostLNBlock.forward` | 复用 ex008 的组件，实现两条相加后归一化的支路 |

构造函数、参数与子模块的登记方式已经给出，不需要改动。

## 本次修正：真正切换块内排列

原导出包的两个模式都调用 ex008 的 Pre-LN 块，仅以 `final_norm` 区分，
无法完成两种排列的对照。现在构造函数按 `norm_style` 选择不同的块：

| 配置 | 块内任意一条支路的顺序 | 堆栈末尾 |
|---|---|---|
| `pre` | `x + Dropout(F(LN(x)))` | 再做 final LayerNorm |
| `post` | `LN(x + Dropout(F(x)))` | 直接进入词表投影 |

这里 `F` 是 Attention 或 FFN，`x` 是该支路输入。两者相加前后的 shape 相同。
Pre-LN 的直接相加路径绕过 LN；Post-LN 的相加结果整体经过 LN。
即使支路输出为零，前者得到 `x`、后者得到 `LN(x)`，所以反向路径也不同。

`PostLNBlock` 继承 ex008 的构造函数，沿用同名参数和子模块，只让你实现 `forward`。
不需要修改 ex008，也不用重写 Attention、FFN、LayerNorm 或 Dropout。
`MiniGPT.forward` 仍统一调用 `block(X, input_valid)`，具体排列由块自己的 `forward` 决定。
这个调用机制就是前面讨论的 `nn.Module.__call__` 到 `forward` 的分派。

`build_two_styles` 要对齐除 `final_norm` 之外的同名参数，同时保证两模型不共享存储。
块的类型差异已由构造函数处理，不必在该函数里修改子模块。
额外的 LayerNorm 用 ones/zeros 初始化，不消耗随机数；重新设置相同 seed 可以对齐初始化。

## 契约要点

完整说明见各函数的 docstring，这里只列容易踩的几条：

- `input_valid` 是**输入位置有效性**（能否作为 key 被读取），不是 `target_valid`。
  本模型不用它筛选损失，也不清零 PAD query 行。
- 位置一律从 0 开始，**只在入口加一次**位置表示，层间不重复加。
- `norm_style == "pre"` 时末尾做一次 `final_norm`；`"post"` 时跳过。
  Post-LN 堆栈的每个子层输出都已归一化，末尾不需要再补。
- `forward` 返回原始 logits：不做 Softmax、不取 argmax、不算损失、不更新参数。
- 允许 `T == 1`；`T > max_positions` 应抛 `ValueError`，`T == max_positions` 必须正常返回。
- CPU、float64；数值比较容差见测试文件顶部。

可以直接复用的既有实现：

| 来源 | 用途 |
|---|---|
| `ex003` 的 `embed_with_positions` | 内容表 + 位置表 → `(B,T,C)` |
| `ex008` 的 `TransformerBlock` / `LayerNorm` | 块与归一化，已在构造函数中登记 |
| `ex002` 的 `make_batch` | 变长序列右侧补齐（训练时用） |
| `ex005` 的 `prepare_next_token_batch` / `masked_cross_entropy` | 错位标签与损失 |

核心数学手写，不使用 `nn.Transformer`、`nn.MultiheadAttention`
或现成的交叉熵/归一化封装。

## 运行

```bash
.venv/bin/python -B -m unittest tests.exercises.test_mini_gpt -v
```

全仓回归：

```bash
.venv/bin/python -B -m unittest discover -s tests -t .
```

**全部四项都未填写时的预期状态**：本练习 33 项测试中，4 项构造/参数登记检查通过，
29 项报 `NotImplementedError`，全仓其余 164 项仍应通过。
已经完成部分实现时，报错数会减少；新建的 Post-LN 路径在填完前会明确报未实现。
这不代表完整任务已经通过。

## 完成标准

本练习 33 项全部通过、无跳过，且全仓回归仍为绿。测试覆盖：

- `text_to_ids` 的起止标记、空文本、多余空白、未知 token 抛 `KeyError`
- 输出 shape/dtype、单位置、长度边界、输入不被修改、保留求导路径
- 位置表确实被使用，且只加一次（对照一层模型的手工展开）
- 因果性：改动末位置不影响更早位置的 logits
- 前缀一致性：喂前 3 个与喂 4 个 token，位置 0–2 的 logits 相同
- PAD key 不泄漏到有效位置
- 参数注册：每个块都在 `named_parameters` 中、块之间参数独立、
  `state_dict` 覆盖全部参数、反向后每个参数都有梯度
- 两种排列：`final_norm` 的有无、两模型是不同对象、共享参数数值对齐、
  模型确实选择对应块、Post-LN 的前向与输入/参数梯度符合独立数学参照、
  两处 Dropout 被调用且遵循 train/eval，Post-LN 路径跳过末尾归一化
- 训练整合：损失下降且**块内参数确实被更新**

新增测试含教师数学参照，仅供验证，不得导入作业实现；教师自检不计作学习者的掌握证据。

## 完成后的一个观察任务

测试不覆盖这一项，但它是 `modern.pre-post-ln` 验收标准要求的实验部分：

用 `build_two_styles` 构造一对模型，同一输入下分别反向，
比较**第一层与最后一层**参数的梯度范数。记录你观察到的差异，
并说明这与两种排列的残差路径有什么关系。

注意：单次随机初始化的观察不能证明某种排列总是更优，
结论要限定在你实际测到的配置上。
