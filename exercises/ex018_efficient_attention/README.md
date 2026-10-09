# 018：实现滑动窗口与在线分块 Attention

本目录提供一份带详细中文注释的参考答案，对应已读完的
[联合讲义](../../notes/sliding-window-and-flash-attention.md)。
从同一份 Q/K/V 开始，先实现局部读取，再实现分块在线归一化，最后复用 ex013 的
模型参数和组件，验证完整滑窗模型的全量、逐 token 与多 token 追加结果。

四个核心函数均已补全；[第 5 节](#5-实现步骤与关键推导)按代码中的步骤编号解释实现。
本次答案由助手补全，运行通过只能验证这份代码的行为，不作为学习者独立掌握的证据。

## 四处核心实现

| 文件 | 函数 | 已实现的行为 |
|---|---|---|
| [attention.py](attention.py) | `local_attention` | 只对窗口内的位置打分、归一化并读取 V |
| 同上 | `online_attention` | 每个 query 逐块合并归一化状态，支持全历史与窗口 |
| [model.py](model.py) | `window_block` | 将在线 Attention 接入既有块，并维护紧凑、有限的层缓存 |
| 同上 | `window_step` | 用独立绝对位置推进请求，处理本次全部新 token |

建议按表格顺序阅读。每个核心函数都按步骤标明关键张量的 shape、计算原因和边界处理；
辅助函数、参数注册、请求容器、全量模型包装器和运行示例沿用原有代码。
ex013 的原答案保留，新的模型入口直接复用它的参数，不重复实现 RMSNorm、RoPE、SwiGLU。

## 1. 两种 Attention 共用的输入合同

Q/K/V 已经过投影和拆头；使用 RoPE 时，Q/K 已完成旋转。这里只算各头输出，
合头与 Wo 投影留给模型块。

```text
q: (B,H,Tq,Dk)       k: (B,H,Tk,Dk)       v: (B,H,Tk,Dv)
q_positions: (Tq,)   k_positions: (Tk,)   key_valid: (B,Tk)
返回: (B,H,Tq,Dv)
```

B 是 batch 数，H 是头数，Tq/Tk 是本次 query/key 数，Dk/Dv 是匹配/内容宽度。
所有轴长度为正，Tq 可以不同于 Tk，Dk 可以不同于 Dv。
Q/K/V 使用 CPU float32 或 float64，同一次调用 dtype 相同。
位置使用 CPU long，严格递增但可以不连续，各 batch 共用一组位置。
key_valid 是 CPU bool，True 表示有效 key，沿 head 轴广播，不是 query mask。

窗口宽度 W 包含当前位置自身。一个 key 可读，当且仅当：

```text
key_valid[b,j] and 0 <= q_positions[i] - k_positions[j] < W
```

`online_attention(window=None)` 只保留其中的因果与 key 有效性约束。
窗口按绝对位置差定义，不是删除 PAD 后再数 W 个 key。
点积按 sqrt(Dk) 缩放；同一个 query 的权重在它的全部可读 key 上归一化。

支持非连续输入，输出保留 dtype/device 和 Q/K/V 梯度。不能修改输入或已有 `.grad`，
不能在函数内部调用 backward、detach 或关闭求导。输入和屏蔽前点积分数均有限。
shape、dtype、device、正整数窗口及块尺寸由调用方保证合法，不考通用参数校验。

唯一必需的 Attention 异常是：任意 batch/query 整行没有合法 key 时抛 `ValueError`。
当前块暂时没有合法 key 不等于整行无合法 key；它不能导致 NaN 输出或梯度。

## 2. 数学正确之外，还要满足执行约束

local_attention 的 QK 乘法只计算因果与窗口范围内的位置。PAD key 可以在局部
打分后屏蔽。不能先计算完整分数表或构造完整权限表，再切成窗口。

online_attention 的 query_block/key_block 分别限制一个块的 query/key 轴长度。
需要处理尾块、不等长 Q/K、当前块中部分行为空以及最大分数变化的情况。
完整 score、权重、mask 都不能预先分配；输入小于指定单块容量时允许一个块完成。
完全没有合法配对的块要在 QK 前跳过，边界块可以计算后屏蔽。允许遍历候选块，
本题不要求进一步优化 Python 的块调度开销。

QK 点积使用 `@`、`torch.matmul/mm/bmm/einsum` 中任意一种，其他部分使用基础 Tensor
运算。教师测试通过执行时的矩阵形状观察配对数和分数块大小，不需要你填写计数日志。
local_attention 可用 torch.softmax；online_attention 自行完成指数、归约与在线合并，
不调用 softmax/log_softmax、局部实现或融合 Attention 代做。
两者均不能调用高级 Attention、测试参照或已有教师探针作为实现。

必要 API 提示：torch.maximum 逐元素取较大值；amax(..., keepdim=True) 沿给定轴取
最大值并保留该轴；torch.where 在两个已经计算出的候选 Tensor 间选择，并非短路分支。
具体在线数学见讲义 §5；下面第 5.2 节将公式对应到实现变量。

## 3. 接入 ex013 的完整模型

model.py 的 project_rotary_qkv 和 finish_block 复用已学会的投影、RoPE、残差与 FFN；
辅助函数的输入输出 shape 见其 docstring。模型所有层使用同一窗口，无 PAD、无 Dropout。
全量 window_forward 保留求导路径；有缓存的调用由外部放在 torch.no_grad() 中，
本题不考跨缓存调用的反向传播。

每层 LayerWindowKV 保存：

```text
k / v: (B,Hkv,t,D)    positions: (t,)
```

Hkv 是 KV 头数，D 是头宽。GQA 允许临时把 KV 扩展到 query 头进行计算，但持久缓存
必须保持 Hkv 份。缓存中的 K 已按绝对位置旋转，V 未旋转。本次可以追加多个 token，
首个 chunk 也可以长于窗口。一次调用返回本次每个位置的 logits；调用结束后每层保存
最新 min(W, 已处理数) 条 KV，使用与保留内容大小相符的独立底层存储，不能只是仍引用
整段旧数据的切片。clone() 会复制底层存储；contiguous() 在输入已连续时不保证复制。

WindowState 用 next_position 记录下一个绝对位置，初值由 start_position 指定。
缓存长度、绝对位置和 token ID 是不同对象。reset() 清空该请求、恢复起始位置，
每次 new_window_state 创建独立请求。不要在模型参数或全局变量中保存请求状态。

model.L 是允许的绝对位置上界，位置必须小于它。window_step 若发现整个新 chunk
将越界，必须在改变任一层缓存和 next_position 前抛 ValueError。
其他输入合法性由调用方保证，不要求为任意异常增加事务回滚。
原模型模式、参数和已有 .grad 必须保持不变。

完整模型的正确性参照使用相同参数、绝对位置与逐层窗口的稠密前向。
测试覆盖每个位置的 logits、各层缓存内容、GQA 布局和实际存储大小。
原全历史模型、只重跑最后 W 个原始 token 的结果，都不作为滑窗模型的相等参照。

## 4. 运行与完成标准

在仓库根目录执行：

```bash
# 验证两个 Attention 函数
.venv/bin/python -B -m unittest tests.exercises.test_efficient_attention -v

# 验证完整模型与缓存
.venv/bin/python -B -m unittest tests.exercises.test_window_llama -v

# 展示局部/在线输出与全量/多 token 追加 logits 的对齐
.venv/bin/python -B -m examples.efficient_attention_exercise_demo
```

要求这两组测试全部实际通过、无跳过，demo 正常完成，并检查实现符合上述局部/分块约束。
教师测试中的稠密数学参照可读，属于教师提供的验收代码，不是学习者独立设计测试的证据。
如果你自行补充反例测试，后续 review 会另行记录它实际证明的排错能力。

验证使用 CPU、固定种子，具体 dtype、种子和浮点容差写在测试文件开头。
配对数与最大单块大小不是进程峰值内存；普通 autograd 可以保存多个块供反向使用。
这里不要求自定义反向、GPU 融合实现、训练总显存线性或实测加速。
后续能力上报需依据学习者自己的解释、实现或排错证据；阅读或运行本参考答案不自动改变掌握状态。

## 5. 实现步骤与关键推导

### 5.1 `local_attention`：先找范围，再打分

这个函数一次处理一个 query 位置，同时计算所有 batch 和 head。
不同 query 的窗口长度可能不同，因此逐行切出 K/V 更容易直接满足“只计算窗口内配对”的要求。

1. **把位置条件转换成切片范围。** 对 query 的绝对位置 `p`，允许的 key 位置是
   `[p-W+1, p]`。`searchsorted` 在递增的 `k_positions` 中查找插入下标；
   默认返回第一个大于等于目标值的位置，`right=True` 返回第一个严格大于目标值的位置。
   因此 `left:right` 恰好覆盖这个闭区间内的 key。
2. **检查当前窗口内的 PAD。** `key_valid[:, left:right]` 的 shape 是 `(B,w)`，
   其中 `w=right-left`。每个 batch 至少要有一个 True；否则整行无定义，直接抛 `ValueError`。
3. **只计算局部分数。** 当前 Q 是 `(B,H,1,Dk)`，局部 K 转置为 `(B,H,Dk,w)`，
   点积得到 `(B,H,1,w)`，并乘 `1/sqrt(Dk)`。窗口外的位置从未进入这个矩阵乘法。
4. **屏蔽 PAD、归一化并读取。** 把 PAD 分数设为 `-inf`，使其 softmax 权重为零。
   权重乘局部 V `(B,H,w,Dv)`，得到当前输出 `(B,H,1,Dv)`。
   最后沿 query 轴拼成 `(B,H,Tq,Dv)`。

例如 `p=14、W=4、k_positions=[7,10,11,14,17]`：

| key 绝对位置 | 7 | 10 | 11 | 14 | 17 |
|---|---|---|---|---|---|
| 与 query 的距离 | 7 | 4 | 3 | 0 | -3 |
| 是否进入局部 QK | 否 | 否 | 是 | 是 | 否 |

这里 `left=2、right=4`。距离等于 W 的位置 10 已经在窗口外。
若某个 batch 的位置 11 是 PAD，只屏蔽它，不会为了凑够 W 个 key 再把位置 10 补进来。
若位置 11 和 14 都是 PAD，该 batch 的这一行报错。

### 5.2 `online_attention`：用可合并的分子、分母替代整张权重表

“在线”表示读到一个 key 块就更新累计状态。`query_block` 控制一次处理多少行 query，
`key_block` 控制一次读取多少列 key，两者都不改变允许读取的集合。
下面固定一个 batch、一个 head 和一个 query 行解释；代码同时处理整个 query 块。

令 `s_j=q·k_j/sqrt(Dk)` 为第 j 个合法 key 的分数，`v_j` 是它的内容向量，shape 为 `(Dv,)`。
对目前已读过的合法 key 集合 J，维护：

```text
m = max(s_j)                    代码：running_max
l = sum(exp(s_j - m))           代码：normalizer
u = sum(exp(s_j - m) * v_j)     代码：weighted_sum
最终输出 = u / l
```

m、l 是每行一个标量；u 是每行一个 `(Dv,)` 向量。对实际 query 块，它们分别是
`(B,H,Qb,1)`、`(B,H,Qb,1)`、`(B,H,Qb,Dv)`，Qb 表示当前块的实际 query 数。
u 尚未除以分母，不是单块的 Attention 输出。

**为什么必须缩放旧状态？** 假设新块使最大分数从 m 变成 m'，旧状态用的是
`exp(s_j-m)`，新状态必须统一成 `exp(s_j-m')`。对任一旧 key 都有：

```text
exp(s_j - m') = exp(m - m') * exp(s_j - m)
```

因此对已有有效历史和有效新块，合并公式为：

```text
m'    = max(m, 当前块的最大合法分数)
alpha = exp(m - m')
p_j   = exp(当前块分数_j - m')
l'    = alpha * l + sum(p_j)
u'    = alpha * u + sum(p_j * v_j)
```

代码中的 `old_scale` 是 alpha，`unnormalized` 是当前块的 p。
`unnormalized @ values` 一次完成所有 query 行的加权内容求和。
旧分子和旧分母必须同时乘 alpha；只缩放其中一个会改变输出。
有效行的 m' 不小于参与合并的任何分数，指数中的差不大于零，因此不会计算巨大正数的指数。

看一个能区分正确合并与错误平均的例子：固定 `B=H=Tq=Dk=Dv=1`、`q=1`，
让四个 K 值产生分数 `[0,0,ln(3),ln(3)]`，V 值为 `[0,0,10,10]`，
query 位置为 3、key 位置为 `[0,1,2,3]`，所有 key 有效。取 `key_block=2、window=None`。

| 处理进度 | m | l | u | u/l |
|---|---:|---:|---:|---:|
| 第一个 key 块 | 0 | 2 | 0 | 0 |
| 合并第二块，旧状态乘 `1/3` | `ln(3)` | `2/3+2=8/3` | 20 | 7.5 |

整行权重应为 `[1,1,3,3]/8`，输出确实是 7.5。
若各块分别做 softmax，再平均块输出，会得到 `(0+10)/2=5`，因为两个块的总权重本来就不同。
给所有分数统一加 1000 或减 1000，不会改变正确输出。

**对应代码的执行顺序：**

1. 切出一个 query 块，把 m/l/u 初始化为 `-inf/0/0`。
2. 遍历 key 块，仅生成当前 `(B,1,Qb,Kb)` 权限，Kb 是当前块的实际 key 数。
   全块在全部 batch/query 中都不可读时，直接跳过 QK。
3. 计算当前 `(B,H,Qb,Kb)` 分数并屏蔽非法项；逐行求新最大值。
4. 先构造安全的减法基准，再计算 alpha 和 p，按上式更新 l/u/m。
5. 全部 key 块读完，确认每行 l 大于零，才执行 `u/l`。
6. 沿 query 轴拼接输出块。代码始终没有拼回完整的分数、权重或权限表。

### 5.3 空块与空行：为什么 `safe_max` 必不可少

屏蔽分数为 `-inf`，因为 `exp(-inf)=0`，被屏蔽位置不贡献权重。
但在还没有读到合法 key 的行，旧 m 和本块最大值都可能是 `-inf`；
直接计算 `m-m'` 或 `score-m'` 会出现 `-inf-(-inf)=NaN`。

实现保留真实状态 `new_max=-inf`，只在作为减法基准时把它替换为零：

```python
safe_max = torch.where(
    torch.isfinite(new_max), new_max, torch.zeros_like(new_max),
)
old_scale = (running_max - safe_max).exp()
unnormalized = (scores - safe_max).exp()
```

`isfinite` 判断元素是否为有限数；在本题契约下，最大值要么有限，要么是空状态的 `-inf`。
`torch.where` 按条件选择张量元素，并不是短路执行，所以先选安全基准，再计算减法和指数。
这样可以在前向和求导过程中避免产生上述 NaN。

| 旧历史有合法 key | 当前块这一行有合法 key | 更新行为 |
|---|---|---|
| 否 | 否 | 安全基准为 0，alpha 和 p 都为 0，保持 `m=-inf、l=0、u=0` |
| 否 | 是 | 新最大值有限，alpha 为 0，直接用当前块建立状态 |
| 是 | 否 | 新最大值等于旧 m，alpha 为 1、p 为 0，完整保留旧状态 |
| 是 | 是 | 按正常合并公式更新 |

不能把真实 m 从一开始就固定初始化为 0：若所有合法分数都是很大的负数，
减 0 后的指数可能全部下溢为零。保留 `m=-inf` 能让第一份有效分数建立正确的最大值基准。
也不能因当前块某一行为空而直接报错，因为后续块可能为它提供合法 key。
只有处理完所有块，该行 l 仍然为零时，才抛 `ValueError`。

### 5.4 `window_block`：把 Attention 接回既有模型

模型输入 x 为 `(B,n,C)`，n 是本次新 token 数，C 是模型特征宽度。
Hq/Hkv 分别是 query/KV 头数，D 是每头宽度，`C=Hq*D`。

1. **投影新 token。** `project_rotary_qkv` 先调用 `norm1`，再投影、拆头，
   得到 `Q:(B,Hq,n,D)` 和 `K/V:(B,Hkv,n,D)`。
   它按传入的绝对位置对新 Q/K 应用 RoPE；V 不旋转。
2. **拼接读取集合。** 有历史时沿 token 轴拼接旧 K/V 和新 K/V，同时拼接对应绝对位置。
   历史 K 已在生成时旋转过，直接复用。没有历史时，读取集合就是本次全部 K/V。
3. **按 GQA 规则临时扩头。** 每个 KV 头连续供 `Hq/Hkv` 个 query 头读取。
   例如 Hq=6、Hkv=2，query 头 0/1/2 读 KV 头 0，3/4/5 读 KV 头 1。
   `repeat_interleave(..., dim=1)` 生成这种顺序；持久缓存仍保留扩展前的 Hkv 份。
4. **读取并完成块输出。** `online_attention` 返回 `(B,Hq,n,D)`，
   `finish_block` 合头、乘 Wo、加第一条残差，再完成 norm2/FFN 和第二条残差，得到 `(B,n,C)`。
5. **最后裁剪缓存。** 成功后保留末尾至多 W 条 K/V/位置，并调用 `clone` 分配独立存储。
   无缓存的全量调用不创建请求状态，整个计算保留求导路径。

为什么第 5 步必须在读取之后？例如 W=3，旧缓存位置为 `[8,9,10]`，
本次新位置为 `[11,12,13,14,15]`：

| 本次 query 位置 | 应读取的 key 位置 |
|---|---|
| 11 | 9、10、11 |
| 12 | 10、11、12 |
| 13 | 11、12、13 |
| 14 | 12、13、14 |
| 15 | 13、14、15 |

如果一拼接就裁成最后三条 `[13,14,15]`，位置 11 和 12 的计算已经失去所需数据。
正确过程是：完整拼接 `[8..15]`，由每行窗口决定读取集合，算完后再保存 `[13,14,15]`。
第一批输入长于 W 时也遵守同一个顺序。

`clone` 解决的是存储问题：切片虽然只显示三条，但仍可能引用旧大张量的完整底层存储。
对 K/V 使用 `clone(memory_format=torch.contiguous_format)`，明确获得独立、连续且大小匹配的存储。
本接口按约定保留 W 条；下一次紧邻位置实际最多读取其中最后 W−1 条历史，再加自身的新 key。
多保存的最旧一条会被位置权限屏蔽，不影响数学结果。

### 5.5 `window_step`：绝对位置、缓存长度和 token ID 各司其职

`input_ids` 存词表编号，用于查 embedding；`state.next_position` 存下一 token 的绝对位置，
用于 RoPE 和窗口判断；缓存长度只表示当前还保留多少条 KV。
例如处理完上面的 chunk 后，`next_position=16`，缓存长度为 3，二者不能互换。

1. 计算 `end_position=state.next_position+n`。若超出 `model.L`，立即抛 `ValueError`，
   此时还没改任何一层缓存。`end_position==model.L` 合法，因为右端不包含在位置序列中。
2. 用 `arange(next_position,end_position)` 构造本次位置，用 `token_table[input_ids]` 查新输入表示。
3. 逐层调用 `window_block`，每层传入该请求自己的缓存和同一份本次位置。
4. 经过最终归一化和词表投影，返回所有新位置的 logits，shape 为 `(B,n,N)`，N 是词表大小。
5. 将 `next_position` 更新为 `end_position`。之后追加从这个新位置继续。

缓存路径由调用方使用 `torch.no_grad()` 关闭求导，核心函数不会自行切换模式或清理已有梯度。
每次 `new_window_state` 都创建独立请求状态；`reset()` 清空各层缓存并恢复配置的起始位置。
窗口驱逐不会重置绝对位置，也不会扩大 `model.L` 的容量。

### 5.6 怎样阅读验证结果

底层测试同时验证前向、Q/K/V 梯度、非连续输入、PAD、非连续绝对位置、极端分数和空行，
并观察实际矩阵乘法的 shape，检查局部读取确实少算配对、在线版本没有超出块预算。
模型测试使用独立的同窗口稠密参照，比较每个位置的 logits、参数梯度、缓存内容与存储大小，
还覆盖逐 token、长 chunk、请求交错、重置和越界失败后的状态保持。

示例展示两条对齐关系：

- 相同 Q/K/V 和窗口下，局部实现与在线实现的输出在容差内一致。
- 相同参数、起始绝对位置和逐层窗口下，全量前向与分段追加的每个位置 logits 在容差内一致。

不要改用原来的全历史模型作为滑窗模型的相等参照：两者允许读取的集合不同。
也不要只拿最后 W 个原始 token 重跑来替代多层滑窗缓存，缓存中的高层表示可能已包含更早信息。

这份实现验证的是 CPU 数学、求导和执行范围。全历史在线 Attention 仍有二次数量的 QK 配对；
分块只限制一次生成的分数块大小。Python 循环和权限遍历本身有开销，普通 autograd 也可能保存
多个块用于反向传播，所以本题结果不代表已经获得 GPU 加速或训练总内存的线性上界。
