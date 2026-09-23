# 013：把三个新组件接成可训练、可缓存的 LLaMA 风格模型

对应已读完的[合并讲义](../../notes/llama-style-model.md)。本题把 RMSNorm、SwiGLU、RoPE 放进同一个模型，复用你已经验收的 GQA、KV 容器、损失和训练循环。

不再追加口头题，也不要求重新写 GQA、缓存容器或生成循环。你完成六处核心实现；教师提供构造、调用入口和独立测试。

## 六处 TODO，一份数据流

| 文件 | 需要完成 | 输入 → 输出 |
|---|---|---|
| [components.py](components.py) | `RMSNorm.forward` | `(B,T,C)` → 同 shape |
| 同上 | `SwiGLU.forward` | `(B,T,C)` → 同 shape 的 FFN 更新量 |
| 同上 | `apply_rope` | 已拆头 `(B,H,n,D)` 与位置 `(n,)` → 同 shape |
| [model.py](model.py) | `LlamaBlock.forward` | 本次表示、绝对位置、权限、可选缓存 → 本次块输出 |
| 同上 | `LlamaLM.forward` | 全量 ID 与 key 有效性 → `(B,T,N)` logits |
| 同上 | `llama_model_step` | 新增 ID 与各层缓存 → `(B,n,N)` logits，追加缓存 |

推荐按表中顺序写。构造器已经注册好正确参数，不需要修改；核心实现只能写在这两个文件内，不改旧练习或教师测试。

**全量和增量共用同一个 `LlamaBlock.forward`。** 两个模型入口只负责构造各自的位置、权限和状态连接，不复制第二份块内数学。这样可以集中验证 RoPE 与缓存组合，而不重复劳动。

整体计算仍是：内容查表 → 多层块 → final RMSNorm → 独立词表投影。本次没有可训练 P 表；不得继承调用旧 MiniGPT 那个自动加 P 的 forward。

## 本题固定约定

尺寸名称：B 是 batch，C 是主干宽度，N 是词表大小；T 是全量长度；t 是调用前缓存长度，n 是本次新增位置数。`Hq/Hkv` 是查询/KV 头数，`D=C//Hq`，G 是 FFN 中间宽度。所有轴长为正。

- Pre-Norm 的两条残差；每层两份独立 RMSNorm，模型末尾再有一份。
- RMSNorm 只沿最后一轴计算均方根，不减均值，eps 放根号内；只有 `gamma(C,)`，无 beta。
- SwiGLU 有独立的 Wgate/Wup/Wdown，无偏置。SiLU 用基础乘法与 `torch.sigmoid` 表达，不直接调用 `F.silu`。
- Q/K 头宽相同且 D 为偶数；RoPE 对头内相邻特征配对，旋转全部 D 维，V 不旋转。
- `apply_rope` 的第 j 对频率为 `theta**(-2*j/D)`，j 从 0 起；角度来自传入的 positions。不能把非零起点偷偷重置成 0。
- GQA 采用已学的连续分组，每个 Q 头 h 读取 KV 头 `h//(Hq//Hkv)`。
- Dropout 为 0；词表投影独立，不与内容表共享权重。不要求改造成任一官方型号或加载官方 checkpoint。
- CPU float32/float64，输入浮点类型与模型一致，允许非连续张量；本题不测半精度、GPU或性能收益。

完整 shape 与局部接口约束写在每个 TODO 的 docstring 中；下面把最容易混的连接边界集中列清。

### 一个块为什么多收 positions 和 allowed

`LlamaBlock.forward(x, positions, allowed, cache=None)` 不猜本次调用处于哪个阶段：

| 对象 | 含义与 shape |
|---|---|
| x | 本次 n 个位置的本层输入 `(B,n,C)` |
| positions | 本次绝对位置，CPU long `(n,)`，所有 batch/head 共用 |
| allowed | CPU bool `(B,n,S)`，True 表示可读；所有 head 共用 |
| cache=None | 无历史状态，S=n；直接读取这批 K/V |
| cache 非空容器 | 历史长 t，S=t+n；追加后读取全部 K/V |
| 返回值 | 仅这次 n 个位置的输出 `(B,n,C)` |

无缓存时 positions 可以有非零起点，块只执行传入的位置和权限。有缓存时，调用方保证 positions 为 `t..t+n-1`；缓存数据没有 PAD。块不自己加一张从零开始的方阵 mask。

全量模型入口生成 `0..T-1` 和“因果 AND key 有效性”的权限；可复用 `make_causal_allowed`。PAD query 不清零，任一 query 没有可读 key 时沿用 ValueError。

增量模型入口生成 `t..t+n-1` 和 `(B,n,t+n)` 权限，本次第 i 个 query 只能读 `j<=t+i` 的 key。不使用 target_valid；它属于损失，不属于注意力权限。

### 缓存里究竟保存什么

继续使用 ex011 的 `LayerKVCache`，但末轴是紧凑 KV 宽度：

```text
cache.k：本层 norm1(x) 投影后、按原绝对位置旋转的 K
cache.v：本层 norm1(x) 投影后、没有执行旋转的 V
两者的持久 shape： (B, 已处理长度, Hkv*D)
```

旧容器注释中的 C 指其末轴宽度，不强制等于本模型 C。旧注释“不参与求导”不能理解为允许 detach：本题数学入口保留同一次计算图的历史依赖，推理调用者才用 no_grad。不同训练步之间不续用缓存。

只旋转新 Q/K，旧 K 不重复旋转。调用 `grouped_query_attention` 之前合回三轴布局；该旧接口接收的不是 `(B,H,n,D)`。临时拆合或临时 GQA 展开允许，但持久 KV 不能保存 Hq 份复制、额外整段容量或未旋转 K 的另一份历史。

请求缓存由调用方持有，每层、每请求独立；不登记为模型参数或 buffer，不进入 state_dict。

### 只检查与本课有关的失败边界

- `apply_rope` 遇到奇数 D 抛 ValueError。
- 块追加超出其 `cache.max_length` 时抛 ValueError，该缓存的值与长度保持原样。
- 全量长度 T 超过 `model.L` 时抛 ValueError。
- 增量 `t+n>model.L` 时，在改变任意一层缓存前抛 ValueError；所有层状态保持原样。

模型级调用保证 caches 长度正确、各层同长、batch/dtype/头宽合法，并且每个缓存的容量是 None 或至少 model.L。无需为这些已保证条件编写额外检查；不要求任意异常的事务回滚。

所有核心前向均不改输入、参数、已有 `.grad` 或 train/eval 模式，不内部调用 backward/no_grad/detach。允许按约定追加缓存。只返回 Tensor，不生成 token、不计算损失。

## 能复用什么、哪些留给你写

可以直接复用已有 `project_qkv`、`split_heads/merge_heads`、`make_causal_allowed`、`grouped_query_attention`、`LayerKVCache`。它们不替你完成本次归一化、门控、旋转或模型连接。

基础 Tensor 运算、autograd、`nn.Module`/`nn.Parameter` 可用。禁止 `nn.Transformer`、`nn.MultiheadAttention`、融合 Attention、现成 RMSNorm/完整 MLP 或导入教师参照代做核心。数学前向采用批量运算，不写逐 token/head/特征对的循环；堆叠块的层循环允许。

必要 API 提示见[讲义末尾](../../notes/llama-style-model.md#实现备忘新-api-只解释一次)：`torch.sigmoid`、`cos/sin`、`[...,0::2]`、`stack`。`cache=None` 表示该参数可以省略；类型注解中的 `LayerKVCache | None` 表示两种允许的对象，不会自动帮你处理分支。`new_caches(model)` 只创建独立空容器，没有任何 K/V 计算。

## 运行顺序

在仓库根目录运行；不要省略全仓命令的 `-t .`。

先完成前三个组件：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_llama_components -v
```

完成块与两个模型入口后，运行全部专项：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_llama_components tests.exercises.test_llama_model -v
```

然后跑综合验证：

```bash
.venv/bin/python -B -m exercises.ex013_llama_style.demo
```

[demo.py](demo.py) 是教师调用器，不用填写。它实际调用你的六处实现，依次检查：

1. 同一模型的全量与 `[3,2,1]` 分块 logits 对齐。
2. 讲义配置的 12936 个参数，以及长度 6、B=2、float32 下 1536 字节紧凑 KV；同时核对底层存储，不把逻辑 shape 当成实际节省。
3. 复用 ex010 的复制/反转任务与训练循环，在固定 8 条序列上拟合；最终 loss<0.05、整条输出准确率 100%。
4. 对这 8 个训练前缀，全量与缓存贪心生成相同，并匹配已训练的目标序列。
5. 在内存保存配置、参数和 Adam 状态，重建新模型后核对；不写磁盘 checkpoint。

固定配置为 C=24、Hq=6、Hkv=2、G=64、两层、N=11、容量128、eps=1e-5、theta=10000、CPU float32、单线程、初始化seed1301、采样seed17、Adam学习率0.01、batch8、160次参数更新。`--steps` 可调整实验迭代数，但不要用增加迭代替代排查错误。它验证小批次学习与连接，不要求重做泛化实验，不据此宣称通用语言能力。

教师适配缓存生成与新模型存档，是为了复用你已学的流程，不隐藏新组件答案。Python 的 `model: MiniGPT` 类型注解不会在运行时强制对象属于该类；旧训练函数只调用兼容接口，所以可以复用。旧 `load_checkpoint` 则在代码中真的构造 MiniGPT，不能直接用于本题，demo 已改为构造 LlamaLM。

最后跑回归：

```bash
.venv/bin/python -B -m unittest discover -s tests -t .
```

### 怎么读空框架的失败

本题一共 **35 项专项测试**：组件 15，块/模型 20。空框架只实际运行 6 个实现状态测试，各有一条明确的 TODO 失败；另外 7 个行为测试组暂跳过，因此会看到 `Ran 6`、`failures=6, skipped=7`，而不是 35 项全跑。这是未完成，不是完成或依赖错误。

逐步填写后，相关行为测试组自动启用；只跳过 NotImplementedError，其他异常会正常报错。最终要求 35 项全部运行通过、无跳过，demo 通过，旧测试无回归。demo 在空框架下明确提示未完成并以非零状态退出。

## 测试为什么能发现“同样写错却互相对齐”

教师测试有独立的逐对旋转、逐头读取及整模型参照，不用你的新组件产生预期答案。RoPE 除长度/共同位移性质外，还有非零位置、两种频率、非对称向量的已知值检查，能排除直接返回输入的假实现。

集成测试检查真实浮点 logits、每层缓存数值、历史 K 不变、V 未旋转、位置偏移、多 token 追加、参数与历史梯度、容量失败及请求隔离；不以形状相同或 argmax 相同替代数值验证。

专项种子为 1301/1307/1313；float64 使用 `rtol=1e-8, atol=1e-10`，float32 使用 `rtol=1e-5, atol=1e-6`；整数、权限与不应变化的数据按精确条件检查。非零角度与较明显的测试权重避免差异被小初始化掩盖。CPU 不做墙钟性能验收。

交付前教师在隔离的临时环境中用正确实现通过专项，并注入典型错误检查测试区分力；这些不是你的实现证据。答案未写进作业文件，六个 TODO 仍待完成。测试中的数学参照仅供排错，不允许从作业调用，也不算你独立设计了这些测试。

完成后把代码和 demo 输出交来即可，不需要再交一份问答作业。验收时分别记录 RMSNorm、SwiGLU、RoPE 和整模型的真实证据；创建框架不提前改变掌握状态。
