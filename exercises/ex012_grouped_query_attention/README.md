# 012：MQA / GQA 与紧凑 KV 缓存

把已完成的 MHA 推广到独立配置查询头数和 KV 头数，再接回同一套 Pre-LN 模型与增量前向。
讲义见 [MQA / GQA](../../notes/grouped-query-attention.md)。
概念检查已经收尾，本题直接通过实现验证，不重复讲义 A/B 口头题。

## 四处实现，一份完整数据流

只填写下面四处；构造、参数登记和 demo 已给出，不修改以前的作业或教师测试。

| 文件 | 待实现入口 | 验证什么 |
|---|---|---|
| [gqa.py](gqa.py) | `grouped_query_attention` | MHA / MQA / GQA 共用一套正确的分组读取 |
| [gqa.py](gqa.py) | `GQABlock.forward` | GQA 接入已学的 Pre-LN 残差块 |
| [model.py](model.py) | `gqa_block_step` | 只投影新位置，保存紧凑 K/V 并读取完整历史 |
| [model.py](model.py) | `gqa_model_step` | 根据缓存偏移取位置，返回每个新位置的真实 logits |

全量路径由已完成的 `MiniGPT.forward` 继承而来：

```text
input_ids → 内容与位置表示 → 多层 GQABlock → final_norm → vocab_proj → logits
```

增量入口返回本次所有新位置的 logits，不负责 argmax 或 EOS。
这样测试可以直接比较完整浮点分数；不以相同的生成 ID 掩盖数值错误，
也不要求再写一次上一课已经实现的生成循环。

## 固定的计算约定

`Hq=num_query_heads`，`Hkv=num_kv_heads`，`D=C//Hq`，`R=Hq//Hkv`。
本题要求 C 可被 Hq 整除、Hq 可被 Hkv 整除，所有尺寸为正。
查询头 h 对应 KV 头 `h//R`。交错分组也可以是一种模型设计，
但本题参数、全量路径与缓存路径统一采用**连续分组**，不能中途更换。

| 对象 | shape |
|---|---|
| 已投影 Q | (B,Tq,C) |
| 已投影紧凑 K、V | 各 (B,Tk,Hkv×D) |
| 分头后的 Q / K / V | (B,Hq,Tq,D) / (B,Hkv,Tk,D) / (B,Hkv,Tk,D) |
| 权限 allowed | bool (B,Tq,Tk)，True 表示可读 |
| 注意力 weights | (B,Hq,Tq,Tk) |
| 合头并乘 Wo 后的输出 | (B,Tq,C) |
| Wq、Wo | 各 (C,C) |
| Wk、Wv | 各 (C,Hkv×D) |

沿 key 位置轴归一化，分数缩放用 sqrt(D)；拼接 Hq 个读取结果后乘 Wo。
Hkv=Hq 时退化为旧 MHA，Hkv=1 时是 MQA。

可以复用自己的 `split_heads`、`merge_heads`、`project_qkv`、
`make_causal_allowed`。主 Attention 的打分、归一化、读取由你实现，
不能调用旧 MHA、高级 Attention 或融合算子代做；`torch.softmax` 可以使用。
原 project_qkv 的说明按 Q/K 等宽书写；其实现实际是三次独立矩阵乘，
本题已核对它能够接受不同输出宽度，复用时以本题 shape 表为准，无需改旧答案。
采用批量 Tensor 运算，不写逐 batch/head/query/key 的循环。
允许用分组轴广播，也允许在一次读取中临时 `repeat_interleave`，
两种写法都必须保持梯度。

### 权限与梯度

- 底层 `allowed=None` 表示全可见，不自动增加因果限制。
- 任意 query 的权限行全 False，抛 ValueError；头数配置错误同样抛 ValueError。
  参数构造有现成校验 helper，其余合法 shape 由调用方保证，不考通用防御性校验。
- 完整块使用因果规则与 key 有效性；input_valid 不是目标标签 mask。
  PAD query 不人为清零。缓存路径限定无 PAD，避免增加本课无关的缓存掩码状态。
- CPU float32/float64、输入和参数同 dtype。支持非连续输入，不要求固定输出 stride。
- 两个 Attention 返回值都保留求导路径；共享 K/V 的梯度要包含组内各查询头的贡献。
- 不修改输入/参数/已有 .grad，不在数学前向里调用 backward、no_grad 或 detach。

## 缓存接口：复用容器，明确宽度

复用 ex011 的 `LayerKVCache`，采用**拆头前**的持久布局：

```text
cache.k / cache.v：各 (B,S,Hkv×D)
位置轴仍是第 1 轴；len(cache) 仍表示已处理的 token 数。
```

原容器注释里的末轴 C，在本题实际传入的是 KV 宽度 `Hkv×D`。
容器本身沿时间追加，不依赖末轴等于模型宽度，因此无需改旧代码。

进入块时缓存已有 t 个位置，本次处理 n 个新位置：
本次 query i 对所有 key j 的权限为 `j<=t+i`；
输出只包含 n 个新位置，缓存追加后长度为 t+n。
K/V 必须来自本层 `norm1(x)` 的投影，不能保存原始 x 或别层的 K/V。

每层缓存只持有紧凑投影数据，不保存扩展到 Hq 的副本、不预分配额外容量。
这保证实际持有的 K/V 存储也能体现 Hkv 的减少。
临时展开的工作区另算；本题不要求优化这部分或达到某个加速倍数。

`gqa_model_step` 接收调用方提供的每层缓存，取它们的当前长度作为位置起点。
调用方保证缓存列表合法、同长、每层独立；若超过位置表容量，
必须在改变任何一层缓存前抛出 ValueError。
块级 cache.append 的容量错误同样不能破坏该容器已有内容。

缓存是请求状态，不写进模型成员、参数或 state_dict。
缓存前向本身保留梯度，调用者在推理时自行使用 `model.eval()` 和 `torch.no_grad()`。
本课固定 p=0，不重考 Dropout；梯度测试仅在同一次计算图里分块前向，
不要求跨训练步携带缓存。

## 已提供的框架与 API 提醒

- `GQABlock` 注册本课正确宽度的四份投影，复用已有 LayerNorm / FFN / Dropout(p=0)。
- `GQAMiniGPT` 只提供参数登记，并继承你已经实现的 `MiniGPT.forward`；
  它创建新模型，不转换或修改旧 MHA 存档。
- `nn.Module.__init__(self)` 初始化模块登记容器。本题构造直接调用它，
  是为了复用原 forward 而不先分配一套随后丢弃的旧 MHA 参数。
- `unsqueeze(dim)` 插入长度 1 的轴，可用于组内广播。
- `repeat_interleave(R,dim=...)` 将该轴的每项连续重复 R 次；
  `repeat` 的整段重复顺序不同，不能未经核对替换。
- `torch.arange` 产生位置编号；`reshape` / `transpose` 改变轴组织，
  先对应清楚 head、组内 head、token 和 D，再做矩阵乘。
- `tensor.numel()*tensor.element_size()` 是逻辑元素字节数；
  `tensor.untyped_storage().nbytes()` 是其底层 storage 字节数，
  共享 storage 需去重，不能冒充进程总内存。
- `named_parameters()` 提供登记后的参数名与张量，可用来核对投影参数量。
  本题不要求再写字节公式函数，直接复用 ex011 的 `kv_cache_bytes`。

## 运行与完成标准

在仓库根目录执行全部专项：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_grouped_query_attention tests.exercises.test_grouped_query_cache -v
```

完成前两个入口时，也可以只跑第一份测试：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_grouped_query_attention -v
```

实现后运行模型连接与账本 demo：

```bash
.venv/bin/python -B -m exercises.ex012_grouped_query_attention.demo
```

它会比较同一 GQA 模型的全量与分块 logits，
列出实际投影参数量、KV 有效字节数及去重后的底层存储。
对照讲义的公式核对输出即可，不再做已通过的概念题。

最后运行原有回归与新测试：

```bash
.venv/bin/python -B -m unittest discover -s tests -t .
```

专项共 **29 项**：Attention/Block 15 项，缓存/模型 14 项。
覆盖连续分组、MHA/MQA 退化、两种 dtype 与非连续输入、权限与梯度累加、
前缀与 PAD 保护、逐 token/多 token 缓存对齐、容量、请求隔离及实际存储账本。

**空脚手架预期**：四个核心入口尚未实现，给出 4 个明确的实现状态失败；
其余行为测试所在的 4 个类暂跳过。unittest 此时只计入实际运行的 4 项测试，
不会显示完成后的 29 项运行数。demo 给出未完成提示并以非零状态退出。
这不算完成，也不是依赖环境损坏。最终要求所有专项实际通过、无跳过，旧回归仍通过。

测试使用 CPU、小尺寸确定性输入及固定种子。float64 对齐使用
`rtol=1e-8,atol=1e-10`；float32 使用 `rtol=1e-5,atol=1e-6`；
精确映射、屏蔽项为零、存储与整数账本按精确结果检查。
模型输出接近零时主要看绝对容差，不要求不同分块运算逐位相等。
测试种子为 1201/1207/1213、401/409/419/421/431/433；demo 使用 421。
具体夹具见测试，容差不会因学习者输出不符而放宽。

教师测试包含可读的独立数学参照，供诊断与对齐，不供作业直接调用。
教师自检证明测试可用，不构成你已经完成实现或独立设计测试的证据。
本题完成后再根据真实实现与运行结果上报，不因创建框架更新掌握状态。
