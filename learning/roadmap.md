# Transformer 能力关卡路线

本路线服务于后端工程师转向 AI Infra：先独立实现并验证模型，再解释模型执行的成本。
不按时间安排，不要求先学完一个学科或背完全部 API。

`map.json` 保存知识点和必要依赖；本文件组织学习与工程关卡；`progress.json` 保存
实际掌握证据。图中的主题分组不是授课顺序，节点数量也不是完成度百分比。
同一份实现可为多个节点提供各自可核验的证据，不能因综合测试通过就批量推定所有知识已会。

## 从当前进度继续

本次审查的起点是 `learning-mechanics.logits-probability`。已有四份基础练习，
基础张量、输入表示、链式法则、autograd断图识别和稳定Softmax的证据继续有效。
当前状态随 `progress.json` 更新，不依赖本段快照。

下一项综合作业规划是一个很小的模型：

```text
文字/编号序列
  → 错开一位的输入与目标标签
  → E[input_ids]
  → 共享输出矩阵产生词表分数
  → 稳定交叉熵
  → 自动求导
  → 手写SGD更新与清梯度
  → 用更新后的参数继续预测
```

SGD指沿负梯度方向更新参数的基础方法。先讲清输出层、标签、对数与损失，再讲更新，
逐步完成同一份作业。本审查只确定任务，不提前填入学习者实现或上报新掌握状态。
上次尚未回答的输出词表映射检查可并入作业，不需要停留在反复确认定义上。

这份基线只根据当前token预测下一token，不包含上下文混合。使用每个token有确定后继的
玩具数据验证它能学习；再展示同一token因更早上下文而需要不同答案的例子，说明加入
Attention的具体动机。不能让表达能力不足的基线在任意语料上强行拟合到零损失。

## 关卡与产物

推荐顺序为 G1 → G2 → G3，之后优先 G4；G5 从 G3 分出，G6 从 G4 的基线继续。
G4/G5/G6 都完成后进入 G7。分支表示先修允许的安排，不要求学习者同时学习多个新知识块。

| 关卡 | 集中解决的问题 | 可验收产物 |
|---|---|---|
| G1：最小训练闭环 | 模型怎样根据错误改变参数 | 当前token预测下一token的小模型、损失曲线、标签/PAD/更新测试、一次独立故障定位 |
| G2：Attention 与 MHA | 一个位置怎样按内容读取其他位置 | 基础Tensor实现的单头与多头、shape与索引说明、前向/梯度对齐、可见范围测试 |
| G3：mini-GPT | 怎样把组件组成可训练、可生成的完整模型 | 完整输入管线、Pre-LN块、小数据训练与验证、生成、保存恢复、第一份性能基线 |
| G4：LLaMA 风格与增量推理 | 怎样替换组件并复用历史计算 | RMSNorm/SwiGLU/RoPE、MHA/MQA/GQA、缓存模型、逐步logits对齐和KV字节账本 |
| G5：原版与架构对照 | 可见范围、输入来源和训练目标怎样改变架构 | 原版Encoder–Decoder、mini-BERT、mini-T5，共享基础组件并分别验证目标构造 |
| G6：高效执行与迁移 | 优化究竟改变数学、存储还是执行方式 | ALiBi、真实局部窗口读取、在线归一化分块Attention、MoE/ViT玩具实现、性能对照报告 |
| G7：独立重建 | 离开现有答案还能否实现、测试和解释 | 从接口规格重建核心模型，通过未见测试；完成一个未实现过的变体与消融实验 |

### G1：训练闭环的边界

主要节点：`block.linear`、`learning-mechanics.forward`、`learning-mechanics.logits-probability`、
`learning-mechanics.loss`、`learning-mechanics.backward-update`、`learning-mechanics.train-infer`、
`learning-mechanics.next-token`、`learning-mechanics.tiny-training-loop`。

先使用基础Tensor与autograd完成一次手动更新。随后解释参数注册、`nn.Module`、
`nn.Parameter`及保存状态等组织机制；这些是新内容，不能由“Python/PyTorch基础已掌握”推定。
SGD的更新先自己写，使用更复杂优化器前解释其状态和作用，允许随后使用 `torch.optim`。

验收包括：输入与标签错开一位、只平均有效标签的损失、极端分数下交叉熵有限、
参数确实更新、梯度清理正确。使用有限差分或参考梯度校验，并独立补一个能抓到实际错误的测试。
这里开始积累 `implementation.testing-debugging` 的证据，不等待 NumPy 或 Attention。

### G2：把 Attention 当作一个完整计算来理解

按数据流逐步讲解点积 → Q/K/V来源 → 两两分数 → 缩放 → 权重 → 读取 → mask。
讲缩放前就地补 `foundation.mean-variance`；不要求提前修完概率论。
之后先完成单头，再补 `foundation.tensor-layout`，把相同计算推广到多头。

基础约定是 `C = H × Dh`，需要区分它与更一般的投影维度选择。头对应不同参数投影，
不能把多头解释成对完全相同的分数重复运算。

统一验收前向、梯度、非连续输入、不同Q/K长度及拆合头的精确索引关系。
在已讲清规则后，用未来token扰动、PAD扰动和全屏蔽行检查定位错误。
数值对齐预先声明dtype、设备和误差容限，不以逐位相等作为一般浮点算法的要求。
`implementation.pytorch-mha` 和概念节点可以使用同一份实现分别验收。

`implementation.numpy-attention` 保留为可选后端比较，不进入默认先修链。

### G3：先完成一个 Transformer

逐步补非线性激活、FFN、残差、LayerNorm和Dropout，说明Pre-LN与Post-LN排列。
mini-GPT使用明确的Pre-LN配置，原版分支再实现Post-LN对照。
概念块 `block.complete` 与代码块 `implementation.transformer-block` 合并验收，
不用把同一模块写两遍来满足两个状态字段。

把已有查表、补齐、位置表示串成原始文字到模型输入的完整管线，随后验证
`implementation.mini-gpt`。先用简单字符或小词表，不强制手写BPE tokenizer。

必须提供：

- 因果信息不泄漏、PAD处理正确、参数被正确注册且有梯度的测试。
- 一个明确可拟合小批次的训练结果，并用独立验证集区分记忆与泛化；不要求小模型理解自然语言。
- 生成循环及模型配置、参数、优化器状态的保存恢复；对比恢复前后的确定性输出。
- 对初始化、学习率、梯度范数和NaN的基本排错。首次使用Adam/AdamW、梯度裁剪等机制先解释。
- `systems.cost-model` 的参数/激活账本，以及 `systems.benchmarking` 的未优化基线。

### G4：把模型变成可以正确缓存的生成器

先在已通过测试的mini-GPT上实现KV Cache，分清两个计算阶段：
prefill一次处理已有前缀并建立缓存，decode每次处理新token并追加缓存。
缓存不是模型参数，而是与具体输入和请求绑定的运行状态。

再逐项加入RMSNorm、SwiGLU、RoPE和可配置KV头数。MHA/MQA/GQA、缓存和新位置表示
各有独立测试，然后在 `implementation.llama` 中集成。RoPE与ALiBi分开学习，
ALiBi不再作为实现LLaMA风格模型的前置条件。

缓存验收逐位置比较全量与增量logits；先固定等价位置、可见范围及模型参数，并关闭随机失活。
覆盖前缀长度变化、多token追加、非零位置偏移、容量边界和重置；请求间状态不得串用。
KV头可以为Q头共享，但不能把共享后的KV永久复制成所有Q头又宣称获得相同缓存节省。

普通等长全量缓存的账本需能从张量形状推导：

```text
KV有效数据字节数
  = 2 × 层数 × B × 缓存长度 × KV头数 × 每头维度 × 每元素字节数
```

2对应K和V；该式不包含权重、工作区、分页浪费和分配器预留。
混合精度和 `systems.serving-bridge` 可在基线与缓存已经成立后加入，不阻塞核心模型。

### G5：保留原目标，按结构差异实现

复用模型积木，新增的每个分支必须有独立的输入/目标语义和测试。

| 架构 | 必须新增或明确的内容 | 代表性验证 |
|---|---|---|
| 原版Transformer | 正弦位置、Post-LN、源/目标两侧、跨序列Attention、目标右移 | 长度不同的源目标、全部mask、复制/反转的未见组合；声明与原论文训练配方的差别 |
| BERT | 双向可见、句段表示、被选中位置的遮盖目标、分类头 | 遮盖选择与替换策略分离、仅正确标签受监督、小型句对预测；若省略NSP，标注为BERT风格简化模型 |
| T5 | 选定版本的归一化/FFN、相对位置分桶偏置、片段遮盖与哨兵token | 哨兵顺序、输入输出构造、decoder标签对齐及小型去噪训练 |

先列配置，再编码，避免把普通Encoder–Decoder改个名称就当成T5，或把自定义LLaMA风格配置
声称为某个已发布模型的完整复现。小词表、小隐藏维度可以用于结构验证，训练规模不作为通关门槛。

### G6：优化要区分语义、算法和实现

ALiBi与RoPE都要实现；滑窗除了窗口mask，还要有不生成完整 `T×T` 分数矩阵的局部读取实现。
稠密矩阵算完再加mask，不会自动减少已经执行的算术量。

FlashAttention先学习在线归一化与分块：每次只处理一部分分数，并正确合并归一化状态。
CPU版本用于验证数学与存储行为；Python循环不等于高性能GPU内核。
完整精确Attention的二次算术量与显存读写量是不同问题。

MoE做到玩具路由、专家计算、负载分布和梯度检查；ViT做到图像分块、映射为序列并复用Encoder
完成小型分类。前者的跨设备专家通信、后者的大规模视觉训练不属于此处的基础验收。

性能报告采用“假设 → 基线 → 一项改动 → 正确性复测 → 性能复测 → 解释”的结构。
没有加速也可以是合格结论，前提是实验可复现且能解释成本；不规定必须快几倍。

### G7：完成标准与范围

闭卷是不给现成模型答案，可以查Python语法和基础Tensor API。根据接口规格重建，
覆盖隐藏测试与一个未讲过具体答案、但所需概念已学的排错案例。

核心毕业包括：GPT、原版Transformer、BERT/T5、LLaMA风格、Pre/Post-LN、RMSNorm、
SwiGLU、正弦位置、RoPE、ALiBi、MHA/MQA/GQA、KV Cache、采样、滑窗、分块Attention、MoE与ViT。
实现标准和理解/玩具实验的边界以上述关卡及 `map.json` 为准。

AI Infra桥接单独记录：能解释自己的模型怎样占用算力与内存，能定位一次执行瓶颈，
能描述有状态推理的服务接口。它不等于完整掌握AI Infra岗位的全部能力。

## 硬件与性能记录

核心正确性、布局实验、缓存一致性、资源账本与分块算法均可在CPU上缩小验证。
有MPS或CUDA时可以做对应后端实验，记录实际硬件和支持范围；不假定当前有CUDA设备。

基准至少记录模型配置、输入长度、batch、输出长度、dtype、设备、软件版本、线程设置、
热身、重复测量和计时范围。GPU异步执行需要正确同步或事件计时；profiling本身也有开销。
优化对比固定workload与数值约定，分别测prefill和decode，报告内存和延迟/吞吐。

模型prefill耗时不等于服务端到端首token延迟；后者还可能包含排队、分词和传输。
“prefill偏计算、decode偏带宽”只作为待验证的常见瓶颈假设，不是所有batch/模型/设备的定律。

融合GPU内核、CUDA计时和设备显存收益需要相应硬件实测，不能用CPU对齐替代。
硬件不足时明确记为未验证；这不阻塞Transformer核心数学与代码毕业。

## 后续独立 AI Infra 项目的接点

这里只建立接点，不将以下工程变成学会Transformer前的必修阻塞：

- 推理服务：连续批处理、分页KV管理、流式输出、取消与资源释放、背压、抢占及服务压测。
- 训练系统：数据/张量/流水线/专家并行、优化器与参数分片、通信成本和故障恢复。
- 算子与运行时：CUDA/Triton、算子融合、图编译、内存调度和跨设备通信。

选择后续方向时，以已完成的模型与性能报告作为输入，再定义一个有负载、指标和正确性约束的项目。
无需在本次审查中创建新项目、购买GPU或实现生产服务。

## 教学与上报约定

1. 每次集中讲清一个知识块，新术语和新API先定义。
2. 作业可覆盖多个已讲解的相关节点；教师提供接口与测试，学习者完成核心代码。
3. 优先用反例、性质和组合任务验证，不为填状态反复要求复述定义。
4. 通过教师测试、独立解释、独立设计测试是不同证据，分别记录。
5. 只有可观察的新证据才调用上报脚本升级掌握状态；课程重排不改变历史掌握结论。
6. 使用基础Tensor和autograd实现核心数学；官方模块与融合Attention在自写版本验收后才用于参照或优化。
7. 文档中的G编号表示能力关卡，不是课时，也不附带时间估计。

## 核查依据

以下来源用于核对技术边界；推荐学习顺序是针对本学习者的教学判断。

- 原版结构、Post-LN与正弦位置：[Attention Is All You Need，§3](https://arxiv.org/html/1706.03762v7#S3)。
- T5的位置分桶、归一化和片段去噪：[T5论文，§2.1与§3.1.4](https://arxiv.org/html/1910.10683v4)。
- LLaMA的预归一化、RMSNorm、SwiGLU与RoPE：[LLaMA论文，§2.2](https://arxiv.org/html/2302.13971v1#S2.SS2)。
- 视图与复制：[PyTorch Tensor Views](https://docs.pytorch.org/docs/2.14/tensor_view.html)；浮点容差：[Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)。
- 自动求导与模式差异：[Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html)；设备异步执行：[CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)。
- 分块减少设备内存读写：[FlashAttention原论文](https://arxiv.org/abs/2205.14135)。
- 推理调度与延迟/吞吐取舍：[vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)；KV内存管理：[PagedAttention论文](https://arxiv.org/abs/2309.06180)。
