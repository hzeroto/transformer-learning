# 学习计划审查：从后端经验连接到模型实现与 AI Infra

审查对象为本仓库的知识地图、掌握记录、学习者画像与四份练习。此日期是审查时间，
不是教学进度或掌握期限。后续执行路线见 [roadmap.md](../roadmap.md)。

## 判断

原计划覆盖面较完整，强调shape、数据流和可验证证据，这些原则应保留。
主要问题是知识节点被过度拆成教学停顿，缺少早期训练闭环，部分依赖与验收标准不适合
“已有后端经验、目标转向AI Infra”的学习者。优化重点是更早整合与测量，不跳过数学先修。

审查时确认15个节点已有掌握证据，当前为 `learning-mechanics.logits-probability`。
本次重新执行的36项Python测试和7项前端测试全部通过，无跳过。
这支持已有实现的正确性，不证明尚未实现的训练循环或Attention已经掌握。

## 具体发现与改动

| 发现 | 原仓库证据 | 改动及原因 |
|---|---|---|
| 训练机制有概念、缺首个集成产物 | 七个learning-mechanics节点，没有可训练基线节点 | 增加tiny-training-loop，以一份作业验证输出、损失、更新、标签与生成 |
| 测试能力被无关技术栈阻塞 | testing-debugging唯一先修为numpy-attention | 测试前移至现有PyTorch基础之后；NumPy作业保留为可选 |
| 因果语义被优化器知识间接阻塞 | next-token依赖train-infer，再阻塞attention.mask | next-token基于词表、输出概率与有效标签即可讲解；完整训练行为仍在G1验收 |
| GPT路径包含不必要的双序列前置 | decoder依赖sequence-roles | 双序列放到Encoder–Decoder分支，mini-GPT先完成 |
| 缩放和归一化需要未显式安排的统计基础 | scaling要求点积方差推导；layer-norm要求手写 | 增加轻量mean-variance桥接，在用到时学习 |
| 拆头合头缺少存储语义 | multi-head仅依赖轴交换 | 增加tensor-layout，验证非连续输入和实际复制 |
| 概念与实现要求重复完成同一块 | block.complete与transformer-block均要求闭卷实现 | 同一关卡分别提供概念与实现证据，不重复写一遍 |
| 性能反馈太晚 | performance先修包含flash-attention | 增加cost-model与benchmarking，mini-GPT后先测基线，最后再综合分析优化 |
| T5缺乏具体实现验收 | 只有architecture.t5的文本到文本任务 | 增加implementation.t5及结构/片段去噪要求 |
| LLaMA组件零散，缺少组合验收 | 有RMSNorm/SwiGLU等节点，无模型节点 | 增加implementation.llama，逐项替换并验证缓存与位置一致性 |
| 原目标要求RoPE与ALiBi，地图只要求二选一 | rope-alibi标准为至少一种 | 独立RoPE节点；保留旧ID作为ALiBi与RoPE对照节点，两种都实现 |
| 原版位置表示没有单独落地 | 已掌握position只覆盖learned table | 增加sinusoidal-position，仅在原版分支前置 |
| 关键技术措辞可能误导 | KV“严格对齐”、滑窗“降低计算量” | 改为浮点容差对齐；要求区别稠密mask与实际局部算法 |
| 毕业依赖可绕过用户原目标 | capstone未覆盖BERT/T5/LLaMA组合/滑窗/MoE | 补齐依赖与关卡覆盖表，维持各任务适当实现深度 |
| 画像留有旧时间预期与初始能力描述 | “约两周”、仍需建立基础轴直觉 | 去除时间目标，记录已验证基础与后端转AI Infra方向 |

知识图保留原有ID及历史证据，新增节点用于表示真正缺口；节点变多不代表需要增加同等数量的作业。
原有掌握状态没有降级、重置或因审查而升级。未作答的输出列映射、完整输入管线、
Attention及模型实现仍按原状态等待真实证据。

## 边界判断

- 现有 `foundation.python-pytorch` 只覆盖基础Tensor与断图识别；参数注册、优化器状态、
  布局、设备与混合精度要按需讲解，不能拿新API突然出题。
- 已有位置作业只证明内容表与可学习位置表查表相加，不能覆盖正弦、RoPE或ALiBi。
- 简单模型能过拟合一个批次，仅证明训练与表达能力通过该检查；另用未见输入检测泄漏、
  缓存状态和泛化边界，不把生成几个合理字符当作唯一毕业证据。
- 后端接口、状态和测试经验可以迁移；矩阵求导、数值稳定和设备执行机制仍需要实际实验。
- 本仓库完成后，AI Infra仍有服务调度、分布式训练与算子运行时等独立工作；不把它们提前
  变成理解Transformer的先修，也不承诺完成本仓库即能覆盖岗位全部要求。

技术事实核查来源列在 [路线的核查依据](../roadmap.md#核查依据)。学习顺序是根据当前证据、
知识依赖与职业方向做的教学判断，不是论文规定的唯一顺序。
