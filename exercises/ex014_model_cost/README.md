# 014：给自己的模型建立成本与执行报告

复用 [ex013 的模型](../ex013_llama_style/model.py)，逐步完成：计算账本 →
存储核对 → 固定工作量计时 → 用执行记录解释一处差异。
本目录会承接整份实验；**FLOPs、存储统计和固定工作量计时代码已通过，
当前进入整节课第三个目标：[用执行记录解释一处差异](../../notes/model-cost-and-benchmark.md#61-用你已完成的计时函数找一个值得解释的差异)。**
前三个实现步骤分别完成了成本账本与基准计时，现在分析实测数据。

当前主线是解释 **decode 的主要矩阵 FLOPs 约为 prefill 的 1/16，耗时却还有约 72%**。
完整实测解释见[讲义第 6.4–6.6 节](../../notes/model-cost-and-benchmark.md#64-先检查矩阵乘法本身真的快了-16-倍吗)：
先重放真实矩阵乘法，再比较同样乘加量的一次批量计算与多次小调用，最后分区计时定位
完整模型保留下来的成本。教师提供可复跑入口，不要求学习者另写一套诊断框架：

```bash
.venv/bin/python -B -m exercises.ex014_model_cost.latency_breakdown_demo --output /tmp/latency-breakdown.json
```

## 第一步：FLOPs 账本（已通过，以下保留题面供查阅）

打开 [cost.py](cost.py)，实现 `estimate_matmul_flops`：输入模型尺寸和一次调用的
位置数，返回各部分及总 FLOPs。模型核心继续沿用 ex013。

这个函数就像一个计算预算器：改变输入长度，就能预测工作量怎样变化。
它是后续性能报告的第一列数据，现在使用 Python 整数运算即可。

函数签名中的 `*` 表示后面的参数按名称传递，例如 `B=2, n=7, S=7`；
`dict[str, int]` 是返回类型提示，表示“字符串键、整数值”的字典。

## 题目条件

| 参数 | 含义 | 讲义示例 |
|---|---|---:|
| B | 一批等长、无 PAD 的序列数 | 2 |
| n | 本次新输入的位置数 | prefill 为 7，单步 decode 为 1 |
| S | 本次追加后的总长度；历史长度是 S−n | 上述两次调用分别为 7、8 |
| C | 主干特征宽 | 24 |
| Hq / Hkv | query 头数 / KV 头数 | 6 / 2 |
| G | SwiGLU 中间宽 | 64 |
| layers | 块数 | 2 |
| N | 词表大小 | 17 |

输入都保证合法。每头宽 `D=C//Hq`，紧凑 KV 宽 `K=Hkv*D`；此处 K 是整数。
沿用 ex013 无偏置、独立词表投影的模型。统计一次前向，包含本次全部 n 个位置的
logits；历史已经有缓存时，不重新计算历史位置的投影和 FFN。

计数采用刚才讲过的规则：`(M,K0) @ (K0,N0)` 约为 `2*M*K0*N0` FLOPs。
有 batch 或头轴时，计入这些轴上重复执行的矩阵乘法。

## 沿模型前向填完七项

下面的块内 shape 都是**单层**；返回时前六项需要累计全部 layers。
所有输入表示均为本次新位置的 `(B,n,C)`，来自这一层相应的归一化或残差分支。

| 返回键 | 要数的操作 | 单层参与计算的 shape |
|---|---|---|
| `q_proj` | 本次表示投影成 Q | `(B,n,C) @ (C,C)` |
| `kv_proj` | 本次表示分别投影成新 K、新 V；合计两次 | 各 `(B,n,C) @ (C,K)` |
| `score` | 新 Q 读取追加后的全部 K，按 query 头分别打分 | `(B,Hq,n,D) @ (B,Hq,D,S)` |
| `value_read` | Softmax 后的权重读取对应的 V | `(B,Hq,n,S) @ (B,Hq,S,D)` |
| `out_proj` | 各头读取结果合并后乘 Wo | `(B,n,C) @ (C,C)` |
| `ffn` | SwiGLU 的 gate、up、down 三次投影 | 两次 `(B,n,C) @ (C,G)`；一次 `(B,n,G) @ (G,C)` |
| `vocab` | 全部块和 final RMSNorm 之后的词表投影，整模型仅一次 | `(B,n,C) @ (C,N)` |

GQA 每个 query 头读取自己所属的 KV 头，所以 `score` 和 `value_read` 按 Hq
份计算；持久 KV 只保存 Hkv 份。因果 mask 在完整打分之后应用，按稠密矩阵计数。
`total` 是七项之和，不再重复乘层数。

本账本只计上述矩阵乘法，忽略归一化、旋转、Softmax、SiLU、门控逐元素乘法、
残差、查表、mask 和复制。它不是全部精确 FLOPs，也不是运行时间、训练步成本或内存量。
查讲义 [第 2 节](../../notes/model-cost-and-benchmark.md#2-从矩阵乘法得到计算账本)
是允许的；不要从教师示例或测试导入答案，也不要把示例数字写死。

## 写完怎么检查

在仓库根目录运行：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_model_cost -v
.venv/bin/python -B -m exercises.ex014_model_cost.demo
```

本步已经验证为 **10 项通过、无跳过**。
讲义配置下，prefill 总量应为 374304，历史 7 追加 1 的总量应为 53856。
测试还覆盖不同 batch、宽度、层数、MHA/MQA/GQA 和多 token 追加，不能只通过这两个例子。

教师测试有基于 ex013 实际权重 shape 的独立计数参照，只用于验证；测试通过是
这份计算账本实现的证据，不代表整节成本/性能课程完成或独立测试设计通过。

存储题面保留在 [STORAGE.md](STORAGE.md)，两个函数的 16 项测试已通过。
[benchmark.py](benchmark.py) 的 12 项测试已通过（含教师补充的同步边界检查），题面和命令保留在 [BENCHMARK.md](BENCHMARK.md)。
算子记录入口由教师提供，可直接运行；本步由你根据记录判断性能改动是否值得验证：

```bash
.venv/bin/python -B -m exercises.ex014_model_cost.profiler_demo
```

基准原始数据保存在[这里](../../notes/assets/model-cost-and-benchmark/cpu-baseline-20260924.json)，
讲解与当前判断题见[讲义第 6 节](../../notes/model-cost-and-benchmark.md#6-账本给出怀疑profiler-找到实际执行的位置)。

已完成一项教师对照实验：[同一步复用 RoPE 系数](ROPE_REUSE.md)。
它核对输出和缓存，再在关闭 profiler 的条件下比较整个 decode，保留原始样本与复跑入口。
