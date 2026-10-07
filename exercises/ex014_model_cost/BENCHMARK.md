# 第三步：每一次计时，都测同一份工作

FLOPs 和存储统计代码已经通过。现在给同一份报告补上实际耗时，仍使用你的 ex013 模型。
这对应整节课第二个目标；本目录将计算、存储、计时分成三个实现步骤，目标没有合并。

## 先固定工作量，再按秒表

你的 `llama_model_step` 会追加缓存。如果历史长 8，连续对同一个缓存调用三次，
每次输入一个 token，实际状态是：

| 第几次调用 | 不恢复历史：进入 → 退出 | 固定历史实验：进入 → 退出 |
|---|---|---|
| 1 | 8 → 9 | 8 → 9 |
| 2 | 9 → 10 | 8 → 9 |
| 3 | 10 → 11 | 8 → 9 |

左边每次要读取的历史都不同，平均下来不能叫“历史 8 的单步耗时”。
因此本步每次都创建新缓存，并在计时前重新处理相同前缀，建立同一长度的历史。
这增加了实验整体运行时间，却让被测调用始终具有相同输入和缓存状态。
历史刚处理过会影响硬件缓存的冷热，所以报告也要注明这个恢复方法。

| 测什么 | 每次计时前准备 | 计时内只做什么 |
|---|---|---|
| prefill | 新建空缓存，准备固定 `(B,T)` 输入 | 一次 step，处理全部 T 个位置 |
| 固定历史 decode | 新建缓存，用固定 `(B,t)` 前缀恢复历史 | 一次 step，处理固定 `(B,1)` 新输入 |

两者都包含本次 step 内部的投影、注意力、FFN、KV 追加和词表投影。
不把模型构造、前缀恢复、日志、统计或正确性断言放进计时。
本模型输出本次所有位置的 logits，prefill 没有偷偷改成只投影最后一个位置。

`time.perf_counter()` 返回高精度时钟的秒数；后一次减前一次，得到经过的秒数。
我们关心两次读数之差，而不是它的绝对数值。这个计时方式会计入期间的等待或抢占，
因此单次结果可能波动。[Python 计时 API](https://docs.python.org/3.12/library/time.html#time.perf_counter)

第一次调用还可能包含初始化成本。**热身**就是先按同样工作量运行几次，不将它们
计入正式结果。热身也会追加 KV，因此每次热身和每个正式样本都要恢复起点。
先使用 10 次热身、50 次采样；次数是实验起点，不保证完全消除噪声。
[PyTorch 基准教程](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)

本步只测 CPU，使用 `model.eval()` 和 `torch.no_grad()`。
前者管理模块的训练/推理行为，后者关闭这段代码的梯度记录；两者分工不同。
demo 固定计算线程数为 1 并记录环境，防止对比时同时改变线程与输入。

## 有了样本，怎样描述“快慢”

**延迟**是一次模型调用花了多久。多次重复后，用中位数表示典型耗时：
将样本排序，取中间值，偶数个时取中间两项的平均。
第 25、75 百分位是排序分布中约四分之一、四分之三处的值，用来观察中间一半样本的波动。
对有限样本可以有不同插值约定，本题固定 `inclusive`，并允许直接调用标准库。

```python
from statistics import median, quantiles

middle = median(samples_s)
q25, q50, q75 = quantiles(samples_s, n=4, method="inclusive")
```

以上仍是秒。返回延迟以毫秒报告，1 秒 = 1000 毫秒。
这个约定中的分位点由排序后相邻样本线性插值得到，不要求你重写插值算法。
[Python 分位数 API](https://docs.python.org/3.12/library/statistics.html#statistics.quantiles)

**吞吐量**是每秒处理多少 token。本题按“所有正式样本的新 token 总数 / 正式样本总秒数”计算。
单次新 token 数为 B*n：prefill 时 n=T，单步 decode 时 n=1。
计时外恢复的历史不能计入分子，也不要对每个样本的 token/s 直接取平均。

例如有 3 次调用，每次处理 2 个新 token，总共花了 0.006 秒，吞吐量就是
6/0.006=1000 token/s。prefill 指输入 token 处理吞吐；固定 ID 的 decode 只测
模型追加调用，不包含采样、网络或排队，因此不是服务端到端延迟。

## 现在填写什么

打开 [benchmark.py](benchmark.py)：

1. `measure_cpu_step`：准备每次的缓存状态，热身，再采集逐次耗时。
2. `summarize_samples`：把原始样本汇总成中位延迟、p25/p75 和吞吐量。

可直接复用已完成的两个接口：

```python
caches = new_caches(model)  # 新建每层的空缓存容器
logits = llama_model_step(model, ids, caches)  # 前向并追加 KV
```

`prefix_ids=None` 表示没有要提前恢复的历史；非 None 时是固定历史的 ID。
`clock` 参数默认是 `perf_counter`；它是可调用函数，执行 `clock()` 读一次秒数。
测试会替换成确定性时钟来检查你是否把准备时间混入计时，不依赖机器快慢打分。
`Callable[[], float]` 只是“无参数、返回 float 的函数”的类型提示。

先测第一个函数，再一起运行：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_model_benchmark.MeasureStatusTest tests.exercises.test_model_benchmark.MeasureCpuStepTest -v
.venv/bin/python -B -m unittest tests.exercises.test_model_benchmark -v
.venv/bin/python -B -m exercises.ex014_model_cost.benchmark_demo --output /tmp/ex014-benchmark.json
```

空脚手架有 **2 项明确的未实现失败、8 项行为测试暂跳过**，demo 报 `NotImplementedError`。
完成后要求 **10 项通过、无跳过**。教师测试覆盖缓存起点、热身、计时边界和数值对齐；
没有“必须小于某个毫秒数”或“decode 必须比 prefill 快多少倍”的断言。

## 怎样看最终输出

demo 使用同一模型、固定种子、CPU float32、单线程，对 B∈{1,2}、T/t∈{8,16}
分别测 prefill 与单步 decode。相同 batch 的长输入以前半段作为短输入，
decode 新 ID 固定；模型参数在全部实验中保持不变。
demo 先在计时外检查全量与缓存 logits，容差 rtol=1e-5、atol=1e-6，
再将你的 FLOPs、KV 预测、计时结果放在同一行。
JSON 保留环境、配置、实际输入、原始样本和汇总值；日志在全部测量结束后打印。
若系统限制读取 CPU 型号，报告会明确标记未获取，保留架构等可用信息；提交时补充型号。

只改变一个条件比较，例如固定 B 和历史长度比较 prefill/decode，或固定调用类型和
长度比较 batch。先观察 FLOPs 比例是否等于耗时比例，不强行解释噪声中的小差异。
操作数不能决定硬件并行程度、数据访问与每次算子的固定开销；具体时间花在哪，
下一步再用执行记录定位，复用[讲义第 6 节](../../notes/model-cost-and-benchmark.md#6-账本给出怀疑profiler-找到实际执行的位置)。

这一阶段只准备并验证计时工具。提交真实运行结果后再解释观察，不把教师工具自检当作
学习者已完成性能分析。

## 已加入的设备同步边界

CPU 练习完成后，教师按要求在 `measure_cpu_step` 中加入了设备同步；函数名保留，
输入在 CPU 时不调用显卡 API，CUDA 输入同步对应的 device，MPS 输入调用
`torch.mps.synchronize()`。MPS 是 PyTorch 使用 Apple GPU 的后端。

每个正式样本现在按以下顺序执行：

```text
新建缓存 → 恢复历史 → 等设备完成旧工作 → start
→ 目标 step → 等设备完成本次工作 → end → 记录样本、释放输出
```

开始前的同步防止尚未完成的历史准备或热身混入样本；结束前的同步防止只测到
CPU 提交任务的时间。logits 保留到结束读数后，避免把输出释放计入延迟。
这里测量含 CPU 提交和同步等待的墙钟延迟。
[CUDA 同步](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.synchronize.html)、
[MPS 同步](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html)

新增测试用“提交只排队、同步才完成”的模拟队列核验同步位置，覆盖热身、无历史
prefill、有历史 decode、CUDA 指定设备及 MPS，并验证 CPU 不调用 GPU 同步。
这类测试验证计时边界，不代表真实 GPU 性能实测。当前 ex013 仍有默认在 CPU
创建的 mask/索引；完整 GPU 运行还需要迁移这些张量并验证数值。本次没有修改模型数学。
