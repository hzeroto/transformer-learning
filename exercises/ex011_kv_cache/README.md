# 011：KV Cache 与两阶段生成

把 ex010 的 `greedy_generate`（每步重算整个前缀）换成 prefill + decode 两阶段，
并证明换完之后结果不变。

讲义见[KV Cache：把重复计算换成可管理的状态](../../notes/kv-cache.md)。

## 待完成

只修改 [cache.py](cache.py) 中的六处 TODO，不改测试：

| 位置 | 要实现的内容 |
|---|---|
| `LayerKVCache.__len__` | 当前缓存长度；它同时是下一个 token 的位置下标 |
| `LayerKVCache.append` | 沿位置轴追加，含容量检查 |
| `LayerKVCache.reset` | 清空，回到新建状态 |
| `block_step_with_cache` | 一个 Pre-LN 块处理若干新 token 并更新缓存 |
| `generate_with_cache` | prefill + decode 两阶段贪心生成 |
| `kv_cache_bytes` | 按形状推导缓存字节数 |

**不修改** ex006/ex007/ex008/ex009/ex010 的任何已验收实现。

## 契约要点

完整说明见各函数 docstring，这里只列容易踩的：

- **先追加再计算**。新 token 必须能读到自己（因果规则允许 `j == i`）。
  写成「先算后追加」时，第一个 token 会面对空缓存。
- **decode 阶段的 mask 是全 True**，不是下三角。唯一的 query 在最后一个位置，
  缓存里不存在未来的 key，没有东西需要屏蔽。套用 `torch.tril` 会让模型
  只看见位置 0 —— 不报错，但答案错。
- **位置下标取自缓存长度**，不要用 `prefix_len + step` 之类从参数推算的值。
  后者在「分两次调用、续用同一份缓存」时会错（讲义检查 B）。
- **缓存不是模型参数**：不注册到 `nn.Module`，不进 `state_dict`，不求导，
  每个请求各一份。两个请求共用会相互污染。
- **不能用 `multi_head_self_attention`**：它内部写死了方阵因果 mask，
  无法表达「1 个 query 读 L 个 key」。要直接调用底层的 `multi_head_attention`。
- 容量检查在**追加之前**完成；抛出后缓存必须保持原样，不能半途写入。
- 生成必须在 eval + `no_grad` 下进行，返回后恢复调用前的 `training` 状态。

可以直接复用的既有实现：

| 来源 | 用途 |
|---|---|
| `ex006` 的 `project_qkv` | X → Q/K/V |
| `ex007` 的 `multi_head_attention` | 支持 Tq ≠ Tk，正是缓存需要的 |
| `ex008` 的 `TransformerBlock` | 提供 norm1/norm2/ffn/drop 与四个投影矩阵 |
| `ex009` 的 `MiniGPT` | 模型本体、token/position 表、vocab_proj |

CPU、float32/float64。本练习要求 `p=0`（无 Dropout），
否则缓存版与全量版的差异会混入随机失活。

## 运行

```bash
.venv/bin/python -B -m unittest tests.exercises.test_kv_cache -v
```

全仓回归：

```bash
.venv/bin/python -B -m unittest discover -s tests -t .
```

**空脚手架的预期状态**：26 项全部报 `NotImplementedError`
（`subTest` 会让报告里的条目数多于 26，属正常）。这是未完成状态，不是环境问题。

## 关于容差

对齐检查用**事先声明的容差**，不是逐位相等：float64 用 `1e-12`，float32 用 `1e-5`。

原因是全量前向时多个 query 一次矩阵乘完成，增量时每个 query 单独乘，
分块与累加顺序不同，浮点结果可能差最后几位。实测差值分别在 3.6e-16 和 3.0e-07 量级
（讲义第 6 节）。这不是 bug，所以不能要求 `torch.equal`。

但**生成出来的 token 序列要求完全相同** —— 贪心下微小的 logits 差异不改变 argmax。

## 完成标准

26 项全部通过、无跳过，且全仓回归仍为绿。测试覆盖：

- 缓存容器：空缓存长度、沿位置轴增长且顺序正确、reset 后 k 与 v 都清空、
  容量检查在写入前完成、默认不限容量
- 块级对齐：prefill 与整段一次前向一致、逐 token 喂入每步都一致、
  先 3 个再逐个追加 2 个、新 token 能读到自己、不修改输入、
  decode 能看见全部缓存的 key
- 生成对齐：prefill logits 与全量一致、float64/float32 两种精度下
  生成序列与无缓存贪心完全相同、前缀长度 1/2/5/9 均可、
  续用缓存时位置取自缓存长度（逐层比较缓存内容）、EOS 停止且保留、
  请求之间互不影响、模式与梯度不被改变、返回 (1,T+K) long、容量超限抛错
- 字节账本：与公式一致、与真实张量的 `numel × element_size` 一致、
  对各维度线性、长度为 0 时为 0

## 完成后的两个观察任务

测试不覆盖这两项，它们是 `modern.kv-cache` 第三条验收标准要求的分析部分。

**一、prefill 与 decode 的计算量差异**

统计两个阶段各自的 K/V 投影行数与 Attention 分数矩阵元素数（`Tq × Tk`），
前缀长度分别取 5、50、200，各生成 10 个 token。

记录：随前缀变长，prefill 的分数矩阵元素数怎样增长？decode 单步呢？
两个阶段哪个更接近「一次大矩阵乘」，哪个更接近「很多次小矩阵乘」？

**二、缓存内存的增长**

用 `kv_cache_bytes` 算出几组配置的缓存大小，至少覆盖：
改变 batch、改变上下文长度、改变层数。再用真实张量核对其中一组。

记录：如果显存预算固定，batch 和上下文长度之间是什么关系？
这个式子**没有**计入哪些内存开销？

两项都要求把结论限定在你实际测到的配置上。特别注意：
**算术量减少不等于墙钟时间变短** —— 这一课不做计时，
真实加速需要 `systems.benchmarking` 的方法才能测量。
