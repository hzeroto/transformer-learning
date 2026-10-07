# 第二步：这次前向涉及多少数据

FLOPs 账本已通过。继续使用同一个模型，先填写 [storage.py](storage.py) 中的
`estimate_storage_bytes`，再填写 `measure_storage_bytes`。

## 先按 shape 算字节

规则只有一个：**元素个数 × 每个元素的字节数**。
例如 `(2,7,8)` 的 float32 张量共有 112 个元素，每个元素占 4 字节，共 448 字节。
float64 和 int64 每个元素占 8 字节，bool 占 1 字节。

沿用之前的 B、n、S、C、Hq、Hkv、G、layers、N；其定义也写在函数注释中。
本题统计的是关闭求导的缓存推理，先不加入梯度和优化器状态。

| 返回键 | 数据来自哪里 | 要数的对象 |
|---|---|---|
| `weights` | 模型全部可训练参数 | 调用方提供 `parameter_count`，含 embedding、投影和 RMSNorm 参数 |
| `kv` | 各层处理新位置后保存的 K、V | 每层两份 `(B,S,Hkv*(C//Hq))`，计全部层 |
| `score_one_layer` | 某层 Q 与历史及新 K 打分 | 一份 `(B,Hq,n,S)` |
| `ffn_one_layer` | 某层 FFN 的中间投影输出 | 一份 `(B,n,G)` |
| `logits` | 最后一次词表投影 | 一份 `(B,n,N)` |

所有浮点数据统一使用参数 `element_bytes`，默认 4。函数只做整数运算。
权重不随 batch 或请求长度改变；缓存包含历史，用 S；本次 FFN 和 logits 只处理新位置，用 n。
GQA 的持久 KV 保存 Hkv 个头；打分仍有 Hq 个 query 头。

这里单层中间量只统计**一份**，不统计整个 FFN 的所有临时张量，也不乘层数。
不同层的临时数据可能先后释放，而一次操作又可能产生多份临时数据。
因此不要给这五项加一个 `total` 或把它称为模型峰值内存。

## 再读取真实张量

同一批数字可以有不同的 shape，而不多存一份。例如：

```python
x = torch.zeros(2, 3, dtype=torch.float32)
y = x.view(6)
```

x 和 y 各自按 shape 都是 24 字节，但它们共用一块 24 字节的底层存储。
把 `[x,y]` 交给统计函数，应得到 `logical_bytes=48, storage_bytes=24`。
这里 **storage（底层存储）就是实际放置元素数据的一块内存**。
`clone()` 会复制数据到另一块 storage，即使数值相同，也需要单独计数。

所需 API：

| API | 含义 |
|---|---|
| `x.numel()` | 按 shape 计算的元素个数 |
| `x.element_size()` | 当前 dtype 的每元素字节数 |
| `x.untyped_storage()` | 获取 x 依赖的底层存储，不复制数据 |
| `x.untyped_storage().nbytes()` | 整块底层存储的字节数 |
| `x.untyped_storage().data_ptr()` | 整块存储的起始地址，本题可用作去重标识 |

切片 `x[1:]` 仍依赖原来的完整 storage，所以 storage 字节数可能大于它自己的
逻辑字节数。不要使用 `x.data_ptr()` 来去重：它指向张量第一个元素，切片后可能偏移。
本练习限定 CPU、非空张量、同时保留对象，让地址去重的含义明确；空列表返回两个 0。

`expand` 可以让多个逻辑位置读取同一个元素，所以 shape 变大也不一定增加 storage。
例如把一行数据扩展成三行，只是三行都读同一行。
`reshape` 则可能共享，也可能复制，要看现有布局是否支持目标形状。
demo 给出讲义中的 GQA 例子：先共享展开，再合并头轴；后一步在这个布局下产生复制。
具体地址步长原因留到你跑出读数后，结合结果解释。
[PyTorch 的视图与复制说明](https://docs.pytorch.org/docs/2.14/tensor_view.html)。

## 实现和核对

先只运行预测函数的测试：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_model_storage.EstimateStatusTest tests.exercises.test_model_storage.EstimateStorageTest -v
```

再实现实际统计函数，运行全部测试和 demo：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_model_storage -v
.venv/bin/python -B -m exercises.ex014_model_cost.storage_demo
```

空脚手架会有 **2 项明确的未实现失败、14 项行为测试暂跳过**，demo 报 `NotImplementedError`。
完成后须 **16 项通过、无跳过**。教师测试可读，不要从测试导入答案。
demo 会用 ex013 的真实参数、缓存和 logits 对照你的预测；score/FFN 只打印单份大小预测，
没有将它们冒充实际执行峰值。demo 中的 FLOPs 来自你已完成的 cost.py。

完成这两个函数是存储账本的实现证据；还要结合共享与复制读数解释模型行为。
CPU 张量 storage 统计不是整个进程占用；计时和执行分析将在本练习后续继续。
