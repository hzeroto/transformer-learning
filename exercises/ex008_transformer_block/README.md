# 008：把已学组件组成一个可训练的 Transformer Block

这次不是四份小作业，而是一份完整组件实现。你只需填写
[`block.py`](block.py) 中四个类的 `forward`：`LayerNorm`、`FeedForward`、
`Dropout`、`TransformerBlock`。构造函数、参数初始化和测试由框架提供。
复用你已经完成的 MHA，不改旧练习，不补一遍拆头代码。

最终得到的是 **Pre-LN、因果自注意力、ReLU FFN** 的一个 Block，
不是原论文的 Post-LN，也不是完整 GPT。这里不加入 embedding、位置编码、
词表投影或整座模型末尾的归一化。

需要回看时：

- [残差、LayerNorm 与 FFN](../../notes/transformer-block.md)
- [Dropout 与模块组织](../../notes/dropout-and-module-organization.md)
- [参数流与数据流](../../notes/parameter-data-flow.md)

## 1. 输入、参数与返回值

`B` 是 batch 大小，`T` 是序列长度，`C` 是 token 的特征宽度；
`H = num_heads`，每头宽度 `Dh = C/H`；`F = ffn_hidden` 是 FFN 中间宽度。
所有维度均为正整数，`C` 必须能被 `H` 整除，但 **F 不必等于 4C**。

| 对象 | 来源与含义 | shape |
| --- | --- | --- |
| `X` | 本次前向传入的 token 表示 | `(B,T,C)` |
| `input_valid` | 本次输入的有效位置；True 是真实 token，False 是 PAD | `(B,T)`，bool |
| `Wq/Wk/Wv/Wo` | Block 持有的四个可训练投影矩阵 | 各 `(C,C)` |
| `norm1/2.gamma, beta` | 两套独立、跨 token 共享的归一化参数 | 各 `(C,)` |
| `ffn.W1, b1` | FFN 第一层矩阵与偏置 | `(C,F)`、`(F,)` |
| `ffn.W2, b2` | FFN 第二层矩阵与偏置 | `(F,C)`、`(C,)` |
| `Y` | Block 返回的新表示，只有这一个返回值 | `(B,T,C)` |

`p`、`eps`、头数和 `self.training` 是控制配置，不是可训练参数。
Dropout 的随机 mask 是本次计算的数据，不是长期参数，也不是 `input_valid`。

范围统一为 CPU、`float32` 或 `float64`；输入和参数的 dtype/device 一致。
默认参数用 `float64` 方便验算；你也可以构造时传 `dtype=torch.float32`。
支持连续、转置或切片得到的非连续输入。不要原地修改输入、mask、参数或已有 `.grad`，
也不要在 `forward` 里重新创建或替换参数/子模块。

本次不要求为所有错误输入编写通用校验；构造参数的基本检查已经给出。
唯一需要保留的前向异常是 MHA 已有的“某行没有任何可读 key”错误。

## 2. 四个 forward 的具体契约

### LayerNorm：每个 token 内归一化

对 `X[b,t,:]` 的 C 个数计算均值和**总体方差**，方差分母是 C。
使用 `(X - mean) / sqrt(var + eps)`，再乘 gamma、加 beta。
不跨 B 或 T 统计；gamma/beta 在所有位置共享。
常量向量和 C=1 也要有限且正确，不能漏掉 eps。

输入、gamma、beta 都必须能反向传播。
如果使用 `torch.var`，不要依赖其默认方差修正；也可以直接用基础算子计算平均平方偏差。

### FeedForward：取回内容之后，逐 token 重新组合特征

按已学顺序执行：第一层仿射变换 → ReLU → 第二层仿射变换。
中间激活 `(B,T,F)`，最终回到 `(B,T,C)`，不能混合 token 轴。
本练习 ReLU 在输入恰好为 0 时采用梯度 0。
可用讲义介绍过的 `torch.where` 表达条件选择。

### Dropout：只由训练模式决定随机失活

- `self.training=True` 且 `p>0`：各元素独立以 p 的概率置零；保留值除以 `1-p`。
- eval 或 p=0：恒等映射，**不消耗随机数**，梯度路径仍然保留。
- `torch.no_grad()` 不负责关 Dropout；不要用梯度开关判断训练模式。
- 随机种子由调用者设置；前向不重设种子，不复用上次的随机 mask。
- 本题明确只处理 `0 <= p < 1`，不用实现 p=1 的特殊分支。

不限定某一种正确随机采样写法，也不要求每次抽样的结果必然不同。
反向传播应沿着本次前向实际抽到的 mask 计算，不重新抽签。

### TransformerBlock：保护原表示，再加入两次更新

把已学过的两条分支接起来；这里只给数学接口，不给完整 Python 实现：

```text
A = MHA(norm1(X), input_valid) 的 output
U = X + drop1(A)
D = ffn(norm2(U))
Y = U + drop2(D)

X、A、U、D、Y 全部为 (B,T,C)。
```

`MHA` 指现成的函数，不是另建一个模块。它的真实签名是：

```python
multi_head_self_attention(X, Wq, Wk, Wv, Wo, input_valid, num_heads)
# 返回 (output, weights)，分别为 (B,T,C) 和 (B,H,T,T)。
```

使用 `self.Wq` 等四个矩阵，并取它返回的 `output`。
Dropout 不施加在整条残差结果上，也不加到 softmax 权重上。
本题不要求返回 weights。

权限是：query 位置 i 能读 key 位置 j，当且仅当 `j <= i` 且
`input_valid[b,j]` 为 True。MHA 内部已经处理 `(B,T,T)` 权限在 head 轴的共享。
常见输入是右侧 PAD；测试也会用有效性有空洞的输入检查 key 是否真正被屏蔽。
本题允许 PAD query 保留自己的输出，**不要把它们清零**。
训练时哪些位置计入 loss 是 `target_valid` 的职责，不属于 Block。

## 3. 几处 Python/PyTorch 写法说明

- `model(X)` 会调用对象的 `forward(X)`；组合时可以写子模块调用，不用绕过模块机制。
- 构造函数参数里的 `*` 表示后面的参数只能用名字传，例如 `dtype=torch.float32`。
- `torch.randn(rows, cols)` 抽取均值为 0、标准差为 1 的随机数；框架乘 0.05
  给矩阵一个小的起始值。初始化策略不是这次考点，也不把这个常数当成通用最佳选择。
- `torch.rand_like(X)` 可以生成与 X 同 shape/dtype/device、取值在 `[0,1)` 的
  均匀随机数。与阈值比较会得到 bool 张量，可用来生成 Dropout 保留 mask。
- 浮点 Tensor 乘 bool Tensor 时，True/False 按 1/0 参与计算。
- `self.training` 是 `nn.Module` 自带的 bool 状态；新建模块默认 True。
  `model.eval()` 会递归切换已注册的子模块，不会关闭 autograd。
- `(output, weights)` 是二元组，可以用 `output, _ = ...` 解包。
  `_` 在这里是普通变量名，惯例上表示这个结果后续不用。

允许基础 Tensor 操作、autograd、`nn.Module`、`nn.Parameter` 和已有 MHA。
不要使用 `nn.LayerNorm`、`nn.Linear`、`nn.Dropout` 或对应的 functional 封装
代做本题；ReLU 也用基础算子实现。不要调用高层 Transformer、Attention 或融合接口。
测试中的官方数值参照仅属于教师测试，不可导入到作业实现中。

## 4. 运行与完成标准

从仓库根目录运行：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_transformer_block
.venv/bin/python -B -m exercises.ex008_transformer_block.demo
```

可以按 `LayerNormTest`、`FeedForwardTest`、`DropoutTest`、`BlockTest` 单独运行一组，
例如在测试模块名后追加 `.LayerNormTest`。建议按上述顺序填写，但这是一次综合验收。

空框架能导入。运行整份测试会有**一个明确的未实现失败**；各行为测试组遇到对应
`NotImplementedError` 暂时跳过，避免刷出几十份相同报错。这不是通过。
构造函数测试可能已通过，因为它们检查的是教师提供的框架。
最终要求本练习全部测试实际执行并通过，**无跳过**。

验收覆盖前向与输入/参数梯度对齐、归一化轴、ReLU、Dropout 缩放和模式、
未来/PAD 信息隔离、两条残差的恒等通路，以及重复前向不丢失注册状态。
浮点参照容差：float32 使用 rtol=1e-5 / atol=1e-6；float64 使用
rtol=1e-9 / atol=1e-11。随机比例另用明确的统计容差，不对齐某个库的随机序列。
随机种子写在测试中，测试不修改训练环境依赖。

测试中存在可读的教师数学参照；这是可查基础 API 的日常实现练习，
不是闭卷重建或“学习者独立设计测试”的证据。
参数注册及初始化已由框架给出，也不能据此声称你独立完成了这部分。
创建框架不改变掌握状态；你提交后再按实现和排错证据验收。

demo 检查 Block 可前向、反向和切换模式，不证明学会语言或训练了 GPT。
