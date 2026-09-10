# 完整多头 Attention：从多路读取到输出投影

前置：单头 Attention 和拆头/合头实现均已通过验收。本节直接组合完整模块，
不重做基础 shape 口头题；代码与测试一起验收新的组合约束。

## 放回完整模型

Attention 接收每个 token 当前的特征 X，返回读过允许上下文的新特征 Y。
在后续完整模型里，X 可以来自输入表示或前一个块；这里不在函数内部创建 E/P。
Y 仍是特征，不是词表概率：后续模型计算和词表矩阵才产生 logits。
ex005 的逐位置模型尚未与这些 Attention 组件端到端集成。

## 一次完成所有头，而不是循环复制单头

本题使用等宽配置：输入宽度 C，头数 H，每头宽度 Dh=C/H；
Wq/Wk/Wv/Wo 均为 (C,C)，不含偏置。

| 阶段 | 实际计算或来源 | shape |
|---|---|---|
| 输入 | X | (B,T,C) |
| 投影 | Q=X@Wq，K=X@Wk，V=X@Wv | 各 (B,T,C) |
| 拆头 | 分别调用已写的 split_heads | 各 (B,H,T,Dh) |
| 匹配并缩放 | Qh@Kh.transpose(-2,-1) / sqrt(Dh) | (B,H,T,T) |
| 权限 | make_causal_allowed(input_valid)，再补 head 轴 | (B,1,T,T) |
| 归一化 | 屏蔽后对每个 query 的 key 轴做 Softmax | (B,H,T,T) |
| 读取 | weights@Vh | (B,H,T,Dh) |
| 合头 | merge_heads，得到 Z | (B,T,C) |
| 输出投影 | Y=Z@Wo | (B,T,C) |

Wq 的输出列按头划分；第 h 头的参数就是
`Wq[:, h*Dh:(h+1)*Dh]`，shape (C,Dh)，Wk/Wv 同理。
因此每个头的投影可以使用 X 的全部 C 个特征，不是先把原始 X 切成互不相通的几份。
大矩阵乘法同时算出这些不同参数投影，拆头只是把结果组织起来。
不同头有各自的分数和归一化过程，允许学习不同读取比例；不保证一定学出不同语义角色。
“分别计算”也不意味着这些随机变量统计独立。

`(B,H)` 是批量矩阵乘法的前缀；每次点积实际只包含 Dh 对分量，
所以除以 sqrt(Dh)，不是 sqrt(C)、sqrt(H) 或 sqrt(T)。
Softmax 只沿最后的 key 轴进行，不能跨 head 归一化。

C=H*Dh 是当前“投影总宽度等于输入宽度，且各头等宽”的配置约定，不是 Attention 的普遍定律。
更一般的投影可将 C 映射到 H*Dk 或 H*Dv；本题不增加这些配置，避免接口分散。

## Mask 的 batch/head 轴不能错配

已有 allowed[b,i,j] 表示样本 b 中 query i 是否允许读取 key j。
它的 shape 是 (B,T,T)，且每个 head 都遵守同一份权限：

```python
allowed_heads = allowed.unsqueeze(1)  # (B,1,T,T)
```

unsqueeze(1) 只增加长度为 1 的 head 轴；与 (B,H,T,T) 的 scores 计算时，
广播把该样本的权限应用于它的全部头，不需要 repeat 成 H 份实体数据。
与单头相同，True=允许，禁止项在 Softmax 前填 -inf；
任一 query 全被禁止时先抛 ValueError，不给 Softmax 喂全 -inf。
input_valid 只约束 key；不把 PAD query 行清空，也不换成 target_valid。

直接使用三维 mask 时，广播从右向左对齐：

```text
scores:          (B,H,T,T)
三维 mask 对齐为: (1,B,T,T)   ← B 会去对齐 H
```

当 B==H 时可能不报错，但样本权限错配成头权限；当 B 与 H 不兼容时会直接报错。
无论哪种形状，都应显式补出正确的 head 轴。

## 拼接不是融合；Wo 学的是同一个位置的特征组合

merge_heads 只把各头结果并排放回同一个 token：

```text
Z[b,t] = [head0 的 Dh 个数, head1 的 Dh 个数, ...]
Y[b,t] = Z[b,t] @ Wo
```

例如双头每头两个数，Z[b,t]=[a,b,c,d]，则某个输出分量为：

```text
Y[b,t,j] = a*Wo[0,j] + b*Wo[1,j] + c*Wo[2,j] + d*Wo[3,j]
```

Wo 为这些来自不同头的内容提供可训练的线性组合。
它在所有样本和 token 位置间共享，不再次读取其他位置，也不改变已经算出的 weights。
当前拼接宽度已经是 C；Wo 的意义不是修补 shape。
不加 Wo 也能返回合法张量，只是去掉了本模块中这个可学习的输出变换。
Wo 与输出词表矩阵不是同一个东西：这里 (C,C)，词表投影通常 (C,N)，N 为词表大小。

训练时 Wo 与 Wq/Wk/Wv 一样是持久参数；本次 Q/K/V、weights、Z/Y 是前向激活。
任务 loss 经后续计算传回 Y，autograd 再把梯度传回 Wo、各头和 Q/K/V 投影参数。
本次 forward 函数既不更新参数，也不主动调用 backward。

## 综合实现，不再按运算符拆作业

只补 [attention.py](../exercises/ex007_multi_head_attention/attention.py) 的两个函数：

1. multi_head_attention：输入已经投影的 Q/K/V，完成逐头读取与 Wo。
   沿用 ex006 的长度约定：Q 是 (B,Tq,C)，K/V 是 (B,Tk,C)，允许 Tq!=Tk。
   外部 allowed 是 (B,Tq,Tk) 或 None；None 表示全部可读，不自动加因果 mask。
2. multi_head_self_attention：复用已有投影、因果权限生成和第一个函数。
   自注意力从同一个 X 得到 Q/K/V，因此 Tq=Tk=T。

第一层采用批量 Tensor 运算，不遍历 batch/head/token；
复用 split_heads/merge_heads。第二层只做已有接口组合。
ex006 的 scaled_dot_product_attention 当前接口明确只收三维输入，
本次不偷偷改变它的合同，也不要求重写已有答案。

返回顺序固定为 (output, weights)，weights 保留全部 head，不求平均。
仅使用此前已学的 Tensor 操作与 math.sqrt；框架已准备好 import，
无需先学 nn.Module 或新的 Python 抽象。测试参照按 head 逐一计算，是为了走独立验证路径，
不是要求你使用相同的循环实现。

## 给后续 Infra 学习留一个成本坐标

在固定 C、四个 (C,C) 无偏置投影的配置下，参数量为 4*C*C，
增加头数并不使投影参数量按 H 成倍增长，因为 Dh 会相应缩小。
显式保存的 scores/weights 则各有 B*H*T*T 个元素：
拆出更多头会增加这些中间张量的元素数。
这只是特定配置下的参数/激活账本，不是完整峰值内存估计，也不是性能实测。
