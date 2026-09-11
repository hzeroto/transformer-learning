# Dropout 与模块组织：随机失活、参数注册与两种开关

前置：Pre-LN 主干的残差、LayerNorm 与 FFN 已在[块讲义](transformer-block.md)讲过，
完整多头 Attention 的原理与实现均已验收。本节补齐 `block.complete` 还缺的最后一块
（Dropout），以及把这些散装函数变成可训练、可保存模型所需的组织机制。

当前接着读第 3、4 节：把散装参数和函数组织成一个可更新、可保存、可切换模式的对象。
第 1、2 节保留作 Dropout 计算的参考，不需要重新逐段学习。
`nn.Module`/`nn.Parameter`、参数的递归收集和 `train()`/`eval()` 是本次新内容；
已有的 Tensor 计算、手写 SGD 和 `no_grad` 直接复用。

第 3、4 节的参数注册、统一更新、恢复遗漏、容器及模式对照已在本仓库
PyTorch 2.14.0、CPU float64 环境核对；教师演示不代表学习者实现已经通过。

## 1. 为什么块里还要放一个「随机丢弃」

到目前为止，你的每个计算都是确定的：同一份输入和参数，前向必然给出同一个结果。
Dropout 打破这一点，而且只在训练时打破。

先说它想解决的问题。你已经在 `ex005` 观察过：训练损失可以降到 0.0034，
但那是「每个 token 都有确定后继」的玩具数据。当模型参数很多而数据有限时，
模型可以把训练集的特定组合背下来，而不是学到可推广的规律。
这种「训练集损失很低、没见过的数据上表现差」的现象叫**过拟合**。

Dropout 是应对过拟合的一种手段：训练时随机让一部分中间数值变成 0，
使模型不能依赖某几个固定分量总是存在。
它不是唯一手段，也不保证在所有任务上都提升效果 —— 本课只要求你准确掌握它的计算和两阶段行为。

### 精确定义

**Dropout** 有一个超参数 p，表示丢弃概率，取值在 `[0,1)` 常用区间（PyTorch 允许 `p=1`）。
p 是人为设定的配置，不是训练参数，没有梯度。

固定一个输入张量 S，Dropout 在**训练态**这样计算：

```text
对 S 的每一个元素位置，独立采样一个伯努利随机变量：
  以概率 p     丢弃 → 该位置输出 0
  以概率 1-p   保留 → 该位置输出 S[...] / (1-p)
```

关键在保留位置**要除以 (1-p)**。这一步叫**倒置 Dropout（inverted dropout）**。
在**推理态**，Dropout 什么都不做：

```text
输出 = 输入          逐元素完全相等，不采样、不置零、也不缩放
```

固定 S 的 shape 为 `(1,1,4)`、值为 `[2,4,6,8]`、p=0.5，下面仅显示 `S[0,0,:]`：

| 同一输入的处理 | 训练态 | 推理态 |
|---|---|---|
| 本次保留位置示例 | [否,是,否,否] | 不采样 |
| 本次输出 | [0,8,0,0] | [2,4,6,8] |
| 保留项的处理 | 除以 0.5 | 原值返回 |
| 重复前向 | 重新采样，可能相同也可能不同 | 本操作始终原值返回 |

### 那个 1/(1-p) 到底在补什么

这是本节最容易含糊的一点，用实测数字说清。

仍取 `S = [2,4,6,8]`、p=0.5。训练态每次重新采样，但两次也可能抽到相同结果；
单次输出不能代表期望。原讲义的 200000 次采样对照如下，用来观察多次平均：

```text
原始 S            : [2, 4, 6, 8]
倒置 Dropout 均值 : [2.0016, 4.0068, 6.0058, 8.0222]   ← 回到原值
只丢弃、不缩放的均值: [1.0034, 2.0012, 3.0086, 3.9989]   ← 系统性减半
```

所以 `1/(1-p)` 补的是**被丢弃造成的期望损失**：
每个位置有 `1-p` 的概率贡献 `S/(1-p)`，有 p 的概率贡献 0，因此期望恰好是 S。

这件事为什么重要：推理态是恒等映射，输出期望就是 S。
如果训练态的期望是 `S` 的一半，那么训练时后续层看到的数值尺度
和推理时看到的尺度差一倍，模型在两个阶段面对的输入分布不一致。
倒置缩放让两个阶段的**期望**对齐，代价是训练时每次的实际值有波动。

必须区分清楚三句话：

| 说法 | 是否成立 |
|---|---|
| 训练态输出的**期望**等于输入 | 成立，由上述概率加权推导；有限次平均只作观察 |
| 训练态**每一次**输出等于输入 | 不成立，单次是 `[0,8,0,0]` 这样的稀疏结果 |
| 推理态输出等于输入 | 成立，且是逐元素严格相等 |

第二行是常见误解。期望相等不代表任何一次前向的数值相等，
也不代表方差不变 —— 训练态引入了额外的波动，这正是它起作用的方式。

这个保证不能直接穿过非线性。取 s=2、p=0.5，后接 `f(y)=ReLU(y-3)`：
Dropout 产生等概率的 0 或 4，经过 f 后成为 0 或 1，最终期望是 0.5；
不做 Dropout 时 `f(2)=0`。本例已经在课堂验证，不再重复出题。

### 两个边界

实跑确认：

```text
p=0.0 训练态输出: [1.0, 2.0, 3.0, 4.0]     全部保留，除以 1，等于恒等
p=1.0 训练态输出: [0.0, 0.0, 0.0, 0.0]     全部丢弃
```

`p=0` 时训练态也退化成恒等映射，但这是「p 取了 0」而不是「切到了推理态」，
两件事的原因不同，不要混说。`p=1` 在 PyTorch 中合法且输出全零；
本课的块不使用这个配置，它只用来构造后面的验证例子。

## 2. Dropout 放在块里的哪个位置

位置错了会破坏你已经建立的残差性质，这是有实际后果的选择。

在 Pre-LN 主干里，Dropout 作用在**支路的输出上**，然后才与残差相加：

```text
正确：  U = X + Dropout(A)
错误：  U = Dropout(X + A)
```

用 `p=1.0` 让支路必然全丢，就能看出区别（实跑，X 为全 1、A 为全 9）：

```text
正确写法 U = X + Dropout(A) = [1, 1, 1, 1]    支路贡献归零，退回 X，X 的信息完整保留
错误写法 U = Dropout(X + A) = [0, 0, 0, 0]    整个块的输出被清零，X 也没了
```

这与你已经掌握的残差语义一致：残差的作用是提供一条**直接路径**，
让子层只需学习「在已有表示上加什么」。
把 Dropout 套在相加结果外面，等于连那条直接路径一起丢弃，
残差就不再保证「支路无贡献时退回输入」。

所以本课的块结构是：

```text
N1 = LN1(X)
A  = MHA(N1, ...)
U  = X + Dropout(A)          ← Dropout 在支路上

N2 = LN2(U)
D  = FFN(N2)
Y  = U + Dropout(D)          ← 同理
```

两条支路各有自己的 Dropout 模块。它们可以共用同一个 p 配置，
但作为模块实例是两个对象；这一点在下一节讲参数注册时会再用到。

原版 Transformer 还在其他位置（如注意力权重上、输入表示相加后）放 Dropout。
本课主干只在两条支路输出上放，这是一个明确的简化配置，
不是「Dropout 只能放这一处」，也不是某个 GPT 版本的完整复刻。

## 3. 从散装函数到模块：参数注册

到目前为止你的所有参数都是手动管理的：`ex005` 里把 `E`、`W` 作为函数外部的变量传进去，
自己写 `E.grad = None`，自己在 `no_grad()` 里更新。

对于本课的无偏置 MHA、带偏置 FFN、两套 LayerNorm，一个块就有
`Wq/Wk/Wv/Wo` 四个投影、`LN1/LN2` 各一组 `gamma/beta`、FFN 的 `W1/b1/W2/b2`，
共 12 个参数张量。若 6 层各用独立参数，就是 72 个，再加上输入和词表投影的参数。

问题不是 Python 传不进这么多参数，而是**计算、更新、保存必须指向同一批对象**。
如果前向用了某份 W，更新清单却漏掉它，模型仍可能靠其他参数降低 loss；
如果保存清单又漏掉它，恢复后的模型就和保存前不同。能运行、loss 下降，都抓不住这类遗漏。

我们需要一份由模型结构维护的参数名册，而不是训练代码再手写一份。
`nn.Module` 是 PyTorch 的模块基类：保存已登记的参数和子模块，并提供统一的访问方式。
它不会自动实现 Attention，也不会替你写 LayerNorm 的数学。

### 用一个已经会算的小操作，理解新的组织方式

先不实现整个 LayerNorm，只包装它最后的缩放平移：

```text
输入 X：(B,T,C)，CPU float64
gamma、beta：(C,)，在 B/T 轴共享
输出 Y = X * gamma + beta：(B,T,C)
```

下面的 `FeatureAffine` 只做这一步，不求均值或方差，也不是完整 Block。

```python
import torch
import torch.nn as nn

class FeatureAffine(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(C, dtype=torch.float64))
        self.beta = nn.Parameter(torch.zeros(C, dtype=torch.float64))

    def forward(self, X):
        return X * self.gamma + self.beta
```

第一次出现的 Python 写法，在这里解释：

- `import torch.nn as nn` 给导入的模块起别名，后面可以写 `nn.Module`。
- `class FeatureAffine(nn.Module)` 定义一个继承 Module 的类；`class` 不是创建实例。
- `__init__` 是实例初始化方法，`FeatureAffine(2)` 创建对象时会调用它。
- `self` 是当前实例，可类比 Go 方法的接收者；`self.gamma` 是此对象持有的属性。
- `super().__init__()` 先初始化父类的参数/子模块管理机制，再登记自己的参数。
- `forward` 是我们实现的前向方法。通常调用 `model(X)`，由框架入口再调用它，不直接调用 `model.forward(X)`。

`nn.Parameter` 是一种特殊的 Tensor。把它赋给 Module 属性时，会自动登记成这个模块的参数；
普通 Tensor 属性不会自动加入参数名册。Parameter 默认开启 `requires_grad`，但**参数登记和自动求导仍是两套机制**。
[Parameter 的规则](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parameter.Parameter.html)

使用这个对象：

```python
model = FeatureAffine(2)
X = torch.tensor([[[2.0, 4.0]]], dtype=torch.float64)  # (1,1,2)
Y = model(X)                                        # (1,1,2)

for name, parameter in model.named_parameters():
    print(name, parameter.shape)
```

`named_parameters()` 可依次遍历名字与参数对象；`for name, parameter` 把每一对结果拆开。
这里列出 `gamma` 和 `beta`，各自 shape 为 `(2,)`。只要不重新创建 model，这两份参数就持续存在；
每次 `model(X)` 重新计算的是 Y，不是重新初始化参数。

### 你的手写 SGD 不用换，只把清单来源换掉

`model.parameters()` 每次返回可遍历的参数对象，不包含名字，也不是参数数值的副本。
沿用上面的 model 和 X，学习率取 0.01，做一次把 Y 推向零的玩具更新：

```python
for parameter in model.parameters():
    parameter.grad = None

Y = model(X)
loss = (Y ** 2).sum()
loss.backward()

with torch.no_grad():
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter -= 0.01 * parameter.grad
```

这里 `** 2` 是逐元素平方，`sum()` 把所有元素相加成标量损失，不是语言模型交叉熵。
`is not None` 是检查这次有没有梯度对象，不是判断梯度是否非零。
`-=` 沿用你已有的原地更新方式，改变的是模型持有的同一份参数。

本例初始 Y 为 `[2,4]`，loss 为 20。更新后 gamma 约为 `[0.92,0.68]`、beta 为 `[-0.04,-0.08]`，
再次前向得到 `[1.8,2.64]`，loss 约为 10.2096。**Module 负责收集；backward 负责求导；更新仍由这段代码执行。**

### 为什么有梯度，仍可能完全没更新？

现在只改构造方法中的一行，其余前向和更新代码保持相同：

```python
self.gamma = torch.ones(C, dtype=torch.float64, requires_grad=True)
```

这份普通 Tensor 明确开启求导，因此不能把问题归因于“没有 requires_grad”。
两种模型都从相同初值出发，使用相同 X、loss 和上面的手写更新：

| 观察量 | gamma 使用 Parameter | gamma 是开启求导的普通 Tensor |
|---|---|---|
| 参数名册 | gamma、beta | 只有 beta |
| 第一次 backward 后 gamma.grad | [8,32] | [8,32] |
| 更新后 gamma | [0.92,0.68] | [1,1] |
| 更新后 loss，初始均为 20 | 10.2096 | 19.208 |

两个 loss 都下降了，但右侧只更新了 beta。自动求导按计算图找到 gamma；更新循环按名册遍历，却没找到它。
这个循环甚至也不会替右侧 gamma 清梯度。可以手动管理普通 Tensor，就像你的 ex005；
但不能一边依赖自动收集，一边漏登记模型实际使用的参数。

反过来，参数已登记也不保证一定有非零梯度：它可能没参与本次 loss，或者局部导数为零。
`nn.Parameter(..., requires_grad=False)` 也仍是已登记参数，只是关闭了自身求导。

### 保存为什么也依赖同一份登记？

`state_dict()` 返回按名称组织的模型状态字典；Python 的字典可类比 Go 的 map。
本例里就是已登记的 gamma/beta。它还可包含登记为持久状态、但不作为参数训练的 Tensor，
这类状态叫 buffer；本例尚未使用，不要求现在实现。

假设手动把漏登记版本的 gamma 改成 `[3,3]`，再把状态加载到一个同类新对象中：

```text
原对象 gamma：[3,3]
state_dict 中的键：只有 beta
同类新对象加载后 gamma：[1,1]，仍是它构造时的初值
strict=True：不报错
```

`new_model.load_state_dict(saved, strict=True)` 把 saved 中的状态装入新对象；
`strict=True` 检查的是状态名称是否匹配，不知道“你本来还想保存 gamma”。
漏登记的版本在保存端和接收端都没有这个键，所以检查照样通过。
[Module 的状态接口](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.state_dict)

边界先记两条：`state_dict()` 不是整个 Python 对象的自动快照，不自动保存本例的模式开关或 Dropout 概率配置；
它返回的状态 Tensor 还可能共享原存储，要在继续训练前冻结一份内存快照需另做复制。
完整磁盘保存、模型配置和恢复训练流程留到 mini-GPT，当前先理解为什么会漏状态。

### 多层怎样一起被找到？

把子模块赋给 Module 属性时，它会成为父模块登记的子模块。
父模块的 `parameters()` 会继续访问这些子模块，不用你展开每一层。
多个子模块可以放入 **ModuleList**：它像列表一样可索引、可遍历，同时登记里面的模块。

沿用 FeatureAffine，做两个顺序相接的小层；X 与输出仍是 `(B,T,2)`。
此处 `nn.Dropout` 是模式行为参照，完整 Block 练习的核心计算仍由你实现。

```python
class AffineStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FeatureAffine(2), FeatureAffine(2)])
        self.drop = nn.Dropout(0.5)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return self.drop(X)
```

`[a,b]` 是 Python 列表；这里两次 `FeatureAffine(2)` 创建两个独立对象，不共享参数。
`nn.Dropout(0.5)` 创建按丢弃概率 0.5 工作的模块；`self.drop(X)` 调用它的前向。
ModuleList 本身只负责容纳，**顺序执行是我们在 forward 里写的循环**。

现在名册中的名称带上了所属路径：

```text
layers.0.gamma    (2,)
layers.0.beta     (2,)
layers.1.gamma    (2,)
layers.1.beta     (2,)
```

如果仅改成普通列表 `self.layers = [FeatureAffine(2), FeatureAffine(2)]`，
前向循环仍能运行，但父模块不会自动沿普通列表登记这些对象；本例父模块的参数名册就变成空的。
“Python 对象能访问到”不等于“PyTorch 管理机制登记到了”。
[ModuleList 的登记规则](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ModuleList.html)

Dropout 没有可训练参数，所以它不贡献参数名册条目，
但它是已登记的子模块，下面的模式切换仍会访问它。

## 4. train() / eval() 与 no_grad()：两个正交的开关

`ex005` 的作业说明里写过「`nn.Module` 与 `eval()` 尚未引入，
不能把 eval 当成 no_grad 的别名」。现在把这句话讲清。

这两个开关控制的是完全不同的事：

| 开关 | 控制什么 | 影响 Dropout 吗 | 影响梯度记录吗 |
|---|---|---|---|
| `model.train()` / `model.eval()` | 模块处于训练态还是推理态 | 是 | 否 |
| `with torch.no_grad():` | 这段计算是否记录到求导图 | 否 | 是 |

沿用 AffineStack，输入为 CPU float64 的 `(1,1,2)` Tensor，参数默认需要求导；
分别组合两种模式与是否启用求导，实跑核对：

```text
train() + 求导开启：Dropout 重新采样，输出需要求导
train() + no_grad()：Dropout 仍重新采样，输出不需要求导
eval()  + 求导开启：Dropout 直接返回输入，整体输出仍需要求导
eval()  + no_grad()：Dropout 直接返回输入，整体输出不需要求导
```

`no_grad()` 不记录这段新计算的求导图，但**没有**关掉 Dropout 的随机性 ——
如果你在生成时只写了 `no_grad()` 而忘了 `eval()`，Dropout 照样在随机丢弃，
logits 可能波动，不能保证两次生成一样；但也不保证每次都不同，argmax 仍可能没变。

`eval()` 则调整模块模式，不关闭求导，也不把参数永久冻结。
本例里，如果只切 eval，参数仍参与需要梯度的计算，因此仍会构建求导图。
这并不矛盾：有时就是需要在推理模式下分析梯度。
[PyTorch 对两类开关的区分](https://docs.pytorch.org/docs/2.14/notes/autograd.html#evaluation-mode-nn-module-eval)

普通的不求梯度推理，可以这样写；这里 model 是上一节的 AffineStack，X 为 `(B,T,2)`：

```python
model.eval()
with torch.no_grad():
    Y = model(X)
```

这只是明确关闭本例 Dropout 和求导记录，不是所有硬件与所有算子上的确定性保证。
恢复训练用 `model.train()`，并在 `no_grad` 代码块之外正常前向、求导、更新。
`train()` 这个名字只表示切模式；不传数据，不计算 loss，也不会自动做一次 SGD。

### 模式切换递归作用于整棵树

沿用上一节的 AffineStack，令 `m = AffineStack()`：

```text
构造后默认 training = True          nn.Module 默认处于训练态

m.eval() 后：m、layers、两个 FeatureAffine、drop 的 training 均为 False
m.train() 后：上述已登记模块的 training 均为 True
```

FeatureAffine 的 forward 没有读取 training，所以切模式不改变它的计算；
Dropout 的 forward 根据 training 选择是否丢弃，所以计算会变。
`eval()` 不会分析并改写任意 Python 代码，只会设置已登记模块的模式。

前面如果把 layers 换成普通列表，这两个 FeatureAffine 就不会被父模块的 `eval()` 递归设置；
它们本身没有模式相关计算，所以目前数值上可能看不出问题。换成含 Dropout 的子层，后果就出现了。

因此登记结构同时服务三件事：找参数做更新、找状态做保存、找子模块切模式。
这才是为什么我们在写完整 Block 之前补模块组织，而不只是换一种类语法。

### 回到你将要实现的 Transformer Block

构造阶段保存每个块自己的参数和子模块；前向阶段复用它们处理本次输入：

```text
构造时：登记 MHA 的 Wq/Wk/Wv/Wo、两个 Norm、FFN、两个 Dropout
前向时：X → LN1 → MHA → Dropout → 加回 X
                  U → LN2 → FFN → Dropout → 加回 U
```

这是顺序摘要，不是精确的分支图；X、U 和块输出均为 `(B,T,C)`，完整分支图见主干讲义。
实际 MHA 前向复用你已有的 `multi_head_self_attention`，只是从 self 取参数传给它。
这些组织机制不代替你实现 LayerNorm、FFN 或 Attention 的数学。

参数在构造阶段创建并保存，forward 复用这些对象，避免每次请求都重新初始化正在学习的数值。
下面把对象的创建时机与模式切换放到一起检查。

## 5. 一个综合排错检查

不用再做 Dropout 的期望手算；下一份综合练习会验证参数注册、数值、梯度及模式切换。
先检查一个关系：**参数更新、模式切换、求导，是否真的走同一条机制？**

某个正常构造的 AffineStack 持有两个 FeatureAffine（各有独立 gamma/beta，均已登记），
但程序员删除了构造方法中的 `self.drop = nn.Dropout(0.5)`，把 forward 改成：

```python
def forward(self, X):
    for layer in self.layers:
        X = layer(X)
    return nn.Dropout(0.5)(X)
```

最后一行先创建一个 Dropout 实例，再立刻调用它；不是在调用原有的子模块。
X 为 CPU float64 的 `(1,8,2)` 全 1 Tensor，gamma 初始全 1、beta 全 0。
对模型先调用 `eval()`，再在 `no_grad()` 下前向；过程中没有更新参数。

这是否已经关闭了 Dropout？如果重复前向时观察到输出变化，应修改的是参数注册、
对象创建的位置，还是再加一次 no_grad？说明原因即可，不要求某两次随机输出必定不同。

## 工程边界与后续衔接

- 本文覆盖 Dropout 的两阶段计算、块内位置、
  `nn.Module`/`nn.Parameter` 参数注册与 `train()`/`eval()` 对 `no_grad()` 的区分。
  已有的残差、LayerNorm、FFN、MHA、链式法则和手写 SGD 直接复用，不在此重考。
- 第 3、4 节新增示例的参数、梯度、统一更新、状态键、恢复遗漏与模式行为已实跑核对。
  第 1 节有限采样仅是观察，期望恒等由公式推导；这些教师演示不算学习者独立实现。
- 仓库中**尚不存在**完整 Transformer 块的实现代码。本文给出的块结构
  （含 Dropout 位置）是可理解的组合方案，不冒充已完成的工程。
- 尚未展开的内容：`torch.optim` 的更复杂优化器（Adam/AdamW）及其状态、
  梯度裁剪、参数初始化策略、完整模型的训练与验证集划分。
  这些按 roadmap 属于 G3 的后续部分，本文只把 `parameters()` 与优化器的接口关系讲到够用。
- 读过本文不构成 `block.dropout`、`block.complete` 或
  `implementation.transformer-block` 的掌握证据。
  按项目约定，掌握需要独立解释、实现、测试或排错的可观察表现；
  实际状态以 `learning/progress.json` 为准。
- 课程准备不自动改变能力状态；课堂已有回答和后续实现，按学习协议分别记录真实证据。
