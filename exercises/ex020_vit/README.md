# 020：ViT——图像入口与分类连接（教师补全版）

对应 [ViT 讲义](../../notes/vision-transformer.md) 及本课的切块、Conv2d、CLS、
位置编码讨论。[model.py](model.py) 已由教师补全，重点阅读两个函数：

1. `patch_embed`：图像 → 不重叠图块 → 展平 → 共享投影。
2. `MiniViT.forward`：图块序列 → CLS 与位置 → 已提供的 Encoder → 图像类别分数。

`VisionEncoderBlock`、参数注册和初始化由教师提供。它复用此前的 MHA、
LayerNorm、GELU，并明确采用 Pre-LN 和全序列双向读取。

## 接口约定

本题固定 CPU float32/float64，输入与参数 dtype 一致；允许非连续输入；
没有 Dropout、PAD、因果 mask、图像尺寸插值或数据增强。
所有维度为正，H/W 可被 P 整除；模型只接收与配置分辨率一致的图片。
无需为契约外非法输入添加校验。每个函数的完整约定见 docstring。

| 对象 | shape / 含义 |
|---|---|
| `images` | `(B,C,H,W)`，原始浮点像素 |
| `patch_weight` | `(C*P*P,D)`，所有图块共享的投影矩阵 |
| `patch_bias` | `(D,)`，共享偏置 |
| `patch_embed` 的返回值 | `(B,N,D)`，`N=(H/P)*(W/P)` |
| `cls_token` | `(1,1,D)`，全 batch 共享，位于每条序列开头 |
| `position` | `(1,N+1,D)`，包含 CLS 的位置，加入一次 |
| `blocks` | 依次调用 `block(tokens)`，输入/输出均为 `(B,N+1,D)` |
| `final_norm` | 逐向量归一化，分类时作用于最终 CLS |
| `class_weight` / `class_bias` | `(D,K)` / `(K,)`，K 为类别数 |
| 最终返回值 | `(B,K)`，每张图对 K 个类别的原始分数 |

图块先按图块行、再按图块列排列。每块内部固定按 **通道、块内行、块内列**
展平；这个顺序决定投影矩阵每一行对应哪个输入数值。不要只保证 shape 相等。

可查基础 API：`unfold` 提取窗口并把窗口内轴追加到末尾；`permute` 调整轴顺序；
`reshape` 合并或拆分轴；`expand` 可沿原长度为 1 的轴广播；`cat` 沿指定轴拼接。
图块投影使用基础张量操作和矩阵乘法；测试使用卷积作独立验算。

## 运行与验收

在仓库根目录运行：

```bash
.venv/bin/python -B -m unittest tests.exercises.test_vit -v
```

当前参考实现 **12 项测试全部通过，无跳过**。
测试包括准确索引、卷积等价、梯度、非连续输入、CLS/位置连接、
图像对分类结果的影响、图块置换性质，以及小批次分类拟合。

测试中的可读参照只供验算，本题不是闭卷测试。你可以查 API 和查看失败信息；
不要修改测试、已完成的前置练习或教师 Encoder 来规避失败。
小批次拟合只证明这份小模型能训练，不代表自然图像泛化。

本课已按机制理解完成验收，独立手写不作为本次要求；实现由教师提供。
