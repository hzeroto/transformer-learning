# 作业 003：内容与位置的相加表示

## 目标

实现本节的基础绝对位置表示，验证你能区分 token ID 与位置编号，并正确使用广播。
这不是正弦位置编码、训练循环或新的 Attention 实现。

只修改 `position_embedding.py` 中的 `embed_with_positions`，不修改测试来迎合实现。
两张表由调用方提供；本函数只查表和相加，不创建或训练参数表。

## 输入与输出

```text
ids:            (B, T)    torch.long
token_table:    (N, C)    浮点 Tensor
position_table: (L, C)    与 token_table 相同的浮点 dtype

返回 X:         (B, T, C)，dtype 与两张表一致

X[b, t, :] = token_table[ids[b, t], :] + position_table[t, :]
```

- 两张表分别用 token ID 与位置编号查行；位置不是 token 的属性。
- 每条序列都使用位置 `0` 到 `T-1`，位置表在 batch 之间共享。
- 要求 `T <= L`；超出时明确抛出 `ValueError`，不能重复或截断位置凑数。
- 不修改输入，不每次重新随机生成表，不沿特征轴拼接。
- 用基础 Tensor 索引、切片和广播完成，不调用高级模型封装。
- 假设输入均在 CPU 上，B、T、N、L、C 均为正数，ID 有效，两张表的 C 相同。
  除长度超限外，不要求额外的输入校验。
- 本函数不接收 valid，也不将 PAD 表示强制归零。接在 Padding 后使用时，仍需
  保留原来的 valid，供后续计算排除补齐位置。

## 指定样例

```text
token_table =
[[1, 2],
 [3, 4],
 [5, 6],
 [7, 8]]

position_table =
[[10, 100],
 [20, 200],
 [30, 300],
 [40, 400]]

ids =
[[2, 0, 2],
 [1, 3, 0]]

X =
[[[15, 106], [21, 202], [35, 306]],
 [[13, 104], [27, 208], [31, 302]]]
```

表中的整数写法只是方便阅读；代码和测试中两张表使用浮点 dtype。

## 语法提示

```python
B, T = ids.shape          # 将两个轴的长度分别赋值
position_table.shape[0]  # 位置表的行数 L
torch.arange(T)          # 整数 Tensor：[0, 1, ..., T-1]
table[index_tensor]      # 按整数 Tensor 中的各个编号查行
table[:T]                # 取前 T 行，不包含第 T 行
raise ValueError("说明") # 抛出明确的输入错误
```

可以用位置编号查表，也可以选取位置表的前 T 行。采用哪一种不影响验收。
函数签名中的 `torch.Tensor` 是类型提示，不会替你自动检查 shape。

## 运行与完成标准

使用项目已安装 PyTorch 的 Python 环境，在仓库根目录执行：

```bash
.venv/bin/python -m unittest tests.exercises.test_position_embedding -v
```

如果已经激活项目环境，也可以将 `.venv/bin/python` 换成 `python`。

未实现时，有一项失败提示需要完成 TODO，功能测试暂时跳过，这是框架的预期状态。
完成后，本题全部测试应通过，不能以 skipped 作为完成依据。

测试覆盖指定输出、重复 token 的不同位置、batch 共享位置、不同 shape/dtype、
位置表长度边界、token ID 重编号后的位置不变，以及输入不被修改。

通过实现与测试后，再结合已完成的位置归属解释，上报 `text-input.position` 的
掌握证据。创建框架不代表通关。
