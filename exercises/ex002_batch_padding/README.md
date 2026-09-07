# 作业 002：批处理与长度补齐

## 任务

只需完成 `batch_padding.py` 中的 `make_batch` 函数。核心实现留给学习者，
自动验证位于 `tests/exercises/test_batch_padding.py`，不要修改测试来迎合实现。

```python
def make_batch(sequences: list[list[int]], pad_id: int = 0):
    # 返回两个 PyTorch Tensor：ids, valid
    ...
```

- `B` 为序列条数，`T` 为当前 batch 中最长序列的长度。
- 将短序列在右侧补齐，保持输入中的序列和 token 顺序。
- `ids` 的 shape 为 `(B, T)`，dtype 为 `torch.long`。
- `valid` 的 shape 为 `(B, T)`，dtype 为 `torch.bool`；真实位置为 `True`，
  补齐位置为 `False`。
- 使用传入的 `pad_id`，不能将补齐编号硬编码为 0。
- 不修改原始 `sequences`。
- 使用基础 PyTorch 张量操作，不调用 `pad_sequence` 等现成补齐函数。

输入约定：batch 和每条序列都非空；ID 均为非负整数；输入序列中不含专用的
`pad_id`。本练习不要求额外处理空输入、非法 ID 或 GPU。

## 指定样例

```text
sequences = [[4, 1, 5], [7], [6, 8]]
pad_id = 0

ids =
[[4, 1, 5],
 [7, 0, 0],
 [6, 8, 0]]

valid =
[[True, True,  True ],
 [True, False, False],
 [True, True,  False]]
```

本例中 ID 1 表示 UNK（未知 token），它是有效内容，不能被当作 PAD 排除。
PAD 取其他编号时，ID 0 也可以是有效内容。判断依据是是否属于补齐位置，
不是编号大小，也不是 embedding 向量是否为零。

## Python / PyTorch 写法提示

这些只是可用操作，不是完整答案：

```python
len(seq)                             # 返回序列长度

for b, seq in enumerate(sequences):
    ...                              # 同时取得下标 b 和该条序列 seq

torch.full((B, T), pad_id, dtype=torch.long)
# 创建指定 shape、全部填入 pad_id 的整数 Tensor

ids[b, :length] = torch.tensor(seq, dtype=torch.long)
# 写入第 b 行的前 length 个位置；切片右端不包含在内
```

函数签名中的 `list[list[int]]` 表示整数列表组成的列表，`tuple[...]` 表示返回
类型，`->` 后面是返回值的类型提示；这些提示本身不会自动检查输入。
`raise NotImplementedError(...)` 是未完成占位，填写实现时将它替换掉。

## 运行测试

需要 Python 3.9 或以上版本，以及安装在同一 Python 环境中的 PyTorch。
CPU 环境即可，不需要 GPU。若尚未安装 PyTorch，可在准备使用的 Python 环境中执行：

```bash
python3 -m pip install torch
```

在仓库根目录运行本题：

```bash
python3 -m unittest tests.exercises.test_batch_padding -v
```

运行所有作业测试：

```bash
python3 -m unittest discover -s tests -t . -p 'test_*.py' -v
```

若系统中的解释器名称是 `python`，可将以上命令中的 `python3` 替换为 `python`。

尚未安装 PyTorch 时，测试会明确提示缺少依赖，而不是全数跳过后显示成功。
装好依赖但尚未填写函数时，会有一项失败提示需要完成 TODO，其余功能测试暂时
跳过。完成实现后，所有本题测试应通过，不能仅以 skipped 作为完成依据。

测试覆盖混合长度、等长序列、单条序列、最后一条最长、非零 PAD 编号、UNK
保留、输出 dtype 和不修改原始输入。通过实现和测试后，再上报
`text-input.batch-padding` 的掌握证据；创建本框架不代表通关。
