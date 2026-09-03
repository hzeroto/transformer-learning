# 作业 001：张量 shape 与索引转换

## 目标

亲自实现高阶张量索引与一维存储位置之间的双向转换，证明你能把 shape 中每个
轴的长度和索引含义落实到代码，而不只是判断简单的输出 shape。

## 已知规则

- `shape` 是每个轴长度组成的元组，例如 `(2, 3, 4, 5)` 表示 4 个轴。
- 每个轴的长度必须是正整数，本作业不处理空 shape 或长度为 0 的轴。
- 索引从 0 开始，并采用“最后一个轴连续存储”的顺序。
- “最后一个轴连续存储”表示：其他索引不变时，最后一个索引增加 1，对应的一维
  位置也增加 1。这也是连续 PyTorch 张量默认使用的顺序。

例如，shape 为 `(2, 3)` 时，二维索引按下面的顺序对应一维位置：

```text
(0, 0) -> 0    (0, 1) -> 1    (0, 2) -> 2
(1, 0) -> 3    (1, 1) -> 4    (1, 2) -> 5
```

## 任务

完成 [`tensor_index.py`](tensor_index.py) 中的两个函数：

```python
def offset(shape: tuple[int, ...], indices: tuple[int, ...]) -> int:
    ...

def indices(shape: tuple[int, ...], offset: int) -> tuple[int, ...]:
    ...
```

要求：

1. 支持任意数量的轴，不能针对四维张量硬编码。
2. `offset` 把多轴索引转换成一维位置。
3. `indices` 完成反向转换。
4. 对合法输入，始终满足：

   ```python
   indices(shape, offset(shape, original_indices)) == original_indices
   ```

5. shape 非法或索引数量与轴数不同时抛出 `ValueError`。
6. 单个索引或一维位置越界时抛出 `IndexError`。
7. 不调用 NumPy、PyTorch 或其他库中现成的索引转换函数。

指定验证样例：

```text
shape   = (2, 3, 4, 5)
indices = (1, 0, 2, 1)
offset  = 71
```

## 运行测试

在仓库根目录执行：

```bash
python -m unittest discover -s tests -t . -p 'test_*.py' -v
```

作业框架最初会因为 `NotImplementedError` 而失败，这是预期行为。完成代码后，所有
基础测试都应通过。如果环境中安装了 PyTorch，对照测试会自动启用；否则该项会显示
为 skipped，不影响基础验证。

通过测试后，代码和测试结果将作为“张量、shape 与索引”的掌握证据。
