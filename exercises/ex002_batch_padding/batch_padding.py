"""将不同长度的 token ID 序列右侧补齐，并标记有效位置。"""

import torch


def make_batch(
    sequences: list[list[int]], pad_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回补齐后的编号张量 ids，以及有效位置张量 valid。

    输入约定：
        sequences 和其中每条序列均非空，元素均为非负整数 ID。
        pad_id 为专用的补齐 ID，不出现在输入序列中。
        本练习不要求验证上述输入约定。

    输出约定：
        B 为序列条数，T 为当前 batch 中最长序列的长度。
        ids: shape 为 (B, T)，dtype 为 torch.long；仅在右侧补齐。
        valid: shape 为 (B, T)，dtype 为 torch.bool；真实位置为 True。
        不修改传入的 sequences，也不改变序列或 token 的顺序。
    所需 Python/PyTorch 写法：
    len(seq)                         # 序列长度

    for b, seq in enumerate(sequences):
        ...                          # 同时取得下标 b 和对应序列 seq

    torch.full((B, T), pad_id, dtype=torch.long)
    # 创建 shape 为 (B,T)、全部填入 pad_id 的整数 Tensor

    ids[b, :length] = torch.tensor(seq, dtype=torch.long)
    # 写入第 b 行的前 length 个位置；切片右端不包含在内
    """
    # TODO 1：根据输入确定 B 和 T。
    B, T = len(sequences), max(len(seq) for seq in sequences)
    # TODO 2：创建整数张量，使用 pad_id 填充。
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    # TODO 3：将每条原始序列写入对应行的左侧。
    for b, seq in enumerate(sequences):
        length = len(seq)
        ids[b, :length] = torch.tensor(seq, dtype=torch.long)
    # TODO 4：生成布尔有效位置标记，返回 (ids, valid)。
    valid = ids != pad_id
    return ids, valid
