"""源序列保持完整，目标序列右移后组成 teacher-forcing batch。"""

import torch


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


def make_teacher_forcing_batch(
    source_sequences: list[list[int]],
    target_sequences: list[list[int]],
) -> dict[str, torch.Tensor]:
    """步骤 1：分别补齐源序列与目标记录，再构造输入和监督标签。

    两个列表各有 B 条序列，B>0；每条均非空，内容 ID 均 >=3，
    不含 PAD/BOS/EOS。每对 source/target 的内容长度可以不同。
    输入由调用方保证合法，不要求额外的通用参数校验；不修改原列表。

    源序列不添加特殊 token，右侧用 PAD 补到本 batch 最长源长度 S。
    每条目标先组成 [BOS] + 内容 + [EOS]，整条记录右 PAD 到
    最长目标内容长度 + 2，再错开一位拆成目标输入与下一 token 标签。
    因此 T = 最长目标内容长度 + 1，与 S 没有必须相等的关系。

    返回普通 dict，恰好以下六个 CPU Tensor：
      src_ids          long (B,S)：补齐后的源 ID；
      src_valid        bool (B,S)：源 token 不是 PAD 时为 True；
      tgt_input_ids    long (B,T)：目标记录去掉末尾一列；
      tgt_input_valid  bool (B,T)：目标输入 token 不是 PAD 时为 True；
      labels           long (B,T)：目标记录去掉开头一列；
      target_valid     bool (B,T)：标签不是 PAD 时为 True，EOS 也要计分。

    有效输入决定 Attention 能读取哪些 key，target_valid 决定 loss
    计分位置，两者不能混用。短目标的 EOS 可留在 tgt_input_ids，
    此处输入有效，但对应的 PAD 标签不计分。此函数不构造 Attention
    的因果 mask，不调用模型、不算 loss。允许用 Python 循环补齐列表。
    """
    # 源和目标分别计算最长长度：两条序列的长度没有必须相等的关系。
    batch_size = len(source_sequences)
    source_length = max(len(sequence) for sequence in source_sequences)
    record_length = max(len(sequence) for sequence in target_sequences) + 2
    src_ids = torch.full((batch_size, source_length), PAD_ID, dtype=torch.long)
    target_records = torch.full((batch_size, record_length), PAD_ID, dtype=torch.long)

    # 先以 PAD 填满容器，再把每条真实内容放到左侧，形成右侧补齐。
    # 列表相加创建新列表，不会往调用方的原始 target 中插入特殊 token。
    for row, (source, target) in enumerate(zip(source_sequences, target_sequences)):
        src_ids[row, :len(source)] = torch.tensor(source, dtype=torch.long)
        record = [BOS_ID] + target + [EOS_ID]
        target_records[row, :len(record)] = torch.tensor(record, dtype=torch.long)

    # [BOS,C,B,A,EOS] 拆成输入 [BOS,C,B,A] 和标签 [C,B,A,EOS]。
    # 每个输入位置只负责预测紧随其后的 token，而不是预测自己。
    tgt_input_ids = target_records[:, :-1]  # (B,T)
    labels = target_records[:, 1:]         # (B,T)

    # 输入权限与标签有效性分别计算。短序列的输入 EOS 可以有效，
    # 但它对应的下一项 PAD 标签不计 loss；作为标签的 EOS 则参与训练。
    return {
        "src_ids": src_ids,
        "src_valid": src_ids != PAD_ID,
        "tgt_input_ids": tgt_input_ids,
        "tgt_input_valid": tgt_input_ids != PAD_ID,
        "labels": labels,
        "target_valid": labels != PAD_ID,
    }
