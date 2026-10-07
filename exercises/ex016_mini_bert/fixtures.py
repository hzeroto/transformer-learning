"""教师提供合成语料与标签，不是学习者待实现的内容。

四个两段文档：[Ai Ai] 后面接 [Bi Bi]，i=0..3。
正例使用真实相邻两段，负例使用 A_i 与 B_(i+1 mod 4)，共 8 对。
MLM 固定遮盖第一段的首个 Ai，右侧仍有 Ai 可作为还原依据。
这是小数据可学习性验证，不模拟自然语言语料或评估泛化。
"""

import torch

from exercises.ex016_mini_bert.data import CLS_ID, SEP_ID, PAD_ID


def make_toy_case():
    rows, nsp, classes = [], [], []
    for i in range(4):
        for is_negative in (0, 1):
            j = (i + is_negative) % 4
            rows.append([CLS_ID, 4 + i, 4 + i, SEP_ID, 8 + j, 8 + j, SEP_ID, PAD_ID])
            nsp.append(is_negative)
            classes.append(j % 2)
    clean = torch.tensor(rows, dtype=torch.long)
    valid = clean != PAD_ID
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 0]], dtype=torch.long).expand(8, -1).clone()
    selected = torch.zeros_like(valid)
    selected[:, 1] = True
    return {
        "clean_ids": clean,
        "input_valid": valid,
        "segment_ids": segments,
        "selected": selected,
        "replacement_kind": torch.zeros_like(clean),  # 这个拟合实验固定全部 MASK。
        "random_ids": torch.full_like(clean, 4),       # 本实验未用，接口仍提供。
        "nsp_labels": torch.tensor(nsp, dtype=torch.long),
        "class_labels": torch.tensor(classes, dtype=torch.long),
    }
