"""玩具数据集：由教师提供，不需要修改。

任务规则：
    序列固定长度 9，形式为
        <bos> m p q r o1 o2 o3 <eos>
    其中 m 是模式标记，p/q/r 是三个内容 token（可重复）。
        m == "A"  ->  (o1,o2,o3) = (p,q,r)      照抄
        m == "B"  ->  (o1,o2,o3) = (r,q,p)      倒序
    内容 token 有 6 种，因此三元组共 6**3 = 216 个，序列共 432 条。

划分方式：
    **按三元组划分**，而不是按序列随机抽取。验证集用到的三元组（连同它的
    两种模式）在训练集中一条都不出现，所以验证集上的表现反映的是对规则的
    掌握，而不是对见过样本的记忆。
"""

import itertools

import torch

TOKENS = ["<pad>", "<bos>", "<eos>", "A", "B", "c0", "c1", "c2", "c3", "c4", "c5"]
VOCAB = {token: index for index, token in enumerate(TOKENS)}
N = len(VOCAB)

PAD_ID = VOCAB["<pad>"]
BOS_ID = VOCAB["<bos>"]
EOS_ID = VOCAB["<eos>"]
MODE_A = VOCAB["A"]
MODE_B = VOCAB["B"]
CONTENT_IDS = [VOCAB[f"c{i}"] for i in range(6)]

SEQ_LEN = 9
# 输出段在“错位之后”的下标范围：input[4:8] 预测 target[4:8]，即 o1 o2 o3 <eos>。
OUTPUT_SLICE = slice(4, 8)


def all_sequences() -> list[list[int]]:
    """返回全部 432 条序列，每条是长度为 9 的 ID 列表。"""
    sequences = []
    for p, q, r in itertools.product(CONTENT_IDS, repeat=3):
        sequences.append([BOS_ID, MODE_A, p, q, r, p, q, r, EOS_ID])
        sequences.append([BOS_ID, MODE_B, p, q, r, r, q, p, EOS_ID])
    return sequences


def split_by_triple(
    seed: int = 0, val_fraction: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor]:
    """按三元组划分训练集与验证集。

    Returns:
        (train_ids, val_ids)，均为 CPU torch.long，shape 分别是 (346, 9) 与 (86, 9)。
        验证集中出现的三元组不会出现在训练集的任何一条里。
    """
    generator = torch.Generator().manual_seed(seed)
    triples = list(itertools.product(CONTENT_IDS, repeat=3))
    order = torch.randperm(len(triples), generator=generator).tolist()
    val_triples = {triples[i] for i in order[: int(len(triples) * val_fraction)]}

    train, val = [], []
    for p, q, r in triples:
        for mode, outputs in ((MODE_A, (p, q, r)), (MODE_B, (r, q, p))):
            row = [BOS_ID, mode, p, q, r, *outputs, EOS_ID]
            (val if (p, q, r) in val_triples else train).append(row)
    return torch.tensor(train), torch.tensor(val)


def split_random_labels(
    seed: int = 0, val_fraction: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor]:
    """对照数据集：输出段随机，与输入之间没有任何规则。

    形状和划分方式与 split_by_triple 完全一致，唯一的区别是 o1/o2/o3 由随机
    抽样得到。这份数据没有可学的规则，降低训练损失的唯一途径是逐条记忆，
    因此可用来观察“训练损失下降”与“学会规则”不是一回事。
    """
    generator = torch.Generator().manual_seed(seed + 777)
    triples = list(itertools.product(CONTENT_IDS, repeat=3))
    order = torch.randperm(len(triples), generator=generator).tolist()
    val_triples = {triples[i] for i in order[: int(len(triples) * val_fraction)]}

    train, val = [], []
    for p, q, r in triples:
        for mode in (MODE_A, MODE_B):
            picks = torch.randint(0, len(CONTENT_IDS), (3,), generator=generator)
            outputs = [CONTENT_IDS[i] for i in picks.tolist()]
            row = [BOS_ID, mode, p, q, r, *outputs, EOS_ID]
            (val if (p, q, r) in val_triples else train).append(row)
    return torch.tensor(train), torch.tensor(val)
