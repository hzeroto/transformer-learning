"""MLM 数据准备：抽取遮盖计划，再生成受损输入与原文标签。

选中哪些位置、如何替换、哪些标签计分是不同对象，不能互相猜测。
当前实现由教师补全，供逐步阅读与后续复习。
"""

import torch

PAD_ID, CLS_ID, SEP_ID, MASK_ID = 0, 1, 2, 3
VOCAB = ("[PAD]", "[CLS]", "[SEP]", "[MASK]",
         "A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3")
VOCAB_SIZE = len(VOCAB)


def make_mlm_batch(
    clean_ids: torch.Tensor,
    input_valid: torch.Tensor,
    segment_ids: torch.Tensor,
    selected: torch.Tensor,
    replacement_kind: torch.Tensor,
    random_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """步骤 1：应用给定遮盖计划，生成模型输入与原位置标签。

    六个输入 shape 都为 (B,T)，B/T>0，CPU，可非连续。
    clean_ids：long，未改写原文；特殊 ID 见上方常量，普通词从 4 开始。
    input_valid：bool，True=可作为 key 的真实位置，与是否选中预测无关。
    segment_ids：long，0/1 表示句段，已经构造好，不在第二段重置位置。
    selected：bool，True=本次还原目标。
    replacement_kind：long；在 selected 位置，0=换 MASK_ID，
        1=换成 random_ids 同位置的 ID，2=保留原词。未选中位置忽略 kind。
    random_ids：long，已抽好的普通词 ID，可恰好等于原词；未用位置忽略。
    本函数不抽随机数；随机计划和数据语义由调用方指定。

    返回普通 dict，恰好五项，每项仍为 (B,T)：
      input_ids：仅按计划改写选中位置，其余保持原值；
      targets：原文同位置 ID，绝不能 shift，也不填 -100；
      input_valid、segment_ids：原样的独立副本；
      target_valid：selected 的独立副本，三种选中分支都 True。
    五个返回 Tensor 彼此、以及与所有输入的存储都独立；修改返回值不能污染输入。
    不修改任何输入，包括 selected/kind/random_ids。

    Raises ValueError：全 batch 没有选中位置；selected 落在 input_valid=False
    或 PAD/CLS/SEP/MASK 上；被选中位置的 kind 不属于 0/1/2。
    允许某条样本不选中、其他样本有选中。未选中处的 kind 不做校验。
    其余 shape/dtype/词表范围由调用方保证，不写通用数据校验框架。
    允许 torch.where、布尔索引、clone；不逐 token 写循环。
    """
    # 1. 只检查本次真正选中的目标；未选中处的 kind 无须有意义。
    if not selected.any():
        raise ValueError("整个 batch 没有选中的 MLM 目标")
    candidates = input_valid & (clean_ids >= 4)
    if (selected & ~candidates).any():
        raise ValueError("MLM 目标必须是有效位置上的普通词")
    selected_kinds = replacement_kind[selected]
    if ((selected_kinds < 0) | (selected_kinds > 2)).any():
        raise ValueError("选中位置的 replacement_kind 必须为 0、1 或 2")

    # 2. 输入允许受损，答案始终是原文。两个 clone 防止它们共享存储。
    input_ids = clean_ids.clone()
    targets = clean_ids.clone()

    # 3. 三种受损策略只改变模型输入；kind=2 直接保留 clone 中的原词。
    mask_positions = selected & (replacement_kind == 0)
    random_positions = selected & (replacement_kind == 1)
    input_ids[mask_positions] = MASK_ID
    input_ids[random_positions] = random_ids[random_positions]

    # 4. 可读取的输入位置与要计分的标签位置各有一份 mask，不能混用。
    # 随机替换恰好等于原词、以及主动保留原词的位置，也仍然需要计分。
    return dict(
        input_ids=input_ids,
        targets=targets,
        input_valid=input_valid.clone(),
        segment_ids=segment_ids.clone(),
        target_valid=selected.clone(),
    )


def sample_mlm_plan(clean_ids, input_valid, *, generator, probability=0.15):
    """教师辅助：近似 15% 选中 + 条件 80/10/10 计划，不替你改写输入。

    本教学版本按普通词独立抽样；全 batch 偶然未选中时固定补第一个候选。
    因此不是官方静态数据生成器的精确采样复刻，短样本比例不保证等于 15%。
    返回 selected、kind、random_ids 三个张量，供 make_mlm_batch 使用。
    generator 是显式 torch.Generator，避免依赖全局随机状态。
    """
    if not 0 < probability <= 1:
        raise ValueError("probability 必须在 (0,1] 内")
    candidates = input_valid & (clean_ids >= 4)
    if not candidates.any():
        raise ValueError("没有可预测的普通词")
    selected = (torch.rand(clean_ids.shape, generator=generator) < probability) & candidates
    if not selected.any():
        first = candidates.nonzero()[0]
        selected[first[0], first[1]] = True
    draws = torch.rand(clean_ids.shape, generator=generator)
    kind = torch.where(draws < 0.8, 0, torch.where(draws < 0.9, 1, 2))
    random_ids = torch.randint(4, VOCAB_SIZE, clean_ids.shape, generator=generator)
    return selected, kind, random_ids
