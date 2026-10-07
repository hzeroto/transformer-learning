"""预训练目标的接线：复用已有稳定 CE，组合词级 MLM 与句对级 NSP。

当前实现由教师补全；此处只计算 loss，参数更新由外部训练循环负责。
"""

import torch

from exercises.ex005_training_loop.loss import masked_cross_entropy


def classification_loss(logits, labels):
    """教师适配：logits(B,K)、labels(B,) → 可导标量 CE。

    把整句视为仅一个监督位置，复用已经验收的逐位置损失。
    CPU float32/64 logits，long 合法标签；每个样本等权，B>0。
    """
    valid = torch.ones_like(labels, dtype=torch.bool).unsqueeze(1)
    return masked_cross_entropy(logits.unsqueeze(1), labels.unsqueeze(1), valid)


def pretraining_loss(model, batch, nsp_labels):
    """步骤 8：实际调用模型，再连接 MLM 与 NSP，返回 (total, mlm, nsp)。

    batch 是 make_mlm_batch 返回的五项 dict，每项 (B,T)。
    nsp_labels：CPU long (B,)，0=实际后续片段，1=不相邻的配对。
    model(input_ids,segment_ids,input_valid) 返回本题四项 dict。
    注意实际送进 model 的必须是受损 input_ids，不能送 targets 原文。
    MLM：同位置 targets，仅 target_valid 计分，复用 masked_cross_entropy。
    NSP：整句二分类，复用上面的 classification_loss。
    三项输出都是与模型 logits 同 dtype 的可导标量 Tensor，total=mlm+nsp；
    不能 .item()/detach，不能把分类头 class_logits 混入本次预训练目标。

    这里只执行一次模型前向与损失计算；不 backward、更新参数、清梯度、
    切换 train/eval、改 batch/labels 或已有 .grad。不重新抽遮盖计划。
    全 batch target_valid 为 False 时沿用旧损失的 ValueError。
    """
    # 1. Encoder 只能看到受损输入；原文 targets 只作为答案交给 loss。
    outputs = model(batch["input_ids"], batch["segment_ids"], batch["input_valid"])

    # 2. MLM 还原同位置的原词，并在全 batch 的选中位置上取平均。
    # 这里不做 next-token 的标签平移，也不根据输入是不是 MASK 推断目标。
    mlm = masked_cross_entropy(
        outputs["mlm_logits"], batch["targets"], batch["target_valid"]
    )

    # 3. NSP 每条句对只计一个二分类目标，与下游 class_logits 分开。
    nsp = classification_loss(outputs["nsp_logits"], nsp_labels)

    # Tensor 相加保留两条求导路径；调用方再决定何时 backward / step。
    total = mlm + nsp
    return total, mlm, nsp
