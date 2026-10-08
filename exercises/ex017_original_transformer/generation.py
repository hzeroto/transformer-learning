"""教师导读：固定源特征，只用模型自己的输出延长目标前缀。"""

import torch


def greedy_generate(
    model: torch.nn.Module,
    src_ids: torch.Tensor,
    src_valid: torch.Tensor,
    *,
    max_new_tokens: int,
    bos_id: int = 1,
    eos_id: int = 2,
) -> list[int]:
    """步骤 8：固定一份源表示，只用自己的预测增长目标前缀。

    输入：
        model 提供 encode(src_ids, src_valid) -> memory(B,S,C)，以及
        decode(tgt_input_ids, memory, src_valid, tgt_input_valid)
        -> logits(B,T,Nt)。本函数不接收目标答案。
        src_ids: CPU long，shape (1,S)，S>=1；src_valid: 同 shape bool，
        True 表示有效源位置，至少一个 True。模型参数为 CPU float32/64。
        max_new_tokens: 非负整数，限制新生成 token 数；负数抛 ValueError。
        bos_id/eos_id: 目标词表中合法、不同的 ID；无需额外通用参数校验。

    生成语义：
        在 eval 模式和 no_grad 范围内编码源序列一次，将同一份 memory 和
        src_valid 交给每一步 decode。目标输入从 [[bos_id]] 开始，每步
        将之前生成的 token 加入前缀，提供同 shape、全 True 的有效标记。
        从最后目标位置的 logits 沿词表轴 argmax，追加该 ID；生成 EOS
        后立即结束并保留 EOS，否则到长度上限结束。返回新 token 的
        Python int 列表，不含初始化的 BOS；不自动过滤 PAD/BOS 等候选，
        只有 eos_id 是停止条件。max_new_tokens=0 返回 []，无需调用模型。

    状态边界：
        每次调用重新编码自己的源输入，不在模型上保留请求缓存。
        不修改源输入、参数或已有 .grad；正常返回时恢复调用前统一的
        train/eval 模式和调用方求导开关。本题不要求恢复混合子模块模式。
        复用完整目标前缀即可，不实现 self-KV Cache，也不预计算 cross-KV。
        eval() 控制 Dropout 等模块；no_grad() 控制计算图，两者用途不同。
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不能为负数")
    if max_new_tokens == 0:
        return []

    was_training = model.training
    generated = []
    # 源输入不变，目标输入则从单独的 BOS 开始；两侧没有拼接。
    prefix = torch.tensor([[bos_id]], dtype=torch.long, device=src_ids.device)
    try:
        model.eval()  # 关闭 Dropout，使相同源输入的表示不受随机失活影响。
        with torch.no_grad():  # 生成不反向传播，不需要保留计算图。
            memory = model.encode(src_ids, src_valid)  # 本次请求仅执行一次。
            for _ in range(max_new_tokens):
                # 前缀全由 BOS 和实际生成的 token 构成，没有批处理补齐位置。
                prefix_valid = torch.ones_like(prefix, dtype=torch.bool)
                logits = model.decode(prefix, memory, src_valid, prefix_valid)

                # 本题 B=1；只取最后目标位置的词表分数，预测紧随前缀的新 token。
                next_token = int(logits[0, -1].argmax(dim=-1).item())
                generated.append(next_token)
                if next_token == eos_id:
                    break  # EOS 是生成结果的一部分，保留它用于结束位置校验。

                # 只把模型自己的预测追加到目标前缀；源和 memory 均保持不变。
                next_ids = torch.tensor([[next_token]], dtype=torch.long, device=prefix.device)
                prefix = torch.cat((prefix, next_ids), dim=1)
    finally:
        # 推理不应永久把调用方的训练模型切到 eval；已有 .grad 也不清除。
        model.train(was_training)
    return generated
