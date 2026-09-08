"""第五部分：学习者实现单条序列的贪心生成，复用已有前向函数。"""

import torch

from exercises.ex005_training_loop.training import forward_logits


def greedy_generate(
    prefix_ids: torch.Tensor,
    E: torch.Tensor,
    W: torch.Tensor,
    eos_id: int,
    max_new_tokens: int,
) -> torch.Tensor:
    """固定参数，将每次预测出的 token 接回输入，返回前缀和续写。

    输入约定：
        prefix_ids: (1,T)，CPU long，T >= 1，单条无补齐的前缀。
        所有 ID 合法；前缀若有 EOS，它只会出现在最后一个位置。
        E: (N,C)，W: (C,N)，相同的 CPU float32 或 float64。
        参数可以需要或不需要求导；本题样例的前向分数均为有限值。
        eos_id 是合法词表 ID；max_new_tokens 是非负整数。
        不要求额外输入校验，不处理多请求 batch 或 PAD 对齐。

    行为要求：
        返回 CPU long，shape 为 (1,T+K)，0 <= K <= max_new_tokens。
        返回值包含原前缀；新生成的 EOS 也必须保留，之后立即停止。
        若前缀已以 EOS 结束，或预算为 0，返回前缀的独立副本。
        每步复用 forward_logits，以最后位置的词表分数选出最高分 ID。
        同分时选较小 ID，沿用 torch.argmax 的首个最大值规则。
        只把 eos_id 作为停止标记，不硬编码 0 或 4。
        不修改 prefix_ids、E、W、已有 .grad 或 requires_grad 状态。
        在 torch.no_grad() 范围内前向；不反向、不更新也不清梯度。
        不永久改变调用方的自动求导开关。

    使用 Tensor、循环和现有前向即可；不用 Softmax、采样或高级生成接口。
    API 与数据流说明见本目录 README 的第五部分。
    """
    pre_new = prefix_ids.clone() # (1, T)
    if prefix_ids[0,-1] == eos_id or max_new_tokens == 0:
        return pre_new 
    
    with torch.no_grad():
        while(max_new_tokens > 0 and pre_new[0,-1] != eos_id):
            max_new_tokens -= 1
            logits = forward_logits(pre_new, E, W) # (1, T, N)
            last_logits = logits[:, -1, :] # (1, N)
            index = last_logits.argmax(dim=-1, keepdim=True)
            pre_new = torch.cat((pre_new, index), dim=1)
        return pre_new