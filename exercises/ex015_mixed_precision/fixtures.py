"""教师提供确定性模型与数据，不包含待实现的精度运行或误差比较逻辑。"""

import torch

from exercises.ex013_llama_style.model import LlamaLM


def make_case(seed=1503, *, B=2, T=6, Hkv=2):
    """返回同一配置的 FP32 模型和已对齐的下一 token 任务输入。"""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = LlamaLM(17, 24, 6, Hkv, 64, 2, 32, dtype=torch.float32).eval()
    generator = torch.Generator().manual_seed(seed + 1)
    sequence = torch.randint(0, model.N, (B, T + 1), generator=generator)
    input_ids, targets = sequence[:, :-1], sequence[:, 1:]
    input_valid = torch.ones_like(input_ids, dtype=torch.bool)
    target_valid = input_valid.clone()
    if B * T > 1:
        target_valid[0, -1] = False  # 有效输入位置仍可不参与 loss。
    batch = dict(input_ids=input_ids, input_valid=input_valid,
                 targets=targets, target_valid=target_valid)
    return model, batch
