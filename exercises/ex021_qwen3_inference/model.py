"""学习者实现区：只使用基础 Tensor 运算，不调用现成模型或融合 Attention。

先读 notes/qwen3-inference-from-checkpoint.md，再按本文件顺序填写。
所有浮点输入/权重为 CPU float32；ID/位置为 CPU long；无 PAD、无 Dropout。
这次只验收推理，不要求反向；调用方用 torch.inference_mode() 关闭求导。
不得修改参数、输入或旧缓存，也不使用全局请求状态。支持非连续 Tensor。
参考 Hugging Face 只允许在独立 compare_hf.py 中作为裁判，不是实现依赖。
"""

import torch

from .support import LayerKV, QwenConfig


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x2 = x.square().mean(dim=-1, keepdim=True)
    x2 = torch.sqrt(x2 + eps)
    x_norm = x / x2
    return x_norm * weight

def apply_rope(
    x: torch.Tensor, positions: torch.Tensor, theta: float,
) -> torch.Tensor:
    """Qwen half-split RoPE：x (B,H,n,D)，positions (n,)，返回同 shape。

    positions 是本次 token 的绝对位置，可能不从 0 开始；D 为正偶数。
    配对是 (j,j+D/2)，不是 (2j,2j+1)。频率 theta**(-2*j/D)。
    正向旋转一对 (a,b) 得到 (a*cos-b*sin, a*sin+b*cos)。
    Q 和 K 用同一约定，V 不调用本函数；只处理新 Q/K，不能重转历史 K。
    """
    # angle = pos * theta ** (-2 *(0 to D /2) /D)
    D = x.shape[-1]
    half_D = D // 2
    j = torch.arange(half_D, dtype=torch.float32)  # (D/2,)
    freq = theta ** (-2 * j / D)  # (D/2,)
    angles = positions[:, None] * freq[None, :]  # (n,D/2)
    cos = torch.cos(angles)  # (n,D/2)
    sin = torch.sin(angles)  # (n,D/2)
    x_pre = x[..., :half_D]  # (B,H,n,D/2)
    x_suf = x[..., half_D:]  # (B,H,n,D/2)
    x_rotated_pre = x_pre * cos[None, None, :, :] - x_suf * sin[None, None, :, :]
    x_rotated_suf = x_pre * sin[None, None, :, :] + x_suf * cos[None, None, :, :]
    x_rotated = torch.empty_like(x)
    x_rotated[..., :half_D] = x_rotated_pre
    x_rotated[..., half_D:] = x_rotated_suf
    return x_rotated


def grouped_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    query_positions: torch.Tensor, key_positions: torch.Tensor,
) -> torch.Tensor:
    """绝对位置因果 GQA，返回 (B,Hq,n,D)，尚未合头/输出投影。

    q (B,Hq,n,D)；k/v (B,Hkv,S,D)；Hq 能被 Hkv 整除。
    query_positions (n,)、key_positions (S,) 为严格递增的绝对位置。
    query 位置 p 只能读取 key 位置 <=p；调用方保证每行至少一个可读 key。
    query head h 使用 KV head h // (Hq // Hkv)。临时复制 KV 允许，
    但缓存保持 Hkv 头。缩放因子用 sqrt(D)，不是 sqrt(C) 或 sqrt(S)。
    mask 的 True 表示允许读取；Softmax 沿 S 轴，无滑窗，不返回权重表。
    允许 torch.softmax，不允许 scaled_dot_product_attention 等融合接口。
    """
    raise NotImplementedError("TODO 4: GQA 与矩形因果权限")


def transformer_block(
    x: torch.Tensor, weights: dict[str, torch.Tensor], config: QwenConfig,
    layer_index: int, positions: torch.Tensor, cache: LayerKV | None = None,
) -> tuple[torch.Tensor, LayerKV]:
    """本层新增 token 的前向 + 功能式缓存追加。

    x (B,n,C) 来自 embedding 或上一层；weights 是完整官方名称字典。
    层参数前缀 model.layers.{layer_index}.，对应键见讲义/expected_shapes。
    positions (n,) 是 t..t+n-1，t=旧缓存长度，无 cache 时 t=0。
    cache.k/v 为 (B,Hkv,t,D)，K 已做本层 Q/K RMSNorm 与 RoPE，V 仅投影。
    返回 x_out (B,n,C)，以及新 LayerKV，k/v 均 (B,Hkv,t+n,D)。

    连接约定：input_layernorm -> Q/K/V 投影 -> 分头 -> Q/K 各自 RMSNorm
    -> Q/K RoPE -> 追加新 K/V -> GQA -> 合头 -> o_proj -> 第一条残差；
    再 post_attention_layernorm -> SwiGLU(gate/up/down) -> 第二条残差。
    post_attention_layernorm 这个权重名不意味着采用 Post-LN 块！
    Q 总宽度 Hq*D 可以不等于 C；不要借用默认 C/Hq 推导 D 的旧模型。
    SwiGLU 中可用 torch.sigmoid 构造 SiLU，不使用 nn.SiLU 或现成 FFN。
    不修改旧 cache 的对象或 Tensor；用新容器返回，不能保存重复后的 Hq 份 KV。
    本函数接收的长度、参数和位置保证合法；总容量由 forward 统一检查。
    """
    raise NotImplementedError("TODO 5: Qwen3 的一个完整块")


def forward(
    input_ids: torch.Tensor, weights: dict[str, torch.Tensor], config: QwenConfig,
    cache: list[LayerKV] | None = None,
) -> tuple[torch.Tensor, list[LayerKV]]:
    """只处理本次输入，返回每个新增位置的 logits 和各层的新缓存。

    input_ids: (B,n) 合法词表 ID，B,n>=1，不含 PAD。
    weights: 已加载、校验的完整字典；lm_head.weight 总存在，可能与 E 共享。
    cache=None: 新请求，从位置 0 开始；非 None: 层数正确且均含 t 个历史 token，
    与本请求/模型/B/dtype 一致。历史缓存的 K/V 布局见 transformer_block。
    返回 logits (B,n,Vocab)，以及长度 num_hidden_layers 的 list[LayerKV]。
    logits 是原始词表分数，不做 Softmax/argmax，只含本次 n 个位置。

    查 model.embed_tokens.weight -> 逐层同一个 block -> model.norm.weight
    -> lm_head.weight。不另加绝对位置 embedding，不缩放 token embedding。
    无论 prefill、单 token decode、多 token append，都走本函数，不复制前向。
    ValueError: t+n > config.max_position_embeddings，在任何计算/状态修改前拒绝。
    本题其余输入合法性由调用方保证，不要求写通用防御式校验框架。
    """
    # 0. 准备参数
    E = weights["model.embed_tokens.weight"]
    rms_gamma = weights["model.norm.weight"]
    W_out = weights["lm_head.weight"]
    # 1. 检查总长度，拒绝超容量请求。
    t = cache[0].k.shape[-2] if cache is not None else 0 # t代表已经处理过的历史长度，当前第一个input下标就是t
    n = input_ids.shape[-1]
    if t + n > config.max_position_embeddings:
        raise ValueError(
            f"历史长度 {t} + 本次长度 {n} 超过最大位置容量 {config.max_position_embeddings}"
        )
    positions = torch.arange(t, t + n, dtype=torch.long) # 生成本次 token 的绝对位置

    # 2. embedding
    x = E[input_ids] # (B,n,C)

    # 3. blocks
    new_cache = []
    for layer_index in range(config.num_hidden_layers):
        old_kv = cache[layer_index] if cache is not None else None
        x, new_kv = transformer_block(x, weights, config, layer_index, positions, cache=old_kv)
        new_cache.append(new_kv)

    # 4. final_norm
    x = rms_norm(x, rms_gamma, config.rms_norm_eps) # (B,n,C)

    # 5. 投影back
    logits = x @ W_out.T # (B,n,Vocab)
    return logits, new_cache


def greedy_generate(
    input_ids: torch.Tensor, weights: dict[str, torch.Tensor], config: QwenConfig,
    *, max_new_tokens: int, eos_token_ids: tuple[int, ...],
) -> torch.Tensor:
    """单请求贪心生成，只返回新增 ID (1,G)，不包含输入 prompt。

    input_ids (1,T)，T>=1，无 PAD；max_new_tokens 为非负整数。
    eos_token_ids: 允许多个结束 ID；空 tuple 表示只按预算结束。
    G<=max_new_tokens；若预测到任一 EOS，把该 ID 收入返回值并立即停止。
    每次调用都是新请求，不接受/保留外部缓存。不得修改 input_ids/weights。
    """
    # 1. 检查输入长度，拒绝超容量请求。
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens 必须为非负整数，但收到 {max_new_tokens}")
    if input_ids.shape[1] + max_new_tokens > config.max_position_embeddings:
        raise ValueError(
            f"输入长度 {input_ids.shape[1]} + 生成预算 {max_new_tokens} "
            f"超过最大位置容量 {config.max_position_embeddings}"
        )
    # 2. 先处理完整 prompt，得到初始 logits 和缓存。
    cache = None
    # 3. 贪心选择第一个新 token，检查是否 EOS。
    new_ids = []
    for _ in range(max_new_tokens):
        logits, cache = forward(input_ids, weights, config, cache=cache)
        next_id = logits[:, -1, :].argmax(dim=-1).item()
        new_ids.append(next_id)
        if next_id in eos_token_ids:
            break
        # 4. 仅送入上次预测的那个 ID，继续 forward。
        input_ids = torch.tensor([[next_id]], dtype=torch.long)
    # 5. 返回新增 ID，形状 (1,G)
    return torch.tensor([new_ids], dtype=torch.long)
