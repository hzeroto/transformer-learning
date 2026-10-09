"""将本课 Attention 接到已验收的 ex013 参数上，提供带注释的参考实现。

教师提供容器、已有运算的组合与全量入口，不新增或复制模型参数。
缓存路径用于调用方 torch.no_grad() 下的推理；无缓存路径保留梯度。
"""
from dataclasses import dataclass, field

import torch

from exercises.ex007_multi_head_attention.heads import merge_heads, split_heads
from exercises.ex013_llama_style.components import apply_rope
from exercises.ex013_llama_style.model import LlamaBlock, LlamaLM
from exercises.ex018_efficient_attention.attention import online_attention


@dataclass
class LayerWindowKV:
    """一个请求的一层状态；K 已按原绝对位置旋转，V 未旋转。

    非空时 k/v: CPU float32/64 (B,Hkv,t,D)，positions: CPU long (t,)。
    空时三个字段都是 None；追加完成后 t=min(已处理 token 数, W)。
    层、请求、K/V 各有自己的存储，不将 Hkv 永久扩成 Hq。
    """

    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    positions: torch.Tensor | None = None

    def __len__(self) -> int:
        return 0 if self.k is None else self.k.shape[-2]


@dataclass
class WindowState:
    """请求状态，不属于参数，也不进入 model.state_dict()。

    window: 每层窗口宽度 W，包含自身；start_position: 请求起始绝对位置。
    next_position: 下一 token 的绝对位置，初始等于 start_position。
    调用方保证状态来自同一模型的固定参数，各层历史合法且一致。
    """

    window: int
    start_position: int = 0
    next_position: int = 0
    layers: list[LayerWindowKV] = field(default_factory=list)

    def reset(self) -> None:
        """教师实现：释放所有层的历史，恢复本请求配置的起始位置。"""
        for layer in self.layers:
            layer.k = layer.v = layer.positions = None
        self.next_position = self.start_position


def _validate_config(window: int, start_position: int) -> None:
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError("window 必须是正整数，且包含当前位置")
    if (not isinstance(start_position, int) or isinstance(start_position, bool)
            or start_position < 0):
        raise ValueError("start_position 必须是非负整数")


def new_window_state(
    model: LlamaLM, window: int, *, start_position: int = 0,
) -> WindowState:
    """教师实现：只分配各层独立的空容器；不做前向或投影。

    通过本函数创建状态，避免手工构造时遗漏 next_position。
    位置容量在处理输入时检查；start_position 可以恰好等于 model.L。
    """
    _validate_config(window, start_position)
    return WindowState(window=window, start_position=start_position,
                       next_position=start_position,
                       layers=[LayerWindowKV() for _ in model.blocks])


def project_rotary_qkv(
    block: LlamaBlock, x: torch.Tensor, positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """教师辅助：复用已掌握的 norm1、投影、分头和新 Q/K 的 RoPE。

    x: (B,n,C)，positions: (n,)。返回 Q:(B,Hq,n,D)，
    K/V:(B,Hkv,n,D)。只处理本次新 x，不接触历史缓存，不旋转 V。
    """
    normalized = block.norm1(x)
    q = split_heads(normalized @ block.Wq, block.num_query_heads)
    k = split_heads(normalized @ block.Wk, block.num_kv_heads)
    v = split_heads(normalized @ block.Wv, block.num_kv_heads)
    return (apply_rope(q, positions, block.rope_theta),
            apply_rope(k, positions, block.rope_theta), v)


def finish_block(
    block: LlamaBlock, x: torch.Tensor, attention_output: torch.Tensor,
) -> torch.Tensor:
    """教师辅助：attention_output:(B,Hq,n,D) 接回原有残差和 FFN。"""
    hidden = x + merge_heads(attention_output) @ block.Wo
    return hidden + block.ffn(block.norm2(hidden))


def window_block(
    block: LlamaBlock, x: torch.Tensor, positions: torch.Tensor, window: int,
    cache: LayerWindowKV | None = None,
) -> torch.Tensor:
    """一个原有 LLaMA 块，使用本课在线滑窗 Attention。

    x: CPU float32/64 (B,n,C)，与 block 参数同 dtype；可非连续。
    positions: CPU long (n,)，本次连续递增的绝对位置，不从缓存长度推导。
    window: 正整数 W，包含自身；无 PAD；每个 query 的自身 key 总存在。
    cache: None，或同一请求本层历史，布局见 LayerWindowKV。
    返回 (B,n,C)，只包含本次新位置。

    可以调用 project_rotary_qkv 与 finish_block，重点实现：
    - 用本次 Q 读取应有的历史和新 KV；调用自己实现的 online_attention。
    - Hq 头 h 读取 KV 头 h//(Hq//Hkv)。允许临时扩展 KV 供底层计算，
      持久缓存必须仍是 Hkv 头；历史 K 不再次施加 RoPE。
    - window 的可读权限由真实绝对位置决定。多 token chunk 的每一行
      都要得到正确结果；完整 chunk 算完才驱逐其后不再需要的历史。
    - 成功后 cache 保存最后 min(W,已处理数) 份 K/V/绝对位置，使用独立
      紧凑存储；仅取 view 不能释放旧的大存储。可使用 Tensor.clone()，
      它复制数据到独立存储并在启用求导时保留求导关系。

    无 cache 时必须保留 x 与所有参数的梯度，不新建请求状态。
    有 cache 时只在调用方 no_grad 下验收，不要求跨调用反向传播。
    不改 x/positions/参数/已有 .grad，不切换模式或在内部调用 backward。
    合法 shape、dtype、正整数 W、历史顺序由调用方保证，不考通用校验。
    """
    # 1. 只为新 token 生成 Q/K/V，并按本次绝对位置旋转新 Q/K。
    # Q: (B,Hq,n,D)，新 K/V: (B,Hkv,n,D)。历史 K 已旋转，直接复用。
    q, new_k, new_v = project_rotary_qkv(block, x, positions)

    # 2. 本次读取集合由「进入时的缓存 + 整个新 chunk」构成。
    # 必须先算完本次所有 query，再裁到 W 条；否则 chunk 前部可能丢 key。
    # 无缓存的全量路径直接读取本次全部 K/V，由窗口权限约束每一行。
    if cache is not None and cache.k is not None:
        all_k = torch.cat((cache.k, new_k), dim=-2)
        all_v = torch.cat((cache.v, new_v), dim=-2)
        all_positions = torch.cat((cache.positions, positions))
    else:
        all_k, all_v, all_positions = new_k, new_v, positions

    # 3. GQA 中每 Hq/Hkv 个相邻 query 头共用一个 KV 头。
    # repeat_interleave 按头逐个重复，例如 Hq=6/Hkv=2 -> [0,0,0,1,1,1]。
    # 普通 repeat 会给出不同顺序。扩展只用于本次运算，缓存仍取 all_k/v。
    group_size = block.num_query_heads // block.num_kv_heads
    if group_size == 1:
        attention_k, attention_v = all_k, all_v
    else:
        attention_k = all_k.repeat_interleave(group_size, dim=1)
        attention_v = all_v.repeat_interleave(group_size, dim=1)
    key_valid = torch.ones(
        (x.shape[0], all_positions.numel()), dtype=torch.bool, device=x.device,
    )  # 模型接口无 PAD；因果与窗口权限由 online_attention 按位置生成。

    # 4. 新 Q 读取历史和新 KV，随后复用 Wo、两条残差及逐 token FFN。
    attention_output = online_attention(
        q, attention_k, attention_v, positions, all_positions, key_valid, window,
    )
    output = finish_block(block, x, attention_output)  # (B,n,C)

    # 5. 整个 chunk 计算成功后，持久化最新 min(W,已处理数) 条紧凑 KV。
    # 只切片会继续引用整个大存储；clone 明确分配独立存储，释放旧内容。
    # contiguous() 在切片本来就连续时可能不复制，不能代替这里的 clone。
    # clone 不会断开梯度；推理是否关闭求导由调用方控制。
    if cache is not None:
        cache.k = all_k[:, :, -window:, :].clone(memory_format=torch.contiguous_format)
        cache.v = all_v[:, :, -window:, :].clone(memory_format=torch.contiguous_format)
        cache.positions = all_positions[-window:].clone()

    return output


def window_forward(
    model: LlamaLM, input_ids: torch.Tensor, window: int, *, start_position: int = 0,
) -> torch.Tensor:
    """教师全量入口：CPU long (B,T) 无 PAD -> 原始 logits (B,T,N)。

    复用 window_block 的无缓存路径；梯度保留，不保存 KV，不更改模式。
    位置 start_position..start_position+T-1；超出 model.L 时 ValueError。
    """
    _validate_config(window, start_position)
    end = start_position + input_ids.shape[1]
    if end > model.L:
        raise ValueError("本次绝对位置超出 model.L")
    positions = torch.arange(start_position, end, device=input_ids.device)
    x = model.token_table[input_ids]
    for block in model.blocks:
        x = window_block(block, x, positions, window)
    return model.final_norm(x) @ model.vocab_proj


def window_step(
    model: LlamaLM, input_ids: torch.Tensor, state: WindowState,
) -> torch.Tensor:
    """只处理新增 ID，并推进本请求的窗口状态。

    input_ids: CPU long (B,n)，B,n>=1，无 PAD，词表 ID 合法。
    state: new_window_state 创建的本请求状态，各层合法、同模型同 batch。
    返回所有新增位置的原始 logits (B,n,N)，dtype 与参数一致。
    本次位置从 state.next_position 开始，成功后该字段增加 n。
    复用 model.token_table、每层 window_block、final_norm、vocab_proj；
    不复制块内数学，不生成或采样 token，不重算整个原始 token 历史。

    若 state.next_position+n>model.L，必须在写入任何缓存或修改游标前
    抛 ValueError；只要求该容量错误保持原样，不增加其他失败回滚要求。
    窗口裁剪不会延长或重置 model.L；合法位置为 0..model.L-1。
    调用方在 torch.no_grad() 下调用。不得修改输入/参数/已有 .grad/模式，
    不登记新参数或把请求状态放进模型，不影响同模型的其他请求。
    """
    # 1. 先检查整个 chunk 的位置上界，确保容量失败不会改动任一层缓存。
    # next_position 是绝对位置游标；缓存裁到 W 条后，len(cache) 不再是它。
    end_position = state.next_position + input_ids.shape[1]
    if end_position > model.L:
        raise ValueError("本次绝对位置超出 model.L")

    # 2. 只查本次新 ID 的 embedding，并生成所有层共用的绝对位置。
    # arange 的右端不包含在结果中；最后一个合法位置为 model.L-1。
    positions = torch.arange(
        state.next_position, end_position, device=input_ids.device,
    )
    x = model.token_table[input_ids]  # (B,n,C)

    # 3. 每层处理同一批新位置，读取并更新本请求专属的这一层缓存。
    # 传给下一层的是本次所有 n 个位置的表示，无需重算旧 token。
    for block, cache in zip(model.blocks, state.layers):
        x = window_block(block, x, positions, state.window, cache)

    # 4. 返回每个新位置的词表分数，然后推进游标。
    # 保留完整 (B,n,N)，多 token 追加时不能只返回最后一个位置。
    logits = model.final_norm(x) @ model.vocab_proj
    state.next_position = end_position
    return logits
