"""原版 Encoder–Decoder 教师完整导读版：从两侧输入到目标词表分数。

复用 ex007 的通用 MHA 与 ex008 的 LayerNorm、ReLU FFN、Dropout。
这里是双向源 Encoder + 因果目标 Decoder，不是扩展讨论中的 CED。
阅读主线：forward → encode / decode → DecoderBlock → CrossAttention。
"""

import math

import torch
from torch import nn

from exercises.ex006_single_head_attention.attention import make_causal_allowed
from exercises.ex007_multi_head_attention.attention import multi_head_attention
from exercises.ex008_transformer_block.block import Dropout, FeedForward, LayerNorm
from exercises.ex017_original_transformer.positions import sinusoidal_positions


def _positive(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


def _matrix(rows, cols, dtype):
    parameter = nn.Parameter(torch.empty(rows, cols, dtype=dtype))
    nn.init.xavier_uniform_(parameter)
    return parameter


def _feed_forward(width, hidden, dtype):
    module = FeedForward(width, hidden, dtype=dtype)
    nn.init.xavier_uniform_(module.W1)
    nn.init.xavier_uniform_(module.W2)
    return module


def source_allowed(src_valid: torch.Tensor, query_length: int) -> torch.Tensor:
    """步骤 3（教师补全）：源 key 权限 (B,S) → (B,query_length,S)。

    src_valid 是 CPU bool，可非连续，True 表示可以被读取的源位置。
    query_length>0；query 可以是源位置（Encoder）或目标位置（cross）。
    每个 query 都能读取全部有效源 key，既没有三角限制，也不按 query
    是否 PAD 清空整行。仅广播 query 轴，head 轴由通用 MHA 内部处理。
    任一样本完全没有有效源 key，抛 ValueError；其余输入保证合法。
    返回 CPU bool，不修改输入，不必物理复制相同的权限行。
    """
    B, S = src_valid.shape
    # 每条样本都要有可读的源位置，否则后续 Softmax 会遇到全屏蔽行。
    if not src_valid.any(dim=-1).all():
        raise ValueError("每条样本至少需要一个有效源 key")
    # (B,S) → (B,1,S) → (B,query_length,S)，所有 query 共享源 key 权限。
    # expand 只创建广播视图，不实际复制多份布尔矩阵。
    return src_valid.unsqueeze(1).expand(B, query_length, S)


class ProjectedAttention(nn.Module):
    """教师提供：每处 Attention 独立拥有四份投影，省略投影偏置。"""

    def __init__(self, C, num_heads, *, dtype=torch.float64):
        super().__init__()
        _positive("C", C)
        _positive("num_heads", num_heads)
        if C % num_heads:
            raise ValueError("C 必须能被 num_heads 整除")
        self.C = C
        self.num_heads = num_heads
        # 参数属于当前 Attention 实例。不同层、self/cross 各有独立的投影。
        self.Wq = _matrix(C, C, dtype)
        self.Wk = _matrix(C, C, dtype)
        self.Wv = _matrix(C, C, dtype)
        self.Wo = _matrix(C, C, dtype)


class SelfAttention(ProjectedAttention):
    """教师提供：复用已验收的同源投影与 MHA，读取权限由调用者指定。"""

    def forward(self, x, allowed):
        # Self Attention 的三份投影都来自本层输入 x；因果与否由 allowed 决定。
        return multi_head_attention(
            x @ self.Wq, x @ self.Wk, x @ self.Wv,
            self.Wo, self.num_heads, allowed,
        )


class CrossAttention(ProjectedAttention):
    def forward(self, query_states, memory, src_valid):
        """步骤 4：目标状态查询源特征，返回 (output, weights)。

        query_states：(B,T,C)，本层目标 self-attention 残差/LN 后的 U。
        memory：(B,S,C)，整个 Encoder 堆栈的最终 E，不是上一层 Decoder。
        src_valid：(B,S) CPU bool，True=可读取的源 key；T 与 S 可以不同。
        两个浮点输入都是 CPU float32/64，与参数 dtype 一致，可非连续。
        Q 的输入是 query_states；K/V 的输入是 memory。用本对象注册的
        Wq/Wk/Wv/Wo 与 source_allowed，接到 ex007.multi_head_attention。
        通用 MHA 接收已投影的三轴张量，返回 output(B,T,C)、
        weights(B,H,T,S)，已经包含 Wo 投影；不要再次乘 Wo。

        不添加因果限制、位置编码、残差或归一化，不调用因果 self wrapper。
        保留 query_states、memory 及全部参数的梯度；不 detach/no_grad。
        不修改输入、参数或已有 .grad，不在此保存 KV 或其他请求状态。
        任一样本全无源 key 的 ValueError 由 source_allowed 传出。
        """
        # 源已完整给出，目标第一个位置也能读取源的最后一个有效位置。
        allowed = source_allowed(src_valid, query_states.shape[1])  # (B,T,S)

        # Q 问的是“当前目标状态需要什么”，因此随本层目标状态而变化。
        query = query_states @ self.Wq  # (B,T,C)
        # K/V 仍在当前 cross-attention 内产生，输入却是源 Encoder 的最终 E。
        # 多个 Decoder 层接收同一份 E，但使用各自的 Wk/Wv，结果通常不同。
        key = memory @ self.Wk         # (B,S,C)
        value = memory @ self.Wv       # (B,S,C)

        # 通用 MHA 完成拆头、/sqrt(D)、mask、softmax、读取、合头及 Wo 投影。
        # output 保留 T 个 query 位置，weights 形状是 (B,H,T,S)。
        return multi_head_attention(
            query, key, value,
            self.Wo, self.num_heads, allowed,
        )


class EncoderBlock(nn.Module):
    """教师提供：双向 Self Attention + ReLU FFN，两条 Post-LN 残差。"""

    def __init__(self, C, num_heads, ffn_hidden, p=0.0, eps=1e-5, *, dtype=torch.float64):
        super().__init__()
        self.C, self.num_heads = C, num_heads
        self.self_attn = SelfAttention(C, num_heads, dtype=dtype)
        self.norm1 = LayerNorm(C, eps, dtype=dtype)
        self.norm2 = LayerNorm(C, eps, dtype=dtype)
        self.ffn = _feed_forward(C, ffn_hidden, dtype)
        self.drop1 = Dropout(p)
        self.drop2 = Dropout(p)

    def forward(self, x, src_valid):
        # Encoder 的每个源位置都能读取完整的有效源序列，包括右边的词。
        allowed = source_allowed(src_valid, x.shape[1])
        attention, _ = self.self_attn(x, allowed)
        # Post-LN：先将更新加回本层输入，再对每个位置做 LayerNorm。
        state = self.norm1(x + self.drop1(attention))
        # FFN 对每个位置分别变换特征，形状仍然是 (B,S,C)。
        return self.norm2(state + self.drop2(self.ffn(state)))


class DecoderBlock(nn.Module):
    def __init__(self, C, num_heads, ffn_hidden, p=0.0, eps=1e-5, *, dtype=torch.float64):
        super().__init__()
        self.C, self.num_heads = C, num_heads
        self.self_attn = SelfAttention(C, num_heads, dtype=dtype)
        self.cross_attn = CrossAttention(C, num_heads, dtype=dtype)
        self.norm1 = LayerNorm(C, eps, dtype=dtype)
        self.norm2 = LayerNorm(C, eps, dtype=dtype)
        self.norm3 = LayerNorm(C, eps, dtype=dtype)
        self.ffn = _feed_forward(C, ffn_hidden, dtype)
        self.drop1 = Dropout(p)
        self.drop2 = Dropout(p)
        self.drop3 = Dropout(p)

    def forward(self, x, memory, src_valid, tgt_input_valid):
        """步骤 5：目标 self → cross → FFN，返回本层目标特征 (B,T,C)。

        x：(B,T,C)，本层目标输入；memory：(B,S,C)，最终源特征 E。
        src_valid：(B,S)、tgt_input_valid：(B,T) 是 CPU bool 权限。
        浮点输入 CPU float32/64、可非连续、与参数同 dtype；S/T 均正。
        显式 valid 是权限的唯一来源，不通过某个 token ID 重新推断。

        本层有三条独立的残差分支，顺序与模块名对应：
        1. self_attn + drop1 + 残差 + norm1，得到 U；权限复用
           make_causal_allowed(tgt_input_valid)，不是 target_valid。
        2. cross_attn 查询 U、读取 memory；drop2 + 加回 U + norm2，得到 R。
        3. ffn 处理 R；drop3 + 加回 R + norm3，得到本层输出。
        每条残差都先相加后归一化；不要把长度可能不同的 memory 加进残差。
        目标 PAD query 行保留；所有 query 至少一个可读 key，否则 ValueError。
        不增加 embedding/词表头，不改模式/输入/参数，不 detach 或缓存 memory。
        训练梯度必须同时流入目标状态、源特征与本层参数。
        """
        # 1. 先读取已经可见的目标前缀。训练虽一次给出整段目标输入，
        #    位置 i 仍只能读取 j<=i 的有效 key，不能偷看后面的目标答案。
        allowed = make_causal_allowed(tgt_input_valid)  # (B,T,T)
        self_update, _ = self.self_attn(x, allowed)
        target_state = self.norm1(x + self.drop1(self_update))  # U: (B,T,C)

        # 2. 用刚得到的 U 查询源 E。读取输出与 U 都有 T 个位置，才能相加；
        #    memory 有 S 个位置，它提供被读取的信息，不直接加进目标残差。
        cross_update, _ = self.cross_attn(target_state, memory, src_valid)
        conditioned_state = self.norm2(target_state + self.drop2(cross_update))  # R

        # 3. 对已经融合目标前缀与源信息的特征做逐位置 FFN，再做第三条残差。
        ffn_update = self.ffn(conditioned_state)
        return self.norm3(conditioned_state + self.drop3(ffn_update))  # (B,T,C)


class MiniTransformer(nn.Module):
    """固定配置边界：原版主干、小词表、独立双侧 embedding 与词表头。"""

    def __init__(
        self, src_vocab_size, tgt_vocab_size, C, num_heads, ffn_hidden,
        n_encoder_layers, n_decoder_layers, p=0.0, eps=1e-5, *, dtype=torch.float64,
    ):
        super().__init__()
        for name, value in (
            ("src_vocab_size", src_vocab_size), ("tgt_vocab_size", tgt_vocab_size),
            ("n_encoder_layers", n_encoder_layers), ("n_decoder_layers", n_decoder_layers),
        ):
            _positive(name, value)
        self.C, self.num_heads = C, num_heads
        self.src_vocab_size, self.tgt_vocab_size = src_vocab_size, tgt_vocab_size
        # 两侧内容表独立注册；只有参数长期属于模型，E 是每次输入产生的激活。
        self.src_table = _matrix(src_vocab_size, C, dtype)
        self.tgt_table = _matrix(tgt_vocab_size, C, dtype)
        # 每次构造新 Block，避免列表复制导致多层意外共享同一套参数。
        self.encoders = nn.ModuleList([
            EncoderBlock(C, num_heads, ffn_hidden, p, eps, dtype=dtype)
            for _ in range(n_encoder_layers)
        ])
        self.decoders = nn.ModuleList([
            DecoderBlock(C, num_heads, ffn_hidden, p, eps, dtype=dtype)
            for _ in range(n_decoder_layers)
        ])
        self.src_drop, self.tgt_drop = Dropout(p), Dropout(p)
        self.W_vocab = _matrix(C, tgt_vocab_size, dtype)
        self.b_vocab = nn.Parameter(torch.zeros(tgt_vocab_size, dtype=dtype))

    def encode(self, src_ids, src_valid):
        """步骤 6：完整源输入 → 最后一层 Encoder 特征 E(B,S,C)。

        src_ids：(B,S) CPU long，所有 ID（包括 PAD 位置）在源词表范围内。
        src_valid：(B,S) bool 是权威权限，每条样本至少一个 True。
        两者可非连续，B/S>0。全无有效源 key 必须 ValueError。
        查 self.src_table，乘 sqrt(C)，加源位置 0..S-1 的固定正弦编码。
        PE 的 dtype 与 src_table 一致；两侧位置都独立从 0 开始。
        输入和 PE 相加后应用 src_drop，再依次经过 self.encoders。
        复用 sinusoidal_positions 与已提供的 EncoderBlock；不额外加
        embedding LN、final LN、segment embedding、pool 或输出头。
        返回全部 S 个位置，不切最后位置或 CLS，不清零 PAD query。
        保留计算图，不改模式/输入/参数/已有 .grad，不保存跨调用的 E。
        """
        # 源侧从自己的第 0 个位置开始。PE(S,C) 可沿 batch 轴广播相加。
        position_ids = torch.arange(src_ids.shape[1], device=src_ids.device)
        position_vectors = sinusoidal_positions(
            position_ids, self.C, dtype=self.src_table.dtype,
        )  # (S,C)
        token_vectors = self.src_table[src_ids]  # (B,S,C)
        states = self.src_drop(math.sqrt(self.C) * token_vectors + position_vectors)

        # 每一层都更新全部源位置；只有整个堆栈的最终输出被命名为 E/memory。
        for block in self.encoders:
            states = block(states, src_valid)
        # 不 detach：训练时目标 loss 要通过 cross KV 回传到源 Encoder。
        return states  # E: (B,S,C)

    def decode(self, tgt_input_ids, memory, src_valid, tgt_input_valid):
        """步骤 7：目标前缀 + 固定源特征 → logits(B,T,tgt_vocab_size)。

        tgt_input_ids：(B,T) CPU long，合法目标 ID；tgt_input_valid：(B,T) bool。
        每条前缀第 0 项有效，T>0；第 0 项无效导致无可读 key，须 ValueError。
        memory：(B,S,C)，encode 返回的最终 E；src_valid：(B,S) bool。
        所有 Tensor 可非连续。memory 与模型同 dtype（CPU float32/64）。
        显式 valid 决定权限，即使无效位置放了其他合法 ID，也仍然无效。

        使用 tgt_table、sqrt(C)、目标位置 0..T-1 的 PE 与 tgt_drop；
        再依次经过 self.decoders，每层接收同一个 memory、src_valid、
        tgt_input_valid。各层自行投影各自的 cross KV，不把 E 当作参数。
        堆栈后使用 W_vocab 与 b_vocab 得到原始词表分数。
        不加 final LN/softmax/argmax，不使用任何目标标签，不计算 loss。
        不重跑 Encoder，不把源与目标沿 token 轴拼接，不改变 memory。
        不 detach/no_grad，不改变模型模式，不创建或更新参数/请求状态。
        """
        # 目标位置也从 0 开始；BOS 是目标位置 0，不接在源长度 S 后面。
        position_ids = torch.arange(tgt_input_ids.shape[1], device=tgt_input_ids.device)
        position_vectors = sinusoidal_positions(
            position_ids, self.C, dtype=self.tgt_table.dtype,
        )  # (T,C)
        token_vectors = self.tgt_table[tgt_input_ids]  # (B,T,C)
        states = self.tgt_drop(math.sqrt(self.C) * token_vectors + position_vectors)

        # states 逐层变化；memory 始终是同一份最终源表示 E。
        # 各层在自己的 cross_attn 内把 E 投影成各自的 KV。
        for block in self.decoders:
            states = block(states, memory, src_valid, tgt_input_valid)

        # 每个目标位置得到一组“下一个 token”的原始分数。
        # 训练交给 CE；生成循环才会选最后位置并 argmax。
        return states @ self.W_vocab + self.b_vocab  # (B,T,Nt)

    def forward(self, src_ids, tgt_input_ids, src_valid, tgt_input_valid):
        """教师提供：训练时一次编码源，再对真实目标前缀并行计算 logits。"""
        # 标签不进入模型，forward 只接收源输入、目标输入和各自的读取权限。
        memory = self.encode(src_ids, src_valid)
        return self.decode(tgt_input_ids, memory, src_valid, tgt_input_valid)
