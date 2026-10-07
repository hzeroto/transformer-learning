"""Mini-BERT 教师补全版：表示输入 → 双向 Encoder → MLM / 整句任务头。

固定教学配置：CPU float32/64、Post-LN、带偏置 MHA、GELU，全部 Dropout=0。
没有 GQA/RoPE/KV Cache，不要求官方 checkpoint 兼容。
"""

import math

import torch
from torch import nn

from exercises.ex007_multi_head_attention.attention import multi_head_attention
from exercises.ex008_transformer_block.block import LayerNorm


def _positive(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须为正整数")


def _weight(rows, cols, dtype):
    return nn.Parameter(torch.randn(rows, cols, dtype=dtype) * 0.04)


def _bias(size, dtype):
    return nn.Parameter(torch.zeros(size, dtype=dtype))


def gelu(x: torch.Tensor) -> torch.Tensor:
    """步骤 2：讲义中的 tanh 近似 GELU，shape/dtype/device 保持不变。

    x 为 CPU float32/64 浮点 Tensor，可非连续，可有任意非空 shape。
    用基础算术与 torch.tanh 实现；不调用 F.gelu/nn.GELU，不改成 ReLU/SiLU。
    保留 x 的梯度，不原地修改。公式见讲义 §7，不要求重新推导。
    """
    # 逐元素非线性；乘以平滑系数，不混合 token 或改变特征宽度。
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
    ))


class EncoderBlock(nn.Module):
    def __init__(self, C, num_heads, ffn_hidden, eps=1e-5, *, dtype=torch.float64):
        super().__init__()
        for name, value in (("C", C), ("num_heads", num_heads), ("ffn_hidden", ffn_hidden)):
            _positive(name, value)
        if C % num_heads:
            raise ValueError("C 必须能被 num_heads 整除")
        self.C, self.num_heads, self.ffn_hidden = C, num_heads, ffn_hidden
        self.Wq = _weight(C, C, dtype)
        self.Wk = _weight(C, C, dtype)
        self.Wv = _weight(C, C, dtype)
        self.Wo = _weight(C, C, dtype)
        self.bq, self.bk = _bias(C, dtype), _bias(C, dtype)
        self.bv, self.bo = _bias(C, dtype), _bias(C, dtype)
        self.norm1 = LayerNorm(C, eps, dtype=dtype)
        self.norm2 = LayerNorm(C, eps, dtype=dtype)
        self.W1, self.b1 = _weight(C, ffn_hidden, dtype), _bias(ffn_hidden, dtype)
        self.W2, self.b2 = _weight(ffn_hidden, C, dtype), _bias(C, dtype)

    def forward(self, x: torch.Tensor, input_valid: torch.Tensor) -> torch.Tensor:
        """步骤 3：双向 MHA + Post-LN，再 GELU FFN + Post-LN。

        x：(B,T,C)，本层输入，CPU float32/64，与参数 dtype 相同；可非连续。
        input_valid：(B,T) bool，True=可读 key。各 head 共享权限。
        返回 y：(B,T,C)，本层输出，不是词表 logits。
        参数 shape 已在构造器列明。Q/K/V 由同一份 x 做带偏置投影。
        复用 multi_head_attention；它接收已投影的三轴 Q/K/V 和
        allowed(B,T,T)，返回 (输出, 权重)，内部不隐含 causal mask。
        Wo 的矩阵乘法由该函数执行，输出偏置 bo 由本层处理。

        每个 query 都允许读取全部有效 key；不要调用因果 self wrapper，
        不屏蔽自身、不按句段隔离，不清空 PAD query 行或强制输出为零。
        两次残差相加之后才分别 norm1/norm2；FFN 中间激活用本题 gelu。
        固定没有 Dropout，不需要新写 Dropout 或模式管理代码。
        沿用 MHA 的 ValueError：任一样本没有有效 key。
        不修改 x/valid/参数/已有 .grad，不切模式，不 detach/no_grad/backward。
        不创建新参数，不保存请求状态；不逐 head/token 手写循环。
        """
        # 1. 同一份输入分别投影出 Q/K/V；每份仍为 (B,T,C)。
        q = x @ self.Wq + self.bq
        k = x @ self.Wk + self.bk
        v = x @ self.Wv + self.bv

        # 2. 每个 query 可以读全部有效 key，包括右侧位置和另一句段。
        # allowed[b,i,j] = input_valid[b,j]；PAD 列不可读，query 行不清零。
        B, T, _ = x.shape
        allowed = input_valid.unsqueeze(1).expand(B, T, T)  # (B,T,T)
        attention, _ = multi_head_attention(q, k, v, self.Wo, self.num_heads, allowed)

        # 3. Attention 已乘 Wo，这里只补 bo；先残差相加，再归一化。
        y = self.norm1(x + attention + self.bo)  # (B,T,C)，Post-LN

        # 4. FFN 独立处理每个位置：扩宽、非线性、投影回 C，再残差和 LN。
        hidden = gelu(y @ self.W1 + self.b1)  # (B,T,ffn_hidden)
        update = hidden @ self.W2 + self.b2  # (B,T,C)
        return self.norm2(y + update)


class MiniBert(nn.Module):
    def __init__(
        self, vocab_size, C, num_heads, ffn_hidden, n_layer, max_positions,
        num_classes=2, eps=1e-5, *, dtype=torch.float64,
    ):
        super().__init__()
        for name, value in (("vocab_size", vocab_size), ("n_layer", n_layer),
                            ("max_positions", max_positions), ("num_classes", num_classes)):
            _positive(name, value)
        self.N, self.C, self.max_positions = vocab_size, C, max_positions
        self.n_layer, self.num_classes = n_layer, num_classes
        self.config = dict(vocab_size=vocab_size, C=C, num_heads=num_heads,
                           ffn_hidden=ffn_hidden, n_layer=n_layer,
                           max_positions=max_positions, num_classes=num_classes, eps=eps)
        # 输入：三张表给出内容、绝对位置、句段的信息，都使用 C 维向量。
        self.token_table = _weight(vocab_size, C, dtype)
        self.position_table = _weight(max_positions, C, dtype)
        self.segment_table = _weight(2, C, dtype)
        self.embedding_norm = LayerNorm(C, eps, dtype=dtype)
        # 主干：每层保持 (B,T,C)，逐层混合可见上下文。
        self.blocks = nn.ModuleList([
            EncoderBlock(C, num_heads, ffn_hidden, eps, dtype=dtype)
            for _ in range(n_layer)
        ])
        # MLM：先变换 C 维表示，最后复用 token_table 得到 N 个词表分数。
        self.Wm, self.bm = _weight(C, C, dtype), _bias(C, dtype)
        self.mlm_norm = LayerNorm(C, eps, dtype=dtype)
        self.bvocab = _bias(vocab_size, dtype)
        # 不另建词表输出权重：MLM 输出真正复用 token_table 参数。
        # 整句任务：共享 CLS pooler，NSP 与下游分类使用各自的输出参数。
        self.Wpool, self.bpool = _weight(C, C, dtype), _bias(C, dtype)
        self.Wnsp, self.bnsp = _weight(C, 2, dtype), _bias(2, dtype)
        self.Wclass, self.bclass = _weight(C, num_classes, dtype), _bias(num_classes, dtype)

    def encode(self, input_ids, segment_ids, input_valid):
        """步骤 4：三表查值、embedding LN、逐层双向块，返回 Z(B,T,C)。

        三个输入 shape 都为 (B,T)，CPU，B/T>0，可非连续。
        input_ids 为合法 long ID（PAD 位置也合法）；segment_ids 为 long 0/1；
        input_valid 为 bool，唯一控制 key 可见性，不能从某个 ID 反推权限。
        第 0 项是有效 CLS；每条样本至少一个有效 key（全 False 仍要求 ValueError）。
        位置取 0..T-1，不在句段切换处归零；三表逐元素相加后调用 embedding_norm。
        按 self.blocks 的顺序把全部层串起来；每层都收到同一份 input_valid。
        Post-LN 堆栈末尾不再额外加 final_norm；不接任务头。
        Raises ValueError：T>self.max_positions，或任一样本全无有效 key。
        不修改输入/参数/已有 .grad，不截断计算图、不改变模式。
        """
        # 1. 容量和可读性检查：每条样本至少要有一个有效 key。
        _, T = input_ids.shape
        if T > self.max_positions:
            raise ValueError("输入长度超出模型容量")
        if not input_valid.any(dim=-1).all():
            raise ValueError("每条样本至少需要一个有效 key")

        # 2. 同一位置的三种信息相加；位置向量在 batch 维广播。
        positions = torch.arange(T, device=input_ids.device)  # (T,)
        token_vectors = self.token_table[input_ids]  # (B,T,C)
        position_vectors = self.position_table[positions]  # (T,C)
        segment_vectors = self.segment_table[segment_ids]  # (B,T,C)
        x = token_vectors + position_vectors + segment_vectors  # (B,T,C)

        # 3. 输入归一化后串起全部 Encoder 层，共用同一份 key 权限。
        x = self.embedding_norm(x)  # (B,T,C)

        for block in self.blocks:
            x = block(x, input_valid)
        return x  # (B,T,C)

    def mlm_logits(self, z):
        """步骤 5：Z(B,T,C) → 原位词表分数 (B,T,N)。

        依次使用 Wm/bm 的仿射变换、gelu、mlm_norm，再用内容表转置作
        输出投影并加 bvocab。token_table 必须是实际共享的同一份参数，
        不复制成新参数、不 detach；两条使用路径的梯度应累加到它。
        z 可非连续，CPU float32/64，与参数同 dtype；不改 z 或已有状态。
        不取 argmax/softmax，不选择目标位置，不在此计算 loss。
        """
        # 前三步都在 C 维完成，最后才变成每个位置的 N 个词表分数。
        x = z @ self.Wm + self.bm  # (B,T,C)
        x = gelu(x)  # (B,T,C)
        x = self.mlm_norm(x)  # (B,T,C)
        # 同一个 Parameter 参与查表和输出投影，autograd 自动累加两条路径的梯度。
        return x @ self.token_table.T + self.bvocab  # (B,T,N)

    def pool(self, z):
        """步骤 6：取 CLS 位置，经 Wpool/bpool 和 tanh，返回 (B,C)。

        z：(B,T,C)，第 0 个位置是 CLS；不是沿 T 求平均，也不是取最后位置。
        这是讲义中的 pooler，不是重新运行 Encoder。不修改 z/已有状态，保留梯度。
        """
        cls = z[:, 0, :]  # (B,C)，这个位置已经通过 Encoder 读过上下文。
        return torch.tanh(cls @ self.Wpool + self.bpool)  # (B,C)

    def forward(self, input_ids, segment_ids, input_valid):
        """步骤 7：复用 encode/mlm_logits/pool，连接两个整句分类头。

        输入合同同 encode；返回 dict，恰好四项：
          encoded：(B,T,C)，主干输出；
          mlm_logits：(B,T,N)，同位置原词预测分数；
          nsp_logits：(B,2)，pooler 输出经 Wnsp/bnsp 得到；
          class_logits：(B,num_classes)，同一 pooler 输出经 Wclass/bclass 得到。
        一个前向只运行一次 Encoder，两个句头共享同一次 pooler 输出。
        只接线，不在这里改写输入、计算 loss、backward 或更新参数。
        输出均保留梯度；不改变模式或输入/参数/已有 .grad。
        一次返回全部头是为了方便教学验收，不是优化后的推理服务接口。
        """
        z = self.encode(input_ids, segment_ids, input_valid)  # (B,T,C)
        pooled = self.pool(z)  # (B,C)，两个整句任务共享这次结果。
        return {
            "encoded": z,
            "mlm_logits": self.mlm_logits(z),  # (B,T,N)，逐位置还原原词。
            "nsp_logits": pooled @ self.Wnsp + self.bnsp,  # (B,2)，片段是否相邻。
            "class_logits": pooled @ self.Wclass + self.bclass,  # (B,num_classes)。
        }
