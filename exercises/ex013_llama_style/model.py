"""教师提供参数登记，学习者实现一个共用块和两个模型入口。

这是明确配置的教学模型，不是官方 LLaMA checkpoint 的加载器。
没有可训练绝对位置表；无 bias、无 Dropout，使用独立词表投影。
"""
import torch
from torch import nn

from exercises.ex006_single_head_attention.attention import make_causal_allowed, project_qkv
from exercises.ex007_multi_head_attention.heads import split_heads, merge_heads
from exercises.ex011_kv_cache.cache import LayerKVCache
from exercises.ex012_grouped_query_attention.gqa import grouped_query_attention, _validate_heads
from exercises.ex013_llama_style.components import RMSNorm, SwiGLU, apply_rope, _positive_int, _weight


class LlamaBlock(nn.Module):
    def __init__(
        self, C, num_query_heads, num_kv_heads, ffn_hidden,
        eps=1e-5, rope_theta=10000.0, *, dtype=torch.float64,
    ):
        super().__init__()
        self.head_dim = _validate_heads(C, num_query_heads, num_kv_heads)
        if self.head_dim % 2:
            raise ValueError("本课 RoPE 要求每头维度为偶数")
        if not rope_theta > 0:
            raise ValueError("rope_theta 必须大于 0")
        self.C = C
        self.num_query_heads, self.num_kv_heads = num_query_heads, num_kv_heads
        self.rope_theta = rope_theta
        self.Wq = _weight(C, C, dtype)
        self.Wk = _weight(C, num_kv_heads * self.head_dim, dtype)
        self.Wv = _weight(C, num_kv_heads * self.head_dim, dtype)
        self.Wo = _weight(C, C, dtype)
        self.norm1 = RMSNorm(C, eps, dtype=dtype)
        self.norm2 = RMSNorm(C, eps, dtype=dtype)
        self.ffn = SwiGLU(C, ffn_hidden, dtype=dtype)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor,
        allowed: torch.Tensor, cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        """一个共用的 Pre-Norm 块：全量和增量都调用此函数。

        x: CPU float32/64 (B,n,C)，本次要处理的表示，与参数同 dtype。
        positions: CPU long (n,)，本次绝对位置；无 cache 时允许非零起点。
        allowed: CPU bool (B,n,S)，True 表示该 query 可以读该 key。
            无 cache：S=n；有 cache：进入时历史长度 t，S=t+n。
            权限由调用者生成，此处不要再强加从零起步的方阵因果 mask。
        cache: None，或本层本请求的 LayerKVCache。
            有 cache 时调用方保证 positions 为 t..t+n-1；缓存合法、无 PAD。
        返回本次 n 个位置的输出 (B,n,C)，不是整个历史的输出。

        实现目标：
        - 两条 Pre-Norm 残差路径；norm1 输入投影出本层 Q/K/V。
        - Q/K 分头后调用本题 apply_rope；V 不旋转。
        - 可合头后直接复用 ex012 grouped_query_attention（它接收三轴张量）。
          Hq 头按 h//(Hq//Hkv) 读取对应组，注意力结果再走已有残差和新 FFN。
        - 有 cache 时，追加已旋转新 K 与未旋转新 V，持久布局均为
          (B,t+n,Hkv*D)。历史 K 不重复旋转，不永久保存 Hq 份复制。
          无 cache 时不创建、保存请求状态。

        Raises: ValueError，追加超出 cache.max_length 时，在修改该缓存前抛出；
            任一 allowed 行全 False 时，沿用已有 GQA 的 ValueError。
            本题只要求容量失败保持缓存原样，不增加其他失败的事务回滚要求。
        支持非连续 x；不修改 x/positions/allowed/参数/已有 .grad，不切换模式。
        保留包含历史 K/V 的同一次计算图；不 detach，不在内部 no_grad/backward。
        合法 batch/dtype/宽度/历史与参数一致性由调用方保证，不考通用校验。
        """
        raise NotImplementedError("TODO 4: LlamaBlock.forward")


class LlamaLM(nn.Module):
    def __init__(
        self, vocab_size, C, num_query_heads, num_kv_heads, ffn_hidden,
        n_layer, max_positions, eps=1e-5, rope_theta=10000.0,
        *, dtype=torch.float64,
    ):
        super().__init__()
        for name, value in (("vocab_size", vocab_size), ("n_layer", n_layer),
                            ("max_positions", max_positions)):
            _positive_int(name, value)
        self.N, self.C, self.L = vocab_size, C, max_positions
        self.n_layer = n_layer
        self.num_query_heads, self.num_kv_heads = num_query_heads, num_kv_heads
        self.token_table = _weight(vocab_size, C, dtype)
        self.blocks = nn.ModuleList([
            LlamaBlock(C, num_query_heads, num_kv_heads, ffn_hidden,
                       eps, rope_theta, dtype=dtype)
            for _ in range(n_layer)
        ])
        self.final_norm = RMSNorm(C, eps, dtype=dtype)
        self.vocab_proj = _weight(C, vocab_size, dtype)

    def forward(self, input_ids: torch.Tensor, input_valid: torch.Tensor) -> torch.Tensor:
        """从位置 0 开始的全量前向，用于训练和确定性对照。

        input_ids: CPU long (B,T)，合法词表 ID，B,T>=1。
        input_valid: CPU bool (B,T)，True 表示有效 key，不是 loss mask。
        返回原始 logits (B,T,N)，不取 argmax、不算 loss，dtype 与参数一致。

        输入只查 token_table，不加绝对位置表。为各层传入位置 0..T-1 以及
        因果+key有效性权限（可复用 make_causal_allowed），调用同一个
        LlamaBlock.forward；末尾 final_norm 与 vocab_proj。
        不清零 PAD query，沿用“每行至少一个可读 key”的要求。
        Raises: ValueError，当 T>self.L，或存在全屏蔽 query 行。
        不修改输入/参数/已有 .grad、不切换模式、不内部 no_grad，不保存 KV。
        """
        raise NotImplementedError("TODO 5: LlamaLM.forward")


def new_caches(model: LlamaLM) -> list[LayerKVCache]:
    """教师辅助：每次调用为一个新请求创建互相独立的空容器，不计算 K/V。"""
    return [LayerKVCache(max_length=model.L) for _ in range(model.n_layer)]


def llama_model_step(
    model: LlamaLM, input_ids: torch.Tensor, caches: list[LayerKVCache],
) -> torch.Tensor:
    """只处理本次新增 ID，返回所有新位置的 logits，并追加各层缓存。

    input_ids: CPU long (B,n)，n>=1，无 PAD，合法 ID，只包含尚未处理的位置。
    caches: 每层一个独立容器，调用方保证全部合法、同长 t、相同 batch/dtype，
        每个 max_length 为 None 或 >=model.L；旧状态来自同模型固定参数。
    返回 (B,n,N) 原始 logits；成功后每层 cache 长度均为 t+n。

    只查内容表；创建本次位置 t..t+n-1 和对应所有 key 0..t+n-1 的权限。
    query 的局部下标 i 只能读 j<=t+i 的 key，allowed 是 (B,n,t+n)。
    逐层调用已经完成的 LlamaBlock.forward 并传入该层容器，再经过
    final_norm/vocab_proj。不要复制一份块内数学或调用旧带 P 的模型入口。

    Raises: ValueError，当 t+n>model.L，必须在改变任何一层缓存前抛出。
    容量是运行配置，不是某张 P 表的行数。无需校验契约保证的其他输入。
    不生成 token、不采样、不计算 loss；不修改输入/参数/已有 .grad/模式。
    不使用 no_grad/detach；推理调用方自己关闭求导。训练步之间不复用 KV。
    """
    raise NotImplementedError("TODO 6: llama_model_step")
