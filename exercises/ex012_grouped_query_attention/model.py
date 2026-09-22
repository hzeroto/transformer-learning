"""教师提供模型参数登记；学习者实现两个缓存前向入口。

全量 forward 继承已完成的 ex009.MiniGPT，缓存容器复用 ex011.LayerKVCache。
本课只处理 Pre-LN、p=0，缓存路径输入不含 PAD；不再重写贪心生成循环。
"""
import torch
from torch import nn

from exercises.ex006_single_head_attention.attention import project_qkv
from exercises.ex008_transformer_block.block import LayerNorm
from exercises.ex009_mini_gpt.model import MiniGPT
from exercises.ex011_kv_cache.cache import LayerKVCache
from exercises.ex012_grouped_query_attention.gqa import (
    GQABlock, grouped_query_attention, _validate_heads,
)


class GQAMiniGPT(MiniGPT):
    """教师提供构造，继承 MiniGPT.forward，使用本题 GQABlock。

    参数表、位置表、词表投影与原模型同一配置，不与旧模型自动共享参数。
    构造的是新模型，不是把旧 MHA 存档无损转换为 GQA。
    """

    def __init__(
        self, vocab_size, C, num_query_heads, num_kv_heads, ffn_hidden,
        n_layer, max_positions, eps=1e-5, *, dtype=torch.float64,
    ):
        # 只初始化模块登记，不先创建一套马上丢弃的旧 MHA 参数。
        nn.Module.__init__(self)
        _validate_heads(C, num_query_heads, num_kv_heads)
        for name, value in (("vocab_size", vocab_size), ("ffn_hidden", ffn_hidden),
                            ("n_layer", n_layer), ("max_positions", max_positions)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        self.N, self.C, self.L = vocab_size, C, max_positions
        self.n_layer = n_layer
        self.norm_style = "pre"
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.token_table = nn.Parameter(torch.randn(vocab_size, C, dtype=dtype) * 0.05)
        self.position_table = nn.Parameter(torch.randn(max_positions, C, dtype=dtype) * 0.05)
        self.blocks = nn.ModuleList([
            GQABlock(C, num_query_heads, num_kv_heads, ffn_hidden, eps, dtype=dtype)
            for _ in range(n_layer)
        ])
        self.final_norm = LayerNorm(C, eps, dtype=dtype)
        self.vocab_proj = nn.Parameter(torch.randn(C, vocab_size, dtype=dtype) * 0.05)


def gqa_block_step(
    block: GQABlock, x: torch.Tensor, cache: LayerKVCache,
) -> torch.Tensor:
    """一个 GQA 块只处理本次 n 个新位置，并追加紧凑 K/V。

    block: 本题 GQABlock，已有固定参数。
    x: CPU float32/64 (B,n,C)，与参数同 dtype，n>=1，可非连续。
    cache: 此层此请求的 LayerKVCache，旧长度 t；空缓存 k/v 为 None。
        非空时 k/v 各为 (B,t,Hkv*D)，沿轴 1 表示时间。
        沿用拆头前布局，不把 Hq 份重复数据存进 cache。
        调用方保证 batch、dtype、KV 宽度与本层匹配。

    返回 (B,n,C) 的本层输出；追加后 cache.k/v 各 (B,t+n,Hkv*D)。
    持久缓存只持有紧凑 K/V，不保留扩展到 Hq 的副本，不预分配额外容量。
    本题常规独立投影得到的紧凑张量即可；不考优化分配器或融合投影。

    需要保持：
        与 GQABlock.forward 同一套 Pre-LN/残差/FFN/投影及分组规则。
        新位置的 K/V 要参与本次读取；缓存里保存本层 norm1(x) 的投影。
        对本次 query i、所有 key j，allowed[b,i,j]=(j<=t+i)，
        shape (B,n,t+n)。无 PAD；n=1 时能读到追加后的全部 key。
        可以调用 cache.append，复用其容量检查。

    Raises:
        ValueError: 本次追加超出 cache.max_length；失败时 k/v/长度保持原样。

    不修改 x/参数/已有 .grad，不切换 train/eval，不调用 no_grad/detach/backward。
    缓存前向也是可求导的算子；是否在推理时关闭梯度，由调用方控制。
    本课不做跨训练步复用缓存或截断反向，梯度测试只覆盖同一次图的分块前向。
    """
    t = len(cache)
    n = x.shape[1]
    if cache.max_length is not None and n + t > cache.max_length:
        raise ValueError("exceed max length")

    n1 = block.norm1(x)
    Q, K_new, V_new = project_qkv(n1, block.Wq, block.Wk, block.Wv)
    cache.append(K_new, V_new)

    # i >= j - t  (n, t + n)
    allowed = torch.arange(t, n + t).view(1, n, 1) >= torch.arange(t + n).view(1, 1, t + n)
    output, _ = grouped_query_attention(Q, cache.k, cache.v, block.Wo, block.num_query_heads, block.num_kv_heads, allowed)
    X1 = x + block.drop1(output)

    n2 = block.norm2(X1)
    return X1 + block.drop2(block.ffn(n2))

def gqa_model_step(
    model: GQAMiniGPT, input_ids: torch.Tensor, caches: list[LayerKVCache],
) -> torch.Tensor:
    """处理一批新 ID，返回它们每个位置的 logits，并更新所有层缓存。

    input_ids: CPU long (B,n)，B>=1，n>=1，无 PAD，ID 均合法。
        若缓存非空，只传本次尚未处理的新 ID，不重复传旧前缀。
    caches: 调用方持有的长度 model.n_layer 的列表，每层一个独立容器。
        调用方保证全部缓存同长 t、batch/dtype/KV宽度合法，
        且每个 cache.max_length 为 None 或 >=model.L；无需增加通用校验。
        新请求用全新的列表和容器。缓存不是 model 参数/成员或 state_dict 内容。

    返回：
        CPU (B,n,model.N) 原始 logits，与模型参数相同 dtype。
        对应绝对位置 t..t+n-1 对下一 token 的预测分数，不只返回最后一行。
        不取 argmax、不采样、不计算 loss。
        每层缓存追加 n 行，最后所有层长度均为 t+n。

    数据连接：
        内容表示来自 token_table[input_ids]。
        位置表示使用 position_table 中对应缓存长度 t 的连续 n 行，且仅加一次。
        依次调用 gqa_block_step，通过 final_norm 后做 vocab_proj。
        既有 MiniGPT.forward 提供从位置 0 开始的全量参照。

    Raises:
        ValueError: t+n>model.L，在改变任何一层缓存前抛出；
            异常后所有缓存的 k/v/长度保持原样。

    不修改 input_ids、参数和已有 .grad；保留求导路径及入口 training 状态。
    不在函数内启用 eval/no_grad；推理调用者自己使用它们。
    本课 p=0；不引入 Dropout 随机性、EOS 处理或旧生成循环的新一轮实现。
    """
    pos = len(caches[0])

    if pos + input_ids.shape[1] > model.L:
        raise ValueError("exceed max positions")

    X = model.token_table[input_ids] + model.position_table[pos:pos+input_ids.shape[1]]# (B, n, C)

    for block,cache in zip(model.blocks, caches):
        X = gqa_block_step(block, X, cache) # (B, n, C) 该层grade

    Y = model.final_norm(X)
    logit = Y @ model.vocab_proj # (B, n, N)

    return logit
        


