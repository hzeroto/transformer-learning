"""学习者完成 grouped_query_attention 和 GQABlock.forward。

CPU float32/float64；核心数学用基础 Tensor，教师构造与已验收组件可直接复用。
完整契约见 README.md。不要从 tests 导入教师参照。
"""
import torch
from torch import nn

from exercises.ex006_single_head_attention.attention import make_causal_allowed, project_qkv
from exercises.ex007_multi_head_attention.heads import split_heads, merge_heads
from exercises.ex008_transformer_block.block import LayerNorm, FeedForward, Dropout


def _validate_heads(C, num_query_heads, num_kv_heads):
    """教师辅助：校验本课等宽、连续等分组配置，返回每头宽度 D。"""
    for name, value in (("C", C), ("num_query_heads", num_query_heads),
                        ("num_kv_heads", num_kv_heads)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} 必须是正整数")
    if C % num_query_heads:
        raise ValueError("C 必须能被 num_query_heads 整除")
    if num_query_heads % num_kv_heads:
        raise ValueError("num_query_heads 必须能被 num_kv_heads 整除")
    return C // num_query_heads


def grouped_query_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Wo: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    allowed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """实现已经投影之后的 GQA 读取，不在本函数中创建参数或重新投影。

    输入：
        Q: (B,Tq,C)，查询投影结果；Hq=num_query_heads，D=C//Hq。
        K,V: 各 (B,Tk,Hkv*D)，紧凑的 key/value 投影；Hkv=num_kv_heads。
        Wo: (C,C)，合并 Hq 个读取结果后的输出投影。
        allowed: None 或 CPU bool (B,Tq,Tk)，True 表示允许读取该 key。
            None 表示全可见；本函数不自动加因果 mask。所有头共享同一权限表。
        Q/K/V/Wo 都是 CPU、相同 dtype 的 float32/float64，支持非连续布局。
        除 head 配置外，shape 匹配且轴长均为正由调用方保证；输入与点积分数有限。

    返回：
        output: (B,Tq,C)，保持 dtype/device。
        weights: (B,Hq,Tq,Tk)，每个 Q 头沿 Tk 轴独立归一化；
            被屏蔽项严格为 0。保留两个返回值的自动求导路径。

    分组约定：
        R=Hq//Hkv；Q 头 h 使用 K/V 头 h//R（连续分组）。
        这是本题固定约定，训练/全量/缓存路径必须一致。
        Hkv=Hq 退化为已有 MHA；Hkv=1 为 MQA。
        缩放使用每个 Q/K 向量的宽度 sqrt(D)。
        每头读出的内容沿特征拼接，随后乘 Wo；不对各头求和或平均。

    Raises:
        ValueError: Hq/Hkv 非正或不满足 C%Hq==0、Hq%Hkv==0，
            可复用 _validate_heads；或任一 query 的 allowed 行全部为 False。
        不要求其他输入校验。

    约束：
        不原地修改输入、参数或任何已有 .grad；不调用 backward/no_grad/detach。
        可以复用 split_heads/merge_heads 和 torch.softmax。
        可用分组轴广播，也允许临时 repeat_interleave；不要求唯一布局或 stride。
        核心打分/归一化/读取由你实现，不调用已有 MHA、高级 Attention 或融合算子代做。
        不写逐 batch/head/query/key 的 Python 循环，采用批量 Tensor 运算。
    """
    raise NotImplementedError("TODO 1：实现分组查询注意力")


class GQABlock(nn.Module):
    """教师构造：Pre-LN 块，只有 K/V 投影宽度改变，p 固定为 0。

    参数已注册；D=C//num_query_heads，KV 宽度 num_kv_heads*D。
    norm1/norm2/ffn/drop1/drop2 复用 ex008，不要求再实现它们。
    """

    def __init__(
        self, C, num_query_heads, num_kv_heads, ffn_hidden, eps=1e-5,
        *, dtype=torch.float64,
    ):
        super().__init__()
        self.head_dim = _validate_heads(C, num_query_heads, num_kv_heads)
        self.C = C
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.Wq = nn.Parameter(torch.randn(C, C, dtype=dtype) * 0.05)
        self.Wk = nn.Parameter(torch.randn(C, num_kv_heads * self.head_dim, dtype=dtype) * 0.05)
        self.Wv = nn.Parameter(torch.randn(C, num_kv_heads * self.head_dim, dtype=dtype) * 0.05)
        self.Wo = nn.Parameter(torch.randn(C, C, dtype=dtype) * 0.05)
        self.norm1 = LayerNorm(C, eps, dtype=dtype)
        self.norm2 = LayerNorm(C, eps, dtype=dtype)
        self.ffn = FeedForward(C, ffn_hidden, dtype=dtype)
        self.drop1 = Dropout(0.0)
        self.drop2 = Dropout(0.0)

    def forward(self, X: torch.Tensor, input_valid: torch.Tensor) -> torch.Tensor:
        """X: (B,T,C)，input_valid: CPU bool (B,T) -> Y: (B,T,C)。

        X 是进入本层的表示，不在本层加内容或位置表。
        复用 norm1/norm2/ffn/drop1/drop2，沿用已实现的两条 Pre-LN 残差路径，
        把原 MHA 换成你本题的 grouped_query_attention。
        norm1 的结果分别乘 Wq/Wk/Wv，形成不同宽度的 Q 与紧凑 K/V。
        project_qkv 仅做三个矩阵乘，可复用；无需重写它。

        因果权限由 make_causal_allowed(input_valid) 得到：
            query i 可读 j<=i 且 input_valid[b,j] 为 True 的 key。
        input_valid 是 key 有效性，不是标签 mask；不清零 PAD query。
        全屏蔽行错误由 grouped_query_attention 传出。

        CPU float32/64，X 与参数相同 dtype，允许非连续 X。
        保留 X 与所有参数梯度；不改 X/input_valid/已有 .grad，不创建参数。
        不切换模式，不在 forward 中启用 no_grad/detach，不保存请求缓存。
        只返回块输出，不返回权重、logits 或 loss。
        """
        raise NotImplementedError("TODO 2：把 GQA 接入 Pre-LN 块")
