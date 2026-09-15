"""构造函数与参数登记由教师提供；请完成三处 TODO。

统一契约见同目录 README.md。基础 Tensor/autograd 可用，核心数学手写。
复用 ex002/ex003/ex005/ex008 的既有实现，不重写块内部。
"""

import torch
from torch import nn

from exercises.ex008_transformer_block.block import LayerNorm, TransformerBlock

# 固定词表：本练习不实现分词器，也不动态扩充词表。
VOCAB = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "A": 3,
    "B": 4,
    "X": 5,
    "U": 6,
    "V": 7,
}
PAD_ID = VOCAB["<pad>"]
BOS_ID = VOCAB["<bos>"]
EOS_ID = VOCAB["<eos>"]


def text_to_ids(text: str, vocab: dict[str, int] = VOCAB) -> list[int]:
    """把一行空格分隔的文本转成带起止标记的 ID 列表。

    Args:
        text: 以单个空格分隔的 token 串，例如 "A X U"。
            允许首尾多余空白；不含任何 token 时视为空序列。
        vocab: token 字符串到整数 ID 的映射，默认使用本模块的 VOCAB。

    Returns:
        Python 的 list[int]，不是 Tensor。
        形式为 [BOS_ID, 各 token 的 ID..., EOS_ID]。
        空文本返回 [BOS_ID, EOS_ID]，长度为 2。

    约定：
        BOS/EOS 由本函数添加，调用方不在原文里书写这两个标记。
        不做补齐，不截断，不修改 vocab。
        `text.split()` 按任意空白拆分并丢弃空串，可直接使用。

    Raises:
        KeyError: 出现 vocab 之外的 token 时，直接让查表抛出即可，
            不要吞掉异常，也不要替换成 UNK。
    """
    # TODO 1：拆分文本，逐个查表，并在两端加上起止标记。
    raise NotImplementedError("请实现 text_to_ids")


class MiniGPT(nn.Module):
    """一个 decoder-only 语言模型：ID 序列 -> 下一 token 的词表分数。

    构造函数已给出全部参数与子模块的登记方式，不需要修改。
    可用的属性：
        self.token_table: (N, C) 内容表，登记为 nn.Parameter。
        self.position_table: (L, C) 位置表，登记为 nn.Parameter。
        self.blocks: 含 n_layer 个 TransformerBlock 的 nn.ModuleList。
        self.final_norm: LayerNorm(C)，仅在 norm_style == "pre" 时存在，
            否则为 None。
        self.vocab_proj: (C, N) 词表投影矩阵，登记为 nn.Parameter。
            本练习使用独立投影，不与 token_table 共享权重。
        self.norm_style: "pre" 或 "post"，见 forward 的说明。
        self.n_layer / self.C / self.N / self.L: 整数配置。
    """

    def __init__(
        self,
        vocab_size: int,
        C: int,
        num_heads: int,
        ffn_hidden: int,
        n_layer: int,
        max_positions: int,
        norm_style: str = "pre",
        p: float = 0.0,
        eps: float = 1e-5,
        *,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if norm_style not in ("pre", "post"):
            raise ValueError("norm_style 必须是 'pre' 或 'post'")
        for name, size in (
            ("vocab_size", vocab_size),
            ("C", C),
            ("num_heads", num_heads),
            ("ffn_hidden", ffn_hidden),
            ("n_layer", n_layer),
            ("max_positions", max_positions),
        ):
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError(f"{name} 必须是正整数")

        self.N = vocab_size
        self.C = C
        self.L = max_positions
        self.n_layer = n_layer
        self.norm_style = norm_style

        self.token_table = nn.Parameter(torch.randn(vocab_size, C, dtype=dtype) * 0.05)
        self.position_table = nn.Parameter(
            torch.randn(max_positions, C, dtype=dtype) * 0.05
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(C, num_heads, ffn_hidden, p=p, eps=eps, dtype=dtype)
                for _ in range(n_layer)
            ]
        )
        # Post-LN 堆栈的每个子层输出都已归一化，末尾不再补这一次。
        self.final_norm = (
            LayerNorm(C, eps, dtype=dtype) if norm_style == "pre" else None
        )
        self.vocab_proj = nn.Parameter(torch.randn(C, vocab_size, dtype=dtype) * 0.05)

    def forward(
        self, input_ids: torch.Tensor, input_valid: torch.Tensor
    ) -> torch.Tensor:
        """返回每个位置对下一 token 的原始分数，shape (B, T, N)。

        Args:
            input_ids: (B, T)，CPU torch.long，所有 ID 在 [0, N) 内。
            input_valid: (B, T)，CPU torch.bool，True 表示该位置是真实
                token、可作为 key 被读取。它不是 target_valid，
                本函数不用它筛选损失，也不用它清零 PAD query 行。

        Returns:
            (B, T, N) 的原始 logits，dtype 与 device 同模型参数。
            不在此处做 Softmax、不取 argmax、不计算损失。

        计算顺序：
            1. 由 input_ids 与位置 0..T-1 得到初始表示 (B, T, C)。
               每条序列的位置一律从 0 开始；所有 batch 共享同一张位置表。
            2. 依次经过 self.blocks 中的每个块，每块都接收同一份
               input_valid。层间不重复加入位置表示。
            3. norm_style == "pre" 时，对最后一个块的输出再做一次
               self.final_norm；norm_style == "post" 时跳过这一步。
            4. 用 self.vocab_proj 把最后一维从 C 变到 N。

        约定：
            保留自动求导路径，不修改输入，不在前向切换 train/eval，
            不启用 no_grad，也不创建或替换参数。
            允许 T 取 1；要求 1 <= T <= self.L。

        Raises:
            ValueError: T 超过 self.L。

        提示：
            位置表示可复用 exercises.ex003_position_embedding 的
            embed_with_positions；它同样在 T > L 时抛 ValueError。
        """
        # TODO 2：按上面四步组装前向。
        raise NotImplementedError("请实现 MiniGPT.forward")


def build_two_styles(
    vocab_size: int,
    C: int,
    num_heads: int,
    ffn_hidden: int,
    n_layer: int,
    max_positions: int,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> tuple["MiniGPT", "MiniGPT"]:
    """返回 (pre_model, post_model) 两个模型，块内参数逐一相同。

    两个模型只在 norm_style 上不同，用于对照实验：
    除 final_norm 外，其余同名参数的数值必须逐元素相等。

    Returns:
        (pre_model, post_model)；两者是不同对象，不共享参数存储。
        pre_model.final_norm 不为 None，post_model.final_norm 为 None。

    约定：
        用同一个 seed 分别构造，使随机初始化一致；
        构造后不得再改动除对齐之外的任何数值。
        不要求两个模型的 final_norm 存在与否之外的结构差异。

    提示：
        torch.manual_seed(seed) 设定全局随机种子；在两次构造前分别调用，
        可让同样的构造顺序得到同样的初始值。
        pre_model 多创建了一个 LayerNorm，若它在其他参数之后创建，
        就不会影响先前已经取用的随机数。请自行确认对齐是否成立。
    """
    # TODO 3：构造两个只在 norm_style 上不同的模型。
    raise NotImplementedError("请实现 build_two_styles")
