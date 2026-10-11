"""ViT 教师补全版：图块投影与完整前向连接，附逐步 shape 和索引说明。

参数登记和 Pre-LN 双向 Encoder 由教师提供，复用此前的数学实现。
请先读 README.md；测试中的参照不是可导入的模型实现。
"""

import torch
from torch import nn

from exercises.ex007_multi_head_attention.attention import multi_head_attention
from exercises.ex008_transformer_block.block import LayerNorm
from exercises.ex016_mini_bert.model import EncoderBlock, gelu


def patch_embed(images, weight, bias, patch_size):
    """步骤 1：图像分块、展平、共享线性投影，返回 patch tokens。

    images: (B,C,H,W)，CPU float32/float64，可非连续。
    patch_size: 正整数 P；H、W 均能被 P 整除，无 padding、无重叠。
    weight: (C*P*P,D)，所有图块共享，采用向量在左、矩阵在右的约定。
    bias: (D,)。输入与参数保证 dtype/device 一致，所有轴长度为正。

    返回 (B,N,D)，N=(H//P)*(W//P)。图块按先行后列的顺序排列；
    单个图块的展平顺序固定为：通道 c → 块内行 r → 块内列 s，
    即先列完通道 0 的 P 行，再列通道 1，依此类推。
    这是接口约定，必须与 weight 的行顺序对应。

    保留 images/weight/bias 的梯度；不修改输入、参数或已有 .grad。
    可用 unfold/reshape/permute/索引/矩阵乘法等基础 Tensor API。
    本函数不调用 Conv2d、F.conv2d、F.unfold、nn.Linear 或测试参照；
    卷积只在教师测试中作为独立参照。不在此加 CLS/位置或做归一化。
    所有输入已满足上述契约，本题不要求额外的通用参数校验。
    """
    # B 是图片数，C 是输入通道数，H/W 是原图的高/宽；P 是图块边长。
    # 后面用 patch_rows/patch_cols 表示图块网格的行数/列数，
    # 与单个图块内部的 P 行、P 列区分开。
    batch, channels, height, width = images.shape
    patch_rows = height // patch_size
    patch_cols = width // patch_size

    # 1. 沿高度切窗口：unfold(轴编号, 窗口长度, 移动步长)。
    # 窗口和步长均为 P，恰好每 P 行一组，不重叠、不跳过像素。
    # unfold 将原来的高度轴变为「第几组行」，将「组内第几行」追加到末尾。
    # (B,C,H,W) -> (B,C,patch_rows,W,P)
    patches = images.unfold(2, patch_size, patch_size)

    # 2. 宽度 W 仍在第 3 号轴；沿宽度也每 P 列分成一组。
    # 新增的最后一个 P 是块内列；倒数第二个 P 是上一步产生的块内行。
    # (B,C,patch_rows,W,P) -> (B,C,patch_rows,patch_cols,P,P)
    patches = patches.unfold(3, patch_size, patch_size)

    # 3. 把「图块位置」放在一起，把「单块的全部像素」放在一起。
    # 原轴顺序：图片、通道、图块行、图块列、块内行、块内列。
    # 调整之后：图片、图块行、图块列、通道、块内行、块内列。
    # (B,C,patch_rows,patch_cols,P,P) -> (B,patch_rows,patch_cols,C,P,P)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    # 此时精确的像素对应关系为（用 P 表示 patch_size）：
    # patches[b,i,j,c,r,s] == images[b,c,i*P+r,j*P+s]。

    # 4. 合并相邻轴：两个图块位置轴合成 N，最后三个轴合成 C*P*P。
    # 图块编号 n = i*patch_cols+j，保证先遍历一行图块，再换到下一行。
    # 块内编号 k = (c*P+r)*P+s，保证先列完一个通道的像素，再换通道；
    # flat_patches[b,n,k] 必须对应 weight[k,:] 这一行投影权重。
    # 不能直接把原图 reshape 成这个 shape，否则可能把不同图块的数混在一起。
    # permute 后的张量可能非连续；reshape 必要时会复制数据来满足新布局，
    # 同时保留自动求导关系，因此非连续 images 也能正确处理。
    flat_patches = patches.reshape(
        batch, patch_rows * patch_cols, channels * patch_size**2,
    )  # (B,N,C*P*P)

    # 5. 所有图片、所有图块共用同一个线性映射：
    # (B,N,C*P*P) @ (C*P*P,D) + (D,) -> (B,N,D)。
    # @ 计算最后两个轴的矩阵乘法；bias 在 B、N 两个轴上广播。
    # 这里才把原始像素组合成 D 个特征；前面的切块和展平只改变组织方式。
    # 整条计算链不修改输入，也不 detach，梯度可回到 images、weight、bias。
    return flat_patches @ weight + bias


class VisionEncoderBlock(EncoderBlock):
    """教师适配器：沿用 ex016 的参数，使用 Pre-LN 和双向读取。

    旧 ex008/mini-GPT 的 self-attention wrapper 内含因果 mask，不能直接
    用于本题；这里复用接收显式 allowed 的基础 MHA。无需修改本类。
    """

    def forward(self, x):
        batch, length, _ = x.shape
        normalized = self.norm1(x)
        allowed = torch.ones(batch, length, length, dtype=torch.bool, device=x.device)
        attention, _ = multi_head_attention(
            normalized @ self.Wq + self.bq,
            normalized @ self.Wk + self.bk,
            normalized @ self.Wv + self.bv,
            self.Wo, self.num_heads, allowed,
        )
        residual = x + attention + self.bo
        hidden = gelu(self.norm2(residual) @ self.W1 + self.b1)
        return residual + hidden @ self.W2 + self.b2


class MiniViT(nn.Module):
    """固定分辨率的玩具 ViT：图像入口、Pre-LN Encoder 与 CLS 分类头。"""

    def __init__(
        self, image_size=(4, 6), in_channels=3, patch_size=2, dim=8,
        num_heads=2, depth=2, num_classes=3, *, dtype=torch.float64,
    ):
        super().__init__()
        height, width = image_size
        for value in (height, width, in_channels, patch_size, dim, num_heads, depth, num_classes):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("尺寸、层数、头数、类别数必须为正整数")
        if height % patch_size or width % patch_size or dim % num_heads:
            raise ValueError("图像宽高须能被 P 整除，D 须能被头数整除")
        self.image_size = (height, width)
        self.in_channels, self.patch_size, self.dim = in_channels, patch_size, dim
        self.num_patches = (height // patch_size) * (width // patch_size)
        self.num_classes = num_classes
        self.patch_weight = nn.Parameter(torch.randn(in_channels * patch_size**2, dim, dtype=dtype) * 0.1)
        self.patch_bias = nn.Parameter(torch.zeros(dim, dtype=dtype))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim, dtype=dtype) * 0.05)
        self.position = nn.Parameter(torch.randn(1, self.num_patches + 1, dim, dtype=dtype) * 0.05)
        self.blocks = nn.ModuleList([
            VisionEncoderBlock(dim, num_heads, 2 * dim, dtype=dtype)
            for _ in range(depth)
        ])
        self.final_norm = LayerNorm(dim, dtype=dtype)
        self.class_weight = nn.Parameter(torch.randn(dim, num_classes, dtype=dtype) * 0.1)
        self.class_bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))

    def forward(self, images):
        """步骤 2：连接图像入口、CLS/位置、Encoder、分类头，返回 logits。

        images: (B,in_channels,H,W)，H/W 与 self.image_size 完全一致，B>0；
        CPU，float32/float64，与模型 dtype 一致，允许非连续。
        所有输入符合契约，不要求增加通用校验。

        使用本模块的 patch_embed 产生图块表示。self.cls_token 是全 batch
        共享的可学习参数，放在每条序列第 0 位；self.position 包括 CLS 的
        位置向量，在进入第一个块前相加一次。
        self.blocks 中的所有层按顺序运行，接口是 block(tokens)，无 PAD。
        最后对第 0 位 CLS 的输出使用 self.final_norm，再经
        self.class_weight/self.class_bias 得到每张图的原始类别分数。

        返回 (B,num_classes)，不返回整段特征、不做 softmax/argmax。
        不改参数/输入/已有 .grad、不创建参数、不切换 train/eval，
        不 detach/no_grad/backward。所有输入与参与计算的参数保留梯度。
        只补连接代码；不另写或调用高级 Encoder/Attention 模块。
        """
        # 1. 将每张图转换成 N 个图块向量，每个向量包含 D=self.dim 个数。
        # 这里的参数已经在 __init__ 中登记，直接复用，不能在 forward 中重建。
        patch_tokens = patch_embed(
            images, self.patch_weight, self.patch_bias, self.patch_size,
        )  # (B,N,D)

        # 2. 为每张图准备序列开头的 CLS，作为后续整张图的分类表示。
        # cls_token 是唯一一份 (1,1,D) 参数，expand 只扩展 batch 轴；
        # 两个 -1 表示对应轴的长度保持不变。它不会创建 B 份独立参数，
        # 反向传播会把各张图经由 CLS 产生的梯度汇总到同一个 cls_token。
        cls_tokens = self.cls_token.expand(images.shape[0], -1, -1)  # (B,1,D)
        # dim=1 是序列轴：CLS 放在第 0 位，原先的图块依次放在第 1..N 位。
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)  # (B,N+1,D)

        # 3. 为内容表示加上位置表示，区分图块位于图片中的哪个位置。
        # position[:,0,:] 对应 CLS，position[:,1:,:] 对应按行排列的图块。
        # (1,N+1,D) 在 batch 轴广播成 (B,N+1,D)，逐元素相加；
        # 位置不增加序列长度或特征维度，并且只在进入 Encoder 前加一次。
        tokens = tokens + self.position

        # 4. 依次经过全部 Encoder 层；每层的输入、输出都是 (B,N+1,D)。
        # 已提供的 VisionEncoderBlock 使用双向 Attention，没有因果 mask；
        # 因而第 0 位的 CLS 可以读取全部图块，逐层形成依赖当前图片的表示。
        for block in self.blocks:
            tokens = block(tokens)

        # 5. 取每张图最后一层输出的 CLS，再沿其 D 个特征做最终 LayerNorm。
        # tokens[:,0,:] 固定序列位置为 0，保留 batch 和特征轴，shape 为 (B,D)。
        # 这里取的是经过 Encoder 更新后的 CLS，而不是最初的共享 cls_token 参数。
        cls_features = self.final_norm(tokens[:, 0, :])  # (B,D)

        # 6. 分类头把每张图的 D 个特征映射到 K=self.num_classes 个类别分数。
        # (B,D) @ (D,K) + (K,) -> (B,K)，偏置沿 batch 轴广播。
        # 返回原始 logits，供外部损失函数或预测逻辑使用；此处不做
        # softmax/argmax，也不执行 backward，完整保留图像和参数的梯度路径。
        return cls_features @ self.class_weight + self.class_bias
