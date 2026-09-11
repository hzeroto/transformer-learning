"""只演示调用；四个核心 forward 都由学习者在 block.py 中完成。"""

import torch

from exercises.ex008_transformer_block.block import TransformerBlock


def main():
    torch.manual_seed(108)
    block = TransformerBlock(C=4, num_heads=2, ffn_hidden=8, p=0.2)
    X = torch.randn(2, 4, 4, dtype=torch.float64, requires_grad=True)
    input_valid = torch.tensor([[True, True, True, True], [True, True, False, False]])

    block.train()
    Y = block(X, input_valid)
    # 仅为检查计算图而选的标量，不是语言模型的交叉熵目标。
    loss = (Y * Y).mean()
    loss.backward()
    parameters = list(block.parameters())
    print("output shape:", tuple(Y.shape))
    print("parameter tensors:", len(parameters))
    print("parameter elements:", sum(p.numel() for p in parameters))
    print("parameter tensors with gradients:", sum(p.grad is not None for p in parameters))
    print("input gradient is finite:", bool(torch.isfinite(X.grad).all()))

    block.eval()
    with torch.no_grad():
        first = block(X, input_valid)
        second = block(X, input_valid)
    print("eval outputs agree:", torch.equal(first, second))
    print("eval + no_grad output requires_grad:", first.requires_grad)
    print("This checks one Transformer Block, not a trained language model.")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as error:
        raise SystemExit(f"尚未实现：{error}") from None
