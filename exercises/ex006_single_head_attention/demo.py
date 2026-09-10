"""教师运行入口；只调用学习者接口，不含 Attention 答案或训练循环。"""

import torch

from exercises.ex006_single_head_attention.attention import single_head_self_attention


def main():
    torch.set_num_threads(1)
    # 手工给定输入特征，分别代表 [BOS,A,X,PAD] 与 [BOS,B,X,PAD]。
    # 不是 token ID；不在本例重复查表或加入新的位置编码。
    x = torch.tensor([
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [7., 8., 9.]],
        [[1., 0., 0.], [0., -1., 0.], [0., 0., 1.], [7., 8., 9.]],
    ], dtype=torch.float64)
    input_valid = torch.tensor([[True, True, True, False], [True, True, True, False]])
    # eye(3) 创建 3x3 单位矩阵；这里故意让匹配特征与输入相同。
    wq = torch.eye(3, dtype=x.dtype)
    wk = wq.clone()
    wv = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=x.dtype)

    with torch.no_grad():
        try:
            output, weights = single_head_self_attention(x, wq, wk, wv, input_valid)
        except NotImplementedError as error:
            raise SystemExit("请先完成 attention.py 的三个新接口：" + str(error)) from error

        print("output shape:", tuple(output.shape))
        print("weights shape:", tuple(weights.shape))
        print("last valid query weights:", weights[:, 2].tolist())
        print("last valid query outputs:", output[:, 2].tolist())

        changed_pad = x.clone()
        changed_pad[~input_valid] += 1000
        pad_output, _ = single_head_self_attention(changed_pad, wq, wk, wv, input_valid)
        print("valid-output difference after PAD change:",
              (pad_output[input_valid] - output[input_valid]).abs().max().item())

        changed_future = x.clone()
        changed_future[:, 2] += 5
        future_output, _ = single_head_self_attention(changed_future, wq, wk, wv, input_valid)
        print("earlier-output difference after position 2 change:",
              (future_output[:, :2] - output[:, :2]).abs().max().item())
        print("This checks content reading and visibility, not trained language ability.")


if __name__ == "__main__":
    main()
