"""教师对照：双向可见范围与 MLM 监督，不提供 Mini-BERT 的核心实现。

运行：.venv/bin/python -B -m examples.encoder_bert_probe
CPU float64；无训练、无随机性。断言只验证此处的固定案例。
"""

import torch

from exercises.ex005_training_loop.loss import masked_cross_entropy
from exercises.ex006_single_head_attention.attention import make_causal_allowed
from exercises.ex007_multi_head_attention.attention import multi_head_attention


def visibility_probe():
    """Q/K 为零让权重可精确解释；只对照权限，不加入 LN/FFN。"""
    x = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [99.0, 99.0]]],
        dtype=torch.float64,
    )
    input_valid = torch.tensor([[True, True, True, False]])
    batch, length, width = x.shape
    causal = make_causal_allowed(input_valid)
    # unsqueeze 增加 query 轴，再把同一份 key 权限扩展给每个 query。
    # expand 是广播视图，不复制 B*T*T 份底层数据；不要原地修改此视图。
    bidirectional = input_valid.unsqueeze(1).expand(batch, length, length)
    wo = torch.eye(width, dtype=x.dtype)

    def read(values, allowed):
        # zeros_like 构造与 values 同 shape、dtype、device 的全零张量。
        q = torch.zeros_like(values)
        k = torch.zeros_like(values)
        return multi_head_attention(q, k, values, wo, 1, allowed)

    causal_output, _ = read(x, causal)
    encoder_output, weights = read(x, bidirectional)
    right_changed = x.clone()
    right_changed[0, 2, 0] = 5.0
    causal_changed, _ = read(right_changed, causal)
    encoder_changed, _ = read(right_changed, bidirectional)
    pad_changed = x.clone()
    pad_changed[0, 3, :] = torch.tensor([-1000.0, 1000.0], dtype=x.dtype)
    encoder_pad, _ = read(pad_changed, bidirectional)
    causal_pad, _ = read(pad_changed, causal)

    torch.testing.assert_close(causal_output[0, 0], causal_changed[0, 0])
    torch.testing.assert_close(
        encoder_output[0, 0], torch.tensor([1.0, 1.0 / 3.0], dtype=x.dtype)
    )
    torch.testing.assert_close(
        encoder_changed[0, 0], torch.tensor([2.0, 1.0 / 3.0], dtype=x.dtype)
    )
    torch.testing.assert_close(encoder_output, encoder_pad)
    torch.testing.assert_close(causal_output, causal_pad)
    assert torch.all(weights[..., 3] == 0)
    assert weights[0, 0, 0, 2] > 0

    print("1. 改右侧有效位置；只看第 0 个 query 的 Attention 输出")
    print("   causal:", causal_output[0, 0].tolist(), "->", causal_changed[0, 0].tolist())
    print("   encoder:", encoder_output[0, 0].tolist(), "->", encoder_changed[0, 0].tolist())
    print("   PAD 扰动：两种权限下的输出均保持不变。")


def mlm_supervision_probe():
    """人工指定三种替换，不模拟原版 15% 的采样频率。"""
    # 词表：PAD=0，CLS=1，SEP=2，MASK=3，缓存=4，命中率=5，恢复=6，磁盘=7。
    clean_ids = torch.tensor([[1, 4, 5, 6, 2, 0]])
    corrupted_ids = torch.tensor([[1, 3, 7, 6, 2, 0]])
    selected = torch.tensor([[False, True, True, True, False, False]])
    input_valid = torch.tensor([[True, True, True, True, True, False]])
    assert torch.all(~selected | input_valid)
    assert corrupted_ids[0, 3] == clean_ids[0, 3] and selected[0, 3]

    # 用独立叶子 logits 观察 loss 的局部导数；这里没有 Encoder。
    # requires_grad=True 使 autograd 跟踪，backward 后叶子的 .grad 可读。
    logits = torch.zeros((1, 6, 8), dtype=torch.float64, requires_grad=True)
    loss = masked_cross_entropy(logits, clean_ids, selected)
    loss.backward()
    assert logits.grad is not None
    assert torch.all(logits.grad[~selected] == 0)
    assert torch.all(logits.grad[selected].abs().sum(dim=-1) > 0)
    # 第 3 个位置没有变成 MASK，但正确标签分数的梯度仍应为负。
    assert logits.grad[0, 3, 6] < 0

    unselected_changed = logits.detach().clone()
    unselected_changed[0, 0, 0] = 20.0
    same_loss = masked_cross_entropy(unselected_changed, clean_ids, selected)
    torch.testing.assert_close(same_loss, loss.detach())
    unchanged_selected_changed = logits.detach().clone()
    unchanged_selected_changed[0, 3, 6] = 2.0
    lower_loss = masked_cross_entropy(unchanged_selected_changed, clean_ids, selected)
    assert lower_loss < loss.detach()

    print("2. MLM：原位置标签；MASK、随机替换、保持原样均受监督")
    print("   clean_ids:    ", clean_ids.tolist())
    print("   corrupted_ids:", corrupted_ids.tolist())
    print("   selected:     ", selected.tolist())
    print(f"   loss: {loss.item():.6f}; 改未选中位置后: {same_loss.item():.6f}")
    print(f"   提高选中但保持原样位置的正确分数后: {lower_loss.item():.6f}")
    print("   logits 梯度：三个选中位置非零，其余位置为零。")
    print("   这不是说未选中 token 的 Encoder 表示或 embedding 没有梯度。")


def main():
    torch.set_num_threads(1)
    visibility_probe()
    mlm_supervision_probe()
    print("固定对照断言通过；没有训练或验收 Mini-BERT。")


if __name__ == "__main__":
    main()
