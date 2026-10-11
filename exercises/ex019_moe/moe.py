"""MoE 完整教学示例：在一个文件中阅读核心、观察中间值、训练与生成。

阅读顺序：
  1. ExpertFFN / SparseMoE.forward：路由、分发、合并和负载统计。
  2. make_inspection_case / inspect_routing：两个 token 的确定性中间值。
  3. MoETransformerBlock / TinyMoELM：接回完整因果语言模型。
  4. train_demo / main：训练、生成和命令行入口。

项目根目录运行：
  .venv/bin/python -B -m exercises.ex019_moe.moe inspect
  .venv/bin/python -B -m exercises.ex019_moe.moe train
  .venv/bin/python -B -m exercises.ex019_moe.moe       # 两部分都跑
也可直接运行：.venv/bin/python -B exercises/ex019_moe/moe.py inspect

只依赖 PyTorch，在 CPU 上运行；不下载数据、不写检查点。
所有输入位置有效，没有 PAD；2 <= k <= E；选中项内重新归一化；
不设容量上限、不丢弃分配。同分选择沿用 torch.topk，不承诺跨设备一致。
使用学习式绝对位置、Pre-LN、普通 MHA、ReLU 专家；无 Dropout、KV Cache
或分布式通信，生成时重算整个前缀。参数注册使用 nn.Module/Parameter，
核心数学直接用 Tensor 运算，CPU float32/float64 用于教学验证。
"""

import argparse
from dataclasses import dataclass
import math

import torch
from torch import nn


# 1. MoE 核心：专家、路由、稀疏执行和辅助损失

def positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是正整数")


def parameter_matrix(rows, columns):
    """随机初始化后登记为可训练参数；输入宽度控制初始尺度。"""
    return nn.Parameter(torch.randn(rows, columns) / math.sqrt(rows))


def stable_softmax(scores):
    """沿最后一轴归一化；每行必须至少有一个有限分数，无 NaN/+Inf。"""
    shifted = scores - scores.amax(dim=-1, keepdim=True)
    exp_scores = shifted.exp()
    return exp_scores / exp_scores.sum(dim=-1, keepdim=True)


class ExpertFFN(nn.Module):
    """一位专家：C → F → C；沿用 ex008 的 ReLU，两次仿射均展开。"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        positive_int("d_model", d_model)
        positive_int("d_ff", d_ff)
        self.W1 = parameter_matrix(d_model, d_ff)
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.W2 = parameter_matrix(d_ff, d_model)
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # x: (M,C)，M 是本次送给这位专家的 token 数。
        hidden = x @ self.W1 + self.b1                    # (M,F)
        activated = torch.where(hidden > 0, hidden, 0.0)  # 手写 ReLU
        return activated @ self.W2 + self.b2              # (M,C)


@dataclass
class RoutingStats:
    """dataclass 自动生成字段初始化；通过 stats.weights 等名字读取。

    所有 token 采用 n=b*T+t 的展平行号。probabilities、weights、
    mean_probabilities、balance_loss 保留梯度；只在打印时 detach。
    不要跨训练步积存这些对象，以免额外保留计算图。
    """

    logits: torch.Tensor              # (N,E)：每个 token 对所有专家的分数
    probabilities: torch.Tensor       # (N,E)：全专家概率，供负载损失使用
    expert_ids: torch.Tensor          # (N,k)：每个 token 选中的专家编号
    weights: torch.Tensor             # (N,k)：选中项内重新归一化的合并权重
    counts: torch.Tensor              # (E,)：各专家实际收到的分配次数
    fractions: torch.Tensor           # (E,)：f，counts / (N*k)
    mean_probabilities: torch.Tensor  # (E,)：P，probabilities.mean(0)
    balance_loss: torch.Tensor        # 标量：E * sum(f * P)，尚未乘 alpha


class SparseMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2):
        super().__init__()
        for name, value in (("d_model", d_model), ("d_ff", d_ff),
                            ("num_experts", num_experts), ("top_k", top_k)):
            positive_int(name, value)
        if not 2 <= top_k <= num_experts:
            raise ValueError("本实现要求 2 <= top_k <= num_experts；Top-1 需另选门控约定")
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k

        # 笔记中的 W_g；C 行接输入特征，E 列各给一个专家打分。
        self.W_g = parameter_matrix(d_model, num_experts)
        # ModuleList 登记所有专家：优化器、state_dict 和 .to() 都能找到它们。
        # 每次构造一个新模块，专家之间不共享参数。
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff) for _ in range(num_experts)
        ])

    def forward(self, x):
        """返回 (output, stats)，output 与输入同 shape (B,T,C)。"""
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x 必须为 (B,T,d_model)")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("不接受空 batch 或空序列")
        if not x.is_floating_point():
            raise ValueError("x 是浮点隐藏向量，不是 token ID")

        # ① 展平位置。每一行仍是一个 token；reshape 支持不连续输入。
        B, T, C = x.shape
        flat = x.reshape(B * T, C)                          # (N,C)
        N = flat.shape[0]

        # ② 同一个 flat 分别送入路由器和专家。这里还没有执行任何专家。
        logits = flat @ self.W_g                           # (N,E)
        probabilities = stable_softmax(logits)             # 每行在 E 个专家间归一化

        # ③ topk 返回选中分数和编号，weights 的第 j 项对应 expert_ids 的第 j 项。
        # 专家编号不可导，但 selected_scores → weights 的计算仍保留梯度。
        selected_scores, expert_ids = torch.topk(logits, self.top_k, dim=-1)
        weights = stable_softmax(selected_scores)          # (N,k)，每行和为 1

        # ④ 按专家收集输入，真正只执行选中的分配，再加回原 token 行。
        output = torch.zeros_like(flat)                    # (N,C)
        counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        for e, expert in enumerate(self.experts):
            # where 在 (N,k) 的比较表中找 True，返回“token 行号、选择槽位”。
            token_rows, slots = torch.where(expert_ids == e)
            counts[e] = token_rows.numel()
            if token_rows.numel() == 0:
                continue                                  # 没人选，真的不执行这个 FFN

            selected_x = flat[token_rows]                 # (M,C)：原向量，不是路由概率
            expert_output = expert(selected_x)            # (M,C)
            selected_weights = weights[token_rows, slots].unsqueeze(-1)  # (M,1)
            contributions = selected_weights * expert_output            # (M,C)
            # index_add 沿第 0 轴按原行号累加，不能用覆盖赋值丢掉另一位专家。
            output = output.index_add(0, token_rows, contributions)

        # ⑤ f 统计硬分配次数；P 仍然是对全部 E 个专家的平均概率。
        # 当前无 PAD/丢弃，因此总分配数严格是 N*k；counts 不需要梯度。
        fractions = counts.to(probabilities.dtype) / (N * self.top_k)
        mean_probabilities = probabilities.mean(dim=0)
        balance_loss = self.num_experts * (fractions * mean_probabilities).sum()

        # ⑥ 恢复位置轴，供 Transformer 块做残差相加。
        stats = RoutingStats(logits, probabilities, expert_ids, weights, counts,
                             fractions, mean_probabilities, balance_loss)
        return output.reshape(B, T, C), stats


# 2. 逐项观察两个 token：把公式与实际中间值对应起来

def make_inspection_case():
    """把聊天中的两行概率实现为真实 x @ W_g，便于逐项手算。

    x 的两行是单位矩阵，因此 x @ W_g 直接取 W_g 的两行。
    概率均为正且各行和为 1，softmax(log(probabilities)) 恢复原概率。
    这是人为设置的参数，不是训练得到的路由规律。
    """
    moe = SparseMoE(d_model=2, d_ff=2, num_experts=4, top_k=2).double()
    x = torch.eye(2, dtype=torch.float64).reshape(1, 2, 2)
    probabilities = torch.tensor([
        [0.60, 0.30, 0.05, 0.05],
        [0.40, 0.20, 0.25, 0.15],
    ], dtype=torch.float64)
    # no_grad 只用于人为设置参数；正式前向仍会建图。
    with torch.no_grad():
        moe.W_g.copy_(probabilities.log())
        for expert, factor in zip(moe.experts, (1.0, 2.0, -1.0, 4.0)):
            expert.W1.copy_(torch.eye(2, dtype=torch.float64))
            expert.W2.copy_(factor * torch.eye(2, dtype=torch.float64))
            expert.b1.zero_()
            expert.b2.zero_()
    # 对本例非负输入，四个真实 FFN 分别等于 x、2x、-x、4x。
    return moe, x


def show(name, value):
    print(f"\n{name}")
    print(value.detach() if isinstance(value, torch.Tensor) else value)


def inspect_routing():
    print("=== 逐项检查：B=1 T=2 C=2 F=2 E=4 k=2 ===")
    moe, x = make_inspection_case()
    # hook 是模块执行后的观察回调，只记录本次真正调用了哪些专家。
    calls = {}
    handles = []
    for e, expert in enumerate(moe.experts):
        def capture(module, inputs, output, e=e):
            calls[e] = (inputs[0].detach().clone(), output.detach().clone())
        handles.append(expert.register_forward_hook(capture))
    try:
        output, stats = moe(x)
    finally:
        for handle in handles:
            handle.remove()

    show("① 输入 x (B,T,C)", x)
    show("② 路由参数 W_g (C,E)", moe.W_g)
    show("③ router logits = x_flat @ W_g (N,E)", stats.logits)
    show("④ 全专家概率 probabilities (N,E)：用于 P", stats.probabilities)
    show("⑤ Top-2 专家编号 expert_ids (N,k)", stats.expert_ids)
    show("⑥ 选中项内重新归一化 weights (N,k)：用于合并", stats.weights)
    for e in range(moe.num_experts):
        if e not in calls:
            print(f"\n专家 {e}：没有分配，forward 没有被调用。")
            continue
        token_rows, slots = torch.where(stats.expert_ids == e)
        inputs, outputs = calls[e]
        weights = stats.weights[token_rows, slots].detach().unsqueeze(-1)
        show(f"专家 {e} 收到原 token 行 {token_rows.tolist()}；输入", inputs)
        show("该专家输出", outputs)
        show("乘路由权重后，加回这些原 token 行的贡献", weights * outputs)
    show("⑦ 合并后的 output (B,T,C)", output)
    expected = torch.tensor([[[4 / 3, 0.0], [0.0, 3 / 13]]], dtype=x.dtype)
    torch.testing.assert_close(output, expected)
    show("⑧ 实际次数 counts", stats.counts)
    show("f = counts / (N*k)", stats.fractions)
    show("P = probabilities.mean(0)", stats.mean_probabilities)
    show("L_balance = E * sum(f*P)，应为 1.4", stats.balance_loss)

    # 用固定目标构造一个小回归任务，只观察两条梯度路径。
    target = x.clone()
    task_loss = (output - target).square().mean()
    task_gradient = torch.autograd.grad(task_loss, moe.W_g, retain_graph=True)[0]
    aux_gradient = torch.autograd.grad(stats.balance_loss, moe.W_g, retain_graph=True)[0]
    show("主任务单独对 W_g 的梯度（专家 3 对应列未选中，因此为 0）", task_gradient)
    show("辅助项单独对 W_g 的梯度（专家 3 对应列也能得到信号）", aux_gradient)
    (task_loss + 0.01 * stats.balance_loss).backward()
    for e, expert in enumerate(moe.experts):
        grads = [p.grad for p in expert.parameters()]
        print(f"专家 {e} 是否收到参数梯度：{any(g is not None for g in grads)}")
    print("这一步没有优化器更新；人为设定的参数保持不变。")


# 3. 完整因果语言模型：Attention 后面的 FFN 替换为 MoE

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        centered = x - x.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        return centered / torch.sqrt(variance + self.eps) * self.scale + self.bias


class CausalSelfAttention(nn.Module):
    """普通多头 Attention：所有头都执行；softmax 沿 key 位置轴。"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        positive_int("d_model", d_model)
        positive_int("num_heads", num_heads)
        if d_model % num_heads:
            raise ValueError("d_model 必须能被 num_heads 整除")
        self.num_heads = num_heads
        self.Wq = parameter_matrix(d_model, d_model)
        self.Wk = parameter_matrix(d_model, d_model)
        self.Wv = parameter_matrix(d_model, d_model)
        self.Wo = parameter_matrix(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.num_heads, C // self.num_heads
        q = (x @ self.Wq).reshape(B, T, H, D).transpose(1, 2)  # (B,H,T,D)
        k = (x @ self.Wk).reshape(B, T, H, D).transpose(1, 2)
        v = (x @ self.Wv).reshape(B, T, H, D).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)    # (B,H,T,T)
        positions = torch.arange(T, device=x.device)
        # 行是 query 位置，列是 key 位置；每行至少能读取自身。
        allowed = positions[:, None] >= positions[None, :]
        weights = stable_softmax(scores.masked_fill(~allowed, -torch.inf))
        mixed = weights @ v                                # (B,H,T,D)
        merged = mixed.transpose(1, 2).reshape(B, T, C)
        return merged @ self.Wo                            # (B,T,C)


class MoETransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts, top_k):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        self.moe = SparseMoE(d_model, d_ff, num_experts, top_k)

    def forward(self, x):
        # MoE 替换的是原 FFN 位置；两条残差和 Attention 的位置都保留。
        h = x + self.attention(self.norm1(x))
        z = self.norm2(h)                                  # (B,T,C)，含有上下文
        delta, stats = self.moe(z)
        return h + delta, stats


class TinyMoELM(nn.Module):
    """ID → embedding/位置 → 多个 MoE 块 → final norm → 词表 logits。"""

    def __init__(self, vocab_size, max_length=32, d_model=24, num_heads=3,
                 d_ff=48, num_layers=2, num_experts=4, top_k=2):
        super().__init__()
        for name, value in (("vocab_size", vocab_size), ("max_length", max_length),
                            ("num_layers", num_layers), ("d_model", d_model)):
            positive_int(name, value)
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_embedding = nn.Parameter(torch.randn(vocab_size, d_model) * 0.1)
        self.position_embedding = nn.Parameter(torch.randn(max_length, d_model) * 0.1)
        self.blocks = nn.ModuleList([
            MoETransformerBlock(d_model, num_heads, d_ff, num_experts, top_k)
            for _ in range(num_layers)
        ])
        self.final_norm = LayerNorm(d_model)
        self.W_vocab = parameter_matrix(d_model, vocab_size)

    def forward(self, input_ids):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids 必须为 (B,T) 的 torch.long")
        B, T = input_ids.shape
        if B == 0 or not 1 <= T <= self.max_length:
            raise ValueError("batch 不能为空，序列长度必须在 1..max_length 内")
        if bool(((input_ids < 0) | (input_ids >= self.vocab_size)).any()):
            raise ValueError("token ID 超出词表范围")
        x = self.token_embedding[input_ids] + self.position_embedding[:T]  # (B,T,C)
        routing = []
        for block in self.blocks:
            x, stats = block(x)
            routing.append(stats)
        logits = self.final_norm(x) @ self.W_vocab          # (B,T,V)，V 是词表大小
        # 明确采用“各层辅助项平均”；alpha 不会随层数隐式放大。
        balance_loss = torch.stack([s.balance_loss for s in routing]).mean()
        return logits, balance_loss, routing


def next_token_loss(logits, targets):
    """所有位置有效的稳定交叉熵；输入是原始词表 logits，不提前 softmax。"""
    if targets.shape != logits.shape[:-1] or targets.dtype != torch.long:
        raise ValueError("targets 必须为与 logits 位置轴匹配的 long 张量")
    if bool(((targets < 0) | (targets >= logits.shape[-1])).any()):
        raise ValueError("target ID 超出词表范围")
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    log_normalizer = shifted.exp().sum(dim=-1).log()         # (B,T)
    target_scores = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_scores).mean()


def greedy_generate(model, prefix, max_new_tokens):
    """贪心生成，不传真实标签；保留调用前模式，且不构建梯度图。"""
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens 必须为非负整数")
    if prefix.ndim != 2 or prefix.shape[0] == 0 or prefix.shape[1] == 0:
        raise ValueError("prefix 必须为非空 (B,T) 张量")
    if prefix.dtype != torch.long or bool(((prefix < 0) | (prefix >= model.vocab_size)).any()):
        raise ValueError("prefix 必须包含有效的 long token ID")
    if prefix.shape[1] + max_new_tokens > model.max_length:
        raise ValueError("生成后的长度超过 max_length")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            result = prefix.clone()
            for _ in range(max_new_tokens):
                logits, _, _ = model(result)
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                result = torch.cat((result, next_id), dim=1)
            return result
    finally:
        model.train(was_training)


# 4. 训练、生成和命令行入口

VOCAB = ("A", "B", "x", "u", "v")


def make_training_batch():
    """同一个 x，在 A/B 两类上下文中分别接 u/v，必须利用上下文。

    只拟合这两条固定序列；这是训练接线演示，不用于宣称泛化或专家专长。
    """
    token_to_id = {token: i for i, token in enumerate(VOCAB)}
    sequences = [
        ["A", "x", "u", "x", "u", "x", "u", "x", "u"],
        ["B", "x", "v", "x", "v", "x", "v", "x", "v"],
    ]
    ids = torch.tensor([[token_to_id[token] for token in row] for row in sequences])
    return ids[:, :-1], ids[:, 1:], ids


def evaluate(model, inputs, targets):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits, aux_loss, routing = model(inputs)
            loss = next_token_loss(logits, targets)
            accuracy = (logits.argmax(-1) == targets).float().mean()
            return loss.item(), accuracy.item(), aux_loss.item(), routing
    finally:
        model.train(was_training)


def train_demo(steps=160, seed=42, alpha=0.01):
    torch.manual_seed(seed)
    inputs, targets, complete = make_training_batch()
    model = TinyMoELM(vocab_size=len(VOCAB))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print(f"\n=== 完整模型：CPU float32 seed={seed} steps={steps} alpha={alpha} ===")
    print("config: B=2 T=8 C=24 H=3 F=48 layers=2 E=4 k=2；所有位置有效")
    print("task: A x u x u ... / B x v x v ...；仅拟合两条固定序列")
    before = evaluate(model, inputs, targets)
    print(f"before: task_loss={before[0]:.6f} accuracy={before[1]:.1%} balance={before[2]:.4f}")

    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits, balance_loss, routing = model(inputs)
        task_loss = next_token_loss(logits, targets)
        # 每层辅助项已在模型中取平均，这里只乘一次 alpha。
        total_loss = task_loss + alpha * balance_loss
        if not bool(torch.isfinite(total_loss)):
            raise RuntimeError("损失出现 NaN/Inf")
        total_loss.backward()
        if step == 1:
            for layer, block in enumerate(model.blocks):
                norm = block.moe.W_g.grad.norm().item()
                print(f"第一次 backward：layer={layer} 路由器梯度范数={norm:.6f}")
        optimizer.step()
        if step == 1 or step % 40 == 0 or step == steps:
            # 这些损失/路由来自本次更新前的前向，不与更新后参数混称。
            print(f"step={step:3d} 更新前 task={task_loss.item():.6f} "
                  f"balance={balance_loss.item():.4f} total={total_loss.item():.6f}")

    task, accuracy, aux, routing = evaluate(model, inputs, targets)
    print(f"after : task_loss={task:.6f} accuracy={accuracy:.1%} balance={aux:.4f}")
    for layer, stats in enumerate(routing):
        print(f"layer={layer} counts={stats.counts.tolist()} "
              f"sum={int(stats.counts.sum())}（应为 B*T*k=32）")
    # prefix 只包含 A x / B x，后续 token 全部由模型自己产生。
    generated = greedy_generate(model, complete[:, :2], max_new_tokens=7)
    for predicted, expected in zip(generated.tolist(), complete.tolist()):
        print("生成：", " ".join(VOCAB[i] for i in predicted))
        print("目标：", " ".join(VOCAB[i] for i in expected))
    print("生成是否逐项匹配：", bool(torch.equal(generated, complete)))

    moe = model.blocks[0].moe
    expert_parameters = sum(p.numel() for p in moe.experts[0].parameters())
    router_parameters = moe.W_g.numel()
    print(f"单层 MoE 参数：总量={moe.num_experts * expert_parameters + router_parameters}；"
          f"每 token 使用={moe.top_k * expert_parameters + router_parameters}")
    print("该账本仅含 MoE，不含 Attention/embedding；活跃参数数目不等于实际延迟。")
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("inspect", "train", "all"), default="all")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.steps < 0 or not 0 <= args.alpha < float("inf"):
        parser.error("steps 必须非负，alpha 必须是有限非负数")
    torch.set_num_threads(1)
    torch.set_printoptions(precision=4, linewidth=110, sci_mode=False)
    if args.mode in ("inspect", "all"):
        inspect_routing()
    if args.mode in ("train", "all"):
        train_demo(args.steps, args.seed, args.alpha)


if __name__ == "__main__":
    main()
