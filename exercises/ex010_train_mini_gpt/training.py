"""训练闭环：学习者实现六个函数，复用 ex005 的损失与 ex009 的模型。

统一契约见同目录 README.md。CPU、float32；核心数学不再手写，本练习的重点是
训练流程本身：批次构造、优化器、训练/验证对照、存档恢复与生成。
"""

import torch

from exercises.ex005_training_loop.loss import masked_cross_entropy
from exercises.ex009_mini_gpt.model import MiniGPT
from exercises.ex010_train_mini_gpt.dataset import (
    EOS_ID,
    N,
    OUTPUT_SLICE,
)


def make_supervised_batch(
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把整齐的序列批次切成「输入 / 目标 / 输入有效性 / 标签有效性」四份。

    Args:
        ids: (B, 9)，CPU torch.long，本练习的数据集没有 PAD，每行都是完整序列。

    Returns:
        四元组 (input_ids, targets, input_valid, target_valid)：
            input_ids:    (B, 8) long，原序列去掉最后一个位置。
            targets:      (B, 8) long，原序列去掉第一个位置（错位一格）。
            input_valid:  (B, 8) bool，本数据集无 PAD，因此全为 True。
            target_valid: (B, 8) bool，**只有输出段为 True**，见下。

    关于 target_valid：
        错位之后，input_ids[:, t] 预测 targets[:, t]。本任务只监督输出段，
        即 dataset.OUTPUT_SLICE（等于 slice(4, 8)）这四个位置：它们对应
        预测 o1、o2、o3 和 <eos>。其余位置一律为 False。
        让模型去预测 <bos> 后面是 A 还是 B 没有意义，那只能靠猜；把这些位置
        计入损失只会往梯度里注入噪声。

    约定：
        不修改输入；允许返回切片视图，不要求复制。
        input_valid 与 target_valid 含义不同，不要相互复用：
        前者表示「该位置能否作为 key 被读取」，后者表示「该位置的标签是否计入损失」。

    提示：
        错位切片的写法与 ex005 的 prepare_next_token_batch 一致。
        构造一个全 False 的 bool 张量后，对 OUTPUT_SLICE 赋 True 即可。
    """
    B, T = ids.shape
    input_ids = ids[:, :-1]
    targets = ids[:, 1:]
    input_valid = torch.ones((B, T - 1), dtype=torch.bool)
    target_valid = torch.zeros((B, T - 1), dtype=torch.bool)
    target_valid[:, OUTPUT_SLICE] = True
    return input_ids, targets, input_valid, target_valid


def evaluate(model: MiniGPT, ids: torch.Tensor) -> tuple[float, float]:
    """在给定数据上评估，返回 (平均损失, 整条输出段全对的比例)。

    Args:
        model: 已构造的 MiniGPT。
        ids: (B, 9) CPU torch.long。

    Returns:
        (loss, accuracy)，两个 Python float（不是 Tensor）。
        loss 为 masked_cross_entropy 在 target_valid 位置上的平均值。
        accuracy 的判定是「一条序列的输出段 4 个位置**全部**预测正确才算对」，
        再对 batch 求比例；不是逐位置正确率。

    约定：
        必须切到推理态并关闭求导 —— 这是两件独立的事，都要做。
        评估不得改变模型参数、不得触发反向、不得残留 train 态：
        函数返回后，模型应回到调用前的 training 状态。
        逐位置预测取 argmax；同分时的选择沿用 torch.argmax 的规则。

    提示：
        `model.eval()` / `model.train()` 切换模式，`model.training` 读当前模式。
        `with torch.no_grad():` 关闭求导。两者的区别见 Dropout 讲义第 4 节。
        判断「整条全对」：先得到逐位置是否正确的 bool，再把 target_valid
        为 False 的位置视为不影响结果，最后沿位置轴取 all。
        `.item()` 把标量 Tensor 变成 Python float。
    """
    is_training = model.training
    model.eval()
    with torch.no_grad():
        input_ids, targets, input_valid, target_valid = make_supervised_batch(ids)
        logits = model(input_ids, input_valid) # (B, T-1, C)
        loss = masked_cross_entropy(logits, targets, target_valid).item()
        predications = logits.argmax(dim=-1) # (B, T-1)
        check = (predications == targets) | (~target_valid) # (B, T-1)
        correct = check.all(dim=-1) # (B, T-1)
        acc = correct.float().mean().item()

    if is_training:
        model.train()
    return loss, acc



def train_model(
    model: MiniGPT,
    train_ids: torch.Tensor,
    *,
    steps: int = 400,
    lr: float = 3e-3,
    batch_size: int = 64,
    seed: int = 0,
    grad_clip: float | None = None,
) -> torch.optim.Optimizer:
    """用 Adam 训练模型，返回使用过的优化器对象。

    Args:
        model: 待训练的 MiniGPT，函数结束时其参数已被更新。
        train_ids: (B, 9) CPU torch.long 训练数据。
        steps: 训练步数，每步取一个随机批次。
        lr: Adam 的学习率。
        batch_size: 每步的批次大小，允许重复采样。
        seed: 采样用的随机种子；相同 seed 必须给出相同的批次序列。
        grad_clip: 为 None 时不裁剪；为正数时按该值裁剪梯度的全局范数。

    Returns:
        本次使用的 torch.optim.Adam 实例（调用方可能要保存它的 state_dict）。

    约定：
        每步都必须切到训练态。
        每步顺序为：取批次 → 前向 → 求损失 → 清梯度 → 反向 → （裁剪）→ 更新。
        梯度必须清零；累积上一步的梯度是本练习明确要避免的错误。
        用 torch.Generator().manual_seed(seed) 产生批次下标，不要用全局随机数，
        以免与模型初始化或 Dropout 的随机性相互干扰。

    提示：
        torch.optim.Adam(model.parameters(), lr=lr) 构造优化器；
        opt.zero_grad() 清梯度，opt.step() 更新参数。
        torch.randint(0, len(train_ids), (batch_size,), generator=g) 取下标。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) 裁剪梯度，
        它会就地修改 .grad 并返回裁剪前的总范数，必须在 backward 之后、
        step 之前调用。
    """
    is_training = model.training
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)

    for step in range(steps):
        batch_indices = torch.randint(0, len(train_ids), (batch_size,), generator=g) # [batch_size]
        train_data = train_ids[batch_indices] # [batch_size, 9]

        inputs, targets, input_valid, target_valid = make_supervised_batch(train_data)
        logits = model(inputs, input_valid) # [batch_size, 8, vocab_size]
        loss = masked_cross_entropy(logits, targets, target_valid)

        opt.zero_grad()
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        opt.step()

    if not is_training:
        model.eval()
    return opt

def save_checkpoint(
    path, model: MiniGPT, optimizer: torch.optim.Optimizer, config: dict
) -> None:
    """把参数、优化器状态和配置存到 path。

    Args:
        path: torch.save 接受的路径或文件对象。
        model / optimizer: 已训练的模型与其优化器。
        config: 重建模型所需的构造参数，例如
            {"vocab_size":11,"C":64,"num_heads":4,"ffn_hidden":128,
             "n_layer":2,"max_positions":16}

    存档必须是一个 dict，且恰好包含这三个键：
        "model"  -> model.state_dict()
        "opt"    -> optimizer.state_dict()
        "config" -> 传入的 config（原样保存）

    为什么三样都要存：
        state_dict 只是「参数名 -> 张量」，不含结构信息。恢复时必须先按 config
        构造出同样形状的模型，才能把参数装回去。优化器状态里是 Adam 的
        exp_avg / exp_avg_sq，丢掉它相当于让 Adam 重新估计每个参数的步长。
    """
    torch.save({
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "config": config
    }, path)


def load_checkpoint(path) -> tuple[MiniGPT, torch.optim.Optimizer, dict]:
    """从 path 恢复，返回 (model, optimizer, config)。

    Returns:
        model: 按存档 config 构造并载入参数的 MiniGPT，处于 eval 态。
        optimizer: 载入了存档状态的 Adam。
        config: 存档中的 config 原字典。

    要求：
        恢复出的模型在相同输入下必须与存档前给出**逐位相同**的 logits
        （CPU、同一 PyTorch 版本、关闭 Dropout 的前提下）。
        必须真的调用 load_state_dict 装载参数，而不是仅按 config 重新构造 ——
        后者会得到一个随机初始化的模型，形状正确但内容完全不同。
        模型返回前切到 eval 态，便于调用方直接推理。

    提示：
        torch.load(path, weights_only=False) 读回 dict；本练习的存档含
        非张量的 config，需要这个参数。
        构造 MiniGPT 时按 config 里的键传参，dtype 用默认值即可。
        优化器需要先绑定 model.parameters() 再 load_state_dict。
    """
    checkpoint = torch.load(path, weights_only=False)
    model = MiniGPT(**checkpoint["config"], dtype=torch.float32)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint["opt"])
    return model, optimizer, checkpoint["config"]


def greedy_generate(
    model: MiniGPT, prefix_ids: torch.Tensor, max_new_tokens: int = 6
) -> torch.Tensor:
    """固定参数，把每步预测出的 token 接回输入，返回前缀加续写。

    Args:
        model: 已训练的 MiniGPT。
        prefix_ids: (1, T) CPU torch.long，单条前缀，T >= 1 且不含 PAD。
        max_new_tokens: 最多新增多少个 token，非负整数。

    Returns:
        (1, T+K) CPU long，0 <= K <= max_new_tokens，包含原前缀。
        新生成的 EOS 必须保留在结果里，生成随即停止。
        若前缀最后一个 token 已是 EOS，或预算为 0，返回前缀的独立副本。

    约定：
        每步用完整的当前序列前向，取**最后一个位置**的 logits 选下一个 ID。
        为什么只看最后一个位置：更早位置的 logits 预测的是它们各自的下一个
        token，而那些答案已经在输入里了。
        同分时取较小 ID，沿用 torch.argmax 的首个最大值规则。
        必须在 eval 态与 no_grad 下进行；函数返回后模型的 training 状态
        应与调用前一致。
        不修改 prefix_ids，不改变参数、不触发反向。
        本练习只做贪心，不做采样、不做 temperature/top-k。

    提示：
        每步的 input_valid 是与当前序列同形状的全 True bool 张量。
        torch.cat([ids, next_id], dim=1) 追加；next_id 的 shape 需为 (1,1)。
    """
    is_training = model.training
    model.eval()
    with torch.no_grad():
        T = prefix_ids.shape[1]
        generated = prefix_ids.clone() # (1, T)
        while generated[0, -1] != EOS_ID and generated.shape[1] < T + max_new_tokens:
            input_valid = torch.ones_like(generated, dtype=torch.bool) # (1, T')
            logits = model(generated,input_valid) # (1, T', vocab_size)
            next_id = logits[:,-1,:].argmax(dim=-1, keepdim=True) # (1, 1)
            generated = torch.cat([generated, next_id], dim=1) # (1, T'+1)
    if is_training:
        model.train()
    return generated
