"""教师导读版训练入口：在 CPU 上训练源到目标的反转任务。

运行：.venv/bin/python -B -m examples.original_transformer_demo --help
目标右移、模型接线、生成循环及训练评估均已实现，可直接运行。
本脚本不保存检查点，不把某次训练结果自动写成掌握记录。
"""

import argparse
import itertools
import random

import torch

from exercises.ex005_training_loop.loss import masked_cross_entropy
from exercises.ex017_original_transformer.data import make_teacher_forcing_batch
from exercises.ex017_original_transformer.generation import greedy_generate
from exercises.ex017_original_transformer.model import MiniTransformer


def make_split(train_size, valid_size, seed):
    """从内容 ID 3..8、长度 1..4 的唯一源序列中切出互不重叠的集合。"""
    sources = [
        source
        for length in range(1, 5)
        for source in itertools.product(range(3, 9), repeat=length)
    ]
    if train_size < 1 or valid_size < 1 or train_size + valid_size > len(sources):
        raise ValueError(f"train-size/valid-size 必须为正数，且总和不超过 {len(sources)}")
    random.Random(seed).shuffle(sources)
    train_sources = sources[:train_size]
    valid_sources = sources[train_size : train_size + valid_size]
    if set(train_sources) & set(valid_sources):
        raise AssertionError("训练/验证不能共享完整源序列")
    return train_sources, valid_sources


def targets_for(sources, task):
    return [list(reversed(source)) if task == "reverse" else list(source) for source in sources]


def make_batch(sources, task):
    return make_teacher_forcing_batch([list(source) for source in sources], targets_for(sources, task))


def teacher_forced_metrics(model, sources, task, batch_size):
    """按有效目标 token 加权；不能把不同长度 batch 的均值直接再平均。"""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    try:
        with torch.no_grad():
            for start in range(0, len(sources), batch_size):
                batch = make_batch(sources[start : start + batch_size], task)
                logits = model(
                    batch["src_ids"], batch["tgt_input_ids"],
                    batch["src_valid"], batch["tgt_input_valid"],
                )
                valid = batch["target_valid"]
                count = int(valid.sum().item())
                loss = masked_cross_entropy(logits, batch["labels"], valid)
                correct = (logits.argmax(dim=-1) == batch["labels"]) & valid
                total_loss += loss.item() * count
                total_correct += int(correct.sum().item())
                total_tokens += count
    finally:
        model.train(was_training)
    return total_loss / total_tokens, total_correct / total_tokens


def free_generation_metrics(model, sources, task):
    """只向生成函数提供源序列；目标仅在函数返回后参与评分。"""
    exact_count = 0
    examples = []
    for source in sources:
        source_tensor = torch.tensor([list(source)], dtype=torch.long)
        source_valid = torch.ones_like(source_tensor, dtype=torch.bool)
        prediction = greedy_generate(
            model, source_tensor, source_valid, max_new_tokens=6
        )
        expected = targets_for([source], task)[0] + [2]
        exact_count += int(prediction == expected)
        if len(examples) < 3:
            examples.append((list(source), expected, prediction))
    return exact_count / len(sources), examples


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600, help="参数更新步数，0 表示只评估初始化模型")
    parser.add_argument("--seed", type=int, default=1701, help="数据、参数和采样随机种子")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.003, help="Adam 学习率")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--valid-size", type=int, default=64)
    parser.add_argument("--task", choices=("reverse", "copy"), default="reverse")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.steps < 0 or args.batch_size < 1 or args.lr <= 0:
        parser.error("steps 必须非负，batch-size 和 lr 必须为正")
    try:
        train_sources, valid_sources = make_split(args.train_size, args.valid_size, args.seed)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    sampling_rng = random.Random(args.seed + 1)
    model = MiniTransformer(
        src_vocab_size=9, tgt_vocab_size=9, C=32, num_heads=4,
        ffn_hidden=64, n_encoder_layers=2, n_decoder_layers=2,
        p=0.0, dtype=torch.float32,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"device=cpu dtype=float32 threads=1 seed={args.seed} task={args.task}")
    print("config: C=32 H=4 FFN=64 encoder=2 decoder=2 Post-LN ReLU sinusoidal p=0")
    print(
        f"data_split: train={len(train_sources)} valid={len(valid_sources)} "
        "unique_source_overlap=0 content_ids=3..8 source_lengths=1..4"
    )
    initial_loss, initial_accuracy = teacher_forced_metrics(
        model, train_sources, args.task, args.batch_size
    )
    print(f"before train_teacher_forced: loss={initial_loss:.4f} token_acc={initial_accuracy:.2%}")
    model.train()
    order = []
    cursor = 0
    for step in range(1, args.steps + 1):
        if cursor >= len(order):
            order = list(range(len(train_sources)))
            sampling_rng.shuffle(order)
            cursor = 0
        selected = order[cursor : cursor + args.batch_size]
        cursor += len(selected)
        batch = make_batch([train_sources[index] for index in selected], args.task)
        # 每个 batch 重新清梯度，再用当前参数计算源 E 和目标 logits。
        # 训练中参数持续更新，不能跨训练步复用之前的 E。
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["src_ids"], batch["tgt_input_ids"],
            batch["src_valid"], batch["tgt_input_valid"],
        )
        # 真实目标只作为标签传给 loss；输入侧只接收右移前缀并由因果 mask 限制。
        loss = masked_cross_entropy(logits, batch["labels"], batch["target_valid"])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"step={step} loss 非有限；先检查接线和数值，再调整训练参数")
        # 梯度从词表头经过 Decoder 的 cross KV，回传到 E 和源 Encoder。
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 5) == 0 or step == args.steps:
            print(f"step={step}/{args.steps} batch_loss={loss.item():.4f}")
    for name, sources in (("train", train_sources), ("valid", valid_sources)):
        loss, accuracy = teacher_forced_metrics(model, sources, args.task, args.batch_size)
        print(f"{name}_teacher_forced: loss={loss:.4f} token_acc={accuracy:.2%}")
    # 最后只给源输入，不提供真实目标前缀，检查模型能否完整生成反转结果和 EOS。
    exact, examples = free_generation_metrics(model, valid_sources, args.task)
    print(f"valid_free_generation: sequence_exact_with_eos={exact:.2%}")
    for source, expected, prediction in examples:
        print(f"source={source} expected={expected} generated={prediction}")
    print("真实目标前缀下的 token_acc 与只给源输入的整句 exact 是不同指标；EOS=2 必须生成且位置正确。")
    print("本次结果只覆盖未见组合、长度 1..4；不代表更长上下文外推，也不自动判定课程掌握。")
    if exact < 1.0:
        print("验证集仍有生成错误；保留这个结果，结合结构测试、训练/验证差距继续排查。")


if __name__ == "__main__":
    main()
