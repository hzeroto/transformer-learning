"""终端连续对话：复用学习者的 Qwen3 前向和贪心生成，权重只加载一次。"""

import argparse
from pathlib import Path
from time import perf_counter

import torch

from exercises.ex021_qwen3_inference import model as learner
from exercises.ex021_qwen3_inference.support import eos_ids, load_weights, read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "checkpoints" / "Qwen3-0.6B",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.threads < 1:
        parser.error("max-new-tokens 和 threads 必须为正")

    torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer

    print("正在加载本地 Qwen3-0.6B…", flush=True)
    config = read_config(args.model_dir)
    if args.max_new_tokens >= config.max_position_embeddings:
        parser.error("输出预算必须小于模型的位置容量，以便保留输入空间")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=False,
    )
    weights = load_weights(args.model_dir, config)
    stop_ids = eos_ids(args.model_dir, tokenizer)
    history = []
    print("已就绪。输入问题开始聊天；/reset 清空对话；/exit 退出。")

    while True:
        try:
            prompt = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n会话已结束。")
            return
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            print("会话已结束。")
            return
        if prompt == "/reset":
            history.clear()
            print("对话已清空。")
            continue

        messages = [*history, {"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt",
        )
        if input_ids.shape[1] + args.max_new_tokens > config.max_position_embeddings:
            print("对话已超出上下文容量，请用 /reset 清空历史，或缩短本次输入。")
            continue

        print("Qwen3 正在回答…（Ctrl+C 可取消本轮）", flush=True)
        try:
            with torch.inference_mode():
                started_at = perf_counter()
                generated = learner.greedy_generate(
                    input_ids, weights, config,
                    max_new_tokens=args.max_new_tokens, eos_token_ids=stop_ids,
                )
                generation_seconds = perf_counter() - started_at
        except KeyboardInterrupt:
            print("\n本轮已取消，之前的对话仍保留。")
            continue

        token_ids = generated[0].tolist()
        answer = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"\nQwen3 > {answer}")
        tokens_per_second = len(token_ids) / generation_seconds
        print(
            f"[输入 {input_ids.shape[1]} token | 新增 {len(token_ids)} token | "
            f"推理 {generation_seconds:.2f} s | "
            f"平均 {tokens_per_second:.2f} tokens/s（含输入处理）]"
        )
        history = [*messages, {"role": "assistant", "content": answer}]
        if len(token_ids) == args.max_new_tokens and token_ids[-1] not in stop_ids:
            print(f"[达到本轮 {args.max_new_tokens} token 上限，可输入“继续”。]")


if __name__ == "__main__":
    main()
