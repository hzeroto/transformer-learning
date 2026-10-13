"""从本地 checkpoint 运行学习者的 CPU FP32 贪心推理，不调用 HF 模型。"""
import argparse

import torch

from .support import DEFAULT_MODEL_DIR, encode_prompt, eos_ids, load_weights, read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--prompt", default="用一句话解释什么是 Transformer。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.max_new_tokens < 0 or args.threads < 1:
        parser.error("max-new-tokens 必须非负，threads 必须为正")
    torch.set_num_threads(args.threads)
    from .model import greedy_generate

    config = read_config(args.model_dir)
    tokenizer, input_ids = encode_prompt(args.model_dir, args.prompt)
    weights = load_weights(args.model_dir, config)
    with torch.inference_mode():
        generated = greedy_generate(
            input_ids, weights, config, max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_ids(args.model_dir, tokenizer),
        )
    # 整段解码，避免把跨 token 的中文 UTF-8 字节逐个解码成乱码。
    print(tokenizer.decode(generated[0].tolist(), skip_special_tokens=True))
    print(f"\n[prompt tokens={input_ids.shape[1]}, new tokens={generated.shape[1]}]")
    print("能生成通顺文本不代表数值实现正确；仍需运行 compare_hf。")


if __name__ == "__main__":
    main()
