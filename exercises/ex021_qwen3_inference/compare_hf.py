"""独立 HF eager 数值参照；不联网，不用 HF generate 代替学习者的 demo。"""
import argparse
import gc

import torch

from .support import QwenConfig, encode_prompt, load_weights, read_config, validate_weights


def _chunk_sizes(length):
    """正常 prompt：长前缀、两 token 追加、两个单 token 追加。"""
    return [length - 4, 2, 1, 1] if length >= 5 else [1] * length


def _hf_outputs(model, ids, sizes):
    model.eval()
    with torch.inference_mode():
        full = model(input_ids=ids, use_cache=False).logits.detach()
        cache, offset, parts = None, 0, []
        for size in sizes:
            out = model(input_ids=ids[:, offset:offset + size], past_key_values=cache, use_cache=True)
            parts.append(out.logits.detach())
            cache, offset = out.past_key_values, offset + size
    return full, torch.cat(parts, dim=1)


def _check(label, actual, expected, *, atol, rtol):
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    error = (actual - expected).abs().max().item()
    agreement = (actual.argmax(-1) == expected.argmax(-1)).float().mean().item()
    print(f"{label}: max_abs_error={error:.8g}, argmax_agreement={agreement:.1%}")
    # 比较每一个词表 logit，不能用 argmax 一致代替数值一致。
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tiny", action="store_true", help="随机小配置，不需要下载任何模型")
    source.add_argument("--model-dir", help="已下载的本地 Qwen3-0.6B 目录")
    parser.add_argument("--prompt", default="用一句话解释什么是 Transformer。")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads 必须为正")
    torch.set_num_threads(args.threads)
    # 只在真正运行参照时导入；--help 可在仅安装 torch 的课程环境中运行。
    import transformers
    if transformers.__version__ != "4.51.3":
        raise RuntimeError("本参照固定 transformers==4.51.3，请使用 requirements-hf.txt 中的可选环境")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from . import model as learner

    if args.tiny:
        torch.manual_seed(20261013)
        config = QwenConfig(
            vocab_size=31, hidden_size=12, intermediate_size=20, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=4,
            max_position_embeddings=64,
        )
        hf_config = Qwen3Config(
            **vars(config), attention_dropout=0.0, attention_bias=False,
            hidden_act="silu", use_sliding_window=False, sliding_window=None, rope_scaling=None,
        )
        hf_config._attn_implementation = "eager"
        reference = Qwen3ForCausalLM(hf_config).to(device="cpu", dtype=torch.float32)
        with torch.no_grad():
            for layer in reference.model.layers:
                # 默认 gamma 全 1 会掩盖“先旋转再 QK norm”的错误，特意打破它。
                layer.self_attn.q_norm.weight.copy_(torch.linspace(0.6, 1.4, config.head_dim))
                layer.self_attn.k_norm.weight.copy_(torch.linspace(1.3, 0.7, config.head_dim))
        ids = torch.tensor([[1, 2, 5, 8, 3, 9, 4]], dtype=torch.long)
        # 词表很小时可持有这份独立拷贝；真实模型分支不会同时常驻两套 FP32 模型。
        weights = validate_weights(
            {name: value.detach().clone() for name, value in reference.state_dict().items()}, config,
        )
        atol, rtol = 5e-5, 5e-5
    else:
        config = read_config(args.model_dir)
        _, ids = encode_prompt(args.model_dir, args.prompt)
        reference = Qwen3ForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=torch.float32, attn_implementation="eager",
            local_files_only=True, trust_remote_code=False, use_safetensors=True,
        ).to("cpu")
        atol, rtol = 5e-4, 1e-4

    sizes = _chunk_sizes(ids.shape[1])
    print(f"CPU float32 / eager / eval / inference_mode; chunks={sizes}; atol={atol}, rtol={rtol}")
    expected_full, expected_cached = _hf_outputs(reference, ids, sizes)
    _check("HF cached vs HF full", expected_cached, expected_full, atol=atol, rtol=rtol)
    del reference
    gc.collect()
    if not args.tiny:
        weights = load_weights(args.model_dir, config)

    with torch.inference_mode():
        actual_full, unused_cache = learner.forward(ids, weights, config)
        del unused_cache
        cache, offset, parts = None, 0, []
        for size in sizes:
            logits, cache = learner.forward(ids[:, offset:offset + size], weights, config, cache)
            parts.append(logits)
            offset += size
        actual_cached = torch.cat(parts, dim=1)
    _check("learner full vs HF full", actual_full, expected_full, atol=atol, rtol=rtol)
    _check("learner cached vs HF cached", actual_cached, expected_cached, atol=atol, rtol=rtol)
    _check("learner cached vs learner full", actual_cached, actual_full, atol=atol, rtol=rtol)
    print("PASS：全部位置、全部词表 logits 数值对齐；这不是性能或生成质量基准。")


if __name__ == "__main__":
    main()
