"""教师固定例子：把 ex013 的参数、KV 和 Tensor 存储对上账。

运行：.venv/bin/python -B -m examples.model_cost_probe
固定 CPU float32 小模型；不计时、不实现通用预算器或基准框架。
矩阵乘 FLOP 采用每次乘加约 2 FLOP 的口径，忽略归一化、RoPE、
Softmax、逐元素运算及数据搬运；这些数字不等于整次前向的精确指令数。
"""
import platform

import torch

from exercises.ex007_multi_head_attention.heads import split_heads
from exercises.ex013_llama_style.model import LlamaLM, llama_model_step, new_caches


def main():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(1401)
        # 这组常量与讲义贯穿示例一致，不是可配置预算工具。
        N, C, Hq, Hkv, G, layers = 17, 24, 6, 2, 64, 2
        B, S, D, element_bytes = 2, 7, 4, 4
        model = LlamaLM(
            vocab_size=N, C=C, num_query_heads=Hq, num_kv_heads=Hkv,
            ffn_hidden=G, n_layer=layers, max_positions=32,
            dtype=torch.float32,
        )
        model.eval()
        ids = torch.tensor([[1, 4, 6, 7, 8, 2, 3], [1, 3, 8, 6, 9, 2, 4]])
        caches = new_caches(model)
        with torch.no_grad():
            full = model(ids, torch.ones_like(ids, dtype=torch.bool))
            pieces, start = [], 0
            for size in (3, 1, 3):
                pieces.append(llama_model_step(model, ids[:, start:start + size], caches))
                start += size
            chunked = torch.cat(pieces, dim=1)
        torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-6)

        print("固定模型：CPU float32；只核对数值与存储，不计时")
        print(f"torch={torch.__version__}; python={platform.python_version()}")
        print(f"平台={platform.system()} {platform.machine()}; 本脚本 CPU 线程数=1")
        print(f"配置 N={N}, C={C}, Hq={Hq}, Hkv={Hkv}, D={D}, G={G}, 层数={layers}, B={B}, S={S}")
        print("logits:", tuple(full.shape), "分块最大绝对误差:", (full - chunked).abs().max().item())

        attention_parameters = 2 * C * C + 2 * C * Hkv * D
        ffn_parameters = 3 * C * G
        norm_parameters = 2 * C
        per_block = attention_parameters + ffn_parameters + norm_parameters
        total_formula = 2 * N * C + layers * per_block + C
        total_actual = sum(p.numel() for p in model.parameters())
        assert attention_parameters == 1536
        assert ffn_parameters == 4608
        assert norm_parameters == 48
        assert total_formula == total_actual == 13224
        for block in model.blocks:
            assert sum(p.numel() for p in block.parameters()) == per_block
            assert sum(p.numel() for p in (block.Wq, block.Wk, block.Wv, block.Wo)) == attention_parameters
            assert sum(p.numel() for p in block.ffn.parameters()) == ffn_parameters
        weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        parameter_storages = {
            p.untyped_storage().data_ptr(): p.untyped_storage().nbytes()
            for p in model.parameters()
        }
        assert sum(parameter_storages.values()) == weight_bytes == total_actual * element_bytes
        print("每层参数 Attention / SwiGLU / RMSNorm:", attention_parameters, ffn_parameters, norm_parameters)
        print("两份词表矩阵 / 最终 RMSNorm:", 2 * N * C, C)
        print("参数总数 / float32 权重有效字节:", total_actual, weight_bytes)

        kv_formula = 2 * layers * B * S * Hkv * D * element_bytes
        kv_tensors = [tensor for cache in caches for tensor in (cache.k, cache.v)]
        kv_logical = sum(t.numel() * t.element_size() for t in kv_tensors)
        # 同一底层存储不能按每个 Tensor 重复相加；此处按地址去重。
        kv_storages = {
            t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
            for t in kv_tensors
        }
        assert kv_formula == kv_logical == sum(kv_storages.values()) == 1792
        assert all(tuple(t.shape) == (B, S, Hkv * D) for t in kv_tensors)
        print("单层单份 K/V shape:", tuple(caches[0].k.shape))
        print("全层 K+V 公式 / 逻辑字节 / 去重 storage 字节:",
              kv_formula, kv_logical, sum(kv_storages.values()))

        # 这里只核算两种固定调用的主要矩阵乘，不实际执行新的性能实验。
        for label, n, key_length in [("空缓存 prefill，n=7/S=7", 7, 7),
                                     ("历史 t=7 的单步 decode，n=1/S=8", 1, 8)]:
            projection = 2 * B * n * attention_parameters
            ffn = 2 * B * n * ffn_parameters
            attention_read = 4 * B * n * key_length * C
            vocabulary = 2 * B * n * C * N
            total = layers * (projection + ffn + attention_read) + vocabulary
            score_elements = B * Hq * n * key_length
            print(label)
            print("  每层投影 / SwiGLU / QK+AV FLOP:", projection, ffn, attention_read)
            print("  全层加词表投影 FLOP:", total, "; 单层一个分数张量字节:", score_elements * element_bytes)
        assert layers * (2 * B * S * attention_parameters + 2 * B * S * ffn_parameters
                         + 4 * B * S * S * C) + 2 * B * S * C * N == 374304
        assert layers * (2 * B * attention_parameters + 2 * B * ffn_parameters
                         + 4 * B * (S + 1) * C) + 2 * B * C * N == 53856
        print("以上只计主要矩阵乘；分数张量字节不是 Attention 峰值，也不是模型峰值。")

        print("实际 GQA 的临时复制：按 kv_indices 高级索引到 Hq 头")
        compact = caches[0].k
        split = split_heads(compact, Hkv)
        selected = split[:, torch.arange(Hq) // (Hq // Hkv)]
        assert split.untyped_storage().data_ptr() == compact.untyped_storage().data_ptr()
        assert selected.untyped_storage().data_ptr() != compact.untyped_storage().data_ptr()
        assert compact.untyped_storage().nbytes() == 448
        assert selected.untyped_storage().nbytes() == 1344
        print("  持久紧凑 K / 临时展开 K storage 字节:",
              compact.untyped_storage().nbytes(), selected.untyped_storage().nbytes())
        print("  V 同样展开；临时副本不改变缓存本身的 Hkv 宽度。")

        print("固定布局对照：shape / stride / logical bytes / storage bytes / 与源共享存储")
        base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        transposed = base.transpose(1, 2)
        flattened = transposed.reshape(2, 12)
        sliced = base[:, :1, :]
        small = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2)
        expanded = small.unsqueeze(2).expand(1, 2, 3, 3, 2)
        merged = expanded.reshape(1, 6, 3, 2)
        # 最后的 expand→reshape 只是独立布局例子，当前 GQA 使用上面的高级索引。
        for label, value, source in [
            ("base", base, base), ("transpose", transposed, base),
            ("transpose 后 reshape", flattened, base), ("小切片", sliced, base),
            ("compact", small, small), ("expand 视图", expanded, small),
            ("expand 后 reshape", merged, small),
        ]:
            print(label, tuple(value.shape), value.stride(),
                  value.numel() * value.element_size(), value.untyped_storage().nbytes(),
                  value.untyped_storage().data_ptr() == source.untyped_storage().data_ptr())
        assert transposed.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
        assert flattened.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
        assert sliced.numel() * sliced.element_size() == 32 and sliced.untyped_storage().nbytes() == 96
        assert expanded.numel() * expanded.element_size() == 144
        assert expanded.untyped_storage().nbytes() == 48
        assert merged.untyped_storage().nbytes() == 144
        assert merged.untyped_storage().data_ptr() != small.untyped_storage().data_ptr()
        torch.testing.assert_close(merged, small.repeat_interleave(3, dim=1), rtol=0, atol=0)
        # 在同一个请求的 S=7 缓存后再处理一个位置，和全量 S=8 的末行对齐。
        next_ids = torch.tensor([[5], [6]])
        ids8 = torch.cat((ids, next_ids), dim=1)
        with torch.no_grad():
            last_logits = llama_model_step(model, next_ids, caches)
            full8 = model(ids8, torch.ones_like(ids8, dtype=torch.bool))
        torch.testing.assert_close(last_logits, full8[:, -1:], rtol=1e-5, atol=1e-6)
        kv8_tensors = [tensor for cache in caches for tensor in (cache.k, cache.v)]
        kv8_logical = sum(t.numel() * t.element_size() for t in kv8_tensors)
        kv8_storages = {
            t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
            for t in kv8_tensors
        }
        kv8_formula = 2 * layers * B * (S + 1) * Hkv * D * element_bytes
        assert all(tuple(t.shape) == (B, S + 1, Hkv * D) for t in kv8_tensors)
        assert kv8_formula == kv8_logical == sum(kv8_storages.values()) == 2048
        print("真实追加：历史 t=7 → S=8；本次 logits shape:", tuple(last_logits.shape))
        print("  与全量 S=8 末行最大绝对误差:", (last_logits - full8[:, -1:]).abs().max().item())
        print("  全层 KV 公式 / 逻辑字节 / 去重 storage 字节:",
              kv8_formula, kv8_logical, sum(kv8_storages.values()))
        print("全部固定数值自检通过；未测延迟、吞吐、进程内存或 GPU 显存峰值。")
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    main()
