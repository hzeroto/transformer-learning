"""教师运行入口：连接学习者的 GQA 模块，不包含待实现的核心数学。"""
import torch

from exercises.ex011_kv_cache.cache import LayerKVCache, kv_cache_bytes
from exercises.ex012_grouped_query_attention.model import GQAMiniGPT, gqa_model_step


def main():
    torch.manual_seed(421)
    model = GQAMiniGPT(
        vocab_size=17, C=12, num_query_heads=6, num_kv_heads=2,
        ffn_hidden=19, n_layer=2, max_positions=16, dtype=torch.float64,
    )
    model.eval()
    ids = torch.tensor([[1, 4, 8, 3, 12, 7, 5], [2, 9, 6, 4, 3, 11, 1]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    caches = [LayerKVCache(max_length=model.L) for _ in model.blocks]
    try:
        with torch.no_grad():
            full = model(ids, valid)
            chunks = []
            offset = 0
            for length in (3, 1, 3):
                chunks.append(gqa_model_step(model, ids[:, offset:offset + length], caches))
                offset += length
            incremental = torch.cat(chunks, dim=1)
    except NotImplementedError as error:
        print(f"尚未完成：{error}。请先填写 README 中的四处 TODO。")
        return 1

    torch.testing.assert_close(full, incremental, rtol=1e-8, atol=1e-10)
    print(f"device=cpu, dtype={full.dtype}, torch={torch.__version__}, seed=421")
    print(f"全量/分块 logits shape={tuple(full.shape)}, 最大绝对差={(full-incremental).abs().max().item():.3g}")
    for i, block in enumerate(model.blocks):
        projections = [block.Wq, block.Wk, block.Wv, block.Wo]
        print(f"层 {i}: 四份投影参数={sum(p.numel() for p in projections)}, "
              f"其中 K/V 参数={block.Wk.numel()+block.Wv.numel()}, "
              f"缓存 K/V shape={tuple(caches[i].k.shape)}")

    tensors = [tensor for cache in caches for tensor in (cache.k, cache.v)]
    logical_bytes = sum(t.numel() * t.element_size() for t in tensors)
    # 同一个 storage 可能被多个视图引用；按地址去重，不重复记账。
    stores = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
    formula = kv_cache_bytes(
        model.n_layer, ids.shape[0], ids.shape[1], model.num_kv_heads,
        model.C // model.num_query_heads, model.token_table.element_size(),
    )
    assert logical_bytes == formula
    print(f"KV 公式/实际有效字节={formula}/{logical_bytes}，去重后 storage 字节={sum(stores.values())}")
    print("同一模型的全量与缓存路径已对齐；正确性测试仍以专项测试为准。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
