"""教师提供模型和数据；预测与统计均调用学习者的 storage.py。"""

import torch

from exercises.ex013_llama_style.model import LlamaLM, llama_model_step, new_caches
from exercises.ex014_model_cost.cost import estimate_matmul_flops
from exercises.ex014_model_cost.storage import estimate_storage_bytes, measure_storage_bytes


def main():
    config = dict(B=2, C=24, Hq=6, Hkv=2, G=64, layers=2, N=17)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(1401)
            model = LlamaLM(17, 24, 6, 2, 64, 2, 32, dtype=torch.float32).eval()
            parameters = list(model.parameters())
            parameter_count = sum(p.numel() for p in parameters)
            caches = new_caches(model)
            inputs = [torch.arange(14).reshape(2, 7), torch.tensor([[5], [6]])]
            for label, ids in zip(("prefill", "decode"), inputs):
                logits = llama_model_step(model, ids, caches)
                n, S = ids.shape[1], len(caches[0])
                predicted = estimate_storage_bytes(
                    **config, n=n, S=S, parameter_count=parameter_count, element_bytes=4,
                )
                flops = estimate_matmul_flops(**config, n=n, S=S)
                objects = {
                    "weights": parameters,
                    "kv": [x for cache in caches for x in (cache.k, cache.v)],
                    "logits": [logits],
                }
                print(f"\n{label}: n={n}, S={S}, FLOPs={flops['total']:,}")
                print("对象             预测字节    实测逻辑字节    去重存储字节")
                for key, tensors in objects.items():
                    measured = measure_storage_bytes(tensors)
                    assert predicted[key] == measured["logical_bytes"], key
                    print(f"{key:16} {predicted[key]:8,} {measured['logical_bytes']:15,} {measured['storage_bytes']:15,}")
                for key in ("score_one_layer", "ffn_one_layer"):
                    print(f"{key}: {predicted[key]:,} 字节（仅单份 shape 预测）")

            base = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2)
            shared = base.unsqueeze(2).expand(1, 2, 3, 3, 2)
            flat = shared.reshape(1, 6, 3, 2)
            print("\nGQA 布局样例：先共享展开，再合并头轴")
            for name, tensors in (("base", [base]), ("shared", [shared]),
                                  ("flat", [flat]), ("三者一起", [base, shared, flat])):
                print(name, measure_storage_bytes(tensors))
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    main()
