"""教师提供的打印入口；所有计算量必须来自学习者的 cost.py。"""

from exercises.ex014_model_cost.cost import estimate_matmul_flops


def main():
    config = dict(B=2, C=24, Hq=6, Hkv=2, G=64, layers=2, N=17)
    workloads = [
        ("prefill 7 个位置", 7, 7),
        ("历史 7，追加 1 个位置", 1, 8),
        ("历史 7，追加 3 个位置", 3, 10),
    ]
    print("当前只输出主要矩阵乘法 FLOPs；不计时。配置：", config)
    for label, n, S in workloads:
        ledger = estimate_matmul_flops(**config, n=n, S=S)
        print(f"\n{label}：n={n}, S={S}")
        for key in ("q_proj", "kv_proj", "score", "value_read", "out_proj", "ffn", "vocab", "total"):
            print(f"  {key:>12}: {ledger[key]:,}")


if __name__ == "__main__":
    main()
