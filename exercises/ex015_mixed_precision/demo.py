"""教师提供的报告入口；两次精度运行和误差计算由学习者函数完成。"""

import argparse
import json
import math
import platform
from pathlib import Path

import torch

from exercises.ex015_mixed_precision.fixtures import make_case
from exercises.ex015_mixed_precision.precision import compare_results, run_precision_pass


def observe_pass(model, batch, use_bf16):
    ffn_dtypes = []

    def observe(module, args, output):
        ffn_dtypes.append(str(output.dtype))

    handle = model.blocks[0].ffn.register_forward_hook(observe)
    try:
        result = run_precision_pass(model, **batch, use_bf16=use_bf16)
    finally:
        handle.remove()
    info = dict(
        parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        first_ffn_output_dtypes=ffn_dtypes,
        logits_dtype=str(result["logits"].dtype),
        logits_bytes=result["logits"].numel() * result["logits"].element_size(),
        loss_dtype=str(result["loss"].dtype),
        loss=float(result["loss"]),
        gradient_dtypes=sorted({str(g.dtype) for g in result["grads"].values()}),
        parameter_gradient_count=len(result["grads"]),
    )
    return result, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="可选：保存 JSON 数值报告")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, batch = make_case()
    try:
        baseline, baseline_info = observe_pass(model, batch, False)
        mixed, mixed_info = observe_pass(model, batch, True)
        differences = compare_results(baseline, mixed)
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}。请填写 precision.py 的两处 TODO。")
        raise SystemExit(1) from None
    except RuntimeError:
        print("数值运行失败；请查看下面的错误，不把失败分支当作已完成的混合精度实验。")
        raise
    report = dict(
        environment=dict(device="cpu", python=platform.python_version(),
                         pytorch=torch.__version__, os=platform.platform(),
                         threads=torch.get_num_threads(), seed=1503),
        workload=dict(B=2, T=6, C=24, Hq=6, Hkv=2, G=64, layers=2, N=17,
                      eps=model.blocks[0].norm1.eps, rope_theta=model.blocks[0].rope_theta,
                      mode="eval_with_grad", **{k: v.tolist() for k, v in batch.items()}),
        fp32=baseline_info, cpu_bf16=mixed_info, differences=differences,
        scope="固定权重与 batch 的一次前向/反向；未更新参数、未计时、未验证长期训练或 CUDA",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        def json_safe(value):
            # JSON 没有 NaN/Infinity 字面量；保留异常文本，不伪造为有限数。
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            if isinstance(value, dict):
                return {k: json_safe(v) for k, v in value.items()}
            if isinstance(value, list):
                return [json_safe(v) for v in value]
            return value

        args.output.write_text(json.dumps(json_safe(report), ensure_ascii=False,
                                         indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"已保存：{args.output}")


if __name__ == "__main__":
    main()
