"""教师提供实验输入、数值核对与报告入口；计时和汇总由学习者实现。"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess

import torch

from exercises.ex013_llama_style.model import LlamaLM, llama_model_step, new_caches
from exercises.ex014_model_cost.benchmark import measure_cpu_step, summarize_samples
from exercises.ex014_model_cost.cost import estimate_matmul_flops
from exercises.ex014_model_cost.storage import estimate_storage_bytes


def cpu_name():
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.partition(":")[2].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return "未获取 CPU 型号（系统查询不可用，请在报告中补充）"


def check_logits(model, ids, prefix):
    """计时外的数值与长度检查，不调用学习者的计时函数。"""
    with torch.no_grad():
        full_ids = ids if prefix is None else torch.cat((prefix, ids), dim=1)
        expected = model(full_ids, torch.ones_like(full_ids, dtype=torch.bool))[:, -ids.shape[1]:]
        caches = new_caches(model)
        history = 0 if prefix is None else prefix.shape[1]
        if prefix is not None:
            llama_model_step(model, prefix, caches)
        assert all(len(c) == history for c in caches)
        actual = llama_model_step(model, ids, caches)
        assert all(len(c) == history + ids.shape[1] for c in caches)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def run_experiment(*, warmup=10, repeats=50):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(1403)
            config = dict(C=24, Hq=6, Hkv=2, G=64, layers=2, N=17)
            model = LlamaLM(17, 24, 6, 2, 64, 2, 32, dtype=torch.float32).eval()
            parameter_count = sum(p.numel() for p in model.parameters())
            all_ids = torch.randint(0, 17, (2, 16))
            all_next_ids = torch.randint(0, 17, (2, 1))
            environment = dict(
                captured_at=datetime.now(timezone.utc).isoformat(),
                cpu=cpu_name(), architecture=platform.machine(),
                os=platform.platform(), python=platform.python_version(),
                torch=torch.__version__, device="cpu", dtype="float32",
                cpu_logical_count=os.cpu_count(), threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads(), seed=1403,
                warmup=warmup, repeats=repeats, model=config, max_positions=32,
                mode="eval + no_grad", clock="time.perf_counter (seconds)",
                scope="一次 llama_model_step，包含 KV 追加与全部新位置 logits；不含历史恢复、统计和打印",
                cache_restore="每次热身和每个样本新建缓存，在计时外重新 prefill 相同前缀",
            )
            rows = []
            for B in (1, 2):
                for length in (8, 16):
                    prompt = all_ids[:B, :length].clone()
                    next_ids = all_next_ids[:B].clone()
                    for label, ids, prefix in (("prefill", prompt, None),
                                               ("decode", next_ids, prompt)):
                        check_logits(model, ids, prefix)
                        samples = measure_cpu_step(model, ids, prefix_ids=prefix,
                                                   warmup=warmup, repeats=repeats)
                        summary = summarize_samples(samples, tokens_per_call=ids.numel())
                        n = ids.shape[1]
                        t = 0 if prefix is None else prefix.shape[1]
                        dimensions = config | dict(B=B, n=n, S=t+n)
                        flops = estimate_matmul_flops(**dimensions)
                        storage = estimate_storage_bytes(**dimensions,
                            parameter_count=parameter_count, element_bytes=4)
                        rows.append(dict(mode=label, **dimensions, history=t,
                            input_ids=ids.tolist(), prefix_ids=None if prefix is None else prefix.tolist(),
                            flops=flops, storage=storage, samples_s=samples, summary=summary))
            return dict(environment=environment, rows=rows)
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, help="可选 JSON 报告路径，保留全部原始样本")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 2:
        parser.error("要求 warmup>=0、repeats>=2")
    report = run_experiment(warmup=args.warmup, repeats=args.repeats)
    print(json.dumps(report["environment"], ensure_ascii=False, indent=2))
    print("\nmode     B  n   t   S     FLOPs  KV bytes  median ms  p25 ms  p75 ms   token/s")
    for row in report["rows"]:
        s = row["summary"]
        print(f"{row['mode']:8} {row['B']:1} {row['n']:2} {row['history']:3} {row['S']:3} "
              f"{row['flops']['total']:9,} {row['storage']['kv']:9,} "
              f"{s['median_ms']:10.4f} {s['p25_ms']:7.4f} {s['p75_ms']:7.4f} {s['tokens_per_second']:9.1f}")
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n完整报告：{args.output}")


if __name__ == "__main__":
    main()
