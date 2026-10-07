"""教师提供采集入口：复用基准报告中的模型和输入，分析调用次数与 shape。"""

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from exercises.ex013_llama_style.model import LlamaLM, llama_model_step, new_caches
from exercises.ex014_model_cost.benchmark_demo import check_logits


DEFAULT_REPORT = (Path(__file__).resolve().parents[2] / "notes/assets/"
                  "model-cost-and-benchmark/cpu-baseline-20260924.json")


def collect_step(model, ids, prefix, label):
    """历史恢复在采集外；两次采集各用新缓存，只保留第二次，减轻首次采集影响。"""
    history = 0 if prefix is None else prefix.shape[1]

    def prepare():
        caches = new_caches(model)
        if prefix is not None:
            llama_model_step(model, prefix, caches)
        assert all(len(c) == history for c in caches)
        return caches

    with torch.no_grad():
        for _ in range(10):
            llama_model_step(model, ids, prepare())
        for _ in range(2):
            caches = prepare()
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function(label + "_target"):
                    logits = llama_model_step(model, ids, caches)
            assert all(len(c) == history + ids.shape[1] for c in caches)
            assert logits.shape == (ids.shape[0], ids.shape[1], model.token_table.shape[0])
    return prof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_REPORT,
                        help="benchmark_demo 输出的 CPU float32 报告；默认使用已保存的基线")
    parser.add_argument("--trace-dir", type=Path, help="可选，导出两份 Chrome trace JSON")
    args = parser.parse_args()
    report = json.loads(args.benchmark.read_text(encoding="utf-8"))
    env = report["environment"]
    if env["device"] != "cpu" or env["dtype"] != "float32":
        parser.error("本入口只复现 CPU float32 报告")
    selected = [r for r in report["rows"] if r["B"] == 2 and
                ((r["mode"] == "prefill" and r["n"] == 16) or
                 (r["mode"] == "decode" and r["history"] == 16 and r["n"] == 1))]
    if len(selected) != 2 or {r["mode"] for r in selected} != {"prefill", "decode"}:
        parser.error("报告须包含 B=2 的长度 16 prefill 和历史 16 单步 decode")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(env["threads"])
    try:
        with torch.random.fork_rng():
            torch.manual_seed(env["seed"])
            c = env["model"]
            model = LlamaLM(c["N"], c["C"], c["Hq"], c["Hkv"], c["G"],
                           c["layers"], env["max_positions"], dtype=torch.float32).eval()
            print(f"CPU float32 | torch={torch.__version__} | threads={torch.get_num_threads()} "
                  f"| interop_threads={torch.get_num_interop_threads()}")
            print(f"参考基准采集时间：{env['captured_at']}；以下 profiler 时间不替代基准延迟。")
            for row in selected:
                ids = torch.tensor(row["input_ids"], dtype=torch.long)
                prefix = (None if row["prefix_ids"] is None else
                          torch.tensor(row["prefix_ids"], dtype=torch.long))
                check_logits(model, ids, prefix)
                prof = collect_step(model, ids, prefix, row["mode"])
                print(f"\n{row['mode']}: FLOPs={row['flops']['total']:,}, "
                      f"原基准 median={row['summary']['median_ms']:.4f} ms")
                counts = {e.key: e.count for e in prof.key_averages()}
                print("调用次数（matmul 包含 mm/bmm 子调用，不要重复相加）：")
                for name in ("aten::matmul", "aten::mm", "aten::bmm", "aten::index", "aten::cat"):
                    print(f"  {name}: {counts.get(name, 0)}")
                print("按输入 shape 分组：")
                for event in prof.key_averages(group_by_input_shape=True):
                    if event.key in {"aten::mm", "aten::bmm", "aten::index", "aten::cat"}:
                        print(f"  {event.key} ×{event.count}: {event.input_shapes}")
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
                if args.trace_dir:
                    args.trace_dir.mkdir(parents=True, exist_ok=True)
                    path = args.trace_dir / f"{row['mode']}.json"
                    prof.export_chrome_trace(str(path))
                    print(f"轨迹：{path}")
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    main()
