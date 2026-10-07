"""教师诊断：相同调用、不同矩阵大小，时间为什么不跟 FLOPs 成比例？

不修改模型文件。先测无 profiler 的完整 step，再分区计时和重放真实矩阵乘法。
分区与重放都会改变执行环境，分别报告口径，不能冒充原始运行的精确时间分解。
"""

import argparse
from collections import defaultdict
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from statistics import mean, median, quantiles
from time import perf_counter
from unittest.mock import patch

import torch

from exercises.ex013_llama_style import model as lm
from exercises.ex013_llama_style.components import RMSNorm, SwiGLU
from exercises.ex014_model_cost.benchmark_demo import check_logits, cpu_name
from exercises.ex014_model_cost.profiler_demo import DEFAULT_REPORT


ROOT = Path(__file__).resolve().parents[2]
STAGES = (
    "norm", "qkv_projection", "rope", "rope_head_layout",
    "cache_append", "attention_including_Wo", "ffn",
)


def summarize(samples):
    q1, _, q3 = quantiles(samples, n=4, method="inclusive")
    return dict(mean_us=mean(samples) * 1e6, median_us=median(samples) * 1e6,
                p25_us=q1 * 1e6, p75_us=q3 * 1e6)


class StageClock:
    """只包围互不嵌套的区段；剩余时间包括主干代码和计时包装自身。"""

    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def reset(self):
        self.sums.clear()
        self.counts.clear()

    def wrap(self, function, name):
        def measured(*args, **kwargs):
            start = perf_counter()
            result = function(*args, **kwargs)
            elapsed = perf_counter() - start
            self.sums[name] += elapsed
            self.counts[name] += 1
            return result
        return measured

    def install(self, stack):
        targets = [
            (RMSNorm, "forward", "norm"),
            (lm, "project_qkv", "qkv_projection"),
            (lm, "apply_rope", "rope"),
            (lm, "split_heads", "rope_head_layout"),
            (lm, "merge_heads", "rope_head_layout"),
            (lm.LayerKVCache, "append", "cache_append"),
            (lm, "grouped_query_attention", "attention_including_Wo"),
            (SwiGLU, "forward", "ffn"),
        ]
        for owner, attr, name in targets:
            stack.enter_context(patch.object(owner, attr, self.wrap(getattr(owner, attr), name)))


def prepare(model, prefix):
    caches = lm.new_caches(model)
    if prefix is not None:
        lm.llama_model_step(model, prefix, caches)
    history = 0 if prefix is None else prefix.shape[1]
    assert all(len(c) == history for c in caches)
    return caches


def sample_steps(model, ids, prefix, *, variant, warmup, repeats):
    clock = StageClock()
    samples, partitions = [], []
    with ExitStack() as stack:
        if variant != "existing_markers":
            stack.enter_context(patch.object(lm, "record_function", lambda *_: nullcontext()))
        if variant == "partitioned":
            clock.install(stack)
        with torch.no_grad():
            for index in range(warmup + repeats):
                caches = prepare(model, prefix)
                clock.reset()  # 历史准备的区段读数不进入目标调用。
                start = perf_counter()
                logits = lm.llama_model_step(model, ids, caches)
                elapsed = perf_counter() - start
                if index >= warmup:
                    samples.append(elapsed)
                    if variant == "partitioned":
                        parts = {name: clock.sums[name] for name in STAGES}
                        parts["outside_sections_including_instrumentation"] = elapsed - sum(parts.values())
                        assert parts["outside_sections_including_instrumentation"] >= 0
                        partitions.append(parts)
                history = 0 if prefix is None else prefix.shape[1]
                assert all(len(c) == history + ids.shape[1] for c in caches)
                assert logits.shape == (ids.shape[0], ids.shape[1], model.N)
                del logits
    return samples, partitions, dict(clock.counts)


def verify_partitioning(model, ids, prefix):
    """计时包装只观察，前后 logits 与每层 K/V 要逐元素一致。"""
    with torch.no_grad():
        expected_cache = prepare(model, prefix)
        actual_cache = prepare(model, prefix)
        expected = lm.llama_model_step(model, ids, expected_cache)
        with ExitStack() as stack:
            stack.enter_context(patch.object(lm, "record_function", lambda *_: nullcontext()))
            StageClock().install(stack)
            actual = lm.llama_model_step(model, ids, actual_cache)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for a, b in zip(actual_cache, expected_cache):
            torch.testing.assert_close(a.k, b.k, rtol=0, atol=0)
            torch.testing.assert_close(a.v, b.v, rtol=0, atol=0)


def capture_matmuls(model, ids, prefix):
    """仅采集一次实际 @ 的输入；正式重放时已移除捕获包装。"""
    calls = []
    original = torch.Tensor.__matmul__

    def capture(a, b):
        result = original(a, b)
        calls.append((a, b, result))
        return result

    with torch.no_grad():
        caches = prepare(model, prefix)
        with patch.object(torch.Tensor, "__matmul__", capture):
            lm.llama_model_step(model, ids, caches)
        assert len(calls) == 19, "模型执行已改变，应先重新核对诊断范围"
        for a, b, expected in calls:
            torch.testing.assert_close(a @ b, expected, rtol=0, atol=0)
    return [(a, b) for a, b, _ in calls]


def replay(calls, *, inner, warmup):
    """每个样本多次执行同一组 @，报告每组平均延迟，减少读时钟开销。"""
    for _ in range(warmup):
        for a, b in calls:
            result = a @ b
    start = perf_counter()
    for _ in range(inner):
        for a, b in calls:
            result = a @ b
    elapsed = perf_counter() - start
    del result
    return elapsed / inner


def run(*, rounds, repeats, warmup, inner):
    source = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    env = source["environment"]
    rows = {r["mode"]: r for r in source["rows"] if r["B"] == 2 and
            ((r["mode"] == "prefill" and r["n"] == 16) or
             (r["mode"] == "decode" and r["history"] == 16))}
    assert set(rows) == {"prefill", "decode"}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(env["threads"])
    try:
        with torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(env["seed"])
            c = env["model"]
            model = lm.LlamaLM(c["N"], c["C"], c["Hq"], c["Hkv"], c["G"],
                              c["layers"], env["max_positions"], dtype=torch.float32).eval()
            inputs, calls = {}, {}
            for name, row in rows.items():
                ids = torch.tensor(row["input_ids"], dtype=torch.long)
                prefix = None if row["prefix_ids"] is None else torch.tensor(row["prefix_ids"], dtype=torch.long)
                inputs[name] = (ids, prefix)
                check_logits(model, ids, prefix)
                verify_partitioning(model, ids, prefix)
                calls[name] = capture_matmuls(model, ids, prefix)

            variants = ("plain", "partitioned", "existing_markers")
            collected = {v: {name: [] for name in rows} for v in variants}
            partitions = {name: [] for name in rows}
            counts, orders = {}, []
            # 轮换测量顺序，避免固定先后顺序；原始样本全部保存。
            for r in range(rounds):
                order = list(variants[r % 3:] + variants[:r % 3])
                names = list(rows) if r % 2 == 0 else list(reversed(rows))
                orders.append(dict(round=r, variants=order, workloads=names))
                for variant in order:
                    for name in names:
                        samples, parts, count = sample_steps(model, *inputs[name], variant=variant,
                                                            warmup=warmup, repeats=repeats)
                        collected[variant][name].extend(samples)
                        if variant == "partitioned":
                            partitions[name].extend(parts)
                            counts[name] = count

            # 同一份 prefill FFN 输入：批量一次算，或拆成 16 次各算一个位置。
            # 切片、连续化和最后拼接都放在计时外，避免把它们混作乘法调用成本。
            a, b = calls["prefill"][6]
            assert a.shape == (2, 16, c["C"]) and b.shape == (c["C"], c["G"])
            split_calls = [(a[:, i:i+1].contiguous(), b) for i in range(a.shape[1])]
            split_output = torch.cat([x @ w for x, w in split_calls], dim=1)
            batched_output = a @ b
            torch.testing.assert_close(split_output, batched_output, rtol=1e-5, atol=1e-6)
            selections = {name: {"all_19": calls[name], "first_ffn_gate": [calls[name][6]]} for name in rows}
            selections["prefill"]["same_gate_work_16_calls"] = split_calls
            replays = {name: {kind: [] for kind in kinds} for name, kinds in selections.items()}
            for r in range(rounds):
                for name in (list(rows) if r % 2 == 0 else list(reversed(rows))):
                    kinds = list(selections[name])
                    if r % 2:
                        kinds.reverse()
                    for kind in kinds:
                        selected = selections[name][kind]
                        replays[name][kind].append(replay(selected, inner=inner, warmup=warmup))

            result = dict(environment=dict(
                captured_at=datetime.now(timezone.utc).isoformat(), cpu=cpu_name(),
                os=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                device="cpu", dtype="float32", threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads(), model=c, seed=env["seed"],
                rounds=rounds, repeats_per_round=repeats, warmup_per_round=warmup,
                replay_inner=inner, mode="eval + no_grad", clock="time.perf_counter",
                baseline_source=str(DEFAULT_REPORT.relative_to(ROOT)),
                scope="target llama_model_step only; rebuild same prefix outside every sample",
            ), verification="full/incremental logits aligned; partitioned logits/K/V exact; 19 captured @ replayed exactly",
                orders=orders, workloads=rows,
                full_step={v: {n: dict(samples_s=s, summary=summarize(s)) for n, s in work.items()}
                           for v, work in collected.items()},
                partitions={n: dict(samples_s=ps, calls=counts[n], mean_us={
                    key: mean(p[key] for p in ps) * 1e6 for key in ps[0]}) for n, ps in partitions.items()},
                matmul_replay={n: dict(shapes=[dict(a=list(a.shape), b=list(b.shape)) for a, b in calls[n]],
                    results={k: dict(call_count=len(selections[n][k]), samples_s=s, summary=summarize(s)) for k, s in kinds.items()})
                    for n, kinds in replays.items()},
                same_work_batching=dict(flops=2 * a.numel() * b.shape[-1],
                    max_abs_error=(split_output - batched_output).abs().max().item(),
                    scope="same prefill gate projection; 1 call versus 16 calls; slicing/contiguous/cat excluded"),
                limitations=[
                    "Partition times include underlying PyTorch dispatch/allocation/arithmetic; not pure FLOP time.",
                    "Outside sections includes model control, residuals, vocab projection, module dispatch and timing wrapper overhead; not pure Python time.",
                    "Only arithmetic means of disjoint partitions sum to the measured total; do not add medians.",
                    "Replay uses saved operands, warm caches and a different tensor lifetime; not an exact in-model share or a valid optimized model.",
                    "CPU intraop threads=1 does not imply scalar execution; no SIMD utilization or memory bandwidth measured.",
                    "New measurements need not reproduce the old 72% ratio exactly.",
                ])
            paths = [Path(__file__), ROOT / "exercises/ex013_llama_style/model.py",
                     ROOT / "exercises/ex013_llama_style/components.py", ROOT / "exercises/ex012_grouped_query_attention/gqa.py",
                     ROOT / "exercises/ex011_kv_cache/cache.py", ROOT / "exercises/ex007_multi_head_attention/heads.py",
                     ROOT / "exercises/ex006_single_head_attention/attention.py"]
            result["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            return result
    finally:
        torch.set_num_threads(old_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--inner", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 3 or args.repeats < 2 or args.warmup < 1 or args.inner < 1:
        parser.error("要求 rounds>=3、repeats>=2、warmup>=1、inner>=1")
    result = run(rounds=args.rounds, repeats=args.repeats, warmup=args.warmup, inner=args.inner)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = dict(full_step={v: {n: x["summary"] for n, x in work.items()} for v, work in result["full_step"].items()},
                   partitions={n: x["mean_us"] for n, x in result["partitions"].items()},
                   replay={n: {k: x["summary"] for k, x in value["results"].items()} for n, value in result["matmul_replay"].items()})
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"原始数据与计时口径：{args.output}")


if __name__ == "__main__":
    main()
