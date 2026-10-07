"""教师对照实验：保持旋转数学不变，在一次模型调用中共享 RoPE 系数。

仅在本进程的顺序 CPU 实验中临时替换 apply_rope；原始模型实现保留。
每个目标调用都会清空系数，不跨位置或请求复用，首次计算也计入延迟。
"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile

from exercises.ex013_llama_style import model as lm
from exercises.ex014_model_cost.benchmark import summarize_samples
from exercises.ex014_model_cost.benchmark_demo import cpu_name
from exercises.ex014_model_cost.profiler_demo import DEFAULT_REPORT


ORIGINAL_ROPE = lm.apply_rope


class StepRope:
    """一次 step 内的位置相同；不同头宽、theta、dtype/device 分开缓存。"""

    def __init__(self):
        self.coefficients = {}

    def reset(self):
        self.coefficients.clear()

    def apply(self, x, positions, theta=10000.0):
        D = x.shape[-1]
        if D % 2:
            raise ValueError("D 必须是偶数")
        key = (D, theta, x.dtype, x.device)
        if key not in self.coefficients:
            # 与原 apply_rope 使用相同的运算顺序和 dtype。
            j = torch.arange(D // 2, dtype=x.dtype, device=x.device)
            freqs = theta ** (-2 * j / D)
            angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
            self.coefficients[key] = (torch.cos(angles), torch.sin(angles))
        cos, sin = self.coefficients[key]
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        rotated_even = x_even * cos[None, None, ...] - x_odd * sin[None, None, ...]
        rotated_odd = x_even * sin[None, None, ...] + x_odd * cos[None, None, ...]
        rotated = torch.empty_like(x)
        rotated[..., 0::2] = rotated_even
        rotated[..., 1::2] = rotated_odd
        return rotated


def target_step(model, ids, caches, reuse):
    if reuse is not None:
        reuse.reset()
    return lm.llama_model_step(model, ids, caches)


def prepare_history(model, prefix):
    """两版都用原实现重新生成相同历史；这部分在计时与采集之外。"""
    caches = lm.new_caches(model)
    with patch.object(lm, "apply_rope", ORIGINAL_ROPE):
        lm.llama_model_step(model, prefix, caches)
    return caches


def make_model(config, max_positions, dtype):
    return lm.LlamaLM(config["N"], config["C"], config["Hq"], config["Hkv"],
                      config["G"], config["layers"], max_positions, dtype=dtype).eval()


def check_equivalence(config, seed):
    """跨 step 的位置、输入长度与 batch 改变，检查 logits 和每层 K/V。"""
    comparisons, max_abs_error = 0, 0.0
    for dtype in (torch.float32, torch.float64):
        torch.manual_seed(seed)
        model = make_model(config, 32, dtype)
        for B in (1, 2):
            ids = torch.randint(config["N"], (B, 32))
            for chunks in ((16, 1), (3, 1, 4, 1), (30, 1, 1)):
                reference, reused = lm.new_caches(model), lm.new_caches(model)
                rope = StepRope()
                offset = 0
                for n in chunks:
                    part = ids[:, offset:offset+n]
                    expected = target_step(model, part, reference, None)
                    with patch.object(lm, "apply_rope", rope.apply):
                        actual = target_step(model, part, reused, rope)
                    pairs = [(actual, expected)]
                    for a, b in zip(reused, reference):
                        assert len(a) == len(b) == offset + n
                        pairs.extend(((a.k, b.k), (a.v, b.v)))
                    for a, b in pairs:
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                        max_abs_error = max(max_abs_error, (a-b).abs().max().item())
                        comparisons += 1
                    offset += n
                full = model(ids[:, :offset], torch.ones_like(ids[:, :offset], dtype=torch.bool))
                tolerance = 1e-6 if dtype == torch.float32 else 1e-12
                torch.testing.assert_close(actual, full[:, -n:], rtol=1e-5, atol=tolerance)
    return dict(tensor_comparisons=comparisons, max_abs_error=max_abs_error,
                dtypes=["float32", "float64"], batches=[1, 2],
                chunks=[[16, 1], [3, 1, 4, 1], [30, 1, 1]],
                checks="每步 logits、各层 K/V 逐元素完全一致；增量末尾 logits 与全量对齐")


def measure_arm(model, ids, prefix, reuse, warmup, repeats):
    samples = []
    fn = ORIGINAL_ROPE if reuse is None else reuse.apply
    with patch.object(lm, "apply_rope", fn):
        for i in range(warmup + repeats):
            caches = prepare_history(model, prefix)
            assert all(len(c) == prefix.shape[1] for c in caches)
            if i < warmup:
                target_step(model, ids, caches, reuse)
            else:
                start = perf_counter()
                logits = target_step(model, ids, caches, reuse)
                elapsed = perf_counter() - start
                samples.append(elapsed)
                del logits  # 输出释放在计时之后。
            assert all(len(c) == prefix.shape[1] + ids.shape[1] for c in caches)
    return samples


def operator_counts(model, ids, prefix, reuse):
    caches = prepare_history(model, prefix)
    fn = ORIGINAL_ROPE if reuse is None else reuse.apply
    with patch.object(lm, "apply_rope", fn):
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            logits = target_step(model, ids, caches, reuse)
    return {e.key: e.count for e in prof.key_averages()
            if e.key in {"aten::cos", "aten::sin", "aten::mm", "aten::bmm"}}


def run_experiment(*, rounds=10, repeats=100, warmup=10):
    baseline = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    env = baseline["environment"]
    row = next(r for r in baseline["rows"]
               if r["mode"] == "decode" and r["B"] == 2 and r["history"] == 16)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # 两组都关闭标记，避免将 record_function 的额外成本混进正式基准。
        with torch.random.fork_rng(), torch.no_grad(), \
             patch.object(lm, "record_function", lambda *_: nullcontext()):
            correctness = check_equivalence(env["model"], env["seed"])
            torch.manual_seed(env["seed"])
            model = make_model(env["model"], env["max_positions"], torch.float32)
            ids = torch.tensor(row["input_ids"], dtype=torch.long)
            prefix = torch.tensor(row["prefix_ids"], dtype=torch.long)
            reuse = StepRope()
            # 固定被测输入也直接核对，避免只验证其他随机输入。
            original_caches = prepare_history(model, prefix)
            reused_caches = prepare_history(model, prefix)
            expected = target_step(model, ids, original_caches, None)
            with patch.object(lm, "apply_rope", reuse.apply):
                actual = target_step(model, ids, reused_caches, reuse)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            runs = []
            all_samples = {"original": [], "reuse": []}
            for index in range(rounds):
                order = ["original", "reuse"] if index % 2 == 0 else ["reuse", "original"]
                run = dict(round=index+1, order=order, arms={})
                for name in order:
                    samples = measure_arm(model, ids, prefix, reuse if name == "reuse" else None,
                                          warmup, repeats)
                    all_samples[name].extend(samples)
                    run["arms"][name] = dict(samples_s=samples,
                        summary=summarize_samples(samples, tokens_per_call=ids.numel()))
                old = run["arms"]["original"]["summary"]["median_ms"]
                new = run["arms"]["reuse"]["summary"]["median_ms"]
                run["speedup"] = old / new
                runs.append(run)

            # 正式计时全部结束后单独采集，只用调用次数验证机制。
            counts = {"original": operator_counts(model, ids, prefix, None),
                      "reuse": operator_counts(model, ids, prefix, reuse)}
            for name in ("aten::cos", "aten::sin"):
                assert counts["original"][name] == 2 * env["model"]["layers"]
                assert counts["reuse"][name] == 1
            for name in ("aten::mm", "aten::bmm"):
                assert counts["original"][name] == counts["reuse"][name]
            aggregate = {name: summarize_samples(samples, tokens_per_call=ids.numel())
                         for name, samples in all_samples.items()}
            old, new = (aggregate[name]["median_ms"] for name in ("original", "reuse"))
            root = Path(__file__).resolve().parents[2]
            files = [Path(__file__).resolve(), root / "exercises/ex013_llama_style/model.py",
                     root / "exercises/ex013_llama_style/components.py"]
            return dict(environment=dict(captured_at=datetime.now(timezone.utc).isoformat(),
                cpu=cpu_name(), os=platform.platform(), python=platform.python_version(),
                torch=torch.__version__, device="cpu", dtype="float32", threads=1,
                interop_threads=torch.get_num_interop_threads(), seed=env["seed"],
                model=env["model"], max_positions=env["max_positions"], mode="eval + no_grad",
                rounds=rounds, repeats_per_arm_per_round=repeats, warmup_per_arm_per_round=warmup,
                scope="完整 llama_model_step；包含每步清空系数与首次构造、KV 追加、logits；历史准备和输出释放不计时",
                profiler="正式计时关闭；两版 record_function 均替换为空上下文；之后单独采集算子次数",
                cache_restore="每个样本使用原版重新 prefill 同一历史，目标步骤开始前长度均为 16",
                source_sha256={str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in files}),
                workload=dict(B=2, n=1, history=16, S=17, input_ids=row["input_ids"], prefix_ids=row["prefix_ids"]),
                correctness=correctness, operator_counts=counts, aggregate=aggregate,
                comparison=dict(speedup=old/new, latency_reduction_pct=100*(old-new)/old,
                    faster_rounds=sum(r["speedup"] > 1 for r in runs),
                    paired_speedup_median=median(r["speedup"] for r in runs)), rounds=runs)
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2 or args.repeats < 2 or args.warmup < 0:
        parser.error("要求 rounds 为至少 2 的偶数、repeats>=2、warmup>=0")
    report = run_experiment(rounds=args.rounds, repeats=args.repeats, warmup=args.warmup)
    print("输出检查：", json.dumps(report["correctness"], ensure_ascii=False))
    print("轮次  原版中位 ms  复用中位 ms  加速比")
    for r in report["rounds"]:
        a, b = (r["arms"][name]["summary"]["median_ms"] for name in ("original", "reuse"))
        print(f"{r['round']:4} {a:12.4f} {b:12.4f} {r['speedup']:7.3f}")
    print("汇总：", json.dumps(report["aggregate"], ensure_ascii=False))
    print("对比：", json.dumps(report["comparison"], ensure_ascii=False))
    print("算子次数：", json.dumps(report["operator_counts"], ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"原始样本与环境：{args.output}")


if __name__ == "__main__":
    main()
