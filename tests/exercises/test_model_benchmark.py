"""ex014 计时教师测试：不以真实耗时大小判分。

CPU float32，种子 1403，单线程；模型 logits 容差 rtol=1e-5、atol=1e-6。
确定性时钟验证计时边界；真实前向验证每次缓存起点、结果及只读约束。
原 CPU 练习包含 10 项；另有异步队列模拟，验证教师加入的设备同步边界。
"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

import torch

from exercises.ex013_llama_style.model import LlamaLM, llama_model_step, new_caches
from exercises.ex014_model_cost import benchmark as learner


def make_model():
    with torch.random.fork_rng():
        torch.manual_seed(1403)
        return LlamaLM(11, 8, 2, 1, 12, 2, 32, dtype=torch.float32).eval()


def probe_measure(model):
    return learner.measure_cpu_step(model, torch.tensor([[2, 3]]), warmup=0, repeats=2)


class MeasureStatusTest(unittest.TestCase):
    def test_implemented(self):
        try:
            probe_measure(make_model())
        except NotImplementedError:
            self.fail("请填写 benchmark.py 的 measure_cpu_step")


class SummaryStatusTest(unittest.TestCase):
    def test_implemented(self):
        try:
            learner.summarize_samples([0.001, 0.002], tokens_per_call=2)
        except NotImplementedError:
            self.fail("请填写 benchmark.py 的 summarize_samples")


class MeasureCpuStepTest(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.model = make_model()
        try:
            probe_measure(self.model)
        except NotImplementedError:
            self.skipTest("计时尚未实现；另有实现状态测试明确失败")

    def assert_samples(self, samples, count):
        self.assertIsInstance(samples, list)
        self.assertEqual(len(samples), count)
        for value in samples:
            self.assertIs(type(value), float)
            self.assertTrue(math.isfinite(value) and value > 0)

    def test_clock_excludes_preparation_and_warmup_keeps_sample_order(self):
        prefix, ids = torch.tensor([[2, 3, 4]]), torch.tensor([[5]])
        now, reads, target_calls, prepare_calls = 0.0, [], [], []
        durations = [0.09, 0.08, 0.004, 0.001, 0.003, 0.002]

        def clock():
            reads.append(now)
            return now

        def prepare(model):
            nonlocal now
            now += 20.0  # 故意让计时外准备昂贵，激活边界错误。
            return new_caches(model)

        def step(model, tokens, caches):
            nonlocal now
            result = llama_model_step(model, tokens, caches)
            if torch.equal(tokens, prefix):
                prepare_calls.append(caches)
                now += 7.0
            else:
                self.assertTrue(torch.equal(tokens, ids))
                now += durations[len(target_calls)]
                target_calls.append(caches)
            return result

        with patch.object(learner, "new_caches", side_effect=prepare), \
             patch.object(learner, "llama_model_step", side_effect=step):
            samples = learner.measure_cpu_step(self.model, ids, prefix_ids=prefix,
                                                warmup=2, repeats=4, clock=clock)
        self.assert_samples(samples, 4)
        self.assertEqual(len(target_calls), 6)
        self.assertEqual(len(prepare_calls), 6)
        self.assertEqual(len(reads), 8, "只在正式样本的前后读时钟")
        for actual, expected in zip(samples, durations[2:]):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_decode_restarts_history_and_matches_full_logits(self):
        prefix = torch.tensor([[1, 2, 3], [4, 5, 6]])
        ids = torch.tensor([[7], [8]])
        full_ids = torch.cat((prefix, ids), dim=1)
        with torch.no_grad():
            expected = self.model(full_ids, torch.ones_like(full_ids, dtype=torch.bool))[:, -1:]
        records = []

        def observe(model, tokens, caches):
            before = tuple(len(c) for c in caches)
            result = llama_model_step(model, tokens, caches)
            records.append((tokens.clone(), caches, before, tuple(len(c) for c in caches), result))
            return result

        with patch.object(learner, "llama_model_step", side_effect=observe):
            samples = learner.measure_cpu_step(self.model, ids, prefix_ids=prefix, warmup=2, repeats=3)
        self.assert_samples(samples, 3)
        self.assertEqual(len(records), 10)
        cache_objects = []
        for i in range(0, 10, 2):
            prep, target = records[i:i+2]
            self.assertTrue(torch.equal(prep[0], prefix))
            self.assertTrue(torch.equal(target[0], ids))
            self.assertEqual((prep[2], prep[3]), ((0, 0), (3, 3)))
            self.assertEqual((target[2], target[3]), ((3, 3), (4, 4)))
            self.assertTrue(all(a is b for a, b in zip(prep[1], target[1])))
            cache_objects.extend(prep[1])
            torch.testing.assert_close(target[4], expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(len({id(c) for c in cache_objects}), 10, "每次用新的缓存容器")

    def test_prefill_always_starts_empty_with_zero_or_nonzero_warmup(self):
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        for warmup in (0, 2):
            records = []

            def observe(model, tokens, caches):
                before = tuple(len(c) for c in caches)
                result = llama_model_step(model, tokens, caches)
                records.append((before, tuple(len(c) for c in caches), result))
                self.assertTrue(torch.equal(tokens, ids))
                return result

            with self.subTest(warmup=warmup), patch.object(learner, "llama_model_step", side_effect=observe):
                samples = learner.measure_cpu_step(self.model, ids, warmup=warmup, repeats=2)
                self.assert_samples(samples, 2)
                self.assertEqual(len(records), warmup + 2)
                for before, after, output in records:
                    self.assertEqual((before, after), ((0, 0), (5, 5)))
                    torch.testing.assert_close(output, records[0][2], rtol=1e-5, atol=1e-6)

    def test_no_grad_and_preserves_model_inputs_and_caller_state(self):
        ids, prefix = torch.tensor([[4]]), torch.tensor([[1, 2, 3]])
        copies = ids.clone(), prefix.clone()
        parameters = list(self.model.parameters())
        values = [p.detach().clone() for p in parameters]
        for p in parameters:
            p.grad = torch.ones_like(p)
        grads = [p.grad for p in parameters]
        modes = [m.training for m in self.model.modules()]
        grad_enabled = []

        def observe(model, tokens, caches):
            grad_enabled.append(torch.is_grad_enabled())
            result = llama_model_step(model, tokens, caches)
            self.assertFalse(result.requires_grad)
            return result

        with torch.enable_grad(), patch.object(learner, "llama_model_step", side_effect=observe):
            learner.measure_cpu_step(self.model, ids, prefix_ids=prefix, warmup=1, repeats=2)
            self.assertTrue(torch.is_grad_enabled(), "返回时恢复调用者的求导开关")
        self.assertEqual(grad_enabled, [False] * 6)
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        self.assertTrue(torch.equal(ids, copies[0]) and torch.equal(prefix, copies[1]))
        self.assertEqual(torch.get_num_threads(), 1)
        for p, value, grad in zip(parameters, values, grads):
            self.assertTrue(torch.equal(p, value))
            self.assertIs(p.grad, grad)
            self.assertTrue(torch.equal(p.grad, torch.ones_like(p)))


class DeviceSynchronizationTest(unittest.TestCase):
    def test_cpu_does_not_call_gpu_synchronization(self):
        with patch.object(torch.cuda, "synchronize") as cuda_sync, \
             patch.object(torch.mps, "synchronize") as mps_sync:
            probe_measure(make_model())
        cuda_sync.assert_not_called()
        mps_sync.assert_not_called()

    def test_async_work_is_completed_with_preparation_outside_timer(self):
        # 无需 GPU：step 只往队列加工作，同步才推进设备完成时间。
        for device_name in ("cuda:1", "mps"):
            for with_prefix in (False, True):
                for warmup in (0, 2):
                    with self.subTest(device=device_name, prefix=with_prefix, warmup=warmup):
                        device = torch.device(device_name)
                        ids = SimpleNamespace(device=device)
                        prefix = object() if with_prefix else None
                        now, pending, target_count, reads = 0.0, 0.0, 0, 0
                        durations = [0.09] * warmup + [0.004, 0.001, 0.003]

                        def prepare(model):
                            nonlocal now
                            now += 20.0  # CPU 侧创建容器的成本不应计入样本。
                            return object()

                        def step(model, tokens, caches):
                            nonlocal pending, target_count
                            self.assertFalse(torch.is_grad_enabled())
                            if tokens is prefix:
                                pending += 7.0  # 提交历史恢复，尚未完成。
                            else:
                                self.assertIs(tokens, ids)
                                pending += durations[target_count]
                                target_count += 1
                            return object()

                        def synchronize(*args):
                            nonlocal now, pending
                            now += pending
                            pending = 0.0

                        def clock():
                            nonlocal reads
                            self.assertEqual(pending, 0.0,
                                             "两次读时钟前都必须完成对应的设备工作")
                            reads += 1
                            return now

                        with patch.object(learner, "new_caches", side_effect=prepare), \
                             patch.object(learner, "llama_model_step", side_effect=step), \
                             patch.object(torch.cuda, "synchronize", side_effect=synchronize) as cuda_sync, \
                             patch.object(torch.mps, "synchronize", side_effect=synchronize) as mps_sync:
                            actual = learner.measure_cpu_step(object(), ids, prefix_ids=prefix,
                                warmup=warmup, repeats=3, clock=clock)
                        for a, b in zip(actual, durations[warmup:]):
                            self.assertAlmostEqual(a, b, places=9)
                        self.assertEqual(len(actual), 3)
                        self.assertEqual(reads, 6)
                        self.assertEqual(target_count, warmup + 3)
                        if device.type == "cuda":
                            self.assertEqual(cuda_sync.call_args_list, [call(device)] * 6)
                            mps_sync.assert_not_called()
                        else:
                            cuda_sync.assert_not_called()
                            self.assertEqual(mps_sync.call_args_list, [call()] * 6)


class SummarizeSamplesTest(unittest.TestCase):
    def setUp(self):
        try:
            learner.summarize_samples([0.001, 0.002], tokens_per_call=2)
        except NotImplementedError:
            self.skipTest("样本汇总尚未实现；另有实现状态测试明确失败")

    def assert_summary(self, samples, tokens, expected):
        before = list(samples)
        actual = learner.summarize_samples(samples, tokens_per_call=tokens)
        self.assertEqual(samples, before, "不要原地排序或修改样本")
        self.assertEqual(set(actual), set(expected))
        for key, value in expected.items():
            self.assertIs(type(actual[key]), float, key)
            self.assertAlmostEqual(actual[key], value, places=8, msg=key)

    def test_unsorted_odd_samples_and_total_throughput(self):
        self.assert_summary([0.020, 0.001, 0.004, 0.002, 0.003], 2,
            dict(median_ms=3.0, p25_ms=2.0, p75_ms=4.0, tokens_per_second=1000/3))

    def test_even_samples_and_interpolated_quartiles(self):
        self.assert_summary([0.008, 0.001, 0.004, 0.002], 3,
            dict(median_ms=3.0, p25_ms=1.75, p75_ms=5.0, tokens_per_second=800.0))

    def test_two_samples(self):
        self.assert_summary([0.001, 0.003], 1,
            dict(median_ms=2.0, p25_ms=1.5, p75_ms=2.5, tokens_per_second=500.0))

    def test_time_and_token_scaling(self):
        samples = [0.001, 0.002, 0.005, 0.004, 0.003]
        base = learner.summarize_samples(samples, tokens_per_call=2)
        slow = learner.summarize_samples([x*3 for x in samples], tokens_per_call=2)
        batch = learner.summarize_samples(samples, tokens_per_call=6)
        for key in ("median_ms", "p25_ms", "p75_ms"):
            self.assertAlmostEqual(slow[key], 3*base[key])
            self.assertAlmostEqual(batch[key], base[key])
        self.assertAlmostEqual(slow["tokens_per_second"], base["tokens_per_second"]/3)
        self.assertAlmostEqual(batch["tokens_per_second"], base["tokens_per_second"]*3)


if __name__ == "__main__":
    unittest.main()
