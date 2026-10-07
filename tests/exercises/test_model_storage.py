"""ex014 存储教师测试：CPU；字节数为整数，要求精确相等。

模型种子 1402，float32/float64，无求导；共享布局另用确定性数据。
空脚手架 2 项失败、14 项跳过；完成后须 16 项通过、无跳过。
参照只用于核对，不能从这里导入学习者实现。
"""

import unittest

import torch

from exercises.ex013_llama_style.model import LlamaLM, llama_model_step, new_caches
from exercises.ex014_model_cost import storage as learner


CONFIG = dict(parameter_count=13224, B=2, n=7, S=7, C=24,
              Hq=6, Hkv=2, G=64, layers=2, N=17, element_bytes=4)
KEYS = ("weights", "kv", "score_one_layer", "ffn_one_layer", "logits")


def estimate(**changes):
    return learner.estimate_storage_bytes(**(CONFIG | changes))


def require_estimator(test):
    try:
        estimate()
    except NotImplementedError:
        test.skipTest("预测函数尚未实现；另有实现状态测试明确失败")


def require_measurement(test):
    try:
        learner.measure_storage_bytes([torch.zeros(1)])
    except NotImplementedError:
        test.skipTest("实测函数尚未实现；另有实现状态测试明确失败")


class EstimateStatusTest(unittest.TestCase):
    def test_implemented(self):
        try:
            estimate()
        except NotImplementedError:
            self.fail("请填写 storage.py 的 estimate_storage_bytes")


class MeasureStatusTest(unittest.TestCase):
    def test_implemented(self):
        try:
            learner.measure_storage_bytes([torch.zeros(1)])
        except NotImplementedError:
            self.fail("请填写 storage.py 的 measure_storage_bytes")


class EstimateStorageTest(unittest.TestCase):
    def setUp(self):
        require_estimator(self)

    def assert_ledger(self, actual, expected):
        self.assertIsInstance(actual, dict)
        self.assertEqual(set(actual), set(KEYS))
        for key in KEYS:
            self.assertIs(type(actual[key]), int, key)
            self.assertEqual(actual[key], expected[key], key)

    def test_known_prefill_and_decode(self):
        self.assert_ledger(estimate(), dict(weights=52896, kv=1792,
            score_one_layer=2352, ffn_one_layer=3584, logits=952))
        self.assert_ledger(estimate(n=1, S=8), dict(weights=52896, kv=2048,
            score_one_layer=384, ffn_one_layer=512, logits=136))

    def test_history_changes_cache_and_score_only(self):
        short, long = estimate(n=1, S=8), estimate(n=1, S=24)
        for key in KEYS:
            self.assertEqual(long[key], short[key] * (3 if key in ("kv", "score_one_layer") else 1), key)

    def test_new_positions_at_fixed_total_length(self):
        single, chunk = estimate(n=1, S=12), estimate(n=3, S=12)
        for key in KEYS:
            self.assertEqual(chunk[key], single[key] * (1 if key in ("weights", "kv") else 3), key)

    def test_batch_layers_and_parameter_count_have_distinct_effects(self):
        base = estimate()
        # 固定 parameter_count 隔离各输入的作用；真实模型参数数在集成测试核对。
        for changes, affected in ((dict(B=6), set(KEYS) - {"weights"}),
                                  (dict(layers=6), {"kv"}),
                                  (dict(parameter_count=39672), {"weights"})):
            actual = estimate(**changes)
            for key in KEYS:
                self.assertEqual(actual[key], base[key] * (3 if key in affected else 1), (changes, key))

    def test_kv_heads_affect_compact_cache_only(self):
        mqa = estimate(Hkv=1)
        for heads in (2, 3, 6):
            actual = estimate(Hkv=heads)
            for key in KEYS:
                self.assertEqual(actual[key], mqa[key] * (heads if key == "kv" else 1), (heads, key))

    def test_element_width_scales_all_parts(self):
        one = estimate(element_bytes=1)
        for width in (2, 4, 8):
            actual = estimate(element_bytes=width)
            for key in KEYS:
                self.assertEqual(actual[key], one[key] * width, (width, key))

    def test_distinct_dimensions_against_materialized_shapes(self):
        # 单份 shape 参照，不代表模型运行期间的峰值。
        cases = [dict(parameter_count=421, B=3, n=2, S=11, C=16, Hq=4,
                      Hkv=2, G=23, layers=3, N=13, element_bytes=8),
                 dict(parameter_count=137, B=1, n=5, S=5, C=12, Hq=3,
                      Hkv=1, G=17, layers=1, N=7, element_bytes=4)]
        for c in cases:
            dtype = torch.float64 if c["element_bytes"] == 8 else torch.float32
            shapes = {
                "weights": (c["parameter_count"],),
                "kv": (c["layers"], 2, c["B"], c["S"], c["Hkv"], c["C"] // c["Hq"]),
                "score_one_layer": (c["B"], c["Hq"], c["n"], c["S"]),
                "ffn_one_layer": (c["B"], c["n"], c["G"]),
                "logits": (c["B"], c["n"], c["N"]),
            }
            expected = {key: torch.empty(shape, dtype=dtype).untyped_storage().nbytes()
                        for key, shape in shapes.items()}
            with self.subTest(config=c):
                self.assert_ledger(learner.estimate_storage_bytes(**c), expected)


class MeasureStorageTest(unittest.TestCase):
    def setUp(self):
        require_measurement(self)

    def assert_bytes(self, tensors, logical, storage):
        actual = learner.measure_storage_bytes(tensors)
        self.assertEqual(actual, dict(logical_bytes=logical, storage_bytes=storage))
        for value in actual.values():
            self.assertIs(type(value), int)

    def test_mixed_dtypes(self):
        tensors = [torch.zeros(3, dtype=torch.float32), torch.zeros(2, dtype=torch.float64),
                   torch.zeros(5, dtype=torch.bool), torch.zeros(2, dtype=torch.int64)]
        self.assert_bytes(tensors, 49, 49)

    def test_offset_slices_share_the_entire_storage(self):
        base = torch.arange(12, dtype=torch.float32)
        # 基底不在列表内；切片首元素地址不同，仍共用完整 48 字节存储。
        self.assert_bytes([base[2:6], base[6:]], 40, 48)
        self.assert_bytes([base[2:6]], 16, 48)
        self.assert_bytes([base.view(3, 4).T, base[::2]], 72, 48)

    def test_repeated_object_and_equal_valued_clone(self):
        base = torch.arange(12, dtype=torch.float32)
        self.assert_bytes([base, base], 96, 48)
        self.assert_bytes([base, base.clone()], 96, 96)

    def test_expand_then_reshape_can_allocate(self):
        base = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2)
        shared = base.unsqueeze(2).expand(1, 2, 3, 3, 2)
        flat = shared.reshape(1, 6, 3, 2)
        self.assert_bytes([shared], 144, 48)
        self.assert_bytes([flat], 144, 144)
        self.assert_bytes([base, shared, flat], 336, 192)

    def test_empty_list(self):
        self.assert_bytes([], 0, 0)

    def test_measurement_is_read_only(self):
        base = torch.arange(12, dtype=torch.float64, requires_grad=True)
        base.grad = torch.ones_like(base)
        grad = base.grad
        tensors = [base, base.view(3, 4).T, base[1::2]]
        objects = list(tensors)
        snapshots = [(x.clone(), x.stride(), x.storage_offset(), x.requires_grad,
                      x.untyped_storage().data_ptr()) for x in tensors]
        self.assert_bytes(tensors, 240, 96)
        self.assertEqual(len(tensors), len(objects))
        for x, original, (value, stride, offset, requires_grad, pointer) in zip(tensors, objects, snapshots):
            self.assertIs(x, original)
            self.assertTrue(torch.equal(x, value))
            self.assertEqual((x.stride(), x.storage_offset(), x.requires_grad,
                              x.untyped_storage().data_ptr()), (stride, offset, requires_grad, pointer))
        self.assertIs(base.grad, grad)
        self.assertTrue(torch.equal(base.grad, torch.ones_like(base)))


class RealModelStorageTest(unittest.TestCase):
    def setUp(self):
        require_estimator(self)
        require_measurement(self)

    def test_actual_weights_caches_and_logits_after_each_append(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng(), torch.no_grad():
                for dtype in (torch.float32, torch.float64):
                    for heads in (1, 2, 4):
                        torch.manual_seed(1402)
                        model = LlamaLM(13, 16, 4, heads, 23, 3, 32, dtype=dtype).eval()
                        parameters = list(model.parameters())
                        caches = new_caches(model)
                        for n in (5, 1, 3):
                            ids = torch.randint(0, 13, (2, n))
                            logits = llama_model_step(model, ids, caches)
                            predicted = estimate(parameter_count=sum(p.numel() for p in parameters),
                                B=2, n=n, S=len(caches[0]), C=16, Hq=4, Hkv=heads,
                                G=23, layers=3, N=13, element_bytes=logits.element_size())
                            groups = dict(weights=parameters, logits=[logits],
                                kv=[x for cache in caches for x in (cache.k, cache.v)])
                            for key, tensors in groups.items():
                                with self.subTest(dtype=dtype, Hkv=heads, n=n, key=key):
                                    expected = sum(x.numel() * x.element_size() for x in tensors)
                                    self.assertEqual(predicted[key], expected)
                                    measured = learner.measure_storage_bytes(tensors)
                                    self.assertEqual(measured["logical_bytes"], expected)
                                    # 这些组内部没有共享 storage；直接相加构成独立参照。
                                    self.assertEqual(measured["storage_bytes"], sum(
                                        x.untyped_storage().nbytes() for x in tensors))
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
