"""教师行为测试：用可预测的模型替身隔离源条件生成的状态与循环。"""

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from exercises.ex017_original_transformer import generation as exercise


if torch is not None:

    class RecordingModel(torch.nn.Module):
        """只模拟接口和分数，不包含任何 Transformer 数学答案。"""

        def __init__(self, schedules=None):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
            self.dropout = torch.nn.Dropout(0.5)
            self.schedules = schedules or {3: [4, 5, 2], 6: [7, 2]}
            self.encode_calls = []
            self.decode_calls = []
            self.modes = []

        def record_mode(self):
            self.modes.append(
                (self.training, self.dropout.training, torch.is_grad_enabled())
            )

        def encode(self, src_ids, src_valid):
            self.record_mode()
            memory = src_ids.to(dtype=self.marker.dtype).unsqueeze(-1) + self.marker * 0
            self.encode_calls.append((src_ids, src_valid, memory))
            return memory

        def decode(self, tgt_input_ids, memory, src_valid, tgt_input_valid):
            self.record_mode()
            self.decode_calls.append(
                (tgt_input_ids.clone(), memory, src_valid, tgt_input_valid.clone())
            )
            source_first = int(memory[0, 0, 0].item())
            schedule = self.schedules[source_first]
            target_length = tgt_input_ids.shape[1]
            successor = schedule[min(target_length - 1, len(schedule) - 1)]
            logits = torch.full((1, target_length, 9), -8.0, dtype=self.marker.dtype)
            logits[..., 8] = 8.0
            logits[0, -1, :] = -8.0
            logits[0, -1, successor] = 8.0
            return logits + self.marker * 0


def source_case(first=3):
    return torch.tensor([[first, 4, 0]]), torch.tensor([[True, True, False]])


@unittest.skipIf(torch is None, "当前 Python 环境未安装 PyTorch")
class EncoderDecoderGenerationTest(unittest.TestCase):
    def assert_generated(self, actual, expected):
        self.assertIsInstance(actual, list)
        self.assertTrue(all(type(token) is int for token in actual))
        self.assertEqual(actual, expected)

    def test_encodes_once_and_reuses_memory_with_growing_own_prefix(self):
        model = RecordingModel()
        source, valid = source_case()
        result = exercise.greedy_generate(model, source, valid, max_new_tokens=9)
        self.assert_generated(result, [4, 5, 2])
        self.assertEqual(len(model.encode_calls), 1, "一次生成只编码源序列一次")
        self.assertEqual(len(model.decode_calls), 3, "生成 EOS 后不能继续调用 decode")
        _, encode_valid, memory = model.encode_calls[0]
        self.assertTrue(torch.equal(encode_valid, valid))
        for call, expected_prefix in zip(model.decode_calls, ([1], [1, 4], [1, 4, 5])):
            prefix, supplied_memory, supplied_valid, input_valid = call
            self.assertEqual(prefix.tolist(), [expected_prefix])
            self.assertEqual(prefix.dtype, torch.long)
            self.assertEqual(prefix.device.type, "cpu")
            self.assertIs(supplied_memory, memory, "每步应复用同一份 Encoder 输出")
            self.assertTrue(torch.equal(supplied_valid, valid))
            self.assertEqual(input_valid.dtype, torch.bool)
            self.assertEqual(input_valid.shape, prefix.shape)
            self.assertTrue(bool(input_valid.all()), "已生成的前缀没有批处理补齐位置")

    def test_limit_counts_new_tokens_and_uses_last_logit_row(self):
        model = RecordingModel()
        source, valid = source_case()
        result = exercise.greedy_generate(model, source, valid, max_new_tokens=2)
        self.assert_generated(result, [4, 5])
        self.assertEqual(len(model.decode_calls), 2)

    def test_stops_at_limit_without_eos(self):
        model = RecordingModel({3: [4]})
        source, valid = source_case()
        result = exercise.greedy_generate(model, source, valid, max_new_tokens=4)
        self.assert_generated(result, [4, 4, 4, 4])
        self.assertEqual(len(model.decode_calls), 4)

    def test_custom_bos_eos_and_pad_are_not_implicitly_filtered(self):
        model = RecordingModel({3: [0, 1, 7]})
        source, valid = source_case()
        result = exercise.greedy_generate(
            model, source, valid, max_new_tokens=8, bos_id=6, eos_id=7
        )
        self.assert_generated(result, [0, 1, 7])
        self.assertEqual(model.decode_calls[0][0].tolist(), [[6]])
        self.assertEqual(model.decode_calls[1][0].tolist(), [[6, 0]])
        self.assertTrue(bool(model.decode_calls[1][3].all()))

    def test_zero_budget_needs_no_encoder_or_decoder(self):
        for training in (False, True):
            with self.subTest(training=training):
                model = RecordingModel().train(training)
                source, valid = source_case()
                result = exercise.greedy_generate(model, source, valid, max_new_tokens=0)
                self.assert_generated(result, [])
                self.assertFalse(model.encode_calls)
                self.assertFalse(model.decode_calls)
                self.assertEqual(model.training, training)

    def test_negative_budget_is_rejected(self):
        model = RecordingModel()
        source, valid = source_case()
        with self.assertRaises(ValueError):
            exercise.greedy_generate(model, source, valid, max_new_tokens=-1)
        self.assertFalse(model.encode_calls)
        self.assertFalse(model.decode_calls)

    def test_eval_no_grad_and_original_mode_inputs_parameters_grads_preserved(self):
        for training in (False, True):
            for caller_grad_enabled in (False, True):
                with self.subTest(training=training, grad_enabled=caller_grad_enabled):
                    model = RecordingModel().train(training)
                    model.marker.grad = torch.full_like(model.marker, 12.0)
                    old_grad = model.marker.grad
                    source, valid = source_case()
                    old_source, old_valid = source.clone(), valid.clone()
                    old_parameter = model.marker.detach().clone()
                    with torch.set_grad_enabled(caller_grad_enabled):
                        exercise.greedy_generate(model, source, valid, max_new_tokens=5)
                        self.assertEqual(torch.is_grad_enabled(), caller_grad_enabled)
                    self.assertTrue(model.modes)
                    self.assertTrue(all(mode == (False, False, False) for mode in model.modes))
                    self.assertEqual(model.training, training)
                    self.assertEqual(model.dropout.training, training)
                    self.assertTrue(torch.equal(source, old_source))
                    self.assertTrue(torch.equal(valid, old_valid))
                    self.assertTrue(torch.equal(model.marker, old_parameter))
                    self.assertIs(model.marker.grad, old_grad)
                    self.assertEqual(model.marker.grad.item(), 12.0)

    def test_two_requests_encode_their_own_source_without_cross_request_cache(self):
        model = RecordingModel()
        first_source, first_valid = source_case(3)
        second_source, second_valid = source_case(6)
        first = exercise.greedy_generate(model, first_source, first_valid, max_new_tokens=5)
        second = exercise.greedy_generate(model, second_source, second_valid, max_new_tokens=5)
        self.assert_generated(first, [4, 5, 2])
        self.assert_generated(second, [7, 2])
        self.assertEqual(len(model.encode_calls), 2)
        first_memory = model.encode_calls[0][2]
        second_memory = model.encode_calls[1][2]
        self.assertIsNot(first_memory, second_memory)
        self.assertEqual(model.decode_calls[3][0].tolist(), [[1]])
        self.assertIs(model.decode_calls[3][1], second_memory)


if __name__ == "__main__":
    unittest.main()
