"""第四部分：学习者自行设计能捕获故障的回归测试。"""

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from exercises.ex005_training_loop.debug_case import candidate_train_step
    from exercises.ex005_training_loop.debug_probe import make_debug_case
    from exercises.ex005_training_loop.gradient_check import finite_difference_one


class TrainingRegressionTest(unittest.TestCase):
    def test_candidate_step_regression(self):
        if torch is None:
            self.fail("请使用项目中已安装 PyTorch 的 Python 环境。")

        # TODO：自己选择有区分力的数据/调用顺序及断言。
        # 可以复用 make_debug_case 和有限差分工具，但不要直接调用诊断 main。
        # 测试必须在修复前因真实的数值/状态错误失败，修复后通过。
        # 不要仅断言 loss 下降、函数返回了 float 或没有抛出异常。
        E, W, batches = make_debug_case()

        lr = 0.2    
        for step, data in enumerate(batches, start=1):
            inputs, targets, valid = data
            # 检查点和差分值都对应本次更新之前的同一组参数。
            before_E = E[2, 0].item()
            before_W = W[0, 2].item()
            numeric_E = finite_difference_one(inputs, targets, valid, E, W, "E", (2, 0))
            numeric_W = finite_difference_one(inputs, targets, valid, E, W, "W", (0, 2))
    
            reported_loss = candidate_train_step(inputs, targets, valid, E, W, lr)
            
            # 由 SGD 公式反推本次实际更新所使用的方向和幅度。
            observed_E = (before_E - E[2, 0].item()) / lr
            observed_W = (before_W - W[0, 2].item()) / lr

            # 断言数值梯度与反推梯度在合理误差范围内接近。
            self.assertAlmostEqual(numeric_E, observed_E, places=5)
            self.assertAlmostEqual(numeric_W, observed_W, places=5)
        



if __name__ == "__main__":
    unittest.main()
