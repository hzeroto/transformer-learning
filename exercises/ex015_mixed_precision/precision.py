"""按学习者请求提供的正确实现；复用原模型与交叉熵，接口说明见 README.md。"""

import torch

from exercises.ex005_training_loop.loss import masked_cross_entropy
from exercises.ex013_llama_style.model import LlamaLM


def run_precision_pass(
    model: LlamaLM,
    input_ids: torch.Tensor,
    input_valid: torch.Tensor,
    targets: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    use_bf16: bool,
) -> dict:
    """从相同参数出发完成一次前向和反向，保存数值快照；不更新参数。

    model: ex013 LlamaLM，CPU FP32 参数，全部 requires_grad=True。
        当前模型没有 Dropout；调用者选择 train/eval，本函数保持原模式。
    input_ids/targets: CPU long (B,T)，合法词表编号。
        targets 是调用者已准备好的下一 token 标签，不在此函数内再次错位。
    input_valid: CPU bool (B,T)，True 为模型可读取的 key。
    target_valid: CPU bool (B,T)，True 为参与 loss 的标签，和 input_valid 分工不同。
        两种 mask 合法；每个 query 有可读 key，至少一个标签参与 loss。
    B,T>=1，T 不超过模型容量；输入可能非连续。调用时求导已开启，无外层 autocast。
    use_bf16: False 为 FP32 基线；True 使用 CPU BF16 autocast。
        本练习要求当前后端支持 BF16 前向及反向；不支持时如实报错，不静默改回 FP32。

    工作要求：
    - 每次调用开始清除旧梯度，避免两次实验累积到一起。
    - 调用原 model 得到 logits；仅 use_bf16=True 时启用 CPU BF16 autocast。
      参数本体保持 FP32；不对整个模型调用 half()/bfloat16()/to(dtype=...)。
    - 将 logits 转为 FP32，调用已写好的 masked_cross_entropy；保留求导路径。
    - 在 autocast 上下文之外反向一次；无需优化器、梯度缩放或 KV Cache。
    - 返回恰好三个键：
      logits: (B,T,N) 的独立、无计算图快照，保留原输出 dtype
              （本模型基线为 FP32，BF16 autocast 下为 BF16）。
      loss: FP32、shape=() 的独立、无计算图快照。
      grads: {参数名: FP32 梯度快照}，覆盖 model.named_parameters() 的全部参数，
             各梯度与对应参数同 shape，独立存储且无计算图。
    - model 的 .grad 可变；输入、参数数值/dtype/requires_grad 和 train/eval 模式不变。
      不用 no_grad/inference_mode 执行前向，不在 backward 之前 detach。
      不重建或重新随机初始化模型，不从 demo/tests 导入目标实现。

    本次没有通用参数校验要求。detach().clone() 可用于反向完成后的结果快照，
    不能替代训练计算。所有参数在本模型前向中都有求导路径。
    """
    # 1. 清除上一轮梯度，保证这次得到的是当前 loss 的梯度。
    model.zero_grad(set_to_none=True)

    # 2. 只改变本次运算的精度策略，参数本体仍然是 FP32。
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(input_ids, input_valid)
        loss = masked_cross_entropy(logits.float(), targets, target_valid)

    # 3. 先反向，再读取参数的 .grad；参数本身与参数梯度是不同对象。
    loss.backward()

    # 4. detach 脱离计算图，clone 取得独立存储，供两次实验比较。
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    return {
        "logits": logits.detach().clone(),
        "loss": loss.detach().clone(),
        "grads": grads,
    }


def compare_results(reference: dict, candidate: dict) -> dict:
    """比较两次 run_precision_pass 的结果；FP32 结果作为 reference。

    输入均有 logits/loss/grads 三个键，各对象形状一致，grads 的非空键集合一致。
    logits 可以分别是 FP32 与 BF16；loss 和 grads 为 FP32。输入不修改。

    返回恰好四项，误差用 Python float，all_finite 用 Python bool：
    - logits_max_abs：logits 全部元素的最大绝对误差。
    - loss_abs：两次 loss 的绝对差。
    - grad_max_abs：所有参数梯度、所有元素中的最大绝对误差，不只看最后一个参数。
    - all_finite：两份结果的 logits、loss、全部梯度都没有 NaN 或正负 inf。

    误差定义为 max(abs(candidate - reference))；先转成 FP32 再计算。
    若某组任一输入出现非有限值，该组误差返回 float("inf")，all_finite=False；
    其余正常组仍按定义计算。梯度作为一整组处理。
    不丢弃异常元素、不只比较 argmax，也不把“有差异”自动当成实现错误。
    """
    def is_finite(tensor):
        # isfinite 得到逐元素布尔张量；all 归约，item 取出 Python bool。
        return torch.isfinite(tensor).all().item()

    def max_abs_error(a, b):
        # Tensor.max() 取全部元素的最大值；item 返回 Python float。
        return (a.float() - b.float()).abs().max().item()

    # 每一组都检查两次运行，不能只检查其中一边或只看 loss。
    logits_finite = is_finite(reference["logits"]) and is_finite(candidate["logits"])
    loss_finite = is_finite(reference["loss"]) and is_finite(candidate["loss"])
    grads_finite = all(
        is_finite(grad)
        for result in (reference, candidate)
        for grad in result["grads"].values()
    )

    logits_error = (
        max_abs_error(reference["logits"], candidate["logits"])
        if logits_finite else float("inf")
    )
    loss_error = (
        (reference["loss"].float() - candidate["loss"].float()).abs().item()
        if loss_finite else float("inf")
    )
    # items() 同时给出参数名和梯度；先逐参数归约，再跨参数取最大值。
    grad_error = (
        max(
            max_abs_error(reference["grads"][name], grad)
            for name, grad in candidate["grads"].items()
        )
        if grads_finite else float("inf")
    )

    return {
        "logits_max_abs": logits_error,
        "loss_abs": loss_error,
        "grad_max_abs": grad_error,
        "all_finite": logits_finite and loss_finite and grads_finite,
    }
