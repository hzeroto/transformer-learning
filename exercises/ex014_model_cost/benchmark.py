"""固定工作量计时：保留 CPU 练习入口，增加 CUDA/MPS 的设备同步边界。"""

from collections.abc import Callable
from statistics import median, quantiles
from time import perf_counter

import torch

from exercises.ex013_llama_style.model import LlamaLM, llama_model_step, new_caches


def _synchronize_device(device: torch.device) -> None:
    """等待输入所在设备完成计算；CPU 无需设备同步。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type != "cpu":
        raise ValueError(f"暂不支持 {device.type} 的计时同步")


def measure_cpu_step(
    model: LlamaLM,
    input_ids: torch.Tensor,
    *,
    prefix_ids: torch.Tensor | None = None,
    warmup: int = 10,
    repeats: int = 50,
    clock: Callable[[], float] = perf_counter,
) -> list[float]:
    """重复测量同一次模型调用，返回 repeats 个按采集顺序排列的耗时，单位秒。

    沿用原练习函数名，按 input_ids.device 处理 CPU/CUDA/MPS 同步。
    model：已构造并处于 eval 模式的模型，参数与输入在同一设备。
    input_ids：int64 (B,n)，合法 ID，无 PAD，B,n>=1。
    prefix_ids=None：测 prefill，每次从空缓存处理 input_ids。
    prefix_ids 非 None：与 input_ids 同设备的 int64 (B,t)，t>=1；测单步 decode，n=1。
      每次都先用相同 prefix_ids 建好长度 t 的历史，再测追加 input_ids。
    调用方保证总长度不超 model.L，warmup>=0，repeats>=2，不考通用参数校验。

    本步统一采用“每次重新 prefill 历史”的恢复方法：每次热身、每次正式样本
    都用 new_caches(model) 创建新容器，若有 prefix_ids 就先调用
    llama_model_step(model, prefix_ids, caches)。这部分放在计时外。
    不复用已经追加过新位置的缓存，也不在本步改用保存历史引用的优化。

    然后用 llama_model_step(model, input_ids, caches) 执行被测调用：
    - 先完成 warmup 次相同工作量，不读 clock、不收集这些耗时。
    - 正式采集 repeats 次，每次只在被测调用前后各读一次 clock()，记录差值。
    - 计时包含一次 step 的完整前向与 KV 追加，不含准备历史、统计、打印。
      logits 保留到结束读数之后，再释放输出。
    - GPU 在准备历史后、start 之前同步，排除旧任务；目标 step 后、end 之前
      再同步，等待本次计算完成。CPU 不调用 CUDA/MPS 同步 API。

    所有模型调用（包括准备历史和热身）都在 torch.no_grad() 下运行。
    不修改模型模式、线程配置、参数、已有 .grad 或输入；返回后恢复调用者
    原有的求导开关。调用方负责固定并记录 CPU 线程数。
    clock 是无参数、返回秒数的函数；测试注入确定性时钟，实际默认 perf_counter。
    必须调用传入的 clock，不另外换计时器。测量的是含提交与同步等待的墙钟延迟。
    当前 ex013 的真实验证仍为 CPU float32/float64；GPU 使用前还需迁移模型内部
    创建的 mask/索引等张量并核对数值。这里增加同步不等于完成模型的 GPU 迁移。
    """
    device = input_ids.device
    with torch.no_grad():
        # 1. 热身
        for _ in range(warmup):
            caches = new_caches(model)
            if prefix_ids is not None:
                llama_model_step(model, prefix_ids, caches)
            llama_model_step(model, input_ids, caches)
        # 2. 正式采样
        samples_s = []
        for _ in range(repeats):
            caches = new_caches(model)
            if prefix_ids is not None:
                llama_model_step(model, prefix_ids, caches)

            _synchronize_device(device)  # 历史准备、热身等旧工作完成后才开始计时。
            start = clock()
            logits = llama_model_step(model, input_ids, caches)
            _synchronize_device(device)  # 本次计算完成后才结束计时。
            end = clock()
            samples_s.append(end - start)
            del logits
        return samples_s


def summarize_samples(samples_s: list[float], *, tokens_per_call: int) -> dict[str, float]:
    """汇总同一工作量的计时样本，不修改输入列表。

    samples_s：至少两个有限且严格为正的耗时，单位秒，只包含正式样本。
    tokens_per_call：每次被测调用处理的新 token 总数，即 B*n，保证为正整数。
    返回恰好四项（Python float）：
    - median_ms：耗时中位数，换成毫秒。
    - p25_ms、p75_ms：耗时第 25、75 百分位，换成毫秒。
      统一采用 statistics.quantiles(samples_s, n=4, method="inclusive") 的口径。
    - tokens_per_second：所有正式样本处理的新 token 总数 / 正式样本总秒数。

    prefill 的 token 数是输入位置数，decode 的 token 数是本次新增位置数；
    历史准备未计时，也不能把历史 token 数加入分子。不包含服务排队等耗时。
    允许使用文件顶部导入的 median、quantiles，不要求手写分位数算法。
    """
    median_ms = median(samples_s) * 1000
    tokens_per_second = len(samples_s) * tokens_per_call / sum(samples_s)
    p25_ms = quantiles(samples_s, n=4, method="inclusive")[0] * 1000
    p75_ms = quantiles(samples_s, n=4, method="inclusive")[2] * 1000
    return {
        "median_ms": median_ms,
        "p25_ms": p25_ms,
        "p75_ms": p75_ms,
        "tokens_per_second": tokens_per_second,
    }
