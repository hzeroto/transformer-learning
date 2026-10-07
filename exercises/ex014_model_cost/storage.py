"""第二步：先预测各类数据的字节数，再读取真实张量核对。题面见 STORAGE.md。"""

import torch


def estimate_storage_bytes(
    *, parameter_count: int, B: int, n: int, S: int, C: int,
    Hq: int, Hkv: int, G: int, layers: int, N: int, element_bytes: int = 4,
) -> dict[str, int]:
    """预测一次缓存推理调用涉及的几类数据量，单位均为字节。

    尺寸含义沿用 cost.py：B 批量，n 本次新位置，S 追加后总位置，C 主干宽，
    Hq/Hkv 为 Q/KV 头数，G 为 FFN 中间宽，layers 为层数，N 为词表大小。
    parameter_count 是调用方提供的全部可训练参数元素数，不需要重新推导。
    所有被估算的浮点张量每个元素均占 element_bytes 字节。
    输入保证合法、为正整数；S>=n，C 可被 Hq 整除，Hq 可被 Hkv 整除。

    返回恰好以下五个键，值为 Python int：
    - weights：全部模型参数的数据量。
    - kv：追加后的全部层 K 和 V 缓存；每层每份为 (B,S,Hkv*(C//Hq))。
    - score_one_layer：单层一份打分张量 (B,Hq,n,S)，按稠密 shape 计数。
    - ffn_one_layer：单层一份 FFN 中间张量 (B,n,G)。
    - logits：本次全部新位置的输出 (B,n,N)。

    本函数只用整数运算，不创建模型或张量。这里没有 total/peak 键：
    这五类数据不是对同时存活对象的完整枚举，不能直接代表推理峰值。
    """
    weights = parameter_count * element_bytes
    kv = 2 * layers * B * S * Hkv * C // Hq * element_bytes
    score_one_layer = B * Hq * n * S * element_bytes
    ffn_one_layer = B * n * G * element_bytes
    logits = B * n * N * element_bytes
    return {
        "weights": weights,
        "kv": kv,
        "score_one_layer": score_one_layer,
        "ffn_one_layer": ffn_one_layer,
        "logits": logits,
    }


def measure_storage_bytes(tensors: list[torch.Tensor]) -> dict[str, int]:
    """测量给定张量列表的逻辑数据量和去重后的底层存储量。

    返回恰好两个键，值为 Python int：
    - logical_bytes：逐个列表项按其逻辑元素数统计，相同对象出现两次也计两次。
    - storage_bytes：这些张量依赖的完整底层 storage，各块只计一次。
      切片可能只使用一部分元素，但仍依赖完整 storage。

    输入限于 CPU 普通稠密张量，可混合 float32、float64、int64、bool；
    支持转置、切片、expand 等非连续张量，每个张量至少有一个元素。
    列表可以为空，此时两项均为 0。所有对象在调用期间存活。
    本题可用 untyped_storage().data_ptr() 识别共享存储；注意它不同于
    tensor.data_ptr()，后者对从中间开始的切片会发生偏移。

    只读：不修改列表、张量内容、布局、requires_grad 或已有 .grad；
    不为统计而 clone/contiguous。结果不是整个进程占用，也不是执行峰值。
    所需 API 及共享存储示例见 STORAGE.md。
    """
    logical_bytes = sum(t.numel() * t.element_size() for t in tensors)


    storage_bytes = 0
    seen = set()
    for t in tensors:
        ptr = t.untyped_storage().data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            storage_bytes += t.untyped_storage().nbytes()
    return {
        "logical_bytes": logical_bytes,
        "storage_bytes": storage_bytes,
    }
