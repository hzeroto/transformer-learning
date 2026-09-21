"""教师小例子：共享 K/V、错误的平均压缩、视图存储与字节账本。

只调用已有 MHA 或直接算两维小例子；不实现通用 GQA、不修改已有模型。
运行：.venv/bin/python -B -m examples.gqa_mechanism_probe
"""
import torch

from exercises.ex007_multi_head_attention.attention import multi_head_attention
from exercises.ex011_kv_cache.cache import kv_cache_bytes


def main():
    print(f"device=cpu torch={torch.__version__}")
    dtype = torch.float64
    k = torch.eye(2, dtype=dtype)
    v = torch.tensor([[10., 0.], [0., 20.]], dtype=dtype)
    q = torch.tensor([[2., 0.], [0., 2.]], dtype=dtype)
    weights = torch.softmax(q @ k.T / (2 ** .5), dim=-1)
    result = weights @ v
    print("shared K/V, different Q")
    print("weights:", weights.tolist())
    print("head outputs:", result.tolist())

    # 两个查询头显式共享同一 K/V，用已验收的 MHA 作数值参照。
    # cat(..., dim=-1) 按特征拼接两个头，形成原接口的 (B,T,H*D)。
    q_wide = q.reshape(1, 1, 4)
    k_wide = torch.cat([k, k], dim=-1).unsqueeze(0)
    v_wide = torch.cat([v, v], dim=-1).unsqueeze(0)
    out, actual_weights = multi_head_attention(
        q_wide, k_wide, v_wide, torch.eye(4, dtype=dtype), 2
    )
    torch.testing.assert_close(out, result.reshape(1, 1, 4), rtol=0, atol=1e-12)
    torch.testing.assert_close(
        actual_weights, weights.reshape(1, 2, 1, 2), rtol=0, atol=1e-12
    )
    print("existing MHA reference agrees")

    print("same Q/V, changing K")
    one_q = q[:1]
    for label, key in [
        ("old head 0", k),
        ("old head 1", k.flip(0)),
        ("mean keys", (k + k.flip(0)) / 2),
    ]:
        print(label, (torch.softmax(one_q @ key.T / (2 ** .5), dim=-1) @ v).tolist())

    print("storage example: shape / stride / logical bytes / storage bytes / shares K")
    compact = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    expanded = compact.unsqueeze(2).expand(1, 2, 2, 3, 4)
    flattened = expanded.reshape(1, 4, 3, 4)
    repeated = compact.repeat_interleave(2, dim=1)
    for label, tensor in [
        ("compact", compact), ("expanded view", expanded),
        ("flattened", flattened), ("repeated", repeated),
    ]:
        print(
            label, tuple(tensor.shape), tensor.stride(),
            tensor.numel() * tensor.element_size(),
            tensor.untyped_storage().nbytes(),
            tensor.untyped_storage().data_ptr() == compact.untyped_storage().data_ptr(),
        )
    torch.testing.assert_close(flattened, repeated, rtol=0, atol=0)

    print("Hkv / K+V params per layer / all four projection params per layer / all-layer KV MiB / score elements")
    C, D, Hq = 512, 64, 8
    for hkv in [8, 2, 1]:
        kv_params = 2 * C * hkv * D
        params = 2 * C * C + kv_params
        size = kv_cache_bytes(12, 4, 1024, hkv, D, 4)
        print(hkv, kv_params, params, size / 2**20, 4 * Hq * 1 * 1024)
        # 小尺寸真实分配，分别检查三种头数，避免为示例分配数百 MiB。
        tensors = [
            torch.empty(2, hkv, 5, D, dtype=torch.float32)
            for _ in range(3 * 2)
        ]
        assert sum(x.numel() * x.element_size() for x in tensors) == kv_cache_bytes(
            3, 2, 5, hkv, D, 4
        )
    print("small real KV allocations agree with byte formula")


if __name__ == "__main__":
    main()
