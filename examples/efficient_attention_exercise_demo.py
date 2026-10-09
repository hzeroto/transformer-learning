"""调用 ex018 学习者实现的 CPU 演示；不包含 Attention 或缓存参考答案。"""

import torch

from exercises.ex013_llama_style.model import LlamaLM
from exercises.ex018_efficient_attention import attention, model


def run_demo():
    generator = torch.Generator().manual_seed(1803)
    q = torch.randn(2, 2, 9, 4, generator=generator, dtype=torch.float64)
    k = torch.randn(2, 2, 9, 4, generator=generator, dtype=torch.float64)
    v = torch.randn(2, 2, 9, 3, generator=generator, dtype=torch.float64)
    positions = torch.arange(7, 16)
    valid = torch.ones(2, 9, dtype=torch.bool)
    with torch.no_grad():
        local = attention.local_attention(q, k, v, positions, positions, valid, 3)
        online = attention.online_attention(
            q, k, v, positions, positions, valid, 3, query_block=2, key_block=4,
        )
    torch.testing.assert_close(local, online, atol=1e-10, rtol=1e-8)
    print(f"局部/在线输出最大绝对差：{(local - online).abs().max().item():.3e}")

    with torch.random.fork_rng():
        torch.manual_seed(1811)
        network = LlamaLM(17, 24, 6, 2, 31, 2, 64, dtype=torch.float64)
    ids = torch.randint(0, 17, (2, 11), generator=generator)
    state = model.new_window_state(network, 3, start_position=7)
    with torch.no_grad():
        full = model.window_forward(network, ids, 3, start_position=7)
        pieces, offset = [], 0
        for size in (5, 2, 1, 3):
            pieces.append(model.window_step(network, ids[:, offset:offset + size], state))
            offset += size
        incremental = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full, incremental, atol=1e-10, rtol=1e-8)
    print(f"全量/分段 logits 最大绝对差：{(full - incremental).abs().max().item():.3e}")
    print(f"下一绝对位置：{state.next_position}；各层缓存长度："
          f"{[entry.k.shape[-2] for entry in state.layers]}")
    storage_bytes = sum(
        tensor.untyped_storage().nbytes()
        for entry in state.layers for tensor in (entry.k, entry.v)
    )
    print(f"持久 K/V 底层存储合计：{storage_bytes} 字节（不含位置元数据、工作区及参数）")
    print("示例对齐完成；完整验收请运行两组独立参照测试。")


def main():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run_demo()
    except NotImplementedError as error:
        print(f"尚未完成：{error}")
        print("请填写 exercises/ex018_efficient_attention/ 中的核心函数。")
        return 1
    finally:
        torch.set_num_threads(previous_threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
