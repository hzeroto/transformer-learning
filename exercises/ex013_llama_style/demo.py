"""教师运行器：调用你的模型，不包含六处待实现数学的答案。

复用 ex010 的训练与全量生成；仅适配新模型的缓存入口和重建类型。
所有训练与检查在内存中进行，不覆盖旧 checkpoint、不自动上报进度。
"""
import argparse
import io

import torch

from exercises.ex010_train_mini_gpt.dataset import all_sequences, EOS_ID
from exercises.ex010_train_mini_gpt.training import (
    evaluate, train_model, greedy_generate, save_checkpoint,
)
from exercises.ex013_llama_style import model as learner


CONFIG = dict(
    vocab_size=11, C=24, num_query_heads=6, num_kv_heads=2,
    ffn_hidden=64, n_layer=2, max_positions=128, eps=1e-5, rope_theta=10000.0,
)


def cached_greedy(model, prefix, max_new_tokens=4, eos_id=EOS_ID):
    """已学生成循环的教师适配，限定单请求、无 PAD、新建缓存。

    不重做旧练习的续用/EOS综合契约；返回前缀加新 ID，不返回缓存。
    最后一个生成 ID 若不再用于预测，就无需再送入模型。
    """
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            result = prefix.clone()
            if max_new_tokens == 0 or result[0, -1].item() == eos_id:
                return result
            caches = learner.new_caches(model)
            logits = learner.llama_model_step(model, result, caches)
            for index in range(max_new_tokens):
                next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
                result = torch.cat((result, next_id), dim=1)
                if next_id.item() == eos_id or index + 1 == max_new_tokens:
                    break
                logits = learner.llama_model_step(model, next_id, caches)
            return result
    finally:
        model.train(was_training)


def checkpoint_roundtrip(model, optimizer, config):
    """内存存档：沿用三字段格式，但显式重建 LlamaLM，不调用旧加载器。"""
    stream = io.BytesIO()
    save_checkpoint(stream, model, optimizer, config)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    restored = learner.LlamaLM(**state["config"], dtype=model.token_table.dtype)
    restored.load_state_dict(state["model"])
    restored.eval()
    restored_optimizer = torch.optim.Adam(restored.parameters())
    restored_optimizer.load_state_dict(state["opt"])
    return restored, restored_optimizer


def run_demo(steps=160):
    """返回实测摘要；steps 是可复现实验配置，不是课程进度判据。"""
    if steps <= 0:
        raise ValueError("steps 必须大于 0")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(1301)
        model = learner.LlamaLM(**CONFIG, dtype=torch.float32)
        model.eval()
        ids = torch.tensor([[1, 4, 6, 7, 8, 2], [1, 3, 8, 6, 9, 2]])
        with torch.no_grad():
            full = model(ids, torch.ones_like(ids, dtype=torch.bool))
            caches, parts, start = learner.new_caches(model), [], 0
            for size in (3, 2, 1):
                parts.append(learner.llama_model_step(model, ids[:, start:start + size], caches))
                start += size
            chunked = torch.cat(parts, dim=1)
        torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-6)
        parameter_count = sum(p.numel() for p in model.parameters())
        kv_bytes = sum(t.numel() * t.element_size() for c in caches for t in (c.k, c.v))
        storage_sizes = {
            t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
            for c in caches for t in (c.k, c.v)
        }
        storage_bytes = sum(storage_sizes.values())
        assert parameter_count == 12936, "参数账本与本题配置不符"
        assert kv_bytes == storage_bytes == 1536, "检查 KV 宽度、布局与实际存储"
        max_error = (chunked - full).abs().max().item()
        print(f"full/chunked logits max abs diff: {max_error:.8g}")
        print(f"parameters: {parameter_count}; KV logical/storage bytes: {kv_bytes}/{storage_bytes}")
        # 参数更新后缓存必须作废；本次训练不复用上面的请求状态。
        del caches, parts, chunked, full

        rows = all_sequences()
        train_ids = torch.tensor([rows[i] for i in (2, 3, 16, 17, 84, 85, 214, 215)])
        initial_loss, _ = evaluate(model, train_ids)
        opt = train_model(model, train_ids, steps=steps, lr=1e-2, batch_size=8, seed=17)
        final_loss, accuracy = evaluate(model, train_ids)
        print(f"tiny batch loss: {initial_loss:.6f} -> {final_loss:.6f}; sequence accuracy: {accuracy:.1%}")
        assert final_loss < 0.05 and accuracy == 1.0, "小批次尚未拟合，先检查梯度/连接与运行配置"
        for row in train_ids:
            prefix = row[:5].unsqueeze(0)
            whole = greedy_generate(model, prefix, max_new_tokens=4)
            incremental = cached_greedy(model, prefix, max_new_tokens=4)
            assert torch.equal(whole, incremental), "全量/缓存生成不同；用完整 logits 定位"
            assert torch.equal(whole, row.unsqueeze(0)), "没有生成该已训练序列的完整输出"

        restored, restored_opt = checkpoint_roundtrip(model, opt, CONFIG)
        with torch.no_grad():
            valid = torch.ones_like(ids, dtype=torch.bool)
            torch.testing.assert_close(restored(ids, valid), model(ids, valid), rtol=0, atol=0)
        assert len(restored_opt.state) == len(opt.state) > 0, "Adam 状态未正确恢复"
        for original_parameter, restored_parameter in zip(model.parameters(), restored.parameters()):
            original_state, restored_state = opt.state[original_parameter], restored_opt.state[restored_parameter]
            for name in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(original_state[name], restored_state[name], rtol=0, atol=0)
        print("8 training prefixes: full/cache generation and expected sequence matched")
        print("checkpoint: logits and Adam state matched; no checkpoint file written")
        print("This verifies this tiny training task, not generalization or GPU performance.")
        return dict(initial_loss=initial_loss, final_loss=final_loss, accuracy=accuracy,
                    parameter_count=parameter_count, kv_bytes=kv_bytes, max_error=max_error)
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args()
    try:
        run_demo(args.steps)
    except NotImplementedError as exc:
        raise SystemExit(f"本练习尚未完成：{exc}。先填写 components.py / model.py 的六处 TODO。") from exc


if __name__ == "__main__":
    main()
