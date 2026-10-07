"""训练演示：调用完整 Mini-BERT，运行预训练—恢复—分类微调闭环。

运行：.venv/bin/python -B -m exercises.ex016_mini_bert.demo
训练与评估均经过 model.py / training.py 的实际入口。
"""

import argparse
import copy

import torch

from exercises.ex016_mini_bert.data import VOCAB_SIZE, make_mlm_batch
from exercises.ex016_mini_bert.fixtures import make_toy_case
from exercises.ex016_mini_bert.model import MiniBert
from exercises.ex016_mini_bert.training import classification_loss, pretraining_loss


def measure_pretraining(model, batch, labels):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            total, mlm, nsp = pretraining_loss(model, batch, labels)
            output = model(batch["input_ids"], batch["segment_ids"], batch["input_valid"])
            predicted = output["mlm_logits"].argmax(-1)
            mask = batch["target_valid"]
            mlm_acc = (predicted[mask] == batch["targets"][mask]).float().mean().item()
            nsp_acc = (output["nsp_logits"].argmax(-1) == labels).float().mean().item()
            return total.item(), mlm.item(), nsp.item(), mlm_acc, nsp_acc
    finally:
        model.train(was_training)


def run_demo(steps=300, finetune_steps=160):
    torch.manual_seed(1601)
    torch.set_num_threads(1)
    case = make_toy_case()
    batch = make_mlm_batch(
        case["clean_ids"], case["input_valid"], case["segment_ids"],
        case["selected"], case["replacement_kind"], case["random_ids"],
    )
    model = MiniBert(VOCAB_SIZE, 16, 4, 32, 2, 16, num_classes=2, dtype=torch.float32)
    initial = measure_pretraining(model, batch, case["nsp_labels"])
    print(f"initial: total={initial[0]:.6f}, MLM={initial[1]:.6f}, NSP={initial[2]:.6f}")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = pretraining_loss(model, batch, case["nsp_labels"])
        total.backward()
        optimizer.step()
    final = measure_pretraining(model, batch, case["nsp_labels"])
    print(f"pretrained: total={final[0]:.6f}, MLM={final[1]:.6f}, NSP={final[2]:.6f}")
    print(f"training-set accuracy: MLM={final[3]:.3f}, NSP={final[4]:.3f}")
    assert final[0] < 0.08 and final[3] == 1.0 and final[4] == 1.0, "先排查目标和前向，不靠盲目加迭代过关"

    # 内存存档只验证参数和配置恢复，不重复要求实现磁盘文件工程。
    checkpoint = copy.deepcopy(model.state_dict())
    restored = MiniBert(**model.config, dtype=model.token_table.dtype)
    restored.load_state_dict(checkpoint)
    model.eval()
    restored.eval()
    with torch.no_grad():
        before = model(batch["input_ids"], batch["segment_ids"], batch["input_valid"])
        after = restored(batch["input_ids"], batch["segment_ids"], batch["input_valid"])
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    print("checkpoint: four outputs identical after restoring parameters/config")

    # 分类阶段读取未改写的完整输入；标签 j%2 不是 NSP 的相邻/不相邻标签。
    token_table_before = restored.token_table.detach().clone()
    restored.train()
    optimizer = torch.optim.Adam(restored.parameters(), lr=0.005)
    for _ in range(finetune_steps):
        optimizer.zero_grad(set_to_none=True)
        output = restored(case["clean_ids"], case["segment_ids"], case["input_valid"])
        loss = classification_loss(output["class_logits"], case["class_labels"])
        loss.backward()
        optimizer.step()
    restored.eval()
    with torch.no_grad():
        output = restored(case["clean_ids"], case["segment_ids"], case["input_valid"])
        cls_loss = classification_loss(output["class_logits"], case["class_labels"]).item()
        cls_acc = (output["class_logits"].argmax(-1) == case["class_labels"]).float().mean().item()
    assert cls_loss < 0.05 and cls_acc == 1.0, "分类闭环未拟合当前固定小批次"
    assert not torch.equal(restored.token_table.detach(), token_table_before), "微调应更新主干而非只更新头"
    print(f"fine-tuned training-set classification: loss={cls_loss:.6f}, accuracy={cls_acc:.3f}")
    print("This checks wiring and tiny-data learning, not natural-language ability or a pretraining benefit.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--finetune-steps", type=int, default=160)
    args = parser.parse_args()
    run_demo(args.steps, args.finetune_steps)


if __name__ == "__main__":
    main()
