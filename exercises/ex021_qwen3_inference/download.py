"""显式下载固定 revision；--help / --dry-run 不联网，不下载权重。"""
import argparse
import json

from .support import DEFAULT_MODEL_DIR, MODEL_ID, REVISION


FILES = [
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "model.safetensors", "LICENSE",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--dry-run", action="store_true", help="只打印下载计划")
    args = parser.parse_args()
    plan = dict(repo_id=MODEL_ID, revision=REVISION, local_dir=args.model_dir, allow_patterns=FILES)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    from huggingface_hub import snapshot_download

    print(f"下载 {MODEL_ID}@{REVISION}，权重约 1.50 GB；仅此命令会联网。")
    print(snapshot_download(**plan))


if __name__ == "__main__":
    main()
