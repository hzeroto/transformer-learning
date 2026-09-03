# Project guidance

## Repository identity — mandatory preflight

- Resolve the canonical local repository root from the operating environment before accessing project files:
  - On macOS (`uname -s` reports `Darwin`), it is exactly `/Users/bytedance/go/src/github.com/hzeroto/transformer-learning`.
  - On Windows through WSL (`WSL_DISTRO_NAME` is set, or the Linux kernel release contains `microsoft`), it is exactly `/home/hzeroto/playground/transformer-learning`.
- Do not select a repository path from the process working directory. Detect the operating environment first, then use only the exact canonical path mapped above.
- The canonical GitHub repository is `hzeroto/transformer-learning`, with expected `origin` URL `https://github.com/hzeroto/transformer-learning.git` and default branch `main`.
- When the user says “这个项目”, “项目仓库”, “我们的仓库”, or refers to the Transformer learning project, resolve it to the canonical repository for the detected operating environment.
- Before reading, creating, or editing project files, detect the operating environment, select the mapped canonical path, and verify that the Git root exactly matches it. Use that canonical root explicitly as the working directory for project commands.
- The Codex-generated directory under `/Users/bytedance/Documents/Codex/` is not this project repository and must not be used for project artifacts.
- If the operating environment is not mapped above, the selected canonical path is unavailable, its Git root differs, or `origin` points elsewhere, stop and ask the user before continuing.

- This repository is for learning and implementing Transformer models from first principles.
- Prefer small, runnable examples and explain non-obvious mathematical or implementation choices.
- Use Chinese for explanations and documentation unless the user requests another language.
- Keep reusable implementations in `src/`, verification in `tests/`, learning notes in `notes/`, and runnable demonstrations in `examples/`.
- Do not copy or derive code, configuration, data, or documentation from company repositories.

## Learning progress protocol

- Treat `learning/map.json` as the canonical Transformer curriculum and dependency map.
- Treat `learning/progress.json` as the canonical current capability boundary and knowledge-point status.
- Treat `learning/learner-profile.md` as the canonical source for the learner's stable background, goals, and teaching preferences.
- Never edit `learning/progress.json`, `learning/records.jsonl`, `docs/learn_record.md`, or `docs/assets/learning-data.generated.js` directly.
- When the learner demonstrates mastery of a knowledge point, always run:

  ```bash
  npm run learning:report -- master <node-id> --evidence "<specific observable evidence>"
  ```

- A mastery report must cite observable evidence from the learner's explanation, implementation, test, or debugging work. Exposure to an explanation is not mastery.
- Use `verify` when understanding is partial, `relearn` when a concept was introduced before its prerequisites, and `current` for the single active learning point.
- Run `npm run learning:validate` after curriculum edits or if generated state may be stale.
- Before teaching a new concept, inspect prerequisite statuses. Define every new term before using it.

## Teaching context

- The ChatGPT project and its conversation history provide lesson continuity; do not turn this repository into a transcript or per-message session store.
- Before teaching, use `learning/learner-profile.md` for stable preferences and use `learning/progress.json` plus `learning/map.json` to understand the current capability boundary and prerequisites.
- Read recent `learning/records.jsonl` entries only when the evidence behind a status is needed.
- Keep repository updates focused on durable capability evidence and overall progress. Do not record ordinary conversational details.

## Git safety

- Before any remote change or push, detect the operating environment and verify that the repository root exactly matches its mapped canonical path above, the effective Git identity is expected, and the destination owner is exactly `hzeroto`.
- Only push this repository to `https://github.com/hzeroto/transformer-learning.git` or its equivalent SSH URL owned by `hzeroto`.
- Stop and ask the user if the source path, Git identity, or destination remote is unexpected.
