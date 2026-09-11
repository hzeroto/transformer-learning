# Transformer Learning

A personal repository for learning Transformer models from first principles.

面向有后端经验、准备学习 AI 基础原理和 Infra 相关知识的学习者：从最小训练闭环到独立实现
Transformer，再验证缓存推理和分析模型执行成本。

当前教学顺序与完成标准见 [能力关卡路线](learning/roadmap.md)，本次调整依据见
[学习计划审查](learning/reviews/2026-09-05-plan-review.md)。知识地图的主题分组不等于授课顺序。

## Goals

- Build an intuitive understanding of attention and Transformer architecture.
- Connect the mathematical formulation to small, runnable implementations.
- Record experiments, notes, and lessons learned along the way.

## Planned structure

- `notes/` — concepts, derivations, and reading notes
- `src/` — implementations and reusable code
- `tests/` — correctness checks
- `examples/` — small runnable experiments
- `exercises/` — learner-completed programming assignments and task instructions

## Learning map

The repository contains a persistent, evidence-driven learning system:

- `learning/map.json` — complete curriculum, dependencies, and mastery criteria
- `learning/roadmap.md` — capability gates, recommended order, implementation boundaries, and AI Infra bridge
- `learning/learner-profile.md` — stable learner background and teaching preferences
- `learning/progress.json` — current capability boundary and evidence for each knowledge point
- `learning/records.jsonl` — append-only learning event log
- `docs/learn_record.md` — generated human-readable progress report
- `docs/index.html` — interactive learning-map frontend
- `scripts/report-learning.mjs` — the only supported progress update entrypoint

Start the local map viewer with:

```bash
npm run serve
```

Report demonstrated mastery with concrete evidence:

```bash
npm run learning:report -- master foundation.batch-matmul \
  --evidence "能把(7,3,4)@(7,4,6)解释为7次(3,4)@(4,6)，并正确得到(7,3,6)"
```

Set the single current learning point with:

```bash
npm run learning:report -- current text-input.token-unit \
  --note "从文字如何变成可计算输入开始补齐背景"
```

Validate the dependency graph and refresh generated frontend data with:

```bash
npm run learning:validate
```

## Teaching context

The ChatGPT project keeps the teaching conversation continuous. The repository only supplies durable context:

- `learning/learner-profile.md` describes the learner's stable background and preferences.
- `learning/progress.json` records the current capability boundary and evidence.
- `learning/map.json` provides the overall path and prerequisite relationships.

Ordinary lesson details stay in the project conversation rather than being duplicated in the repository.

## Project skills

三个仓库级 skill 位于 `.agents/skills/`，共用现有学习地图、画像和进度协议：

| Skill | 职责 | 示例请求 |
|---|---|---|
| [transformer-lesson-design](.agents/skills/transformer-lesson-design/SKILL.md) | 设计可直接学习的讲解、必要图解与理解检查 | `使用 $transformer-lesson-design 设计下一块教学内容` |
| [transformer-exercise](.agents/skills/transformer-exercise/SKILL.md) | 提供接口、TODO、教师测试与运行入口，核心实现留给学习者 | `使用 $transformer-exercise 准备这块的练习框架` |
| [transformer-review](.agents/skills/transformer-review/SKILL.md) | 审查真实代码/回答，按证据验收并同步进度 | `使用 $transformer-review 看看我写的，不改答案` |

普通的“继续推进”“没懂”留在自然课堂对话中，不启动一次新的课程设计。
教学设计不设固定的“为什么学”环节；map 和进度服务于实际教学内容，不替代讲解。
创建课案或练习本身不构成掌握证据。

这些 skill 依赖本仓库的协议与材料，不是脱离仓库的独立教材包。
仓库级自动发现与启动目录有关：从本仓库或其子目录启动时可使用 `.agents/skills/`；
从仓库外的项目镜像聊天时，不能仅凭文件已存在就断言已自动加载。可明确引用上述
`SKILL.md` 路径并要求使用，或另外配置用户级入口；本仓库不自动安装全局副本或软链接。
跨设备同步源文件仍使用仓库，不在 skill 中写死最新进度。
加载规则参见 [OpenAI 官方文档](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills)。
