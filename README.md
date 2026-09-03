# Transformer Learning

A personal repository for learning Transformer models from first principles.

## Goals

- Build an intuitive understanding of attention and Transformer architecture.
- Connect the mathematical formulation to small, runnable implementations.
- Record experiments, notes, and lessons learned along the way.

## Planned structure

- `notes/` — concepts, derivations, and reading notes
- `src/` — implementations and reusable code
- `tests/` — correctness checks
- `examples/` — small runnable experiments

## Learning map

The repository contains a persistent, evidence-driven learning system:

- `learning/map.json` — complete curriculum, dependencies, and mastery criteria
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
