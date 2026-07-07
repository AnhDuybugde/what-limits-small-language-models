# Task: Implement Paper Or Experiment Repo

Use this for paper reproduction, repo re-implementation, notebook adaptation, SLM limit experiments, method checks, and novelty preparation.

## Required Read

- Read `.agent/rule-implement.md` before planning or coding.
- Inspect notebooks, data files, configs, outputs, and logs with non-destructive commands first.

## Default Project Path

- Build Kaggle-first notebooks, not a large script-based codebase.
- Keep every inference notebook independently runnable with a config cell at the top.
- Implement inference-only experiments. Do not train or fine-tune unless the user changes scope.
- Save prediction records as JSONL and make runs resume-safe with `unique_key`.
- Extract final answers before computing metrics.
- Use deterministic decoding and log efficiency details when feasible.

## Required Separation

- Project rule: what `.agent/rule-implement.md` already fixes for this repository.
- Current state: what the notebooks, data, outputs, and logs actually contain now.
- Adaptation: what must change for Kaggle, local paths, model size, shards, or hardware.
- Novelty: any new idea not already present in the project rules. Mark it explicitly.

## Implementation Areas

- Data preparation: normalize datasets into shared JSONL/CSV records.
- Inference notebooks: run knowledge, reasoning, context, and robustness conditions by model, condition, and shard.
- Evaluation: merge JSONL records, remove duplicates, compute metrics, and export CSV summaries.
- Plotting: generate the required matplotlib figures without adding unnecessary plotting dependencies.
- Error analysis: sample wrong predictions, assign preliminary error types, and preserve manual notes.

## Response Shape

- Start with a short summary.
- Give phase-based notebook or code steps.
- State sanity checks, expected outputs, and known constraints.
- Keep code compact, readable, and easy to debug.

## Cycle End

- Update `.agent/short_memory.md` with concise notes when a meaningful phase finishes.
- Promote durable decisions to `.agent/long_memory.md` only when they are stable and useful.
