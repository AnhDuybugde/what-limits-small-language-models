# Working Memory

Operating rules for Codex in this repository.

## Before Work

- At the start of every meaningful work cycle, read this file again.
- Read `long_memory.md` for durable preferences.
- Read `short_memory.md` for current-session context.
- Read the relevant file in `tasks/` when it matches the request.
- For paper, repo, notebook, reproduction, adaptation, novelty, or experiment work, read `rule-implement.md` before planning or coding.
- Treat `.agent/` as the official memory folder.

## Work Cycle

- Cycle start: read memory files, implementation rules, and the matching task guide.
- Grounding: inspect notebooks, data files, configs, outputs, logs, and environment details with non-destructive commands first.
- Plan: separate established project rules, current notebook/data state, Kaggle constraints, and any new adaptation decisions.
- Implementation: keep changes small, runnable, and paired with smoke checks.
- Cycle end: update `short_memory.md` with concise session notes; promote stable decisions to `long_memory.md` only when they will help future sessions.

## Notebook Rules

- Keep notebooks independent so each one can run in a fresh Kaggle session.
- Put a clear config cell at the top of each inference notebook.
- Run small samples first: smoke, then pilot, then full run after prompts and metrics are frozen.
- Use deterministic decoding for main experiments.
- Require `Final answer:` in prompts and extract the final answer before scoring.
- Save JSONL records incrementally and never overwrite useful previous outputs.
- Use `unique_key` to skip completed successful records and support resume after crashes.
- Track latency, tokens, memory, status, and errors for every prediction when feasible.
- Keep merge/evaluation notebooks responsible for combined CSV tables and matplotlib figures.

## Coding Rules

- Prefer small, focused changes that match the existing notebook or script style.
- Keep Python code Pythonic: direct names, shallow control flow, specific exceptions, clear data shapes.
- Avoid code bloat: do not add abstractions, dependencies, or configuration unless they remove real complexity.
- Use comments sparingly. Explain why something exists, not what obvious code does.
- Preserve user changes. Do not revert unrelated edits.

## Memory Rules

- Do not delete `.agent/` files unless the user explicitly requests it.
- Keep `short_memory.md` concise and current.
- Promote stable lessons from short memory to long memory when they are likely to help future sessions.
- Do not store secrets, credentials, private tokens, or sensitive personal data in memory files.
- Do not store raw logs or noisy transient outputs; summarize only what changes future work.
