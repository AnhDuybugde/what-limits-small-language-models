# Long Memory

Durable preferences and project-level decisions for future Codex sessions.

## User Preferences

- Write Python and notebook code that is Pythonic, readable, compact, and easy to debug.
- Prefer direct data flow, clear names, and small helper functions over framework-like layers.
- Use short comments only when they explain intent, constraints, or non-obvious experiment choices.
- Preserve `.agent/` files. Do not delete them unless the user explicitly asks.
- Treat `.agent/` as the official agent memory folder for this repository.

## Project Notes

- Project name: What Limits Small Language Models?
- The source of truth for experiment implementation is `.agent/rule-implement.md`.
- The project uses Kaggle-first, inference-only notebooks to analyze limits of small language models.
- Do not fine-tune, train, or introduce a complex `.py` codebase unless the user changes scope.
- Each notebook should be independently runnable on Kaggle and start with an editable config cell.
- Prefer one generic inference runner (`02-run-experiment.ipynb`) over separate near-duplicate notebooks for knowledge, reasoning, context, and robustness.
- Use deterministic decoding for main results: `temperature = 0.0` and `top_p = 1.0`.
- Require model outputs to support `Final answer:` extraction before metric computation.
- Always evaluate extracted final answers, not raw chain-of-thought text.
- Save prediction records as JSONL with resume-safe unique keys.
- Follow the run order: smoke test, pilot, freeze prompts/extraction/metrics, then full run.
- Avoid tuning prompts on the final test set. Use train/validation subsets for debugging.

## Memory Practice

- Move only distilled, durable facts from `short_memory.md` into this file.
- Start meaningful work by rereading `.agent/working_memory.md`, `.agent/long_memory.md`, `.agent/short_memory.md`, and the relevant task guide.
- For paper, repo, notebook, reproduction, adaptation, or experiment implementation work, read `.agent/rule-implement.md` before planning or coding.
