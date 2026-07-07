# Task: Python Refactor

Use this when refactoring Python code in scripts or notebooks.

## Rules

1. Preserve behavior unless the user asks for a behavior change.
2. Read callers, notebook cells, or usage examples before changing public functions.
3. Make the smallest useful refactor.
4. Prefer functions over classes unless state or polymorphism is genuinely useful.
5. Keep notebook cells easy to run independently and easy to debug on Kaggle.
6. Avoid adding dependencies unless they clearly reduce real complexity.
7. Validate with focused smoke checks, import checks, compile checks, or small notebook runs when available.

## Notebook-Friendly Outcomes

- Clear config at the top.
- Short helper functions with explicit inputs and outputs.
- Stable JSONL writing and resume behavior.
- Metric code that is readable enough to audit.
- No hidden global state that makes rerunning cells unsafe.

## Good Outcomes

- Shorter or clearer code.
- Fewer branches or repeated blocks.
- Better names.
- Comments reduced or improved.
- No new unnecessary dependencies.
