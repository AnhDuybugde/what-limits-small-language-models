# Task: Code Quality Review

Use this when asked to review code quality, maintainability, notebook reliability, or Pythonic style.

## Review Order

1. Correctness risks, behavioral bugs, and metric errors.
2. Data leakage, prompt tuning on final test data, or train/test contamination.
3. Notebook reliability: resume safety, output overwrites, missing config cells, and Kaggle path assumptions.
4. Hard-to-read control flow or data flow.
5. Unnecessary size, duplication, abstraction, or dependencies.
6. Comment quality, naming, and missing focused smoke checks.

## SLM Experiment Checks

- Confirm final-answer extraction happens before scoring.
- Confirm deterministic decoding settings for main results.
- Confirm JSONL records include status and enough fields to reproduce metrics.
- Confirm failed JSON parsing, empty outputs, truncation, and extraction errors are logged explicitly.
- Confirm figures and CSV summaries are derived from merged records, not hand-entered numbers.

## Style

- Findings first, ordered by severity.
- Use file and line references when available.
- Separate bugs from style preferences.
- Avoid rewriting everything when a narrow fix is enough.
