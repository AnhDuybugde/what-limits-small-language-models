# Task: Read And Analyze Files

Use this when asked to inspect, understand, or explain files.

## Steps

1. Identify the smallest set of files needed for the question.
2. Prefer structural tools for code relationships when available.
3. Use literal search for exact strings, comments, messages, config keys, and notebook text.
4. Inspect notebooks, JSONL/CSV outputs, configs, experiment logs, and figures as first-class project artifacts.
5. Summarize behavior in terms of inputs, outputs, side effects, metrics, and dependencies.
6. Mention uncertainty when a file depends on code, data, or outputs that were not inspected.

## SLM Experiment Focus

- Track which model, axis, dataset, condition, split, and shard a file belongs to.
- Check whether outputs are raw predictions, extracted answers, merged records, summaries, plots, or error samples.
- Distinguish smoke/pilot artifacts from main experiment results.

## Output

- Lead with the answer.
- Include file references when useful.
- Keep explanations concise and grounded in the inspected files.
