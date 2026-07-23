# Prompt Templates

This document contains the exact prompt templates used for all 15 diagnostic conditions across the four axes (knowledge, reasoning, context, and prompt sensitivity) in *What Limits Small Language Models? A Diagnostic Study of Qwen2.5-Instruct*.

Curly braces `{}` denote fields filled in from each dataset example.

## Knowledge axis (TriviaQA)

**Closed (no context):**
```
Answer with a short phrase only. Do not explain.
Question: {question}
Final answer:
```

**Oracle:**
```
Answer with a short phrase only. Do not explain.
Use the context when possible.
Context: {reference_context}
Question: {question}
Final answer:
```

**Additional context:**
```
Answer with a short phrase only. Do not explain.
Some context may be irrelevant.
Context: {reference_context} {additional_reference_sentences}
Question: {question}
Final answer:
```

## Reasoning axis (GSM8K)

**Direct:**
```
Solve the math problem. Return only one number.
Do not explain.
Problem: {question}
Final answer:
```

**Chain-of-thought (CoT):**
```
Solve the math problem step by step.
End with "Final answer: <number>".
Problem: {question}
Final answer:
```

**Oracle-step:**
```
Problem: {question}
Helpful intermediate step:
The final numeric answer should be {gold_answer}.
Continue and give only the final numeric answer.
Final answer:
```

## Context QA axis (HotpotQA-Distractor)

**Context QA (none / retrieved / oracle):**
```
Answer with a short phrase only. Do not explain.
Use the context when possible.
Context: {selected_context}
Question: {question}
Final answer:
```

**Context additional / oracle-end:**
```
Answer with a short phrase only. Do not explain.
Some context may be irrelevant.
Context: {selected_context}
Question: {question}
Final answer:
```

## Prompt sensitivity axis (GSM8K-derived)

**Original:**
```
Solve the math problem. Return only one number.
Do not explain.
Problem: {question}
Final answer:
```

**Paraphrase:**
```
Find the final result for this word problem.
Reply with just the number.
Problem: {question}
Final answer:
```

**Typo:**
```
Solve the math problem. Return only one number.
Do not explain.
Problem: {typo_question}
Final answer:
```

**JSON format:**
```
Return JSON only in this exact schema:
{"answer": "<number>"}
Problem: {question}
```