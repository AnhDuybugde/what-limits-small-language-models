# Short Memory

Session-local notes that may be promoted into `long_memory.md` later.

## Current Session

- User is starting a new project named "What Limits Small Language Models?"
- User already updated `.agent/rule-implement.md`; treat it as the source of truth for implementation rules.
- `.agent/` is now the correct official memory folder name.
- The project direction is Kaggle-first, inference-only experiments for analyzing limits of small language models.
- Planned notebooks are:
  - `00_prepare_data.ipynb`
  - `01_smoke_test.ipynb`
  - `02_run_experiment.ipynb`
  - `06_merge_evaluate_plot.ipynb`
  - `07_error_analysis.ipynb`
- Planned outputs include JSONL prediction records, merged CSV summaries, matplotlib figures, and error-analysis files.
- Main experiment axes are knowledge, reasoning, context utilization, robustness, and efficiency.
- Main model family starts with Qwen2.5 Instruct sizes: 1.5B, 3B, and 7B.
- Main datasets start with GSM8K, HotpotQA distractor, and TriviaQA or Natural Questions.

## Current Roadmap

1. Code `00_prepare_data.ipynb`.
2. Code `01_smoke_test.ipynb`.
3. Code `06_merge_evaluate_plot.ipynb`.
4. Run 10 local samples.
5. Fix extraction, metric, and output issues found in local smoke testing.
6. Code and run `02_run_experiment.ipynb` as one generic runner for knowledge, reasoning, context, and robustness.
7. Run 100-sample pilot jobs on Kaggle.
8. Freeze code, prompts, extraction, and metrics.
9. Run full jobs in parallel.
10. Merge results, plot figures, and complete error analysis.

## Latest Progress

- Created `00_prepare_data.ipynb`, `01_smoke_test.ipynb`, and `06_merge_evaluate_plot.ipynb`.
- `00_prepare_data.ipynb` writes shared JSONL/CSV files and uses fast fallback samples locally when HuggingFace access is unavailable.
- `01_smoke_test.ipynb` supports resume-safe JSONL records and defaults to `BACKEND = "mock"` for local extraction/metric/output checks; change to `BACKEND = "transformers"` on Kaggle.
- `06_merge_evaluate_plot.ipynb` merges records, deduplicates by `unique_key`, exports summary CSVs, and saves matplotlib figures with the headless `Agg` backend.
- Local smoke run completed with fallback data: 3 reasoning records, all success, mean EM/F1/accuracy = 1.0.
- Local outputs created under `data/`, `outputs/`, and `merged_outputs/`.
- Current repo notebook filenames use hyphens: `00-prepare-data.ipynb`, `01-smoke-test.ipynb`, and `06-merge-evaluate-plot.ipynb`.
- Updated the three notebooks to be local-first: `00` defaults to `slm_limits_data/` and preserves existing data files, `01` runs 10 samples for each of six required smoke conditions, and `06` defaults to merging only `records_smoke_mock_six_conditions_shard*.jsonl`.
- Latest local verification wrote 60 smoke records across `reasoning_direct`, `reasoning_cot`, `context_none`, `context_oracle`, `knowledge_closed`, and `knowledge_oracle`.
- Latest local merge produced non-empty summaries: `condition_summary.csv` with 3 rows and `model_axis_summary.csv` with 3 rows.
- User correctly rejected mock smoke as insufficient for real validation.
- Updated `01-smoke-test.ipynb` to default to `MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"`, `MODEL_SIZE = "1.5B"`, and `BACKEND = "transformers"` for real 60-record smoke.
- Updated real smoke output naming to `outputs/records_smoke_qwen2p5_1p5B_real_six_conditions_shard0.jsonl`; `06-merge-evaluate-plot.ipynb` now defaults to this real-smoke glob.
- Added real-data guards: `00` and `01` now fail if `slm_limits_data/*.jsonl` is fallback/toy data.
- Current `slm_limits_data/` in workspace is fallback/toy data, so real smoke requires restoring the Kaggle/HF prepared JSONL files before running `01`.
- Reworked `00-prepare-data.ipynb` to load real HuggingFace data directly: `gsm8k/main`, `hotpot_qa/distractor`, and `trivia_qa/rc.nocontext`.
- `00` now defaults to `LOAD_FROM_HF = True`, `ALLOW_FALLBACK_DATA = False`, and overwrites old toy files with real prepared JSONL/CSV.
- Verified `00` with network access: it wrote 1000 rows each for `reasoning_gsm8k.jsonl`, `context_hotpotqa.jsonl`, and `knowledge_data.jsonl`, all with `metadata.source = "huggingface"`.
- User ran real Qwen2.5-1.5B smoke and confirmed inference/data/logging passed, but metrics failed due answer extraction.
- Updated `01-smoke-test.ipynb` v3: `MAX_NEW_TOKENS = 384`, stronger `reasoning_direct` prompt, `PROMPT_TEMPLATE = "smoke_v3"`, and GSM8K extraction now falls back to the last number in `prediction_raw` when `Final answer:` is missing.
- New real smoke output is `outputs/records_smoke_qwen2p5_1p5B_real_six_conditions_v3_shard0.jsonl`; `06` and `check_local_outputs.py` now target this v3 file.
- Updated `01-smoke-test.ipynb` v4 for QA extraction: context/knowledge answers now use the text after `Final answer:` but keep only the first non-empty line and strip prefixes like `Answer:` or `The answer is:`.
- QA prompts now say `Answer with a short phrase only. Do not explain.` for context/knowledge smoke conditions.
- New real smoke output is `outputs/records_smoke_qwen2p5_1p5B_real_six_conditions_v4_shard0.jsonl`; `06-merge-evaluate-plot.ipynb` and `check_local_outputs.py` now target v4.
- `scripts/test_smoke_extraction.py` includes QA long-explanation regression cases and currently passes.
- User changed the experiment plan to prefer one generic runner over four similar notebooks. The new runner is `02-run-experiment.ipynb`.
- `02-run-experiment.ipynb` is generated by `scripts/rebuild_local_first_notebooks.py` and supports config-only switching for `AXIS`, `MODEL`, `CONDITION`, `MAX_SAMPLES`, `SHARD_ID`, and `NUM_SHARDS`.
- `02-run-experiment.ipynb` also supports local environment overrides for the same config fields, so repeated condition runs do not require hand-editing the notebook.
- Supported `02` conditions are: knowledge closed/oracle/distractor; reasoning direct/CoT/oracle-step; context none/retrieved/oracle/oracle-distractor/oracle-end; robustness original/para/typo/format.
- `06-merge-evaluate-plot.ipynb` now merges `records_experiment_*.jsonl` plus smoke v4 records and computes broader comparison pairs for oracle gain, distractor/position drop, reasoning gain, and robustness sensitivity.
- `scripts/test_experiment_runner.py` checks the generic runner prompt/extraction/output naming without loading a model and currently passes.
- User found knowledge QA metrics were still falsely low because answers like `John Entwhistle. To determine...` kept explanations.
- Updated `02-run-experiment.ipynb` generation to use `experiment_v2` and `clean_short_qa_answer()`, which cuts explanatory text after short QA answers.
- Updated `06-merge-evaluate-plot.ipynb` generation to re-extract and re-score from `prediction_raw` during merge, then deduplicate by semantic sample key so newer extraction/prompt versions do not mix with stale metrics.
- After re-running `06`, knowledge metrics match the expected corrected range: `knowledge_oracle` F1 about 0.728 and `knowledge_distractor` F1 about 0.835.
- User confirmed the knowledge direction is acceptable and moved focus to reasoning, context, and robustness checks.
- Updated the generic runner generation so `02-run-experiment.ipynb` reads `OUTPUT_DIR` from the environment, making safe dry-runs possible outside the real `outputs/` folder.
- Expanded `scripts/test_experiment_runner.py` to cover prompt generation for all reasoning, context, and robustness conditions without running model inference.
- Fixed `robust_typo` prompt generation so typo perturbation only affects the instruction text and preserves the exact `Final answer:` marker for extraction.
- Current real outputs remain clean: only knowledge 100-sample batches and smoke v4 records are under `outputs/`; no real reasoning 100, context 100, or robustness records have been run yet.
- User confirmed reasoning is fairly stable and moved to context.
- Updated `02-run-experiment.ipynb` generation for context pilot: run all five conditions (`context_none`, `context_retrieved`, `context_oracle`, `context_oracle_distractor`, `context_oracle_end`).
- Context prompts now use condition-specific selected context rather than always storing the full HotpotQA context: no context, BM25-like top-5 paragraphs, oracle supporting paragraphs, oracle + first 3 distractors, or first 3 distractors + oracle at the end.
- Context records now log top-level metadata fields: `num_context_paragraphs`, `num_oracle_paragraphs`, `num_distractor_paragraphs`, `context_tokens`, `gold_at_position`, `context_source`, `context_truncated`, and `distractor_contains_answer`.
- Updated `06-merge-evaluate-plot.ipynb` generation so `condition_summary.csv` includes `comparison_name` and computes context `Oracle Gain`, `Retrieval Gap`, `Distractor Drop`, and `Position Drop` with oracle as the positive reference.
- Added/kept merge figures for oracle gain, distractor drop, position drop, prompt sensitivity, score-latency, and score-memory.
- `scripts/test_experiment_runner.py` now checks context metadata behavior for all five context conditions and currently passes.
- User ran robustness and liked the results: original 0.36, paraphrase 0.50, typo 0.18, JSON format 0.30.
- Audited `robust_original` vs `robust_para`: paraphrase changes wording only and does not add CoT/step-by-step/reasoning instructions.
- Added a regression guard in `scripts/test_experiment_runner.py` so `robust_para` cannot accidentally include reasoning cues such as step-by-step, reason carefully, show your work, or explain your reasoning.
- Upgraded `06-merge-evaluate-plot.ipynb` generation into the paper-ready merge hub before full runs.
- `06` now scans recursive JSONL files from `outputs/`, root axis folders (`knowledge/`, `reasoning/`, `context/`, `robustness/`), `outputs/<axis>/`, `/kaggle/working/outputs`, and `/kaggle/input/slm-limits-outputs`; env overrides are `MERGE_INPUT_DIRS`, `MERGE_OUTPUT_DIR`, and `MERGE_RECORD_GLOBS`.
- `06` now computes matched condition comparisons by `sample_id` with `n_base`, `n_compare`, and `n_overlap`, plus `delta`, `base_minus_compare`, and `abs_delta`.
- `06` exports `robustness_summary.csv`, `paper_main_table.csv`, `paper_comparison_table.csv`, and `paper_axis_table.csv` in addition to the original merged tables.
- `06` now saves extra paper figures: `figure_cot_gain.png`, `figure_retrieval_gap.png`, and `figure_robustness_gap.png`; the model-axis figure is now an actual heatmap.
- Latest validation ran `06` successfully on 1500 records across 15 conditions, producing 15 leaderboard rows, 11 comparison rows, 4 model-axis rows, and 1 robustness summary row.
- Added two Kaggle end-to-end scale notebooks generated from `scripts/rebuild_local_first_notebooks.py`: `03-kaggle-qwen2p5-3b-end-to-end.ipynb` and `04-kaggle-qwen2p5-7b-end-to-end.ipynb`.
- The new 03/04 notebooks run the frozen 15-condition pipeline for one model, load the model once, save resume-safe JSONL files under `/kaggle/working/outputs`, and default to `MAX_SAMPLES = 100`; change the top config to `1000` for the next scale run.
- `02-run-experiment.ipynb` generation now supports `DATA_DIR` environment override so Kaggle can read prepared JSONL from datasets such as `/kaggle/input/slm-limits-data`.
- Updated 02/03/04 data path detection for Kaggle notebook-output mounts. They now prefer `/kaggle/input/notebooks/nguynnguynhehe/00-prepare-data/slm_limits_data`, also scan `/kaggle/input/notebooks/*/00-prepare-data/slm_limits_data`, then `/kaggle/input/slm-limits-data`, then local `slm_limits_data`.

## Promotion Rule

At the end of meaningful work, summarize durable preferences or project decisions into `long_memory.md`. Remove or rewrite stale short-term notes only when they have been promoted or are no longer useful.
