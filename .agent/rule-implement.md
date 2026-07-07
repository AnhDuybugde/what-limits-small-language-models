# PLAN NOTEBOOK KAGGLE

# Understanding the Limits of Small Language Models

## 1. Mục tiêu

Chạy inference-only experiments để phân tích giới hạn của SLM.

Không fine-tune.

Không train.

Không dùng codebase `.py` phức tạp.

Mỗi notebook phải chạy độc lập trên Kaggle.

Mỗi notebook phải có cell cấu hình ở đầu để đổi model, dataset, condition, shard.

Notebook phải tự lưu records JSONL.

Notebook tổng hợp cuối phải merge kết quả, tính metric và vẽ biểu đồ.

## 2. Vì sao dùng notebook thay vì file .py

Vì chạy trên Kaggle nhiều session song song.

Notebook dễ copy.

Notebook dễ đổi model / shard.

Notebook dễ debug trực tiếp.

Notebook có thể lưu output và biểu đồ ngay trong file.

Notebook phù hợp hơn khi chia việc theo account/session.

## 3. Nguyên tắc quan trọng

Chỉ dùng inference.

Dùng deterministic decoding.

temperature = 0.0.

top_p = 1.0.

Mọi output phải có `Final answer:`.

Không đánh giá toàn bộ CoT.

Luôn extract final answer trước khi tính điểm.

Mỗi notebook chỉ chạy một nhóm job nhỏ.

Không chạy full ngay từ đầu.

Đầu tiên chạy smoke test.

Sau đó pilot.

Sau đó mới chạy full test.

## 4. Quy trình chạy đúng

Bước 1: Chạy 10 samples để test code.

Bước 2: Chạy 100 samples để kiểm tra metric và output.

Bước 3: Chốt prompt, condition, extraction, metric.

Bước 4: Freeze code.

Bước 5: Chạy full public test hoặc full validation/test split.

Bước 6: Merge toàn bộ output.

Bước 7: Vẽ bảng và biểu đồ.

## 5. Lưu ý về full test

Có thể chạy full test nếu dataset có nhãn public.

Vì không train/fine-tune nên không có vấn đề training leakage.

Nhưng không được chỉnh prompt nhiều lần dựa trên test.

Nên dùng subset train/validation để debug prompt.

Sau khi chốt, mới chạy full test.

Nếu test label không public, dùng validation làm evaluation chính và ghi rõ trong paper.

## 6. Scope version đầu

Model chính:

Qwen2.5-1.5B-Instruct

Qwen2.5-3B-Instruct

Qwen2.5-7B-Instruct

Dataset chính:

GSM8K cho reasoning.

HotpotQA distractor cho context utilization.

TriviaQA hoặc Natural Questions cho knowledge.

Robustness tạo từ GSM8K và HotpotQA.

Số sample:

Smoke: 10.

Pilot: 100.

Main: full test nếu chạy nổi.

Nếu full test quá nặng, dùng 1000 samples cố định seed.

## 7. Các notebook cần có

Notebook 00_prepare_data.ipynb

Notebook 01_smoke_test.ipynb

Notebook 02_run_knowledge.ipynb

Notebook 03_run_reasoning.ipynb

Notebook 04_run_context.ipynb

Notebook 05_run_robustness.ipynb

Notebook 06_merge_evaluate_plot.ipynb

Notebook 07_error_analysis.ipynb

## 8. Notebook 00_prepare_data.ipynb

Nhiệm vụ:

Load dataset từ HuggingFace.

Chuẩn hóa dữ liệu về cùng format.

Lưu ra JSONL/CSV để các notebook khác dùng lại.

Output:

knowledge_data.jsonl

reasoning_gsm8k.jsonl

context_hotpotqa.jsonl

robustness_gsm8k.jsonl

robustness_hotpotqa.jsonl

Mỗi sample cần có:

sample_id

axis

dataset

question

context

gold_answer

supporting_facts

metadata

## 9. Notebook 01_smoke_test.ipynb

Nhiệm vụ:

Chạy 10 samples.

Chạy 1 model nhỏ.

Chạy vài condition chính.

Kiểm tra model load.

Kiểm tra prompt.

Kiểm tra extract answer.

Kiểm tra metric.

Kiểm tra file output.

Không dùng notebook này để lấy kết quả paper.

## 10. Notebook 02_run_knowledge.ipynb

Mục tiêu:

Kiểm tra SLM sai vì thiếu knowledge hay vì không dùng được oracle context.

Dataset:

TriviaQA hoặc Natural Questions.

Condition:

knowledge_closed

knowledge_oracle

knowledge_distractor

knowledge_closed:

Chỉ hỏi question.

knowledge_oracle:

Cho oracle context.

knowledge_distractor:

Cho oracle context + distractor context.

Metric:

Exact Match.

Token F1.

Oracle Gain.

Distractor Drop.

Output:

records_knowledge_{model}_{condition}_shard{shard_id}.jsonl

## 11. Notebook 03_run_reasoning.ipynb

Mục tiêu:

Kiểm tra giới hạn reasoning.

Dataset:

GSM8K.

Condition:

reasoning_direct

reasoning_cot

reasoning_oracle_step

reasoning_direct:

Yêu cầu trả lời trực tiếp.

reasoning_cot:

Yêu cầu giải từng bước.

reasoning_oracle_step:

Cung cấp intermediate step đúng nếu có thể tạo được.

Metric:

Numeric Accuracy.

Exact Match.

CoT Gain.

Oracle Step Gain.

Output:

records_reasoning_{model}_{condition}_shard{shard_id}.jsonl

## 12. Notebook 04_run_context.ipynb

Mục tiêu:

Kiểm tra khả năng dùng context.

Dataset:

HotpotQA distractor.

Condition:

context_none

context_retrieved

context_oracle

context_oracle_distractor

context_oracle_end

context_none:

Chỉ hỏi question.

context_retrieved:

BM25 top-k paragraph.

context_oracle:

Gold supporting paragraphs.

context_oracle_distractor:

Gold evidence + distractors.

context_oracle_end:

Distractors trước, gold evidence ở cuối.

Metric:

Exact Match.

Token F1.

Context Recall.

Context Precision.

Oracle Gain.

Distractor Drop.

Position Drop.

Output:

records_context_{model}_{condition}_shard{shard_id}.jsonl

## 13. Notebook 05_run_robustness.ipynb

Mục tiêu:

Kiểm tra độ nhạy prompt.

Dataset:

Dùng lại GSM8K và HotpotQA.

Condition:

robust_original

robust_para

robust_typo

robust_format

robust_original:

Prompt gốc.

robust_para:

Đổi cách diễn đạt instruction.

robust_typo:

Thêm typo nhẹ vào instruction.

robust_format:

Yêu cầu output JSON.

Metric:

Mean Score.

Worst-case Score.

Prompt Sensitivity Gap.

Robustness Drop.

Output:

records_robustness_{model}_{condition}_shard{shard_id}.jsonl

## 14. Notebook 06_merge_evaluate_plot.ipynb

Nhiệm vụ:

Load tất cả records JSONL.

Merge thành một dataframe.

Remove duplicate theo unique_key.

Tính metric tổng hợp.

Tạo leaderboard.

Tạo bảng condition summary.

Vẽ biểu đồ.

Output:

records_all.jsonl

leaderboard.csv

condition_summary.csv

model_axis_summary.csv

efficiency_log.csv

figure_model_axis_heatmap.png

figure_oracle_gain.png

figure_distractor_drop.png

figure_prompt_sensitivity.png

figure_accuracy_latency.png

figure_accuracy_memory.png

## 15. Notebook 07_error_analysis.ipynb

Nhiệm vụ:

Lọc các sample sai.

Lấy mỗi model/axis/condition một số lỗi tiêu biểu.

Tự động gán lỗi sơ bộ.

Cho phép manual notes.

Output:

error_samples.jsonl

error_samples.csv

Các error type:

empty_output

format_error

wrong_answer

numeric_extraction_error

context_ignored

distractor_followed

hallucination_possible

## 16. Cell cấu hình bắt buộc ở đầu mỗi notebook chạy inference

Mỗi notebook chạy inference phải có cell như sau:

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

MODEL_SIZE = "1.5B"

AXIS = "reasoning"

DATASET_NAME = "gsm8k"

CONDITION = "reasoning_cot"

SPLIT = "test"

MAX_SAMPLES = None

SHARD_ID = 0

NUM_SHARDS = 8

SEED = 42

TEMPERATURE = 0.0

TOP_P = 1.0

MAX_NEW_TOKENS = 256

OUTPUT_DIR = "/kaggle/working/outputs"

DATA_PATH = "/kaggle/input/slm-limits-data/reasoning_gsm8k.jsonl"

Nếu MAX_SAMPLES = None thì chạy full split.

Nếu MAX_SAMPLES = 1000 thì chạy 1000 samples đầu hoặc 1000 samples theo seed cố định.

## 17. Sharding trong notebook

Mỗi notebook chỉ chạy một shard.

Chia dữ liệu như sau:

sample_index % NUM_SHARDS == SHARD_ID

Ví dụ NUM_SHARDS = 8.

Notebook 1 chạy SHARD_ID = 0.

Notebook 2 chạy SHARD_ID = 1.

...

Notebook 8 chạy SHARD_ID = 7.

Cách này dễ chia trên Kaggle.

## 18. Unique key

Mỗi prediction phải có unique_key:

model_name + axis + dataset + condition + sample_id + prompt_template

Trước khi chạy, notebook đọc output cũ nếu có.

Nếu unique_key đã có status success thì skip.

Nhờ vậy notebook có thể resume sau khi crash.

## 19. Format records JSONL

Mỗi dòng gồm:

unique_key

sample_id

axis

dataset

condition

model_name

model_size

prompt_template

question

context

gold_answer

prediction_raw

prediction_normalized

exact_match

f1

accuracy

additional_metrics

input_tokens

output_tokens

latency_seconds

tokens_per_second

max_memory_allocated_gb

context_truncated

status

error_message

timestamp

## 20. Inference backend

Ưu tiên dùng Transformers trong notebook cho dễ chạy Kaggle.

Nếu vLLM chạy ổn trên Kaggle thì có thể thêm sau.

Transformers setting:

torch_dtype = bfloat16 nếu GPU hỗ trợ.

Nếu không thì float16.

device_map = "auto".

Không quantization ở main result nếu chưa cần.

Nếu dùng 4-bit để chạy model lớn, phải log rõ quantization = 4bit.

## 21. Prompt chung

Mọi prompt phải kết thúc bằng:

Final answer:

QA direct:

You are given a question. Answer as concisely as possible.

Question: {question}

Final answer:

Context QA:

You are given context and a question. Use the context when possible.

Context:

{context}

Question: {question}

Final answer:

GSM8K direct:

Solve the math problem. Give only the final numeric answer.

Problem: {question}

Final answer:

GSM8K CoT:

Solve the math problem step by step. End with "Final answer: <number>".

Problem: {question}

Oracle step:

Problem: {question}

Helpful intermediate step: {oracle_step}

Continue and give the final answer.

Final answer:

JSON format:

Return your answer as JSON only:

{"answer": "..."}

Question: {question}

## 22. Answer extraction

Không dùng raw output để tính điểm.

Luôn extract answer.

Ưu tiên lấy phần sau:

Final answer:

Nếu không có, lấy dòng cuối không rỗng.

Với JSON, parse trường answer.

Nếu parse fail, đánh dấu format_error rồi fallback.

Với GSM8K, extract số cuối cùng.

Với text QA, normalize:

lowercase.

remove punctuation.

remove articles.

collapse whitespace.

Tính EM và token F1 kiểu SQuAD.

## 23. Efficiency logging

Mỗi prediction phải đo:

latency_seconds

input_tokens

output_tokens

tokens_per_second

max_memory_allocated_gb

Không cần đo energy ở version đầu.

Sau này có thể thêm nếu cần.

## 24. Biểu đồ cần vẽ trong notebook merge

Biểu đồ 1:

Model × Axis heatmap.

Biểu đồ 2:

Oracle Gain theo model.

Biểu đồ 3:

Distractor Drop theo model.

Biểu đồ 4:

Prompt Sensitivity Gap theo model.

Biểu đồ 5:

Accuracy/F1 vs Latency.

Biểu đồ 6:

Accuracy/F1 vs Memory.

Dùng matplotlib.

Không dùng seaborn.

## 25. Bảng cần xuất

leaderboard.csv:

model

axis

dataset

condition

n

exact_match_mean

f1_mean

accuracy_mean

latency_mean

memory_mean

tokens_per_second_mean

condition_summary.csv:

model

axis

dataset

base_condition

compare_condition

base_score

compare_score

delta

metric_name

model_axis_summary.csv:

model

axis

main_score

latency_mean

memory_mean

efficiency_score

## 26. Cách chia job trên Kaggle

Mỗi notebook chạy một combination:

model + axis + condition + shard_id

Ví dụ:

Qwen1.5B + reasoning + cot + shard0

Qwen1.5B + reasoning + cot + shard1

Qwen7B + context + oracle + shard0

Qwen7B + context + oracle + shard1

Không nên để một notebook chạy quá nhiều condition.

Chạy xong lưu output JSONL.

Cuối cùng gom toàn bộ JSONL vào notebook merge.

## 27. Gợi ý thứ tự chạy full

Ưu tiên chạy trước:

reasoning_direct

reasoning_cot

context_none

context_oracle

context_oracle_distractor

knowledge_closed

knowledge_oracle

Sau đó mới chạy:

context_retrieved

context_oracle_end

robust_para

robust_typo

robust_format

Vì các condition đầu tạo finding chính.

## 28. Điều cần kiểm tra sau mỗi batch

Số record có đủ không.

Có prediction rỗng nhiều không.

Có Final answer không.

Metric có bất thường không.

Latency có bị quá cao không.

Context có bị truncate nhiều không.

Output file có duplicate không.

## 29. Anti-bugs

Không chỉnh prompt sau khi đã bắt đầu full test.

Không dùng test để tune prompt.

Không overwrite output cũ.

Không quên lưu output sau mỗi vài samples.

Không đánh giá toàn bộ CoT.

Không để GSM8K sai vì extract answer lỗi.

Không để JSON parse fail thành answer sai mà không log.

Không mix quantized/non-quantized nếu không ghi rõ.

Không report oracle setting như realistic setting.

## 30. Kết quả cuối cần có

Sau khi chạy xong cần có:

records_all.jsonl

leaderboard.csv

condition_summary.csv

model_axis_summary.csv

efficiency_log.csv

figures PNG/PDF

error_samples.csv

Các file này đủ để viết paper analysis.
