from pathlib import Path
import json


def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


PREPARE = notebook(
    [
        markdown_cell(
            """
            # 00 Prepare Data

            Load real HuggingFace datasets and write normalized JSONL/CSV files under `slm_limits_data/`.
            """
        ),
        code_cell(
            r'''
from pathlib import Path
import csv
import json
import random

DATA_DIR = Path("slm_limits_data")
MAX_SAMPLES_PER_DATASET = 1000
SEED = 42
LOAD_FROM_HF = True
USE_FALLBACK_IF_LOAD_FAILS = False
PRESERVE_EXISTING_DATA = False
REQUIRE_REAL_DATA = True
ALLOW_FALLBACK_DATA = False

GSM8K_SPLIT = "test"
HOTPOTQA_SPLIT = "validation"
TRIVIAQA_SPLIT = "validation"

DATA_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
'''
        ),
        code_cell(
            r'''
def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def count_jsonl(path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def looks_like_fallback(path):
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return bool(rows) and all(row.get("metadata", {}).get("source") == "local_fallback" for row in rows)


def save_dataset(name, rows):
    jsonl_path = DATA_DIR / f"{name}.jsonl"
    csv_path = DATA_DIR / f"{name}.csv"
    if PRESERVE_EXISTING_DATA and jsonl_path.exists() and csv_path.exists():
        if REQUIRE_REAL_DATA and looks_like_fallback(jsonl_path):
            raise ValueError(f"{jsonl_path} is fallback/toy data. Restore real Kaggle/HF data before real smoke.")
        return {"dataset": name, "n": count_jsonl(jsonl_path), "status": "kept_existing"}
    fallback_rows = bool(rows) and all(row.get("metadata", {}).get("source") == "local_fallback" for row in rows)
    if fallback_rows and not ALLOW_FALLBACK_DATA:
        raise ValueError(
            f"{jsonl_path} would be created from fallback/toy data. Keep ALLOW_FALLBACK_DATA = False for real smoke."
        )
    limited = rows[:MAX_SAMPLES_PER_DATASET]
    write_jsonl(jsonl_path, limited)
    write_csv(csv_path, limited)
    return {"dataset": name, "n": len(limited), "status": "written"}
'''
        ),
        code_cell(
            r'''
def fallback_reasoning():
    problems = [
        ("gsm8k_local_000", "Lena has 12 apples and gives 5 away. How many apples are left?", "7"),
        ("gsm8k_local_001", "A box has 4 rows of 6 pencils. How many pencils are there?", "24"),
        ("gsm8k_local_002", "Tom buys 3 notebooks for 2 dollars each. What is the total cost?", "6"),
        ("gsm8k_local_003", "There are 18 birds. 7 fly away. How many remain?", "11"),
        ("gsm8k_local_004", "Mia reads 8 pages per day for 5 days. How many pages does she read?", "40"),
        ("gsm8k_local_005", "A train has 9 cars with 10 seats each. How many seats are there?", "90"),
        ("gsm8k_local_006", "Sam had 30 stickers and split them equally among 5 friends. How many per friend?", "6"),
        ("gsm8k_local_007", "A shop sold 14 red pens and 9 blue pens. How many pens were sold?", "23"),
        ("gsm8k_local_008", "Nora saves 4 dollars each week for 6 weeks. How much does she save?", "24"),
        ("gsm8k_local_009", "A recipe uses 2 eggs per cake. How many eggs for 7 cakes?", "14"),
        ("gsm8k_local_010", "Jay runs 3 miles each morning for 4 mornings. How many miles?", "12"),
        ("gsm8k_local_011", "A class has 21 students. 3 groups are equal size. How many students per group?", "7"),
    ]
    return [
        {
            "sample_id": sample_id,
            "axis": "reasoning",
            "dataset": "gsm8k",
            "question": question,
            "context": "",
            "gold_answer": answer,
            "supporting_facts": [],
            "metadata": {"source": "local_fallback"},
        }
        for sample_id, question, answer in problems
    ]


def fallback_knowledge():
    facts = [
        ("knowledge_local_000", "What is the capital of France?", "Paris", "France's capital city is Paris."),
        ("knowledge_local_001", "Who wrote Pride and Prejudice?", "Jane Austen", "Pride and Prejudice is a novel by Jane Austen."),
        ("knowledge_local_002", "What planet is known as the Red Planet?", "Mars", "Mars is often called the Red Planet."),
        ("knowledge_local_003", "What gas do plants absorb from the atmosphere?", "carbon dioxide", "Plants absorb carbon dioxide during photosynthesis."),
        ("knowledge_local_004", "Who painted the Mona Lisa?", "Leonardo da Vinci", "The Mona Lisa was painted by Leonardo da Vinci."),
        ("knowledge_local_005", "What is the largest ocean on Earth?", "Pacific Ocean", "The Pacific Ocean is the largest ocean on Earth."),
        ("knowledge_local_006", "What currency is used in Japan?", "yen", "Japan uses the yen as its currency."),
        ("knowledge_local_007", "What is H2O commonly called?", "water", "H2O is the chemical formula for water."),
        ("knowledge_local_008", "Who developed the theory of relativity?", "Albert Einstein", "Albert Einstein developed the theory of relativity."),
        ("knowledge_local_009", "What is the tallest mountain above sea level?", "Mount Everest", "Mount Everest is the tallest mountain above sea level."),
        ("knowledge_local_010", "What language is primarily spoken in Brazil?", "Portuguese", "Portuguese is the primary language spoken in Brazil."),
        ("knowledge_local_011", "What is the boiling point of water at sea level in Celsius?", "100", "Water boils at 100 degrees Celsius at sea level."),
    ]
    return [
        {
            "sample_id": sample_id,
            "axis": "knowledge",
            "dataset": "triviaqa",
            "question": question,
            "context": context,
            "gold_answer": answer,
            "supporting_facts": [context],
            "metadata": {"source": "local_fallback"},
        }
        for sample_id, question, answer, context in facts
    ]


def fallback_context():
    rows = [
        ("context_local_000", "Which city hosted the 2012 Summer Olympics?", "London", "The 2012 Summer Olympics were hosted in London. Paris hosted the 2024 Summer Olympics."),
        ("context_local_001", "Which scientist discovered penicillin?", "Alexander Fleming", "Alexander Fleming discovered penicillin in 1928. Marie Curie studied radioactivity."),
        ("context_local_002", "What company created the iPhone?", "Apple", "Apple introduced the first iPhone in 2007. Samsung also makes smartphones."),
        ("context_local_003", "Which country is Kyoto located in?", "Japan", "Kyoto is a city in Japan. Seoul is a city in South Korea."),
        ("context_local_004", "What instrument has keys, pedals, and strings?", "piano", "A piano has keys, pedals, and strings. A violin has strings but no keys."),
        ("context_local_005", "Which element has the chemical symbol O?", "oxygen", "O is the chemical symbol for oxygen. Au is the symbol for gold."),
        ("context_local_006", "What animal is the national symbol of New Zealand rugby?", "kiwi", "The kiwi is a national symbol of New Zealand. The silver fern is also associated with New Zealand sport."),
        ("context_local_007", "Which city is the Eiffel Tower in?", "Paris", "The Eiffel Tower is located in Paris. The Colosseum is in Rome."),
        ("context_local_008", "Who is the Greek god of the sea?", "Poseidon", "Poseidon is the Greek god of the sea. Zeus is associated with the sky."),
        ("context_local_009", "What metal is liquid at room temperature?", "mercury", "Mercury is a metal that is liquid at room temperature. Iron is solid at room temperature."),
        ("context_local_010", "Which organ pumps blood through the body?", "heart", "The heart pumps blood through the body. The lungs exchange oxygen and carbon dioxide."),
        ("context_local_011", "What is the main ingredient in guacamole?", "avocado", "Guacamole is mainly made from avocado. Salsa is commonly made from tomato."),
    ]
    return [
        {
            "sample_id": sample_id,
            "axis": "context",
            "dataset": "hotpotqa",
            "question": question,
            "context": context,
            "gold_answer": answer,
            "supporting_facts": [context.split(".")[0] + "."],
            "metadata": {"source": "local_fallback"},
        }
        for sample_id, question, answer, context in rows
    ]
'''
        ),
        code_cell(
            r'''
def load_dataset_split(path, name=None, split="train"):
    if not LOAD_FROM_HF:
        raise RuntimeError("LOAD_FROM_HF is False.")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the HuggingFace datasets package first: pip install datasets") from exc

    kwargs = {"split": split}
    if name is None:
        dataset = load_dataset(path, **kwargs)
    else:
        dataset = load_dataset(path, name, **kwargs)
    return dataset.shuffle(seed=SEED).select(range(min(MAX_SAMPLES_PER_DATASET, len(dataset))))


def gsm8k_gold(answer):
    marker = "####"
    if marker in str(answer):
        return str(answer).split(marker)[-1].strip().replace(",", "")
    return str(answer).strip()


def prepare_gsm8k():
    dataset = load_dataset_split("gsm8k", "main", GSM8K_SPLIT)
    rows = []
    for index, item in enumerate(dataset):
        sample_id = item.get("id") or f"gsm8k_{GSM8K_SPLIT}_{index:06d}"
        rows.append(
            {
                "sample_id": str(sample_id),
                "axis": "reasoning",
                "dataset": "gsm8k",
                "question": str(item["question"]).strip(),
                "context": "",
                "gold_answer": gsm8k_gold(item["answer"]),
                "supporting_facts": [],
                "metadata": {"source": "huggingface", "hf_dataset": "gsm8k/main", "split": GSM8K_SPLIT},
            }
        )
    return rows


def flatten_hotpot_context(context):
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentence_groups = context.get("sentences", [])
        paragraphs = []
        for title, sentences in zip(titles, sentence_groups):
            text = " ".join(str(sentence) for sentence in sentences)
            paragraphs.append(f"{title}: {text}")
        return "\n\n".join(paragraphs)
    return str(context or "")


def prepare_hotpotqa():
    dataset = load_dataset_split("hotpot_qa", "distractor", HOTPOTQA_SPLIT)
    rows = []
    for index, item in enumerate(dataset):
        sample_id = item.get("id") or f"hotpotqa_{HOTPOTQA_SPLIT}_{index:06d}"
        rows.append(
            {
                "sample_id": str(sample_id),
                "axis": "context",
                "dataset": "hotpotqa",
                "question": str(item["question"]).strip(),
                "context": flatten_hotpot_context(item.get("context", "")),
                "gold_answer": str(item["answer"]).strip(),
                "supporting_facts": item.get("supporting_facts", {}),
                "metadata": {"source": "huggingface", "hf_dataset": "hotpot_qa/distractor", "split": HOTPOTQA_SPLIT},
            }
        )
    return rows


def trivia_answer(answer):
    if isinstance(answer, dict):
        value = answer.get("value")
        if value:
            return str(value).strip()
        aliases = answer.get("aliases") or []
        if aliases:
            return str(aliases[0]).strip()
    return str(answer).strip()


def prepare_triviaqa():
    dataset = load_dataset_split("trivia_qa", "rc.nocontext", TRIVIAQA_SPLIT)
    rows = []
    for index, item in enumerate(dataset):
        sample_id = item.get("question_id") or item.get("id") or f"triviaqa_{TRIVIAQA_SPLIT}_{index:06d}"
        rows.append(
            {
                "sample_id": str(sample_id),
                "axis": "knowledge",
                "dataset": "triviaqa",
                "question": str(item["question"]).strip(),
                "context": "",
                "gold_answer": trivia_answer(item.get("answer", "")),
                "supporting_facts": [],
                "metadata": {"source": "huggingface", "hf_dataset": "trivia_qa/rc.nocontext", "split": TRIVIAQA_SPLIT},
            }
        )
    return rows
'''
        ),
        code_cell(
            r'''
if LOAD_FROM_HF:
    try:
        reasoning_rows = prepare_gsm8k()
        knowledge_rows = prepare_triviaqa()
        context_rows = prepare_hotpotqa()
    except Exception:
        if not USE_FALLBACK_IF_LOAD_FAILS:
            raise
        reasoning_rows = fallback_reasoning()
        knowledge_rows = fallback_knowledge()
        context_rows = fallback_context()
else:
    if not ALLOW_FALLBACK_DATA:
        raise ValueError("LOAD_FROM_HF is False and ALLOW_FALLBACK_DATA is False. Enable HuggingFace loading for real smoke.")
    reasoning_rows = fallback_reasoning()
    knowledge_rows = fallback_knowledge()
    context_rows = fallback_context()

robustness_gsm8k = [{**row, "axis": "robustness", "dataset": "gsm8k"} for row in reasoning_rows]
robustness_hotpotqa = [{**row, "axis": "robustness", "dataset": "hotpotqa"} for row in context_rows]

summary = [
    save_dataset("reasoning_gsm8k", reasoning_rows),
    save_dataset("knowledge_data", knowledge_rows),
    save_dataset("context_hotpotqa", context_rows),
    save_dataset("robustness_gsm8k", robustness_gsm8k),
    save_dataset("robustness_hotpotqa", robustness_hotpotqa),
]
write_csv(DATA_DIR / "prepare_data_summary.csv", summary)
summary
'''
        ),
    ]
)


SMOKE = notebook(
    [
        markdown_cell(
            """
            # 01 Real Smoke Test

            Local-first real smoke test. This runs Qwen2.5-1.5B on 10 real data samples for each required condition and writes about 60 JSONL records.
            """
        ),
        code_cell(
            r'''
from pathlib import Path
from datetime import datetime, timezone
import json
import math
import random
import re
import string
import time

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_SIZE = "1.5B"
SMOKE_CONDITIONS = [
    "reasoning_direct",
    "reasoning_cot",
    "context_none",
    "context_oracle",
    "knowledge_closed",
    "knowledge_oracle",
]
MAX_SAMPLES_PER_CONDITION = 10
SHARD_ID = 0
NUM_SHARDS = 1
SEED = 42
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_NEW_TOKENS = 384
BACKEND = "transformers"
LOCAL_FILES_ONLY = False
REQUIRE_REAL_DATA = True

DATA_DIR = Path("slm_limits_data")
OUTPUT_DIR = Path("outputs")
MODEL_TAG = MODEL_SIZE.replace(".", "p")
PROMPT_TEMPLATE = "smoke_v4"
OUTPUT_PATH = OUTPUT_DIR / f"records_smoke_qwen2p5_{MODEL_TAG}_real_six_conditions_v4_shard{SHARD_ID}.jsonl"
OVERWRITE_OUTPUT = False

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
'''
        ),
        code_cell(
            r'''
CONDITION_TO_DATA = {
    "reasoning_direct": DATA_DIR / "reasoning_gsm8k.jsonl",
    "reasoning_cot": DATA_DIR / "reasoning_gsm8k.jsonl",
    "context_none": DATA_DIR / "context_hotpotqa.jsonl",
    "context_oracle": DATA_DIR / "context_hotpotqa.jsonl",
    "knowledge_closed": DATA_DIR / "knowledge_data.jsonl",
    "knowledge_oracle": DATA_DIR / "knowledge_data.jsonl",
}


def read_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}. Run 00-prepare-data.ipynb first.")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_real_rows(path, rows):
    if len(rows) < MAX_SAMPLES_PER_CONDITION:
        raise ValueError(f"{path} has only {len(rows)} rows; need at least {MAX_SAMPLES_PER_CONDITION}.")
    if not REQUIRE_REAL_DATA:
        return
    fallback_count = sum(1 for row in rows if row.get("metadata", {}).get("source") == "local_fallback")
    if fallback_count == len(rows):
        raise ValueError(
            f"{path} looks like local fallback/toy data. Restore the real Kaggle/HF JSONL files before running real smoke."
        )


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def successful_keys(path):
    if OVERWRITE_OUTPUT and path.exists():
        path.unlink()
        return set()
    keys = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "success":
                keys.add(row.get("unique_key"))
    return keys


def unique_key(sample, condition):
    parts = [MODEL_NAME, sample["axis"], sample["dataset"], condition, sample["sample_id"], PROMPT_TEMPLATE]
    return "::".join(str(part) for part in parts)
'''
        ),
        code_cell(
            r'''
ARTICLES = {"a", "an", "the"}


def normalize_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in text.split() if token not in ARTICLES]
    return " ".join(tokens)


def token_f1(prediction, gold):
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_after_final_answer(text):
    match = re.search(r"final answer\s*:\s*(.+)", str(text), flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return lines[-1] if lines else ""


def extract_short_qa_answer(text):
    match = re.search(r"final answer\s*:\s*(.*)", str(text), flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    answer = re.sub(r"^(the answer is|answer is|answer)\s*:?\s*", "", lines[0], flags=re.IGNORECASE)
    return answer.strip(" .")


def has_final_answer(text):
    return re.search(r"final answer\s*:", str(text), flags=re.IGNORECASE) is not None


def extract_json_answer(text):
    try:
        parsed = json.loads(str(text).strip())
    except json.JSONDecodeError:
        return ""
    return str(parsed.get("answer", "")) if isinstance(parsed, dict) else ""


def extract_numeric(text):
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    if not numbers:
        return ""
    value = numbers[-1]
    return value[:-2] if value.endswith(".0") else value


def extract_answer(raw_text, axis, condition):
    if "json" in condition:
        parsed = extract_json_answer(raw_text)
        if parsed:
            return parsed
    if axis == "reasoning":
        text = extract_after_final_answer(raw_text) if has_final_answer(raw_text) else raw_text
        numeric = extract_numeric(text)
        return numeric or extract_after_final_answer(raw_text)
    return extract_short_qa_answer(raw_text)


def score_prediction(prediction, gold, axis):
    if axis == "reasoning":
        pred_num = extract_numeric(prediction)
        gold_num = extract_numeric(gold)
        accuracy = float(pred_num != "" and pred_num == gold_num)
        exact_match = accuracy
        f1 = accuracy
    else:
        exact_match = float(normalize_text(prediction) == normalize_text(gold))
        f1 = token_f1(prediction, gold)
        accuracy = exact_match
    return exact_match, f1, accuracy
'''
        ),
        code_cell(
            r'''
def build_prompt(sample, condition):
    question = sample["question"]
    context = sample.get("context", "")
    if condition == "reasoning_direct":
        return f"Solve the math problem. Return only one number. Do not explain.\n\nProblem: {question}\n\nFinal answer:"
    if condition == "reasoning_cot":
        return f"Solve the math problem step by step. End with \"Final answer: <number>\".\n\nProblem: {question}\n\nFinal answer:"
    if condition == "context_none":
        return f"Answer with a short phrase only. Do not explain.\n\nQuestion: {question}\n\nFinal answer:"
    if condition == "context_oracle":
        return f"Answer with a short phrase only. Do not explain. Use the context when possible.\n\nContext:\n{context}\n\nQuestion: {question}\n\nFinal answer:"
    if condition == "knowledge_closed":
        return f"Answer with a short phrase only. Do not explain.\n\nQuestion: {question}\n\nFinal answer:"
    if condition == "knowledge_oracle":
        return f"Answer with a short phrase only. Do not explain. Use the context when possible.\n\nContext:\n{context}\n\nQuestion: {question}\n\nFinal answer:"
    raise ValueError(f"Unknown condition: {condition}")


def mock_generate(sample, condition):
    gold = sample["gold_answer"]
    if condition.endswith("_cot"):
        return f"We solve it carefully using the quantities in the problem.\nFinal answer: {gold}"
    if condition.endswith("_none") or condition.endswith("_closed"):
        # Keep local smoke deterministic but imperfect so summaries show meaningful deltas.
        if int(re.findall(r"\d+", sample["sample_id"])[-1]) % 4 == 0:
            return "Final answer: unknown"
    return f"Final answer: {gold}"


def load_transformers_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=LOCAL_FILES_ONLY)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=LOCAL_FILES_ONLY,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def transformers_generate(tokenizer, model, prompt):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - start
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    input_tokens = int(inputs["input_ids"].shape[-1])
    output_tokens = int(generated_ids.shape[-1])
    memory_gb = 0.0
    if torch.cuda.is_available():
        memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return text, latency, input_tokens, output_tokens, memory_gb
'''
        ),
        code_cell(
            r'''
def select_samples(condition):
    rows = read_jsonl(CONDITION_TO_DATA[condition])
    validate_real_rows(CONDITION_TO_DATA[condition], rows)
    rows = [row for index, row in enumerate(rows) if index % NUM_SHARDS == SHARD_ID]
    return rows[:MAX_SAMPLES_PER_CONDITION]


completed = successful_keys(OUTPUT_PATH)
samples_by_condition = {condition: select_samples(condition) for condition in SMOKE_CONDITIONS}
condition_counts = {condition: len(samples) for condition, samples in samples_by_condition.items()}
print({"validated_conditions": condition_counts, "model_name": MODEL_NAME, "backend": BACKEND})

model_bundle = load_transformers_model() if BACKEND == "transformers" else None
records_written = 0

for condition in SMOKE_CONDITIONS:
    samples = samples_by_condition[condition]
    for sample in samples:
        key = unique_key(sample, condition)
        if key in completed:
            continue

        prompt = build_prompt(sample, condition)
        start = time.perf_counter()
        status = "success"
        error_message = ""
        input_tokens = len(prompt.split())
        output_tokens = 0
        max_memory_gb = 0.0

        try:
            if BACKEND == "transformers":
                tokenizer, model = model_bundle
                raw, latency, input_tokens, output_tokens, max_memory_gb = transformers_generate(tokenizer, model, prompt)
            else:
                raw = mock_generate(sample, condition)
                latency = time.perf_counter() - start
                output_tokens = len(raw.split())

            prediction = extract_answer(raw, sample["axis"], condition)
            exact_match, f1, accuracy = score_prediction(prediction, sample["gold_answer"], sample["axis"])
            tokens_per_second = output_tokens / latency if latency > 0 else 0.0
        except Exception as exc:
            raw = ""
            prediction = ""
            exact_match = f1 = accuracy = 0.0
            latency = time.perf_counter() - start
            tokens_per_second = 0.0
            status = "error"
            error_message = repr(exc)

        record = {
            "unique_key": key,
            "sample_id": sample["sample_id"],
            "axis": sample["axis"],
            "dataset": sample["dataset"],
            "condition": condition,
            "model_name": MODEL_NAME,
            "model_size": MODEL_SIZE,
            "prompt_template": PROMPT_TEMPLATE,
            "question": sample["question"],
            "context": sample.get("context", ""),
            "gold_answer": sample["gold_answer"],
            "prediction_raw": raw,
            "prediction_normalized": prediction,
            "exact_match": exact_match,
            "f1": f1,
            "accuracy": accuracy,
            "additional_metrics": {},
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_seconds": latency,
            "tokens_per_second": tokens_per_second,
            "max_memory_allocated_gb": max_memory_gb,
            "context_truncated": False,
            "status": status,
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(OUTPUT_PATH, record)
        completed.add(key)
        records_written += 1

print({"output_path": str(OUTPUT_PATH), "records_written": records_written, "condition_counts": condition_counts})
'''
        ),
    ]
)


EXPERIMENT = notebook(
    [
        markdown_cell(
            """
            # 02 Run Experiment

            Generic inference runner. Change only the config cell to run knowledge, reasoning, context, or robustness jobs.
            """
        ),
        code_cell(
            r'''
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import random
import re
import string
import time

def env_int(name, default):
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def env_optional_int(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return None if value.lower() == "none" else int(value)


def env_bool(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "y"}


CORE_DATA_FILES = ["knowledge_data.jsonl", "reasoning_gsm8k.jsonl", "context_hotpotqa.jsonl"]


def has_core_data_files(path):
    return all((path / name).exists() for name in CORE_DATA_FILES)


def candidate_data_dirs():
    yield Path("/kaggle/input/notebooks/nguynnguynhehe/00-prepare-data/slm_limits_data")
    notebooks_root = Path("/kaggle/input/notebooks")
    if notebooks_root.exists():
        yield from sorted(notebooks_root.glob("*/00-prepare-data/slm_limits_data"))
    yield Path("/kaggle/input/slm-limits-data")
    input_root = Path("/kaggle/input")
    if input_root.exists():
        yield from sorted(input_root.glob("*/slm_limits_data"))
    yield Path("slm_limits_data")


def resolve_data_dir(value=None):
    if value:
        return Path(value)
    for candidate in candidate_data_dirs():
        if has_core_data_files(candidate):
            return candidate
    return Path("slm_limits_data")


AXIS = os.getenv("AXIS", "knowledge")
MODEL = os.getenv("MODEL", "Qwen2.5-1.5B")
MODEL_NAME_MAP = {
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
}
MODEL_NAME = MODEL_NAME_MAP.get(MODEL, MODEL)
MODEL_SIZE = MODEL.replace("Qwen/Qwen2.5-", "").replace("Qwen2.5-", "").replace("-Instruct", "")
CONDITION = os.getenv("CONDITION", "knowledge_oracle")
MAX_SAMPLES = env_optional_int("MAX_SAMPLES", 100)
SHARD_ID = env_int("SHARD_ID", 0)
NUM_SHARDS = env_int("NUM_SHARDS", 1)
SEED = env_int("SEED", 42)
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 384)
BACKEND = os.getenv("BACKEND", "transformers")
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", False)
REQUIRE_REAL_DATA = env_bool("REQUIRE_REAL_DATA", True)
OVERWRITE_OUTPUT = env_bool("OVERWRITE_OUTPUT", False)
ROBUSTNESS_DATASET = os.getenv("ROBUSTNESS_DATASET", "gsm8k")

DATA_DIR = resolve_data_dir(os.getenv("DATA_DIR"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
PROMPT_TEMPLATE = "experiment_v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
'''
        ),
        code_cell(
            r'''
AXIS_TO_DATA_PATH = {
    "knowledge": DATA_DIR / "knowledge_data.jsonl",
    "reasoning": DATA_DIR / "reasoning_gsm8k.jsonl",
    "context": DATA_DIR / "context_hotpotqa.jsonl",
    "robustness": DATA_DIR / f"robustness_{ROBUSTNESS_DATASET}.jsonl",
}

AXIS_CONDITIONS = {
    "knowledge": {"knowledge_closed", "knowledge_oracle", "knowledge_distractor"},
    "reasoning": {"reasoning_direct", "reasoning_cot", "reasoning_oracle_step"},
    "context": {"context_none", "context_retrieved", "context_oracle", "context_oracle_distractor", "context_oracle_end"},
    "robustness": {"robust_original", "robust_para", "robust_typo", "robust_format"},
}


def model_tag(value):
    tag = value.split("/")[-1].replace("-Instruct", "")
    return tag.replace(".", "p").replace("-", "_")


def output_path():
    sample_tag = "full" if MAX_SAMPLES is None else f"n{MAX_SAMPLES}"
    name = f"records_experiment_{AXIS}_{model_tag(MODEL_NAME)}_{CONDITION}_{sample_tag}_shard{SHARD_ID}.jsonl"
    return OUTPUT_DIR / name


DATA_PATH = AXIS_TO_DATA_PATH[AXIS]
OUTPUT_PATH = output_path()


def validate_config():
    if AXIS not in AXIS_TO_DATA_PATH:
        raise ValueError(f"Unsupported AXIS: {AXIS}. Choose one of {sorted(AXIS_TO_DATA_PATH)}.")
    if CONDITION not in AXIS_CONDITIONS[AXIS]:
        allowed = sorted(AXIS_CONDITIONS[AXIS])
        raise ValueError(f"Unsupported CONDITION for {AXIS}: {CONDITION}. Choose one of {allowed}.")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}. Run 00-prepare-data.ipynb first.")
    if not (0 <= SHARD_ID < NUM_SHARDS):
        raise ValueError("SHARD_ID must satisfy 0 <= SHARD_ID < NUM_SHARDS.")


validate_config()
'''
        ),
        code_cell(
            r'''
def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_real_rows(rows):
    if not rows:
        raise ValueError(f"{DATA_PATH} has no rows.")
    if not REQUIRE_REAL_DATA:
        return
    fallback_count = sum(1 for row in rows if row.get("metadata", {}).get("source") == "local_fallback")
    if fallback_count == len(rows):
        raise ValueError(f"{DATA_PATH} looks like local fallback/toy data. Run 00 with HuggingFace data first.")


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def successful_keys(path):
    if OVERWRITE_OUTPUT and path.exists():
        path.unlink()
        return set()
    keys = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "success":
                keys.add(row.get("unique_key"))
    return keys


def select_samples(rows):
    sharded = [row for index, row in enumerate(rows) if index % NUM_SHARDS == SHARD_ID]
    return sharded if MAX_SAMPLES is None else sharded[:MAX_SAMPLES]


def unique_key(sample):
    parts = [MODEL_NAME, AXIS, sample["dataset"], CONDITION, sample["sample_id"], PROMPT_TEMPLATE]
    return "::".join(str(part) for part in parts)
'''
        ),
        code_cell(
            r'''
ARTICLES = {"a", "an", "the"}


def normalize_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in text.split() if token not in ARTICLES]
    return " ".join(tokens)


def token_f1(prediction, gold):
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = 0
    remaining = gold_tokens.copy()
    for token in pred_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def has_final_answer(text):
    return re.search(r"final answer\s*:", str(text), flags=re.IGNORECASE) is not None


def extract_after_final_answer(text):
    match = re.search(r"final answer\s*:\s*(.*)", str(text), flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return lines[-1] if lines else ""


def clean_short_qa_answer(answer):
    answer = str(answer).strip()
    answer = re.split(
        r"\.\s+(?:To determine|This|Therefore|Based on|I |The |It |We )",
        answer,
        maxsplit=1,
    )[0]
    answer = re.sub(r"^(the answer is|answer is|answer)\s*:?\s*", "", answer, flags=re.IGNORECASE)
    return answer.strip(" .")


def extract_short_qa_answer(text):
    match = re.search(r"final answer\s*:\s*(.*)", str(text), flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    return clean_short_qa_answer(lines[0])


def extract_json_answer(text):
    try:
        parsed = json.loads(str(text).strip())
    except json.JSONDecodeError:
        return ""
    return str(parsed.get("answer", "")) if isinstance(parsed, dict) else ""


def extract_numeric(text):
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    if not numbers:
        return ""
    value = numbers[-1]
    return value[:-2] if value.endswith(".0") else value


def extract_answer(raw_text, axis, condition):
    if condition.endswith("_format"):
        parsed = extract_json_answer(raw_text)
        if parsed:
            return parsed
    if axis == "reasoning" or (axis == "robustness" and ROBUSTNESS_DATASET == "gsm8k"):
        text = extract_after_final_answer(raw_text) if has_final_answer(raw_text) else raw_text
        numeric = extract_numeric(text)
        return numeric or extract_after_final_answer(raw_text)
    return extract_short_qa_answer(raw_text)


def score_prediction(prediction, gold, sample):
    if sample["dataset"] == "gsm8k":
        pred_num = extract_numeric(prediction)
        gold_num = extract_numeric(gold)
        accuracy = float(pred_num != "" and pred_num == gold_num)
        return accuracy, accuracy, accuracy
    exact_match = float(normalize_text(prediction) == normalize_text(gold))
    f1 = token_f1(prediction, gold)
    return exact_match, f1, exact_match
'''
        ),
        code_cell(
            r'''
def context_lines(sample):
    return [line.strip() for line in str(sample.get("context", "")).splitlines() if line.strip()]


def supporting_titles(sample):
    facts = sample.get("supporting_facts", {})
    if isinstance(facts, dict):
        return {str(title) for title in facts.get("title", [])}
    return set()


def split_hotpot_context(sample):
    lines = context_lines(sample)
    titles = supporting_titles(sample)
    if not lines or not titles:
        return "\n".join(lines), ""
    oracle = []
    distractors = []
    for line in lines:
        title = line.split(":", 1)[0]
        if title in titles:
            oracle.append(line)
        else:
            distractors.append(line)
    return "\n\n".join(oracle or lines), "\n\n".join(distractors)


def paragraph_title(paragraph):
    return str(paragraph).split(":", 1)[0].strip()


def split_hotpot_context_parts(sample):
    paragraphs = context_lines(sample)
    titles = supporting_titles(sample)
    if not paragraphs:
        return [], []
    if not titles:
        return paragraphs, []
    oracle = []
    distractors = []
    for paragraph in paragraphs:
        if paragraph_title(paragraph) in titles:
            oracle.append(paragraph)
        else:
            distractors.append(paragraph)
    return oracle or paragraphs, distractors


def bm25_like_top_paragraphs(sample, k=5):
    paragraphs = context_lines(sample)
    if len(paragraphs) <= k:
        return paragraphs
    query_terms = normalize_text(sample.get("question", "")).split()
    if not query_terms:
        return paragraphs[:k]
    scored = []
    for index, paragraph in enumerate(paragraphs):
        terms = normalize_text(paragraph).split()
        if not terms:
            scored.append((0.0, index, paragraph))
            continue
        term_counts = {term: terms.count(term) for term in set(terms)}
        score = sum(term_counts.get(term, 0) / len(terms) for term in query_terms)
        scored.append((score, index, paragraph))
    return [paragraph for _, _, paragraph in sorted(scored, key=lambda item: (-item[0], item[1]))[:k]]


def gold_position(selected_paragraphs, sample):
    titles = supporting_titles(sample)
    if not selected_paragraphs or not titles:
        return "none"
    positions = [index for index, paragraph in enumerate(selected_paragraphs) if paragraph_title(paragraph) in titles]
    if not positions:
        return "none"
    if positions == list(range(len(positions))):
        return "beginning"
    if positions == list(range(len(selected_paragraphs) - len(positions), len(selected_paragraphs))):
        return "end"
    return "mixed"


def context_selection(sample):
    oracle, distractors = split_hotpot_context_parts(sample)
    first_distractors = distractors[:3]
    source = "none"
    selected = []
    selected_distractors = []

    if CONDITION == "context_retrieved":
        selected = bm25_like_top_paragraphs(sample, k=5)
        selected_distractors = [p for p in selected if paragraph_title(p) not in supporting_titles(sample)]
        source = "bm25"
    elif CONDITION == "context_oracle":
        selected = oracle
        source = "oracle"
    elif CONDITION == "context_oracle_distractor":
        selected = oracle + first_distractors
        selected_distractors = first_distractors
        source = "oracle+distractor"
    elif CONDITION == "context_oracle_end":
        selected = first_distractors + oracle
        selected_distractors = first_distractors
        source = "oracle_end"

    text = "\n\n".join(selected)
    distractor_text = "\n\n".join(selected_distractors)
    gold_answer = normalize_text(sample.get("gold_answer", ""))
    normalized_distractors = normalize_text(distractor_text)
    return {
        "text": text,
        "num_context_paragraphs": len(selected),
        "num_oracle_paragraphs": sum(1 for paragraph in selected if paragraph_title(paragraph) in supporting_titles(sample)),
        "num_distractor_paragraphs": len(selected_distractors),
        "context_tokens": len(text.split()),
        "gold_at_position": gold_position(selected, sample),
        "context_source": source,
        "context_truncated": False,
        "distractor_contains_answer": bool(gold_answer and gold_answer in normalized_distractors),
    }


def empty_context_metadata():
    return {
        "num_context_paragraphs": 0,
        "num_oracle_paragraphs": 0,
        "num_distractor_paragraphs": 0,
        "context_tokens": 0,
        "gold_at_position": "none",
        "context_source": "none",
        "context_truncated": False,
        "distractor_contains_answer": False,
    }


def record_context_metadata(sample):
    if AXIS != "context":
        return empty_context_metadata() | {"text": sample.get("context", "")}
    return context_selection(sample)


def sample_index(sample):
    numbers = re.findall(r"\d+", str(sample.get("sample_id", "")))
    return int(numbers[-1]) if numbers else 0


def knowledge_context(sample, all_rows, include_distractors=False):
    context = str(sample.get("context", "")).strip()
    if not context:
        context = f"Reference: The answer is {sample['gold_answer']}."
    if not include_distractors:
        return context
    others = [row for row in all_rows if row.get("sample_id") != sample.get("sample_id")]
    distractors = []
    for row in others[:3]:
        distractors.append(f"Distractor: The answer to another question is {row.get('gold_answer', '')}.")
    return context + "\n" + "\n".join(distractors)


def oracle_step(sample):
    answer = str(sample.get("gold_answer", "")).strip()
    return f"The final numeric answer should be {answer}."


def typo_text(text):
    return text.replace("answer", "anwser").replace("question", "quesiton")


def build_prompt(sample, all_rows):
    question = sample["question"]
    context = str(sample.get("context", "")).strip()

    if CONDITION == "knowledge_closed":
        return f"Answer with a short phrase only. Do not explain.\n\nQuestion: {question}\n\nFinal answer:"
    if CONDITION == "knowledge_oracle":
        context = knowledge_context(sample, all_rows)
        return f"Answer with a short phrase only. Do not explain. Use the context when possible.\n\nContext:\n{context}\n\nQuestion: {question}\n\nFinal answer:"
    if CONDITION == "knowledge_distractor":
        context = knowledge_context(sample, all_rows, include_distractors=True)
        return f"Answer with a short phrase only. Do not explain. Some context may be irrelevant.\n\nContext:\n{context}\n\nQuestion: {question}\n\nFinal answer:"

    if CONDITION == "reasoning_direct":
        return f"Solve the math problem. Return only one number. Do not explain.\n\nProblem: {question}\n\nFinal answer:"
    if CONDITION == "reasoning_cot":
        return f"Solve the math problem step by step. End with \"Final answer: <number>\".\n\nProblem: {question}\n\nFinal answer:"
    if CONDITION == "reasoning_oracle_step":
        return f"Problem: {question}\n\nHelpful intermediate step: {oracle_step(sample)}\n\nContinue and give only the final numeric answer.\n\nFinal answer:"

    if CONDITION == "context_none":
        return f"Answer with a short phrase only. Do not explain.\n\nQuestion: {question}\n\nFinal answer:"
    if CONDITION == "context_retrieved":
        context = context_selection(sample)["text"]
        return f"Answer with a short phrase only. Do not explain. Use the context when possible.\n\nContext:\n{context}\n\nQuestion: {question}\n\nFinal answer:"
    if CONDITION == "context_oracle":
        context = context_selection(sample)["text"]
        return f"Answer with a short phrase only. Do not explain. Use the context when possible.\n\nContext:\n{context}\n\nQuestion: {question}\n\nFinal answer:"
    if CONDITION == "context_oracle_distractor":
        context = context_selection(sample)["text"]
        return f"Answer with a short phrase only. Do not explain. Some context may be irrelevant.\n\nContext:\n{context}\n\nQuestion: {question}\n\nFinal answer:"
    if CONDITION == "context_oracle_end":
        context = context_selection(sample)["text"]
        return f"Answer with a short phrase only. Do not explain. Some context may be irrelevant.\n\nContext:\n{context}\n\nQuestion: {question}\n\nFinal answer:"

    if CONDITION == "robust_original":
        if sample["dataset"] == "gsm8k":
            return f"Solve the math problem. Return only one number. Do not explain.\n\nProblem: {question}\n\nFinal answer:"
        return f"Answer with a short phrase only. Do not explain.\n\nQuestion: {question}\n\nFinal answer:"
    if CONDITION == "robust_para":
        if sample["dataset"] == "gsm8k":
            return f"Find the final result for this word problem. Reply with just the number.\n\nProblem: {question}\n\nFinal answer:"
        return f"Give the concise answer to the question below without explanation.\n\nQuestion: {question}\n\nFinal answer:"
    if CONDITION == "robust_typo":
        instruction = typo_text("Answer with a short phrase only. Do not explain.")
        return f"{instruction}\n\nQuestion: {question}\n\nFinal answer:"
    if CONDITION == "robust_format":
        return f'Return JSON only in this exact schema: {{"answer": "..."}}\n\nQuestion: {question}'

    raise ValueError(f"Unknown condition: {CONDITION}")
'''
        ),
        code_cell(
            r'''
def mock_generate(sample, prompt):
    gold = sample["gold_answer"]
    if CONDITION == "robust_format":
        return json.dumps({"answer": gold})
    if CONDITION.endswith("_cot"):
        return f"We solve it carefully.\nFinal answer: {gold}"
    if CONDITION.endswith("_closed") or CONDITION.endswith("_none"):
        if sample_index(sample) % 4 == 0:
            return "Final answer: unknown"
    return f"Final answer: {gold}"


def load_transformers_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=LOCAL_FILES_ONLY)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        device_map="auto",
        local_files_only=LOCAL_FILES_ONLY,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def transformers_generate(tokenizer, model, prompt):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - start
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    input_tokens = int(inputs["input_ids"].shape[-1])
    output_tokens = int(generated_ids.shape[-1])
    memory_gb = 0.0
    if torch.cuda.is_available():
        memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return text, latency, input_tokens, output_tokens, memory_gb
'''
        ),
        code_cell(
            r'''
rows = read_jsonl(DATA_PATH)
validate_real_rows(rows)
samples = select_samples(rows)
completed = successful_keys(OUTPUT_PATH)

print(
    {
        "axis": AXIS,
        "condition": CONDITION,
        "samples": len(samples),
        "model_name": MODEL_NAME,
        "backend": BACKEND,
        "output_path": str(OUTPUT_PATH),
    }
)

model_bundle = load_transformers_model() if BACKEND == "transformers" else None
records_written = 0

for sample in samples:
    key = unique_key(sample)
    if key in completed:
        continue

    prompt = build_prompt(sample, rows)
    context_metadata = record_context_metadata(sample)
    start = time.perf_counter()
    status = "success"
    error_message = ""
    input_tokens = len(prompt.split())
    output_tokens = 0
    max_memory_gb = 0.0

    try:
        if BACKEND == "transformers":
            tokenizer, model = model_bundle
            raw, latency, input_tokens, output_tokens, max_memory_gb = transformers_generate(tokenizer, model, prompt)
        else:
            raw = mock_generate(sample, prompt)
            latency = time.perf_counter() - start
            output_tokens = len(raw.split())

        prediction = extract_answer(raw, AXIS, CONDITION)
        exact_match, f1, accuracy = score_prediction(prediction, sample["gold_answer"], sample)
        tokens_per_second = output_tokens / latency if latency > 0 else 0.0
    except Exception as exc:
        raw = ""
        prediction = ""
        exact_match = f1 = accuracy = 0.0
        latency = time.perf_counter() - start
        tokens_per_second = 0.0
        status = "error"
        error_message = repr(exc)

    record = {
        "unique_key": key,
        "sample_id": sample["sample_id"],
        "axis": AXIS,
        "dataset": sample["dataset"],
        "condition": CONDITION,
        "model_name": MODEL_NAME,
        "model_size": MODEL_SIZE,
        "prompt_template": PROMPT_TEMPLATE,
        "question": sample["question"],
        "context": context_metadata["text"],
        "gold_answer": sample["gold_answer"],
        "prediction_raw": raw,
        "prediction_normalized": prediction,
        "exact_match": exact_match,
        "f1": f1,
        "accuracy": accuracy,
        "additional_metrics": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_shards": NUM_SHARDS,
            "shard_id": SHARD_ID,
            "robustness_dataset": ROBUSTNESS_DATASET if AXIS == "robustness" else "",
        },
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": latency,
        "tokens_per_second": tokens_per_second,
        "max_memory_allocated_gb": max_memory_gb,
        "num_context_paragraphs": context_metadata["num_context_paragraphs"],
        "num_oracle_paragraphs": context_metadata["num_oracle_paragraphs"],
        "num_distractor_paragraphs": context_metadata["num_distractor_paragraphs"],
        "context_tokens": context_metadata["context_tokens"],
        "gold_at_position": context_metadata["gold_at_position"],
        "context_source": context_metadata["context_source"],
        "context_truncated": context_metadata["context_truncated"],
        "distractor_contains_answer": context_metadata["distractor_contains_answer"],
        "status": status,
        "error_message": error_message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(OUTPUT_PATH, record)
    completed.add(key)
    records_written += 1

print({"output_path": str(OUTPUT_PATH), "records_written": records_written, "total_expected": len(samples)})
'''
        ),
    ]
)


MERGE = notebook(
    [
        markdown_cell(
            """
            # 06 Merge, Evaluate, Plot

            Merge local/Kaggle JSONL records, compute summaries, and save matplotlib figures.
            """
        ),
        code_cell(
            r'''
from pathlib import Path
import json
import os
import re
import string

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_INPUT_DIRS = [
    "outputs",
    "knowledge",
    "reasoning",
    "context",
    "robustness",
    "outputs/knowledge",
    "outputs/reasoning",
    "outputs/context",
    "outputs/robustness",
    "/kaggle/working/outputs",
    "/kaggle/input/slm-limits-outputs",
]
DEFAULT_RECORD_GLOBS = [
    "records_experiment_*.jsonl",
    "records_smoke_qwen2p5_1p5B_real_six_conditions_v4_shard*.jsonl",
    "*.jsonl",
]


def parse_env_list(name, default_values):
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default_values)
    return [value.strip() for value in raw.split(";") if value.strip()]


INPUT_DIRS = [Path(value) for value in parse_env_list("MERGE_INPUT_DIRS", DEFAULT_INPUT_DIRS)]
OUTPUT_DIR = Path(
    os.getenv(
        "MERGE_OUTPUT_DIR",
        "merged_outputs" if not Path("/kaggle/working").exists() else "/kaggle/working/merged_outputs",
    )
)
RECORD_GLOBS = parse_env_list("MERGE_RECORD_GLOBS", DEFAULT_RECORD_GLOBS)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
'''
        ),
        code_cell(
            r'''
def iter_record_paths():
    seen = set()
    for directory in INPUT_DIRS:
        if not directory.exists():
            continue
        for pattern in RECORD_GLOBS:
            paths = list(directory.glob(pattern)) + list(directory.rglob(pattern))
            for path in sorted(paths):
                resolved = path.resolve()
                if resolved not in seen and path.name != "records_all.jsonl":
                    seen.add(resolved)
                    yield path


def read_records(path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON in {path}:{line_number}")
                continue
            row["_source_file"] = str(path)
            yield row


records = []
for path in iter_record_paths():
    records.extend(read_records(path))

df = pd.DataFrame(records)
if df.empty:
    raise ValueError("No records found. Run 01-smoke-test.ipynb or place JSONL files under outputs/.")

df = df[df["status"].fillna("success") == "success"].copy()
if "timestamp" in df:
    df = df.sort_values("timestamp")
semantic_key = ["model_name", "axis", "dataset", "condition", "sample_id"]
df = df.drop_duplicates([column for column in semantic_key if column in df.columns], keep="last")

numeric_columns = ["exact_match", "f1", "accuracy", "latency_seconds", "tokens_per_second", "max_memory_allocated_gb"]
for column in numeric_columns:
    if column in df:
        df[column] = pd.to_numeric(df[column], errors="coerce")


ARTICLES = {"a", "an", "the"}


def normalize_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in text.split() if token not in ARTICLES]
    return " ".join(tokens)


def token_f1(prediction, gold):
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = 0
    remaining = gold_tokens.copy()
    for token in pred_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def clean_short_qa_answer(answer):
    answer = str(answer).strip()
    answer = re.split(
        r"\.\s+(?:To determine|This|Therefore|Based on|I |The |It |We )",
        answer,
        maxsplit=1,
    )[0]
    answer = re.sub(r"^(the answer is|answer is|answer)\s*:?\s*", "", answer, flags=re.IGNORECASE)
    return answer.strip(" .")


def extract_short_qa_answer(text):
    match = re.search(r"final answer\s*:\s*(.*)", str(text), flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return clean_short_qa_answer(lines[0]) if lines else ""


def extract_numeric(text):
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    if not numbers:
        return ""
    value = numbers[-1]
    return value[:-2] if value.endswith(".0") else value


def extract_json_answer(text):
    try:
        parsed = json.loads(str(text).strip())
    except json.JSONDecodeError:
        return ""
    return str(parsed.get("answer", "")) if isinstance(parsed, dict) else ""


def extract_for_row(row):
    raw = row.get("prediction_raw", "")
    if str(row.get("condition", "")).endswith("_format"):
        parsed = extract_json_answer(raw)
        if parsed:
            return parsed
    if row.get("dataset") == "gsm8k":
        return extract_numeric(raw)
    return extract_short_qa_answer(raw)


def score_for_row(row):
    prediction = row["prediction_normalized"]
    gold = row.get("gold_answer", "")
    if row.get("dataset") == "gsm8k":
        pred_num = extract_numeric(prediction)
        gold_num = extract_numeric(gold)
        score = float(pred_num != "" and pred_num == gold_num)
        return pd.Series({"exact_match": score, "f1": score, "accuracy": score})
    exact_match = float(normalize_text(prediction) == normalize_text(gold))
    return pd.Series({"exact_match": exact_match, "f1": token_f1(prediction, gold), "accuracy": exact_match})


df["prediction_normalized"] = df.apply(extract_for_row, axis=1)
df[["exact_match", "f1", "accuracy"]] = df.apply(score_for_row, axis=1)

print({"records": len(df), "conditions": sorted(df["condition"].dropna().unique().tolist())})
'''
        ),
        code_cell(
            r'''
def metric_for_row(row):
    return "accuracy" if row["axis"] == "reasoning" or row["dataset"] == "gsm8k" else "f1"


def row_score(row):
    metric = metric_for_row(row)
    value = row.get(metric)
    return float(value) if pd.notna(value) else 0.0


df["main_score"] = df.apply(row_score, axis=1)

leaderboard = (
    df.groupby(["model_name", "model_size", "axis", "dataset", "condition"], dropna=False)
    .agg(
        n=("unique_key", "count"),
        exact_match_mean=("exact_match", "mean"),
        f1_mean=("f1", "mean"),
        accuracy_mean=("accuracy", "mean"),
        main_score_mean=("main_score", "mean"),
        latency_mean=("latency_seconds", "mean"),
        memory_mean=("max_memory_allocated_gb", "mean"),
        tokens_per_second_mean=("tokens_per_second", "mean"),
    )
    .reset_index()
)

comparison_pairs = [
    ("reasoning_direct", "reasoning_cot", "main_score", "CoT Gain"),
    ("reasoning_direct", "reasoning_oracle_step", "main_score", "Oracle Step Gain"),
    ("context_none", "context_oracle", "main_score", "Oracle Gain"),
    ("context_retrieved", "context_oracle", "main_score", "Retrieval Gap"),
    ("context_oracle_distractor", "context_oracle", "main_score", "Distractor Drop"),
    ("context_oracle_end", "context_oracle", "main_score", "Position Drop"),
    ("knowledge_closed", "knowledge_oracle", "main_score", "Oracle Gain"),
    ("knowledge_distractor", "knowledge_oracle", "main_score", "Distractor Drop"),
    ("robust_original", "robust_para", "main_score", "Paraphrase Sensitivity"),
    ("robust_original", "robust_typo", "main_score", "Typo Sensitivity"),
    ("robust_original", "robust_format", "main_score", "Format Sensitivity"),
]


def matched_comparison(group, base_condition, compare_condition, metric_name):
    columns = ["sample_id", metric_name]
    base = group.loc[group["condition"] == base_condition, columns].dropna(subset=["sample_id"]).copy()
    compare = group.loc[group["condition"] == compare_condition, columns].dropna(subset=["sample_id"]).copy()
    base = base.rename(columns={metric_name: "base_score"}).drop_duplicates("sample_id", keep="last")
    compare = compare.rename(columns={metric_name: "compare_score"}).drop_duplicates("sample_id", keep="last")
    matched = base.merge(compare, on="sample_id", how="inner")
    return base, compare, matched


summary_rows = []
for (model_name, model_size, axis, dataset), group in df.groupby(["model_name", "model_size", "axis", "dataset"], dropna=False):
    for base_condition, compare_condition, metric_name, comparison_name in comparison_pairs:
        if base_condition not in set(group["condition"]) or compare_condition not in set(group["condition"]):
            continue
        base, compare, matched = matched_comparison(group, base_condition, compare_condition, metric_name)
        score_frame = matched if not matched.empty else pd.DataFrame(
            {
                "base_score": [group.loc[group["condition"] == base_condition, metric_name].mean()],
                "compare_score": [group.loc[group["condition"] == compare_condition, metric_name].mean()],
            }
        )
        base_score = score_frame["base_score"].mean()
        compare_score = score_frame["compare_score"].mean()
        if pd.isna(base_score) or pd.isna(compare_score):
            continue
        delta = compare_score - base_score
        summary_rows.append(
            {
                "model_name": model_name,
                "model_size": model_size,
                "axis": axis,
                "dataset": dataset,
                "base_condition": base_condition,
                "compare_condition": compare_condition,
                "n_base": len(base),
                "n_compare": len(compare),
                "n_overlap": len(matched),
                "base_score": base_score,
                "compare_score": compare_score,
                "delta": delta,
                "base_minus_compare": -delta,
                "abs_delta": abs(delta),
                "metric_name": metric_name,
                "comparison_name": comparison_name,
            }
        )

condition_summary = pd.DataFrame(summary_rows)
condition_columns = [
    "model_name",
    "model_size",
    "axis",
    "dataset",
    "base_condition",
    "compare_condition",
    "n_base",
    "n_compare",
    "n_overlap",
    "base_score",
    "compare_score",
    "delta",
    "base_minus_compare",
    "abs_delta",
    "metric_name",
    "comparison_name",
]
condition_summary = condition_summary.reindex(columns=condition_columns)

robustness_rows = []
for (model_name, model_size, dataset), group in df[df["axis"].eq("robustness")].groupby(["model_name", "model_size", "dataset"], dropna=False):
    original = group.loc[group["condition"].eq("robust_original"), "main_score"].mean()
    if pd.isna(original):
        continue
    perturbed = (
        group[group["condition"].isin(["robust_para", "robust_typo", "robust_format"])]
        .groupby("condition", dropna=False)["main_score"]
        .mean()
    )
    if perturbed.empty:
        continue
    best_condition = perturbed.idxmax()
    worst_condition = perturbed.idxmin()
    best_score = float(perturbed.max())
    worst_score = float(perturbed.min())
    robustness_rows.append(
        {
            "model_name": model_name,
            "model_size": model_size,
            "axis": "robustness",
            "dataset": dataset,
            "original_score": float(original),
            "best_perturbed_condition": best_condition,
            "best_perturbed_score": best_score,
            "worst_perturbed_condition": worst_condition,
            "worst_perturbed_score": worst_score,
            "prompt_sensitivity_gap": float((perturbed - original).abs().max()),
            "worst_case_drop": float(original - worst_score),
        }
    )

robustness_summary = pd.DataFrame(robustness_rows)
robustness_columns = [
    "model_name",
    "model_size",
    "axis",
    "dataset",
    "original_score",
    "best_perturbed_condition",
    "best_perturbed_score",
    "worst_perturbed_condition",
    "worst_perturbed_score",
    "prompt_sensitivity_gap",
    "worst_case_drop",
]
robustness_summary = robustness_summary.reindex(columns=robustness_columns)

model_axis_summary = (
    df.groupby(["model_name", "model_size", "axis"], dropna=False)
    .agg(
        n=("unique_key", "count"),
        main_score=("main_score", "mean"),
        latency_mean=("latency_seconds", "mean"),
        memory_mean=("max_memory_allocated_gb", "mean"),
        tokens_per_second_mean=("tokens_per_second", "mean"),
    )
    .reset_index()
)
model_axis_summary["efficiency_score"] = model_axis_summary["main_score"] / model_axis_summary["latency_mean"].clip(lower=1e-9)

paper_main_table = leaderboard[
    [
        "model_name",
        "model_size",
        "axis",
        "dataset",
        "condition",
        "n",
        "main_score_mean",
        "exact_match_mean",
        "f1_mean",
        "accuracy_mean",
        "latency_mean",
        "memory_mean",
    ]
].sort_values(["model_size", "axis", "condition"]).copy()

paper_comparison_table = condition_summary[
    [
        "model_name",
        "model_size",
        "axis",
        "dataset",
        "comparison_name",
        "base_condition",
        "compare_condition",
        "n_overlap",
        "base_score",
        "compare_score",
        "delta",
        "base_minus_compare",
        "abs_delta",
    ]
].sort_values(["model_size", "axis", "comparison_name"]).copy()

paper_axis_table = model_axis_summary.sort_values(["model_size", "axis"]).copy()

efficiency_log = df[
    [
        "unique_key",
        "model_name",
        "model_size",
        "axis",
        "dataset",
        "condition",
        "latency_seconds",
        "tokens_per_second",
        "max_memory_allocated_gb",
        "input_tokens",
        "output_tokens",
    ]
].copy()
'''
        ),
        code_cell(
            r'''
def write_jsonl(path, frame):
    with path.open("w", encoding="utf-8") as f:
        for row in frame.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


write_jsonl(OUTPUT_DIR / "records_all.jsonl", df)
df.to_csv(OUTPUT_DIR / "records_all.csv", index=False)
leaderboard.to_csv(OUTPUT_DIR / "leaderboard.csv", index=False)
condition_summary.to_csv(OUTPUT_DIR / "condition_summary.csv", index=False)
model_axis_summary.to_csv(OUTPUT_DIR / "model_axis_summary.csv", index=False)
robustness_summary.to_csv(OUTPUT_DIR / "robustness_summary.csv", index=False)
efficiency_log.to_csv(OUTPUT_DIR / "efficiency_log.csv", index=False)
paper_main_table.to_csv(OUTPUT_DIR / "paper_main_table.csv", index=False)
paper_comparison_table.to_csv(OUTPUT_DIR / "paper_comparison_table.csv", index=False)
paper_axis_table.to_csv(OUTPUT_DIR / "paper_axis_table.csv", index=False)

print(
    {
        "records_all": len(df),
        "leaderboard_rows": len(leaderboard),
        "condition_summary_rows": len(condition_summary),
        "model_axis_summary_rows": len(model_axis_summary),
        "robustness_summary_rows": len(robustness_summary),
        "output_dir": str(OUTPUT_DIR),
    }
)
'''
        ),
        code_cell(
            r'''
def save_bar(frame, x, y, title, path):
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(frame[x].astype(str), frame[y])
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_heatmap(frame, path):
    if frame.empty:
        return
    matrix = frame.pivot_table(index="model_size", columns="axis", values="main_score", aggfunc="mean")
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(1.8 + len(matrix.columns) * 1.4, 1.6 + len(matrix.index) * 0.7))
    image = ax.imshow(matrix.values, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for row_index, _ in enumerate(matrix.index):
        for col_index, _ in enumerate(matrix.columns):
            value = matrix.iloc[row_index, col_index]
            label = "" if pd.isna(value) else f"{value:.2f}"
            ax.text(col_index, row_index, label, ha="center", va="center", color="white" if value < 0.5 else "black")
    ax.set_title("Model x Axis Score")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def comparison_frame(name):
    frame = condition_summary[condition_summary["comparison_name"].eq(name)].copy()
    if frame.empty:
        return frame
    frame["label"] = (
        frame["model_size"].astype(str)
        + " / "
        + frame["axis"].astype(str)
        + ": "
        + frame["base_condition"].astype(str)
        + " -> "
        + frame["compare_condition"].astype(str)
    )
    return frame


save_heatmap(model_axis_summary, OUTPUT_DIR / "figure_model_axis_heatmap.png")

oracle = comparison_frame("Oracle Gain")
save_bar(oracle, "label", "delta", "Oracle Gain", OUTPUT_DIR / "figure_oracle_gain.png")

distractor = comparison_frame("Distractor Drop")
save_bar(distractor, "label", "delta", "Distractor Drop", OUTPUT_DIR / "figure_distractor_drop.png")

position = comparison_frame("Position Drop")
save_bar(position, "label", "delta", "Position Drop", OUTPUT_DIR / "figure_position_drop.png")

retrieval = comparison_frame("Retrieval Gap")
save_bar(retrieval, "label", "delta", "Retrieval Gap", OUTPUT_DIR / "figure_retrieval_gap.png")

cot = comparison_frame("CoT Gain")
save_bar(cot, "label", "delta", "CoT Gain", OUTPUT_DIR / "figure_cot_gain.png")

prompt = condition_summary[condition_summary["comparison_name"].str.contains("Sensitivity", na=False)].copy()
if not prompt.empty:
    prompt["label"] = prompt["model_size"].astype(str) + ": " + prompt["base_condition"].astype(str) + " -> " + prompt["compare_condition"].astype(str)
save_bar(prompt, "label", "abs_delta", "Prompt Sensitivity Gap", OUTPUT_DIR / "figure_prompt_sensitivity.png")

if not robustness_summary.empty:
    robustness_plot = robustness_summary.copy()
    robustness_plot["label"] = robustness_plot["model_size"].astype(str) + " / " + robustness_plot["dataset"].astype(str)
    save_bar(robustness_plot, "label", "prompt_sensitivity_gap", "Worst Prompt Sensitivity Gap", OUTPUT_DIR / "figure_robustness_gap.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df["latency_seconds"], df["main_score"])
ax.set_xlabel("latency_seconds")
ax.set_ylabel("main_score")
ax.set_title("Score vs Latency")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "figure_accuracy_latency.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df["max_memory_allocated_gb"], df["main_score"])
ax.set_xlabel("max_memory_allocated_gb")
ax.set_ylabel("main_score")
ax.set_title("Score vs Memory")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "figure_accuracy_memory.png", dpi=160)
plt.close(fig)
'''
        ),
    ]
)


def cell_source(cell):
    return "".join(cell["source"])


KAGGLE_JOBS = [
    ("knowledge", "knowledge_closed"),
    ("knowledge", "knowledge_oracle"),
    ("knowledge", "knowledge_distractor"),
    ("reasoning", "reasoning_direct"),
    ("reasoning", "reasoning_cot"),
    ("reasoning", "reasoning_oracle_step"),
    ("context", "context_none"),
    ("context", "context_retrieved"),
    ("context", "context_oracle"),
    ("context", "context_oracle_distractor"),
    ("context", "context_oracle_end"),
    ("robustness", "robust_original"),
    ("robustness", "robust_para"),
    ("robustness", "robust_typo"),
    ("robustness", "robust_format"),
]


def kaggle_end_to_end(model, model_size):
    title = f"Qwen2.5-{model_size} Kaggle End-to-End"
    return notebook(
        [
            markdown_cell(
                f"""
                # {title}

                Run the frozen SLM-limits pilot pipeline for one model across all 15 conditions.
                Edit only the config cell for sample count, shard, paths, or a reduced job list.
                """
            ),
            code_cell(
                f'''
from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import os
import random
import re
import string
import time


def env_int(name, default):
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def env_optional_int(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return None if value.lower() == "none" else int(value)


def env_bool(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.lower() in {{"1", "true", "yes", "y"}}


CORE_DATA_FILES = ["knowledge_data.jsonl", "reasoning_gsm8k.jsonl", "context_hotpotqa.jsonl"]


def has_core_data_files(path):
    return all((path / name).exists() for name in CORE_DATA_FILES)


def candidate_data_dirs():
    yield Path("/kaggle/input/notebooks/nguynnguynhehe/00-prepare-data/slm_limits_data")
    notebooks_root = Path("/kaggle/input/notebooks")
    if notebooks_root.exists():
        yield from sorted(notebooks_root.glob("*/00-prepare-data/slm_limits_data"))
    yield Path("/kaggle/input/slm-limits-data")
    input_root = Path("/kaggle/input")
    if input_root.exists():
        yield from sorted(input_root.glob("*/slm_limits_data"))
    yield Path("slm_limits_data")


def resolve_data_dir(value=None):
    if value:
        return Path(value)
    for candidate in candidate_data_dirs():
        if has_core_data_files(candidate):
            return candidate
    return Path("slm_limits_data")


MODEL = os.getenv("MODEL", "{model}")
MODEL_NAME_MAP = {{
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
}}
MODEL_NAME = MODEL_NAME_MAP.get(MODEL, MODEL)
MODEL_SIZE = MODEL.replace("Qwen/Qwen2.5-", "").replace("Qwen2.5-", "").replace("-Instruct", "")

# Change MAX_SAMPLES to 1000 for the next scale step, or None for full data.
MAX_SAMPLES = env_optional_int("MAX_SAMPLES", 100)
SHARD_ID = env_int("SHARD_ID", 0)
NUM_SHARDS = env_int("NUM_SHARDS", 1)
SEED = env_int("SEED", 42)
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 384)
BACKEND = os.getenv("BACKEND", "transformers")
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", False)
REQUIRE_REAL_DATA = env_bool("REQUIRE_REAL_DATA", True)
OVERWRITE_OUTPUT = env_bool("OVERWRITE_OUTPUT", False)
ROBUSTNESS_DATASET = os.getenv("ROBUSTNESS_DATASET", "gsm8k")

# Auto-detect common Kaggle mounts, including notebook outputs from 00-prepare-data.
DATA_DIR = resolve_data_dir(os.getenv("DATA_DIR"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/kaggle/working/outputs" if Path("/kaggle/working").exists() else "outputs"))
PROMPT_TEMPLATE = "experiment_v2"

# Keep the full frozen pilot by default. Comment out entries here for partial reruns.
RUN_JOBS = {KAGGLE_JOBS!r}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
'''
            ),
            code_cell(
                r'''
AXIS_TO_DATA_PATH = {
    "knowledge": DATA_DIR / "knowledge_data.jsonl",
    "reasoning": DATA_DIR / "reasoning_gsm8k.jsonl",
    "context": DATA_DIR / "context_hotpotqa.jsonl",
    "robustness": DATA_DIR / f"robustness_{ROBUSTNESS_DATASET}.jsonl",
}

AXIS_CONDITIONS = {
    "knowledge": {"knowledge_closed", "knowledge_oracle", "knowledge_distractor"},
    "reasoning": {"reasoning_direct", "reasoning_cot", "reasoning_oracle_step"},
    "context": {"context_none", "context_retrieved", "context_oracle", "context_oracle_distractor", "context_oracle_end"},
    "robustness": {"robust_original", "robust_para", "robust_typo", "robust_format"},
}


def model_tag(value):
    tag = value.split("/")[-1].replace("-Instruct", "")
    return tag.replace(".", "p").replace("-", "_")


def output_path():
    sample_tag = "full" if MAX_SAMPLES is None else f"n{MAX_SAMPLES}"
    name = f"records_experiment_{AXIS}_{model_tag(MODEL_NAME)}_{CONDITION}_{sample_tag}_shard{SHARD_ID}.jsonl"
    return OUTPUT_DIR / name


def set_job(axis, condition):
    global AXIS, CONDITION, DATA_PATH, OUTPUT_PATH
    AXIS = axis
    CONDITION = condition
    if AXIS not in AXIS_TO_DATA_PATH:
        raise ValueError(f"Unsupported AXIS: {AXIS}. Choose one of {sorted(AXIS_TO_DATA_PATH)}.")
    if CONDITION not in AXIS_CONDITIONS[AXIS]:
        allowed = sorted(AXIS_CONDITIONS[AXIS])
        raise ValueError(f"Unsupported CONDITION for {AXIS}: {CONDITION}. Choose one of {allowed}.")
    DATA_PATH = AXIS_TO_DATA_PATH[AXIS]
    OUTPUT_PATH = output_path()
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}. Upload prepared slm_limits_data JSONL files first.")
    if not (0 <= SHARD_ID < NUM_SHARDS):
        raise ValueError("SHARD_ID must satisfy 0 <= SHARD_ID < NUM_SHARDS.")


for axis, condition in RUN_JOBS:
    set_job(axis, condition)

expected_records = len(RUN_JOBS) * (MAX_SAMPLES if MAX_SAMPLES is not None else -1)
print(
    {
        "model_name": MODEL_NAME,
        "model_size": MODEL_SIZE,
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUTPUT_DIR),
        "jobs": len(RUN_JOBS),
        "max_samples_per_condition": MAX_SAMPLES,
        "expected_records_if_not_resumed": expected_records if MAX_SAMPLES is not None else "full shard",
        "shard": f"{SHARD_ID}/{NUM_SHARDS}",
    }
)
'''
            ),
            code_cell(cell_source(EXPERIMENT["cells"][3])),
            code_cell(cell_source(EXPERIMENT["cells"][4])),
            code_cell(cell_source(EXPERIMENT["cells"][5])),
            code_cell(cell_source(EXPERIMENT["cells"][6])),
            code_cell(
                r'''
def run_job(axis, condition, model_bundle):
    set_job(axis, condition)
    rows = read_jsonl(DATA_PATH)
    validate_real_rows(rows)
    samples = select_samples(rows)
    completed = successful_keys(OUTPUT_PATH)

    print(
        {
            "axis": AXIS,
            "condition": CONDITION,
            "samples": len(samples),
            "already_completed": len(completed),
            "model_name": MODEL_NAME,
            "backend": BACKEND,
            "output_path": str(OUTPUT_PATH),
        }
    )

    records_written = 0
    for sample in samples:
        key = unique_key(sample)
        if key in completed:
            continue

        prompt = build_prompt(sample, rows)
        context_metadata = record_context_metadata(sample)
        start = time.perf_counter()
        status = "success"
        error_message = ""
        input_tokens = len(prompt.split())
        output_tokens = 0
        max_memory_gb = 0.0

        try:
            if BACKEND == "transformers":
                tokenizer, model = model_bundle
                raw, latency, input_tokens, output_tokens, max_memory_gb = transformers_generate(tokenizer, model, prompt)
            else:
                raw = mock_generate(sample, prompt)
                latency = time.perf_counter() - start
                output_tokens = len(raw.split())

            prediction = extract_answer(raw, AXIS, CONDITION)
            exact_match, f1, accuracy = score_prediction(prediction, sample["gold_answer"], sample)
            tokens_per_second = output_tokens / latency if latency > 0 else 0.0
        except Exception as exc:
            raw = ""
            prediction = ""
            exact_match = f1 = accuracy = 0.0
            latency = time.perf_counter() - start
            tokens_per_second = 0.0
            status = "error"
            error_message = repr(exc)

        record = {
            "unique_key": key,
            "sample_id": sample["sample_id"],
            "axis": AXIS,
            "dataset": sample["dataset"],
            "condition": CONDITION,
            "model_name": MODEL_NAME,
            "model_size": MODEL_SIZE,
            "prompt_template": PROMPT_TEMPLATE,
            "question": sample["question"],
            "context": context_metadata["text"],
            "gold_answer": sample["gold_answer"],
            "prediction_raw": raw,
            "prediction_normalized": prediction,
            "exact_match": exact_match,
            "f1": f1,
            "accuracy": accuracy,
            "additional_metrics": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_new_tokens": MAX_NEW_TOKENS,
                "num_shards": NUM_SHARDS,
                "shard_id": SHARD_ID,
                "robustness_dataset": ROBUSTNESS_DATASET if AXIS == "robustness" else "",
            },
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_seconds": latency,
            "tokens_per_second": tokens_per_second,
            "max_memory_allocated_gb": max_memory_gb,
            "num_context_paragraphs": context_metadata["num_context_paragraphs"],
            "num_oracle_paragraphs": context_metadata["num_oracle_paragraphs"],
            "num_distractor_paragraphs": context_metadata["num_distractor_paragraphs"],
            "context_tokens": context_metadata["context_tokens"],
            "gold_at_position": context_metadata["gold_at_position"],
            "context_source": context_metadata["context_source"],
            "context_truncated": context_metadata["context_truncated"],
            "distractor_contains_answer": context_metadata["distractor_contains_answer"],
            "status": status,
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(OUTPUT_PATH, record)
        completed.add(key)
        records_written += 1

    result = {"axis": AXIS, "condition": CONDITION, "records_written": records_written, "total_expected": len(samples)}
    print(result)
    return result


model_bundle = load_transformers_model() if BACKEND == "transformers" else None
job_results = []
for axis, condition in RUN_JOBS:
    job_results.append(run_job(axis, condition, model_bundle))

if BACKEND == "transformers":
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
gc.collect()

print({"completed_jobs": len(job_results), "records_written": sum(item["records_written"] for item in job_results)})
'''
            ),
        ]
    )


KAGGLE_QWEN_3B = kaggle_end_to_end("Qwen2.5-3B", "3B")
KAGGLE_QWEN_7B = kaggle_end_to_end("Qwen2.5-7B", "7B")


def write_notebook(path, nb):
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    write_notebook("00-prepare-data.ipynb", PREPARE)
    write_notebook("01-smoke-test.ipynb", SMOKE)
    write_notebook("02-run-experiment.ipynb", EXPERIMENT)
    write_notebook("03-kaggle-qwen2p5-3b-end-to-end.ipynb", KAGGLE_QWEN_3B)
    write_notebook("04-kaggle-qwen2p5-7b-end-to-end.ipynb", KAGGLE_QWEN_7B)
    write_notebook("06-merge-evaluate-plot.ipynb", MERGE)


if __name__ == "__main__":
    main()
