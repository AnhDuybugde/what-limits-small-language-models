import csv
import json
from pathlib import Path


def count_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def count_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def main():
    records_path = Path("outputs/records_smoke_qwen2p5_1p5B_real_six_conditions_v4_shard0.jsonl")
    condition_path = Path("merged_outputs/condition_summary.csv")
    model_axis_path = Path("merged_outputs/model_axis_summary.csv")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    conditions = sorted({row["condition"] for row in records})
    print(
        {
            "records": count_jsonl(records_path),
            "conditions": conditions,
            "condition_summary_rows": count_csv_rows(condition_path),
            "model_axis_summary_rows": count_csv_rows(model_axis_path),
        }
    )


if __name__ == "__main__":
    main()
