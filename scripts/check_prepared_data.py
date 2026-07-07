import json
from pathlib import Path


REQUIRED_FILES = [
    Path("slm_limits_data/reasoning_gsm8k.jsonl"),
    Path("slm_limits_data/context_hotpotqa.jsonl"),
    Path("slm_limits_data/knowledge_data.jsonl"),
]


def read_rows(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    report = {}
    for path in REQUIRED_FILES:
        if not path.exists():
            report[str(path)] = {"exists": False}
            continue
        rows = read_rows(path)
        sources = sorted({row.get("metadata", {}).get("source", "") for row in rows})
        report[str(path)] = {
            "exists": True,
            "rows": len(rows),
            "sources": sources,
            "looks_real": bool(rows) and sources != ["local_fallback"],
        }
    print(report)


if __name__ == "__main__":
    main()
