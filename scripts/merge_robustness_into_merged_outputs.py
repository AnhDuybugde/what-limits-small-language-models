"""
Replace robustness axis data in merged_outputs with values from
merged_outputs_robustness (corrected re-run / fixed main_score).

Usage (PowerShell):
  python scripts/merge_robustness_into_merged_outputs.py
  python scripts/merge_robustness_into_merged_outputs.py --dry-run
  python scripts/merge_robustness_into_merged_outputs.py ^
      --main "C:\\Users\\Administrator\\Downloads\\merged_outputs" ^
      --robust "C:\\Users\\Administrator\\Downloads\\merged_outputs_robustness"
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

AXIS = "robustness"

# Summary tables: replace rows where axis == robustness (or whole file for robustness-only).
SUMMARY_BY_AXIS = [
    "leaderboard.csv",
    "condition_summary.csv",
    "model_axis_summary.csv",
    "paper_axis_table.csv",
    "paper_comparison_table.csv",
    "paper_main_table.csv",
]

# Full replace (robustness-only tables).
SUMMARY_FULL_REPLACE = [
    "robustness_summary.csv",
]

# Row-level files: drop old robustness rows, append new ones.
ROW_LEVEL_CSV = [
    "records_all.csv",
    "efficiency_log.csv",
]

# Copy figures that are robustness-only or recomputed from full data in robust folder.
FIGURE_COPY = [
    "figure_robustness_gap.png",
    "figure_prompt_sensitivity.png",
    # These two in robust folder only cover robustness samples; optional overwrite.
    # Prefer regenerating from full merge if you re-run the notebook.
    "figure_model_axis_heatmap.png",
    "figure_accuracy_latency.png",
    "figure_accuracy_memory.png",
]


def read_csv_dicts(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def write_csv_dicts(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def align_row(row: dict[str, str], fieldnames: list[str]) -> dict[str, str]:
    """Map source row onto target column order; fill missing keys with empty string."""
    return {k: row.get(k, "") for k in fieldnames}


def replace_axis_rows(
    main_fields: list[str],
    main_rows: list[dict[str, str]],
    rob_rows: list[dict[str, str]],
    axis_value: str = AXIS,
) -> list[dict[str, str]]:
    kept = [r for r in main_rows if r.get("axis") != axis_value]
    added = [align_row(r, main_fields) for r in rob_rows if r.get("axis") == axis_value]
    # If robust file is robustness-only and has no axis filter match, take all rob rows.
    if not added and rob_rows:
        added = [align_row(r, main_fields) for r in rob_rows]
    return kept + added


def merge_summary_by_axis(main_path: Path, rob_path: Path, dry_run: bool) -> dict:
    main_fields, main_rows = read_csv_dicts(main_path)
    _, rob_rows = read_csv_dicts(rob_path)
    before = sum(1 for r in main_rows if r.get("axis") == AXIS)
    after_src = sum(1 for r in rob_rows if r.get("axis") == AXIS) or len(rob_rows)
    merged = replace_axis_rows(main_fields, main_rows, rob_rows)
    after = sum(1 for r in merged if r.get("axis") == AXIS)
    if not dry_run:
        write_csv_dicts(main_path, main_fields, merged)
    return {
        "file": main_path.name,
        "main_rows_before": len(main_rows),
        "main_rows_after": len(merged),
        "robustness_before": before,
        "robustness_after": after,
        "robustness_from_src": after_src,
    }


def merge_row_level_csv(main_path: Path, rob_path: Path, dry_run: bool) -> dict:
    """Stream non-robustness rows from main, then append all rows from robust."""
    main_fields, _ = read_csv_dicts(main_path)  # only need header + count via stream
    # Re-open streaming to avoid holding 80MB+ twice in memory as full list when possible.
    # For simplicity and safety on column align, we still materialize; files are ~45k rows.
    main_fields, main_rows = read_csv_dicts(main_path)
    rob_fields, rob_rows = read_csv_dicts(rob_path)

    before_total = len(main_rows)
    before_rob = sum(1 for r in main_rows if r.get("axis") == AXIS)
    kept = [r for r in main_rows if r.get("axis") != AXIS]
    # Prefer target column order from main.
    added = [align_row(r, main_fields) for r in rob_rows]
    # Ensure main_score exists for every row (main had it at end).
    if "main_score" in main_fields:
        for r in added:
            if r.get("main_score", "") == "" and "main_score" in rob_fields:
                r["main_score"] = r.get("main_score", "")
            # already aligned; if rob has main_score in different position DictReader still has key
            if not r.get("main_score"):
                # fallback: if missing, leave empty
                pass

    merged = kept + added
    if not dry_run:
        write_csv_dicts(main_path, main_fields, merged)

    return {
        "file": main_path.name,
        "main_rows_before": before_total,
        "main_rows_after": len(merged),
        "robustness_before": before_rob,
        "robustness_after": len(added),
        "non_robustness_kept": len(kept),
    }


def merge_records_jsonl(main_path: Path, rob_path: Path, dry_run: bool) -> dict:
    """Rewrite jsonl: keep non-robustness from main, append all from robust."""
    kept_path = main_path.with_suffix(".jsonl.tmp")
    n_kept = 0
    n_dropped = 0
    n_added = 0

    if dry_run:
        with main_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("axis") == AXIS:
                    n_dropped += 1
                else:
                    n_kept += 1
        with rob_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n_added += 1
        return {
            "file": main_path.name,
            "non_robustness_kept": n_kept,
            "robustness_dropped": n_dropped,
            "robustness_added": n_added,
        }

    with main_path.open("r", encoding="utf-8") as src, kept_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as out:
        for line in src:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("axis") == AXIS:
                n_dropped += 1
                continue
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_kept += 1

        with rob_path.open("r", encoding="utf-8") as rob_f:
            for line in rob_f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_added += 1

    kept_path.replace(main_path)
    return {
        "file": main_path.name,
        "non_robustness_kept": n_kept,
        "robustness_dropped": n_dropped,
        "robustness_added": n_added,
    }


def full_replace(main_path: Path, rob_path: Path, dry_run: bool) -> dict:
    if not dry_run:
        shutil.copy2(rob_path, main_path)
    return {"file": main_path.name, "action": "full_copy", "bytes": rob_path.stat().st_size}


def copy_figures(main_dir: Path, rob_dir: Path, dry_run: bool) -> list[dict]:
    results = []
    for name in FIGURE_COPY:
        src = rob_dir / name
        dst = main_dir / name
        if not src.exists():
            results.append({"file": name, "action": "skip_missing_src"})
            continue
        if not dry_run:
            shutil.copy2(src, dst)
        results.append({"file": name, "action": "copied", "bytes": src.stat().st_size})
    return results


def backup_dir(main_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = main_dir.parent / f"{main_dir.name}_backup_before_robustness_{stamp}"
    shutil.copytree(main_dir, backup)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject robustness outputs into merged_outputs")
    parser.add_argument(
        "--main",
        type=Path,
        default=Path(r"C:\Users\Administrator\Downloads\merged_outputs"),
        help="Target merged_outputs directory",
    )
    parser.add_argument(
        "--robust",
        type=Path,
        default=Path(r"C:\Users\Administrator\Downloads\merged_outputs_robustness"),
        help="Source merged_outputs_robustness directory",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only, do not write")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip full directory backup (not recommended)",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Do not copy PNG figures from robust folder",
    )
    args = parser.parse_args()

    main_dir: Path = args.main
    rob_dir: Path = args.robust

    if not main_dir.is_dir():
        raise SystemExit(f"Main dir not found: {main_dir}")
    if not rob_dir.is_dir():
        raise SystemExit(f"Robust dir not found: {rob_dir}")

    print(f"MAIN  : {main_dir}")
    print(f"ROBUST: {rob_dir}")
    print(f"MODE  : {'DRY-RUN' if args.dry_run else 'WRITE'}")

    if not args.dry_run and not args.no_backup:
        print("Creating full backup of main dir ...")
        backup = backup_dir(main_dir)
        print(f"Backup -> {backup}")

    reports: list[dict] = []

    for name in SUMMARY_BY_AXIS:
        m, r = main_dir / name, rob_dir / name
        if not m.exists() or not r.exists():
            reports.append({"file": name, "action": "skip_missing", "main": m.exists(), "rob": r.exists()})
            continue
        reports.append(merge_summary_by_axis(m, r, args.dry_run))

    for name in SUMMARY_FULL_REPLACE:
        m, r = main_dir / name, rob_dir / name
        if not m.exists() or not r.exists():
            reports.append({"file": name, "action": "skip_missing", "main": m.exists(), "rob": r.exists()})
            continue
        reports.append(full_replace(m, r, args.dry_run))

    for name in ROW_LEVEL_CSV:
        m, r = main_dir / name, rob_dir / name
        if not m.exists() or not r.exists():
            reports.append({"file": name, "action": "skip_missing", "main": m.exists(), "rob": r.exists()})
            continue
        reports.append(merge_row_level_csv(m, r, args.dry_run))

    jsonl_main = main_dir / "records_all.jsonl"
    jsonl_rob = rob_dir / "records_all.jsonl"
    if jsonl_main.exists() and jsonl_rob.exists():
        reports.append(merge_records_jsonl(jsonl_main, jsonl_rob, args.dry_run))
    else:
        reports.append(
            {
                "file": "records_all.jsonl",
                "action": "skip_missing",
                "main": jsonl_main.exists(),
                "rob": jsonl_rob.exists(),
            }
        )

    if not args.skip_figures:
        reports.extend(copy_figures(main_dir, rob_dir, args.dry_run))

    print("\n=== REPORT ===")
    for item in reports:
        print(item)

    print("\nDone.")
    if not args.dry_run:
        print(
            "Note: figure_accuracy_latency / figure_accuracy_memory / figure_model_axis_heatmap "
            "from the robust folder only reflect robustness samples. Re-run "
            "06-merge-evaluate-plot.ipynb on the updated records_all for full figures."
        )


if __name__ == "__main__":
    main()
