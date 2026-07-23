"""
Recompute summary tables and redraw all figures from an existing
merged_outputs/records_all.csv (after robustness fix).

Usage:
  python scripts/replot_merged_outputs.py
  python scripts/replot_merged_outputs.py --output-dir "C:\\Users\\Administrator\\Downloads\\merged_outputs"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def metric_for_row(row: pd.Series) -> str:
    if row["axis"] == "robustness" and row["dataset"] == "robustness_gsm8k":
        return "accuracy"
    return "accuracy" if row["axis"] == "reasoning" or row["dataset"] == "gsm8k" else "f1"


def row_score(row: pd.Series) -> float:
    metric = metric_for_row(row)
    value = row.get(metric)
    return float(value) if pd.notna(value) else 0.0


def matched_comparison(group: pd.DataFrame, base_condition: str, compare_condition: str, metric_name: str):
    columns = ["sample_id", metric_name]
    base = group.loc[group["condition"] == base_condition, columns].dropna(subset=["sample_id"]).copy()
    compare = group.loc[group["condition"] == compare_condition, columns].dropna(subset=["sample_id"]).copy()
    base = base.rename(columns={metric_name: "base_score"}).drop_duplicates("sample_id", keep="last")
    compare = compare.rename(columns={metric_name: "compare_score"}).drop_duplicates("sample_id", keep="last")
    matched = base.merge(compare, on="sample_id", how="inner")
    return base, compare, matched


def build_summaries(df: pd.DataFrame):
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

    summary_rows = []
    for (model_name, model_size, axis, dataset), group in df.groupby(
        ["model_name", "model_size", "axis", "dataset"], dropna=False
    ):
        for base_condition, compare_condition, metric_name, comparison_name in comparison_pairs:
            if base_condition not in set(group["condition"]) or compare_condition not in set(group["condition"]):
                continue
            base, compare, matched = matched_comparison(group, base_condition, compare_condition, metric_name)
            score_frame = (
                matched
                if not matched.empty
                else pd.DataFrame(
                    {
                        "base_score": [group.loc[group["condition"] == base_condition, metric_name].mean()],
                        "compare_score": [group.loc[group["condition"] == compare_condition, metric_name].mean()],
                    }
                )
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
    for (model_name, model_size, dataset), group in df[df["axis"].eq("robustness")].groupby(
        ["model_name", "model_size", "dataset"], dropna=False
    ):
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
    model_axis_summary["efficiency_score"] = model_axis_summary["main_score"] / model_axis_summary[
        "latency_mean"
    ].clip(lower=1e-9)

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

    return {
        "leaderboard": leaderboard,
        "condition_summary": condition_summary,
        "model_axis_summary": model_axis_summary,
        "robustness_summary": robustness_summary,
        "efficiency_log": efficiency_log,
        "paper_main_table": paper_main_table,
        "paper_comparison_table": paper_comparison_table,
        "paper_axis_table": paper_axis_table,
    }


def save_bar(frame: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
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


def save_heatmap(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    matrix = frame.pivot_table(index="model_size", columns="axis", values="main_score", aggfunc="mean")
    if matrix.empty:
        return
    # Stable axis order for readability
    preferred = [c for c in ["context", "knowledge", "reasoning", "robustness"] if c in matrix.columns]
    other = [c for c in matrix.columns if c not in preferred]
    matrix = matrix[preferred + other]
    size_order = [s for s in ["1.5B", "3B", "7B"] if s in matrix.index]
    other_sizes = [s for s in matrix.index if s not in size_order]
    matrix = matrix.reindex(size_order + other_sizes)

    fig, ax = plt.subplots(figsize=(1.8 + len(matrix.columns) * 1.4, 1.6 + len(matrix.index) * 0.7))
    image = ax.imshow(matrix.values, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for row_index, _ in enumerate(matrix.index):
        for col_index, _ in enumerate(matrix.columns):
            value = matrix.iloc[row_index, col_index]
            label = "" if pd.isna(value) else f"{value:.2f}"
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white" if (pd.notna(value) and value < 0.5) else "black",
            )
    ax.set_title("Model x Axis Score")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_all(df: pd.DataFrame, summaries: dict, output_dir: Path) -> list[str]:
    condition_summary = summaries["condition_summary"]
    model_axis_summary = summaries["model_axis_summary"]
    robustness_summary = summaries["robustness_summary"]
    written: list[str] = []

    def comparison_frame(name: str) -> pd.DataFrame:
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

    path = output_dir / "figure_model_axis_heatmap.png"
    save_heatmap(model_axis_summary, path)
    written.append(path.name)

    for name, filename, title in [
        ("Oracle Gain", "figure_oracle_gain.png", "Oracle Gain"),
        ("Distractor Drop", "figure_distractor_drop.png", "Distractor Drop"),
        ("Position Drop", "figure_position_drop.png", "Position Drop"),
        ("Retrieval Gap", "figure_retrieval_gap.png", "Retrieval Gap"),
        ("CoT Gain", "figure_cot_gain.png", "CoT Gain"),
    ]:
        frame = comparison_frame(name)
        path = output_dir / filename
        save_bar(frame, "label", "delta", title, path)
        if path.exists():
            written.append(path.name)

    prompt = condition_summary[condition_summary["comparison_name"].str.contains("Sensitivity", na=False)].copy()
    if not prompt.empty:
        prompt["label"] = (
            prompt["model_size"].astype(str)
            + ": "
            + prompt["base_condition"].astype(str)
            + " -> "
            + prompt["compare_condition"].astype(str)
        )
    path = output_dir / "figure_prompt_sensitivity.png"
    save_bar(prompt, "label", "abs_delta", "Prompt Sensitivity Gap", path)
    if path.exists():
        written.append(path.name)

    if not robustness_summary.empty:
        robustness_plot = robustness_summary.copy()
        robustness_plot["label"] = (
            robustness_plot["model_size"].astype(str) + " / " + robustness_plot["dataset"].astype(str)
        )
        path = output_dir / "figure_robustness_gap.png"
        save_bar(
            robustness_plot,
            "label",
            "prompt_sensitivity_gap",
            "Worst Prompt Sensitivity Gap",
            path,
        )
        written.append(path.name)

    path = output_dir / "figure_accuracy_latency.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["latency_seconds"], df["main_score"], alpha=0.35, s=8)
    ax.set_xlabel("latency_seconds")
    ax.set_ylabel("main_score")
    ax.set_title("Score vs Latency")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path.name)

    path = output_dir / "figure_accuracy_memory.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["max_memory_allocated_gb"], df["main_score"], alpha=0.35, s=8)
    ax.set_xlabel("max_memory_allocated_gb")
    ax.set_ylabel("main_score")
    ax.set_title("Score vs Memory")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path.name)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot figures from merged_outputs records")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\Administrator\Downloads\merged_outputs"),
        help="Directory containing records_all.csv and where figures will be written",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="Optional path to records_all.csv (defaults to <output-dir>/records_all.csv)",
    )
    parser.add_argument(
        "--skip-csv-write",
        action="store_true",
        help="Only redraw figures; do not overwrite summary CSVs",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    records_path = args.records or (output_dir / "records_all.csv")
    if not records_path.exists():
        raise SystemExit(f"records not found: {records_path}")

    print(f"Loading {records_path} ...")
    df = pd.read_csv(records_path, low_memory=False)
    if df.empty:
        raise SystemExit("records_all is empty")

    numeric_columns = [
        "exact_match",
        "f1",
        "accuracy",
        "main_score",
        "latency_seconds",
        "tokens_per_second",
        "max_memory_allocated_gb",
        "input_tokens",
        "output_tokens",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Recompute main_score with correct metric rules (robustness -> accuracy).
    df["main_score"] = df.apply(row_score, axis=1)

    print(
        {
            "n_records": len(df),
            "axes": sorted(df["axis"].dropna().unique().tolist()),
            "models": sorted(df["model_size"].dropna().unique().tolist()),
        }
    )

    print("Building summaries ...")
    summaries = build_summaries(df)

    if not args.skip_csv_write:
        print("Writing summary CSVs ...")
        summaries["leaderboard"].to_csv(output_dir / "leaderboard.csv", index=False)
        summaries["condition_summary"].to_csv(output_dir / "condition_summary.csv", index=False)
        summaries["model_axis_summary"].to_csv(output_dir / "model_axis_summary.csv", index=False)
        summaries["robustness_summary"].to_csv(output_dir / "robustness_summary.csv", index=False)
        summaries["efficiency_log"].to_csv(output_dir / "efficiency_log.csv", index=False)
        summaries["paper_main_table"].to_csv(output_dir / "paper_main_table.csv", index=False)
        summaries["paper_comparison_table"].to_csv(output_dir / "paper_comparison_table.csv", index=False)
        summaries["paper_axis_table"].to_csv(output_dir / "paper_axis_table.csv", index=False)
        # Keep records main_score consistent with recomputed metric.
        df.to_csv(output_dir / "records_all.csv", index=False)

    print("Drawing figures ...")
    written = plot_all(df, summaries, output_dir)
    for name in written:
        path = output_dir / name
        print(f"  wrote {name} ({path.stat().st_size} bytes)")

    # Quick sanity: robustness main_score should match accuracy mean on leaderboard
    rob = summaries["leaderboard"][summaries["leaderboard"]["axis"] == "robustness"]
    if not rob.empty:
        sample = rob.iloc[0]
        print(
            "sanity robustness sample:",
            sample["model_size"],
            sample["condition"],
            "main=",
            round(float(sample["main_score_mean"]), 4),
            "acc=",
            round(float(sample["accuracy_mean"]), 4),
        )

    print("Done.")


if __name__ == "__main__":
    main()
