"""
Redraw manuscript figures with clean labels:
- scaling trends: values above the plot (no overlap with lines)
- bar charts: grouped bars + short straight labels (no diagonal text)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "QWEN2.5-merged-outputs"
OUT_DIRS = [SRC, ROOT / "combo_latex" / "Images"]

SIZE_ORDER = ["1.5B", "3B", "7B"]
AXIS_MAP = {
    "knowledge": "Knowledge",
    "reasoning": "Reasoning",
    "context": "Context",
    "robustness": "Prompt",
}


def save_all(fig: plt.Figure, name: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print("saved", path)
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)


def annotate_bars(ax: plt.Axes, bars, values, fmt: str = "{:.2f}", dy: float = 3) -> None:
    for bar, val in zip(bars, values):
        ax.annotate(
            fmt.format(val),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#222222",
        )


def plot_scaling_trends(model_axis: pd.DataFrame) -> None:
    """
    Clean line chart:
    - trend lines free of overlapping text
    - numeric values printed in a small score table ABOVE the chart
    - series names only in the legend
    """
    df = model_axis.copy()
    df["axis_label"] = df["axis"].map(AXIS_MAP)
    order_axis = ["Knowledge", "Reasoning", "Context", "Prompt"]
    pivot = (
        df.pivot(index="axis_label", columns="model_size", values="main_score")
        .reindex(order_axis)[SIZE_ORDER]
    )

    colors = {
        "Knowledge": "#1f77b4",
        "Reasoning": "#ff7f0e",
        "Context": "#2ca02c",
        "Prompt": "#d62728",
    }
    markers = {
        "Knowledge": "o",
        "Reasoning": "s",
        "Context": "D",
        "Prompt": "^",
    }

    fig = plt.figure(figsize=(7.2, 5.4))
    # Title on its own band (above table), then table, then line chart
    ax_title = fig.add_axes([0.10, 0.93, 0.82, 0.05])
    ax_tbl = fig.add_axes([0.10, 0.72, 0.82, 0.20])
    ax = fig.add_axes([0.10, 0.10, 0.82, 0.56])

    ax_title.axis("off")
    ax_title.text(
        0.5,
        0.5,
        "Scaling trends across diagnostic axes",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax_title.transAxes,
    )

    # --- score table (values live OUTSIDE the trend lines) ---
    ax_tbl.axis("off")
    col_labels = ["Axis"] + SIZE_ORDER
    cell_text = []
    cell_colors = []
    for axis in order_axis:
        row = [axis] + [f"{pivot.loc[axis, s]:.2f}" for s in SIZE_ORDER]
        cell_text.append(row)
        cell_colors.append(["#f7f7f7"] + ["white"] * len(SIZE_ORDER))
    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colColours=["#e8eef6"] * len(col_labels),
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    for i, axis in enumerate(order_axis):
        table[(i + 1, 0)].get_text().set_color(colors[axis])
        table[(i + 1, 0)].get_text().set_fontweight("medium")

    # --- line chart (no numeric annotations on the lines) ---
    x = np.arange(len(SIZE_ORDER), dtype=float)
    for axis in order_axis:
        y = pivot.loc[axis].values.astype(float)
        ax.plot(
            x,
            y,
            color=colors[axis],
            marker=markers[axis],
            markersize=9,
            linewidth=2.3,
            label=axis,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_ORDER, fontsize=11)
    ax.set_xlabel("Model size", fontsize=11)
    ax.set_ylabel("Mean axis score", fontsize=11)
    ax.set_ylim(0.0, min(1.05, float(pivot.max().max()) + 0.12))
    style_axes(ax)
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=9, columnspacing=1.0)

    save_all(fig, "figure_scaling_trends.png")


def plot_heatmap(model_axis: pd.DataFrame) -> None:
    matrix = model_axis.pivot_table(index="model_size", columns="axis", values="main_score", aggfunc="mean")
    preferred = [c for c in ["knowledge", "reasoning", "context", "robustness"] if c in matrix.columns]
    matrix = matrix[preferred]
    matrix = matrix.reindex([s for s in SIZE_ORDER if s in matrix.index])
    matrix = matrix.rename(columns=AXIS_MAP)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    image = ax.imshow(matrix.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(list(matrix.columns), rotation=0, ha="center", fontsize=10)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(list(matrix.index), fontsize=10)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = float(matrix.iloc[i, j])
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.45 else "black",
                fontsize=10,
                fontweight="medium",
            )
    ax.set_title("Mean score by model size and diagnostic axis", fontsize=12, pad=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    save_all(fig, "figure_model_axis_heatmap.png")


def plot_oracle_gain(condition: pd.DataFrame) -> None:
    """Grouped bars: x=model size, series=Knowledge / Context oracle gains."""
    frame = condition[condition["comparison_name"].eq("Oracle Gain")].copy()
    # knowledge closed->oracle and context none->oracle
    rows = []
    for size in SIZE_ORDER:
        k = frame[(frame.model_size == size) & (frame.axis == "knowledge")]
        c = frame[(frame.model_size == size) & (frame.axis == "context")]
        rows.append(
            {
                "size": size,
                "Knowledge\n(closed→oracle)": float(k["delta"].iloc[0]) if not k.empty else np.nan,
                "Context\n(none→oracle)": float(c["delta"].iloc[0]) if not c.empty else np.nan,
            }
        )
    wide = pd.DataFrame(rows).set_index("size")

    x = np.arange(len(SIZE_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    b1 = ax.bar(x - width / 2, wide.iloc[:, 0], width, label="Knowledge (closed→oracle)", color="#1f77b4")
    b2 = ax.bar(x + width / 2, wide.iloc[:, 1], width, label="Context (none→oracle)", color="#2ca02c")
    annotate_bars(ax, b1, wide.iloc[:, 0].values)
    annotate_bars(ax, b2, wide.iloc[:, 1].values)
    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_ORDER, fontsize=11)
    ax.set_xlabel("Model size", fontsize=11)
    ax.set_ylabel("Delta", fontsize=11)
    ax.set_title("Oracle Gain", fontsize=12, pad=8)
    ax.set_ylim(0, float(np.nanmax(wide.values)) + 0.08)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    save_all(fig, "figure_oracle_gain.png")


def plot_retrieval_gap(condition: pd.DataFrame) -> None:
    frame = condition[condition["comparison_name"].eq("Retrieval Gap")].copy()
    size_rank = {s: i for i, s in enumerate(SIZE_ORDER)}
    frame["_sr"] = frame["model_size"].map(size_rank)
    frame = frame.sort_values("_sr")
    values = frame["delta"].astype(float).values
    labels = frame["model_size"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    x = np.arange(len(values))
    bars = ax.bar(x, values, color="#3b6ea5", width=0.55)
    annotate_bars(ax, bars, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Model size", fontsize=11)
    ax.set_ylabel("Delta (oracle − retrieved)", fontsize=11)
    ax.set_title("Retrieval Gap", fontsize=12, pad=8)
    ax.set_ylim(0, values.max() + 0.05)
    style_axes(ax)
    fig.tight_layout()
    save_all(fig, "figure_retrieval_gap.png")


def plot_prompt_sensitivity(condition: pd.DataFrame) -> None:
    """Grouped bars: x=model size, series=Paraphrase / Typo / JSON abs deltas."""
    prompt = condition[condition["comparison_name"].str.contains("Sensitivity", na=False)].copy()
    name_map = {
        "Paraphrase Sensitivity": "Paraphrase",
        "Typo Sensitivity": "Typo",
        "Format Sensitivity": "JSON",
    }
    prompt["series"] = prompt["comparison_name"].map(name_map)
    series_order = ["Paraphrase", "Typo", "JSON"]
    colors = {"Paraphrase": "#1f77b4", "Typo": "#ff7f0e", "JSON": "#d62728"}

    x = np.arange(len(SIZE_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, series in enumerate(series_order):
        vals = []
        for size in SIZE_ORDER:
            sub = prompt[(prompt.model_size == size) & (prompt.series == series)]
            vals.append(float(sub["abs_delta"].iloc[0]) if not sub.empty else 0.0)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=series, color=colors[series])
        annotate_bars(ax, bars, vals, dy=2)

    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_ORDER, fontsize=11)
    ax.set_xlabel("Model size", fontsize=11)
    ax.set_ylabel("Absolute delta vs original", fontsize=11)
    ax.set_title("Prompt Sensitivity Gap", fontsize=12, pad=8)
    ax.set_ylim(0, 0.12)
    style_axes(ax)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")
    fig.tight_layout()
    save_all(fig, "figure_prompt_sensitivity.png")


def plot_simple_comparison(condition: pd.DataFrame, name: str, filename: str, title: str) -> None:
    """Grouped-friendly simple bars for optional extras (CoT, distractor, position)."""
    frame = condition[condition["comparison_name"].eq(name)].copy()
    if frame.empty:
        return
    size_rank = {s: i for i, s in enumerate(SIZE_ORDER)}
    frame["_sr"] = frame["model_size"].map(size_rank)
    frame = frame.sort_values(["_sr", "axis"])

    # Prefer short labels: size + short axis/condition if multiple series
    if frame["axis"].nunique() > 1 or frame["base_condition"].nunique() > 1:
        labels = [
            f"{r.model_size}\n{AXIS_MAP.get(str(r.axis), str(r.axis))}"
            for _, r in frame.iterrows()
        ]
    else:
        labels = frame["model_size"].astype(str).tolist()
    values = frame["delta"].astype(float).values

    fig, ax = plt.subplots(figsize=(max(5.5, 1.1 * len(values) + 2), 4.0))
    x = np.arange(len(values))
    bars = ax.bar(x, values, color="#3b6ea5", width=0.6)
    annotate_bars(ax, bars, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
    ax.set_ylabel("Delta", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    pad = 0.05 * max(abs(values.max()), abs(values.min()), 0.01) + 0.02
    if values.min() >= 0:
        ax.set_ylim(0, values.max() + pad)
    else:
        ax.set_ylim(values.min() - pad, values.max() + pad)
    style_axes(ax)
    fig.tight_layout()
    save_all(fig, filename)


def plot_scatter(df: pd.DataFrame, xcol: str, xlabel: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    sample = df if len(df) <= 8000 else df.sample(8000, random_state=0)
    ax.scatter(sample[xcol], sample["main_score"], alpha=0.25, s=10, color="#3b6ea5", edgecolors="none")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Main score", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    style_axes(ax)
    fig.tight_layout()
    save_all(fig, filename)


def main() -> None:
    condition = pd.read_csv(SRC / "condition_summary.csv")
    model_axis = pd.read_csv(SRC / "model_axis_summary.csv")
    records = pd.read_csv(
        SRC / "records_all.csv",
        usecols=["latency_seconds", "max_memory_allocated_gb", "main_score"],
    )

    plot_scaling_trends(model_axis)
    plot_heatmap(model_axis)
    plot_oracle_gain(condition)
    plot_retrieval_gap(condition)
    plot_prompt_sensitivity(condition)

    for name, filename, title in [
        ("Distractor Drop", "figure_distractor_drop.png", "Distractor Drop"),
        ("Position Drop", "figure_position_drop.png", "Position Drop"),
        ("CoT Gain", "figure_cot_gain.png", "CoT Gain"),
    ]:
        plot_simple_comparison(condition, name, filename, title)

    plot_scatter(records, "latency_seconds", "Latency (seconds)", "Score vs latency", "figure_accuracy_latency.png")
    plot_scatter(
        records,
        "max_memory_allocated_gb",
        "Peak allocated memory (GB)",
        "Score vs memory",
        "figure_accuracy_memory.png",
    )
    print("done")


if __name__ == "__main__":
    main()
