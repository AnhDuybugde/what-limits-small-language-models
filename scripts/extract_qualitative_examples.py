"""Extract real qualitative closed/oracle examples from merged records."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "QWEN2.5-merged-outputs" / "records_all.csv"


def short(s, n=140):
    s = "" if pd.isna(s) else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def first_line_f1(pred, gold):
    if pd.isna(pred) or pd.isna(gold):
        return 0.0
    raw = str(pred)
    line = raw.split("\n")[0].strip()
    for mk in ["Context:", "Question:", "Final answer:", "Problem:", "Reference:"]:
        if mk in line:
            line = line.split(mk)[0].strip()
    line = re.sub(r"^(Answer|Final answer)\s*:\s*", "", line, flags=re.I).strip()

    def toks(x):
        return set(re.findall(r"\w+", x.lower()))

    pt, gt = toks(line), toks(str(gold))
    if not pt or not gt:
        return 0.0
    inter = len(pt & gt)
    if inter == 0:
        return 0.0
    p = inter / len(pt)
    r = inter / len(gt)
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    df = pd.read_csv(RECORDS, low_memory=False)
    print("n=", len(df))

    # Case A: knowledge bottleneck 1.5B
    kn = df[(df.axis == "knowledge") & (df.model_size == "1.5B")].copy()
    closed = kn[kn.condition == "knowledge_closed"][
        ["sample_id", "question", "gold_answer", "prediction_normalized", "prediction_raw", "main_score"]
    ].rename(
        columns={
            "prediction_normalized": "closed_pred",
            "prediction_raw": "closed_raw",
            "main_score": "closed_score",
        }
    )
    oracle = kn[kn.condition == "knowledge_oracle"][
        ["sample_id", "prediction_normalized", "prediction_raw", "main_score"]
    ].rename(
        columns={
            "prediction_normalized": "oracle_pred",
            "prediction_raw": "oracle_raw",
            "main_score": "oracle_score",
        }
    )
    m = closed.merge(oracle, on="sample_id")
    case_a = m[(m.closed_score < 0.2) & (m.oracle_score >= 0.8)].copy()
    # Prefer short questions and short answers for tables
    case_a["qlen"] = case_a.question.astype(str).str.len()
    case_a["glen"] = case_a.gold_answer.astype(str).str.len()
    case_a = case_a.sort_values(["qlen", "glen"]).head(20)
    print("\n=== KNOWLEDGE BOTTLENECK 1.5B ===")
    for _, r in case_a.head(8).iterrows():
        print("---")
        print("id:", r.sample_id)
        print("Q:", short(r.question, 180))
        print("Gold:", short(r.gold_answer, 100))
        print("Closed:", short(r.closed_pred, 100), "score", round(float(r.closed_score), 3))
        print("Oracle:", short(r.oracle_pred, 100), "score", round(float(r.oracle_score), 3))

    # Case B: 7B knowledge oracle extraction artifact
    k7 = df[(df.axis == "knowledge") & (df.model_size == "7B") & (df.condition == "knowledge_oracle")].copy()
    k7["first_f1"] = [first_line_f1(a, b) for a, b in zip(k7.prediction_raw, k7.gold_answer)]
    art = k7[(k7.main_score < 0.5) & (k7.first_f1 >= 0.8)].copy()
    c7 = df[(df.axis == "knowledge") & (df.model_size == "7B") & (df.condition == "knowledge_closed")][
        ["sample_id", "prediction_normalized", "prediction_raw", "main_score"]
    ].rename(
        columns={
            "prediction_normalized": "closed_pred",
            "prediction_raw": "closed_raw",
            "main_score": "closed_score",
        }
    )
    artm = art.merge(c7, on="sample_id")
    artm["qlen"] = artm.question.astype(str).str.len()
    artm = artm.sort_values(["first_f1", "qlen"], ascending=[False, True])
    print(f"\n=== 7B KNOWLEDGE ORACLE ARTIFACTS n={len(artm)} ===")
    for _, r in artm.head(10).iterrows():
        print("---")
        print("id:", r.sample_id)
        print("Q:", short(r.question, 180))
        print("Gold:", short(r.gold_answer, 100))
        print("Closed:", short(r.closed_pred, 100), "score", round(float(r.closed_score), 3))
        print("Oracle main:", round(float(r.main_score), 3), "first_f1", round(float(r.first_f1), 3))
        print("Oracle raw:", short(r.prediction_raw, 260))

    # Case C: reasoning
    rs = df[(df.axis == "reasoning") & (df.model_size == "7B")].copy()
    d = rs[rs.condition == "reasoning_direct"][
        ["sample_id", "question", "gold_answer", "prediction_normalized", "prediction_raw", "main_score"]
    ].rename(
        columns={
            "prediction_normalized": "direct_pred",
            "prediction_raw": "direct_raw",
            "main_score": "direct_score",
        }
    )
    o = rs[rs.condition == "reasoning_oracle_step"][
        ["sample_id", "prediction_normalized", "prediction_raw", "main_score"]
    ].rename(
        columns={
            "prediction_normalized": "oracle_pred",
            "prediction_raw": "oracle_raw",
            "main_score": "oracle_score",
        }
    )
    rm = d.merge(o, on="sample_id")
    case_r = rm[(rm.direct_score < 0.5) & (rm.oracle_score >= 0.99)].copy()
    case_r["qlen"] = case_r.question.astype(str).str.len()
    case_r = case_r.sort_values("qlen")
    print(f"\n=== REASONING 7B direct fail / oracle ok n={len(case_r)} ===")
    for _, r in case_r.head(6).iterrows():
        print("---")
        print("id:", r.sample_id)
        print("Q:", short(r.question, 200))
        print("Gold:", short(r.gold_answer, 40))
        print("Direct:", short(r.direct_pred, 80), "score", r.direct_score)
        print("Oracle:", short(r.oracle_pred, 80), "score", r.oracle_score)


if __name__ == "__main__":
    main()
