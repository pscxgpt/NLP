#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline Comparison — TF-IDF + Logistic Regression vs. Deep Learning
=====================================================================
Trains the TF-IDF + Logistic Regression baseline described in the project
guide and compares its metrics against the deep-learning pipeline.

Architecture (from project guide):
  - FeatureUnion of:
      • Character-level TF-IDF (3–5 char n-grams)
      • Word-level TF-IDF (1–2 word n-grams)
  - LogisticRegression (multinomial, lbfgs)

Uses the SAME stratified 80/20 split (seed=42) as icd10_pipeline.py
so that metrics are directly comparable.

Outputs:
  - ../outputs/baseline_classification_report.txt
  - ../outputs/baseline_vs_dl_comparison.txt
  - Printed metrics to stdout

Usage:
    python baseline_tfidf.py
"""

import json
import re
import string
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

# ────────────────────────────────────────────────────────────────────────────
# Configuration  (mirrors icd10_pipeline.py)
# ────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
SEED = 42
CATEGORIES = list(string.digits + string.ascii_uppercase)  # 0-9, A-Z
LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}

ICD10_CHAPTERS = {
    "A": "Infectious diseases",       "B": "Infectious diseases",
    "C": "Neoplasms (malignant)",      "D": "Blood / Benign neoplasms",
    "E": "Endocrine / Metabolic",      "F": "Mental / Behavioural",
    "G": "Nervous system",             "H": "Eye / Ear",
    "I": "Circulatory system",         "J": "Respiratory system",
    "K": "Digestive system",           "L": "Skin / Subcutaneous",
    "M": "Musculoskeletal",            "N": "Genitourinary",
    "O": "Pregnancy / Childbirth",     "P": "Perinatal conditions",
    "Q": "Congenital malformations",   "R": "Symptoms / Signs",
    "S": "Injury (single region)",     "T": "Injury / Poisoning",
    "U": "Special purposes",           "V": "Transport accidents",
    "W": "External causes (falls)",    "X": "External causes (other)",
    "Y": "External causes (surg.)",    "Z": "Health status / Services",
    "0": "Procedures (Med/Surg)",      "1": "Procedures (Obstetric)",
    "2": "Procedures (Placement)",     "3": "Procedures (Administration)",
    "4": "Procedures (Measurement)",   "5": "Procedures (Extracorporeal I)",
    "6": "Procedures (Extracorporeal II)", "7": "Procedures (Osteopathic)",
    "8": "Procedures (Other)",         "9": "Procedures (Chiropractic)",
}


def _resolve_data_path():
    """Find codification_data.csv in DATA/ or project root."""
    candidates = [
        PROJECT_DIR / "data" / "codification_data.csv",
        PROJECT_DIR / "DATA" / "codification_data.csv",
        PROJECT_DIR / "codification_data.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find codification_data.csv in {[str(c) for c in candidates]}"
    )


def preprocess_text(text):
    """Same preprocessing as the main pipeline."""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("BASELINE — TF-IDF + Logistic Regression")
    print("=" * 70)

    # ── 1. Load & prepare  (identical to icd10_pipeline.py) ───────────────
    data_path = _resolve_data_path()
    df = pd.read_csv(data_path)
    df = df[["Literal", "Code"]].copy()
    df["y_category"] = df["Code"].astype(str).str[0].str.upper()

    valid = set(CATEGORIES)
    df = df[df["y_category"].isin(valid)].copy()
    df["Literal"] = df["Literal"].apply(preprocess_text)
    df = df[df["Literal"].str.len() > 0].reset_index(drop=True)
    df["label"] = df["y_category"].map(LABEL2ID)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    print(f"\n  Samples (after filtering): {len(df):,}")

    # ── 2. Stratified 80/20 split  (same seed) ──────────────────────────
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"  Train : {len(train_df):,}  |  Validation : {len(val_df):,}")

    X_train = train_df["Literal"].values
    y_train = train_df["label"].values
    X_val = val_df["Literal"].values
    y_val = val_df["label"].values

    # ── 3. Build the TF-IDF + LogReg pipeline ────────────────────────────
    print(f"\n  Building TF-IDF feature union ...")
    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("char_tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=100_000,
                sublinear_tf=True,
            )),
            ("word_tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=50_000,
                sublinear_tf=True,
            )),
        ])),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            C=1.0,
            random_state=SEED,
            n_jobs=-1,
        )),
    ])

    # ── 4. Train ─────────────────────────────────────────────────────────
    print("  Training baseline ...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")

    # ── 5. Evaluate ──────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_val)

    bl_accuracy = accuracy_score(y_val, y_pred)
    bl_macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    bl_weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    target_names = [
        f"{ID2LABEL[i]} ({ICD10_CHAPTERS.get(ID2LABEL[i], '?')[:20]})"
        for i in range(len(CATEGORIES))
    ]
    report = classification_report(
        y_val, y_pred, target_names=target_names, zero_division=0
    )

    print(f"\n{'─' * 60}")
    print("  BASELINE RESULTS (TF-IDF + Logistic Regression)")
    print(f"{'─' * 60}")
    print(f"\n  Accuracy          : {bl_accuracy:.4f}  ({bl_accuracy*100:.2f}%)")
    print(f"  Macro-F1          : {bl_macro_f1:.4f}  ({bl_macro_f1*100:.2f}%)")
    print(f"  Weighted-F1       : {bl_weighted_f1:.4f}  ({bl_weighted_f1*100:.2f}%)")
    print(f"\n  Per-class classification report:\n")
    print(report)

    # Save report
    report_path = PROJECT_DIR / "outputs" / "baseline_classification_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BASELINE — TF-IDF + Logistic Regression\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy    : {bl_accuracy:.4f}\n")
        f.write(f"Macro-F1    : {bl_macro_f1:.4f}\n")
        f.write(f"Weighted-F1 : {bl_weighted_f1:.4f}\n\n")
        f.write("Per-class classification report:\n\n")
        f.write(report)
    print(f"  ✓ Saved: {report_path.name}")

    # ── 6. Load DL metrics (from error_analysis.py output if available) ──
    dl_metrics_path = PROJECT_DIR / "outputs" / "dl_val_metrics.json"
    if dl_metrics_path.exists():
        with open(dl_metrics_path, "r") as f:
            dl_metrics = json.load(f)
        dl_accuracy = dl_metrics["accuracy"]
        dl_macro_f1 = dl_metrics["macro_f1"]
        dl_source = "exact (from error_analysis.py)"
    else:
        # Approximate from performance_metrics.png training curves
        dl_accuracy = 0.57
        dl_macro_f1 = 0.50
        dl_source = "approximate (from training curves)"

    # ── 7. Comparison table ──────────────────────────────────────────────
    acc_delta = dl_accuracy - bl_accuracy
    f1_delta = dl_macro_f1 - bl_macro_f1

    comparison = f"""
{'─' * 60}
  BASELINE vs. DEEP LEARNING — COMPARISON
{'─' * 60}

  DL metrics source: {dl_source}

  ┌────────────────────────┬────────────────┬────────────────┬────────────┐
  │ Metric                 │ TF-IDF + LogReg│ RoBERTa-Bio-ES │ Δ (DL−BL)  │
  ├────────────────────────┼────────────────┼────────────────┼────────────┤
  │ Accuracy               │   {bl_accuracy:>10.4f}   │   {dl_accuracy:>10.4f}   │  {acc_delta:>+8.4f}  │
  │ Macro-F1               │   {bl_macro_f1:>10.4f}   │   {dl_macro_f1:>10.4f}   │  {f1_delta:>+8.4f}  │
  └────────────────────────┴────────────────┴────────────────┴────────────┘

  Accuracy improvement : {acc_delta*100:>+.2f} percentage points
  Macro-F1 improvement : {f1_delta*100:>+.2f} percentage points
"""
    print(comparison)

    # Save comparison
    comparison_path = PROJECT_DIR / "outputs" / "baseline_vs_dl_comparison.txt"
    comparison_path.parent.mkdir(exist_ok=True)
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(comparison)
    print(f"  ✓ Saved: {comparison_path.name}")

    # Tip for user
    if not dl_metrics_path.exists():
        print(
            "\n  TIP: Run error_analysis.py first to generate exact DL metrics."
            "\n       Then re-run this script for a precise comparison.\n"
        )

    print("=" * 70)
    print("BASELINE COMPARISON COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
