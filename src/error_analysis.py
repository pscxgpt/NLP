#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Error Analysis — Qualitative Evaluation of the Deep Learning Model
===================================================================
Loads best_model.pt, runs inference on the validation split, and performs
detailed error analysis: confusion matrix, top confusion pairs,
per-class accuracy, and qualitative misclassification examples.

Outputs:
  - error_confusion_matrix.png      (36×36 heatmap)
  - error_top_confusions.png        (focused bar chart of top error pairs)
  - dl_val_metrics.json             (exact DL metrics for baseline comparison)
  - Printed qualitative examples to stdout

Usage:
    python error_analysis.py
"""

import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# Import model classes from the pipeline (main() is guarded by __name__)
# ─────────────────────────────────────────────────────────────────────────────
from icd10_pipeline import (
    CFG,
    CATEGORIES,
    ID2LABEL,
    LABEL2ID,
    AttentionPooling,
    ICD10Classifier,
    ICD10Dataset,
    preprocess_text,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """Locate the training data."""
    candidates = [DATA_DIR / "codification_data.csv"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find codification_data.csv in {[str(c) for c in candidates]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS — Deep Learning Model Qualitative Evaluation")
    print("=" * 70)
    print(f"\n  Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # ── 1. Reproduce the EXACT same data split as the pipeline ────────────
    print("\n[1] Loading data and reproducing stratified split ...")
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

    _, val_df = train_test_split(
        df, test_size=0.2, random_state=CFG.seed, stratify=df["label"]
    )
    val_df = val_df.reset_index(drop=True)
    print(f"    Validation samples : {len(val_df):,}")

    # ── 2. Load tokenizer & model ────────────────────────────────────────
    print("\n[2] Loading tokenizer and model ...")
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CFG.backbone)
    config = AutoConfig.from_pretrained(CFG.backbone)
    model = ICD10Classifier(config)

    model_path = OUTPUT_DIR / "best_model.pt"
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"    Model loaded from : {model_path.name}")

    # ── 3. Run inference on validation set ────────────────────────────────
    print("\n[3] Running inference on validation set ...")
    val_dataset = ICD10Dataset(
        texts=val_df["Literal"].values,
        labels=val_df["label"].values,
        tokenizer=tokenizer,
        max_len=CFG.max_len,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
    )

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"]

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ── 4. Compute global metrics ────────────────────────────────────────
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"\n{'─' * 60}")
    print("  DEEP LEARNING MODEL — VALIDATION METRICS")
    print(f"{'─' * 60}")
    print(f"\n  Accuracy    : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Macro-F1    : {macro_f1:.4f}  ({macro_f1*100:.2f}%)")
    print(f"  Weighted-F1 : {weighted_f1:.4f}  ({weighted_f1*100:.2f}%)")

    # Save metrics for baseline comparison script
    dl_metrics = {
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
    }
    metrics_path = OUTPUT_DIR / "dl_val_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(dl_metrics, f, indent=2)
    print(f"\n  ✓ Saved exact DL metrics → {metrics_path.name}")

    # ── 5. Full classification report ────────────────────────────────────
    target_names = [
        f"{ID2LABEL[i]} ({ICD10_CHAPTERS.get(ID2LABEL[i], '?')[:18]})"
        for i in range(len(CATEGORIES))
    ]
    report = classification_report(
        all_labels, all_preds, target_names=target_names, zero_division=0
    )
    print(f"\n  Per-class classification report:\n")
    print(report)

    # ── 6. Confusion matrix heatmap ──────────────────────────────────────
    print("[4] Generating confusion matrix ...")
    cm = confusion_matrix(all_labels, all_preds)

    # Identify which classes actually appear in the data
    present_mask = cm.sum(axis=1) > 0
    present_indices = np.where(present_mask)[0]
    cm_present = cm[np.ix_(present_indices, present_indices)]
    present_labels = [ID2LABEL[i] for i in present_indices]

    # Normalize for display (row-wise = per true class)
    cm_norm = cm_present.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=present_labels,
        yticklabels=present_labels,
        linewidths=0.3,
        linecolor="gray",
        square=True,
        ax=ax,
        cbar_kws={"label": "Proportion of True Class", "shrink": 0.8},
        annot_kws={"size": 7},
    )
    ax.set_xlabel("Predicted Category", fontsize=13)
    ax.set_ylabel("True Category", fontsize=13)
    ax.set_title(
        f"Confusion Matrix  (Normalized by True Class)\n"
        f"Val Accuracy = {accuracy:.4f}  |  Macro-F1 = {macro_f1:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    cm_path = OUTPUT_DIR / "error_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {cm_path.name}")

    # ── 7. Top confusion pairs ───────────────────────────────────────────
    print("\n[5] Extracting top confusion pairs ...")
    error_mask = all_preds != all_labels
    error_pairs = list(
        zip(all_labels[error_mask], all_preds[error_mask])
    )
    pair_counts = Counter(
        (ID2LABEL[true], ID2LABEL[pred]) for true, pred in error_pairs
    )
    top_pairs = pair_counts.most_common(15)

    print(f"\n{'─' * 60}")
    print("  TOP-15 MOST FREQUENT MISCLASSIFICATION PAIRS")
    print(f"{'─' * 60}")
    print(f"\n  {'#':<4} {'True → Pred':>12}  {'Count':>6}  True Chapter → Pred Chapter")
    print(f"  {'─'*4} {'─'*12}  {'─'*6}  {'─'*40}")
    for rank, ((true_lbl, pred_lbl), count) in enumerate(top_pairs, 1):
        true_ch = ICD10_CHAPTERS.get(true_lbl, "?")[:20]
        pred_ch = ICD10_CHAPTERS.get(pred_lbl, "?")[:20]
        print(f"  {rank:<4} {true_lbl:>4} → {pred_lbl:<4}   {count:>5}  {true_ch} → {pred_ch}")

    # Bar chart of top confusions
    fig, ax = plt.subplots(figsize=(12, 6))
    pair_labels = [f"{t} → {p}" for (t, p), _ in top_pairs]
    pair_vals = [c for _, c in top_pairs]

    # Color-code: same type (letter-letter or digit-digit) vs cross-type
    colors = []
    for (t, p), _ in top_pairs:
        if t.isdigit() == p.isdigit():
            colors.append("#DD8452")  # same type confusion
        else:
            colors.append("#C44E52")  # cross-type confusion (letter ↔ digit)

    ax.barh(range(len(pair_labels)), pair_vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Error Count", fontsize=12)
    ax.set_title(
        "Top-15 Most Frequent Misclassification Pairs  (True → Predicted)",
        fontsize=13,
        fontweight="bold",
    )

    for i, v in enumerate(pair_vals):
        ax.text(v + max(pair_vals) * 0.01, i, str(v), va="center", fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#DD8452", label="Within same type (letter↔letter / digit↔digit)"),
        Patch(facecolor="#C44E52", label="Cross-type (letter ↔ digit)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    conf_path = OUTPUT_DIR / "error_top_confusions.png"
    fig.savefig(conf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Saved: {conf_path.name}")

    # ── 8. Per-class accuracy (worst categories) ─────────────────────────
    print(f"\n{'─' * 60}")
    print("  PER-CLASS ACCURACY  (sorted worst → best)")
    print(f"{'─' * 60}")

    per_class_acc = {}
    for cls_id in range(len(CATEGORIES)):
        mask = all_labels == cls_id
        n_total = mask.sum()
        if n_total == 0:
            continue
        n_correct = (all_preds[mask] == cls_id).sum()
        per_class_acc[ID2LABEL[cls_id]] = {
            "accuracy": n_correct / n_total,
            "correct": int(n_correct),
            "total": int(n_total),
        }

    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1]["accuracy"])

    print(f"\n  {'Cat':<4} {'Acc':>7} {'Correct/Total':>15}  Chapter")
    print(f"  {'─'*4} {'─'*7} {'─'*15}  {'─'*25}")
    for cat, info in sorted_classes:
        acc_pct = info["accuracy"] * 100
        desc = ICD10_CHAPTERS.get(cat, "?")[:25]
        print(
            f"  {cat:<4} {acc_pct:>6.1f}% "
            f"{info['correct']:>6}/{info['total']:<6}  {desc}"
        )

    # ── 9. Qualitative error examples ────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  QUALITATIVE ERROR EXAMPLES")
    print(f"{'─' * 60}")

    val_df = val_df.copy()
    val_df["pred_label"] = all_preds
    val_df["pred_category"] = val_df["pred_label"].map(ID2LABEL)
    val_df["true_category"] = val_df["label"].map(ID2LABEL)
    val_df["is_error"] = val_df["pred_label"] != val_df["label"]

    # Confidence of the wrong prediction
    val_df["pred_confidence"] = [
        all_probs[i, all_preds[i]] for i in range(len(all_preds))
    ]

    errors_df = val_df[val_df["is_error"]].copy()
    total_errors = len(errors_df)
    total_samples = len(val_df)

    print(f"\n  Total errors: {total_errors} / {total_samples} "
          f"({total_errors/total_samples*100:.1f}%)")

    # Focus on interesting confusion patterns
    interesting_pairs = [
        ("C", "D", "Malignant neoplasms ↔ Blood/Benign neoplasms"),
        ("O", "0", "Pregnancy (letter O) ↔ Procedures (digit 0)"),
        ("I", "1", "Circulatory (letter I) ↔ Procedures (digit 1)"),
        ("S", "5", "Injury (letter S) ↔ Procedures (digit 5)"),
        ("Z", "0", "Health status (Z) ↔ Procedures (0)"),
        ("A", "B", "Infectious diseases A ↔ B"),
    ]

    for true_cat, pred_cat, description in interesting_pairs:
        subset = errors_df[
            (errors_df["true_category"] == true_cat)
            & (errors_df["pred_category"] == pred_cat)
        ]
        if len(subset) == 0:
            # Try reverse
            subset = errors_df[
                (errors_df["true_category"] == pred_cat)
                & (errors_df["pred_category"] == true_cat)
            ]
            if len(subset) == 0:
                continue
            true_cat, pred_cat = pred_cat, true_cat
            description = f"{description} (reversed)"

        print(f"\n  ── {description} ({len(subset)} errors) ──")
        sample = subset.head(3)
        for _, row in sample.iterrows():
            print(
                f"    Literal : \"{row['Literal'][:80]}\"\n"
                f"    True    : {row['true_category']} ({ICD10_CHAPTERS.get(row['true_category'], '?')})\n"
                f"    Pred    : {row['pred_category']} ({ICD10_CHAPTERS.get(row['pred_category'], '?')})\n"
                f"    Conf    : {row['pred_confidence']:.3f}\n"
            )

    # Show additional random error examples
    print(f"\n  ── Random misclassification samples ──")
    random_errors = errors_df.sample(min(10, len(errors_df)), random_state=42)
    for idx, (_, row) in enumerate(random_errors.iterrows(), 1):
        print(
            f"  [{idx:02d}] \"{row['Literal'][:70]}\"\n"
            f"       True: {row['true_category']} → Pred: {row['pred_category']}  "
            f"(conf={row['pred_confidence']:.3f})"
        )

    # ── 10. High-confidence errors (model was confident but wrong) ───────
    print(f"\n{'─' * 60}")
    print("  HIGH-CONFIDENCE ERRORS (model confident ≥ 0.7 but wrong)")
    print(f"{'─' * 60}")

    high_conf_errors = errors_df[errors_df["pred_confidence"] >= 0.7].sort_values(
        "pred_confidence", ascending=False
    )
    print(f"\n  Count: {len(high_conf_errors)} high-confidence errors")

    for idx, (_, row) in enumerate(high_conf_errors.head(8).iterrows(), 1):
        print(
            f"\n  [{idx}] \"{row['Literal'][:75]}\"\n"
            f"      True: {row['true_category']} ({ICD10_CHAPTERS.get(row['true_category'], '?')[:25]})\n"
            f"      Pred: {row['pred_category']} ({ICD10_CHAPTERS.get(row['pred_category'], '?')[:25]})\n"
            f"      Confidence: {row['pred_confidence']:.4f}"
        )

    # ── Done ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("ERROR ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n  Output files:")
    print(f"    - error_confusion_matrix.png")
    print(f"    - error_top_confusions.png")
    print(f"    - dl_val_metrics.json")
    print()


if __name__ == "__main__":
    main()
