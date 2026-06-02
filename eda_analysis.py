#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis (EDA) -- ICD-10 Codification Dataset
=============================================================
Generates dataset statistics and visualizations for the project report.

Outputs:
  - eda_class_distribution.png    (horizontal bar chart)
  - eda_text_length_distribution.png  (word & character histograms)
  - Printed summary statistics to stdout

Usage:
    python eda_analysis.py
"""

import re
import string
import sys
from pathlib import Path

# Force UTF-8 output on Windows (cp1252 chokes on box-drawing chars)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
CATEGORIES = list(string.digits + string.ascii_uppercase)  # 0-9, A-Z

# ICD-10 chapter descriptions for the first character
ICD10_CHAPTERS = {
    "A": "Infectious diseases",
    "B": "Infectious diseases",
    "C": "Neoplasms (malignant)",
    "D": "Blood / Benign neoplasms",
    "E": "Endocrine / Metabolic",
    "F": "Mental / Behavioural",
    "G": "Nervous system",
    "H": "Eye / Ear",
    "I": "Circulatory system",
    "J": "Respiratory system",
    "K": "Digestive system",
    "L": "Skin / Subcutaneous",
    "M": "Musculoskeletal",
    "N": "Genitourinary",
    "O": "Pregnancy / Childbirth",
    "P": "Perinatal conditions",
    "Q": "Congenital malformations",
    "R": "Symptoms / Signs",
    "S": "Injury (single region)",
    "T": "Injury (multiple) / Poisoning",
    "U": "Special purposes",
    "V": "Transport accidents",
    "W": "External causes (falls/exposure)",
    "X": "External causes (other)",
    "Y": "External causes (surgical)",
    "Z": "Health status / Services",
    "0": "Procedures (Med/Surg Central)",
    "1": "Procedures (Obstetric)",
    "2": "Procedures (Placement)",
    "3": "Procedures (Administration)",
    "4": "Procedures (Measurement)",
    "5": "Procedures (Extracorporeal)",
    "6": "Procedures (Extracorporeal II)",
    "7": "Procedures (Osteopathic)",
    "8": "Procedures (Other)",
    "9": "Procedures (Chiropractic)",
}


def _resolve_data_path():
    """Find codification_data.csv in DATA/ or project root."""
    candidates = [
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS — ICD-10 Codification Dataset")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────────
    data_path = _resolve_data_path()
    df = pd.read_csv(data_path)
    print(f"\n  Data source : {data_path}")
    print(f"  Raw rows    : {len(df):,}")
    print(f"  Columns     : {df.columns.tolist()}")

    # Extract first character → category
    df["y_category"] = df["Code"].astype(str).str[0].str.upper()

    # Filter to valid 0-9 / A-Z categories
    valid = set(CATEGORIES)
    df = df[df["y_category"].isin(valid)].copy()

    # Preprocess text
    df["Literal"] = df["Literal"].apply(preprocess_text)
    df = df[df["Literal"].str.len() > 0].reset_index(drop=True)

    # Derived columns
    df["char_len"] = df["Literal"].str.len()
    df["word_count"] = df["Literal"].str.split().str.len()

    # -- 1. Basic statistics -----------------------------------------------
    n_samples = len(df)
    n_categories = df["y_category"].nunique()

    print(f"\n{'-' * 60}")
    print("  1. BASIC DATASET STATISTICS")
    print(f"{'-' * 60}")
    print(f"\n  Total valid samples       : {n_samples:,}")
    print(f"  Unique categories (classes) : {n_categories}")
    print(f"  Unique full ICD-10 codes  : {df['Code'].nunique():,}")

    print(f"\n  -- Literal character length --")
    print(f"     Mean   : {df['char_len'].mean():.1f}")
    print(f"     Median : {df['char_len'].median():.0f}")
    print(f"     Min    : {df['char_len'].min()}")
    print(f"     Max    : {df['char_len'].max()}")
    print(f"     Std    : {df['char_len'].std():.1f}")

    print(f"\n  -- Literal word count --")
    print(f"     Mean   : {df['word_count'].mean():.1f}")
    print(f"     Median : {df['word_count'].median():.0f}")
    print(f"     Min    : {df['word_count'].min()}")
    print(f"     Max    : {df['word_count'].max()}")
    print(f"     Std    : {df['word_count'].std():.1f}")

    # -- 2. Class distribution ---------------------------------------------
    print(f"\n{'-' * 60}")
    print("  2. CLASS DISTRIBUTION & IMBALANCE")
    print(f"{'-' * 60}")

    cat_counts = df["y_category"].value_counts().sort_values(ascending=False)
    total = cat_counts.sum()

    print(f"\n  {'Cat':<4} {'Count':>7} {'Pct':>7}  Description")
    print(f"  {'-'*4} {'-'*7} {'-'*7}  {'-'*30}")
    for cat, count in cat_counts.items():
        pct = count / total * 100
        desc = ICD10_CHAPTERS.get(cat, "Unknown")
        print(f"  {cat:<4} {count:>7,} {pct:>6.2f}%  {desc}")

    max_class_name = cat_counts.index[0]
    min_class_name = cat_counts.index[-1]
    max_count = cat_counts.iloc[0]
    min_count = cat_counts.iloc[-1]
    imbalance_ratio = max_count / min_count

    print(f"\n  Most frequent  : {max_class_name} ({max_count:,} samples)")
    print(f"  Least frequent : {min_class_name} ({min_count:,} samples)")
    print(f"  Imbalance ratio (max / min) : {imbalance_ratio:.1f}x")

    # Count how many classes have fewer than 100 samples
    rare = (cat_counts < 100).sum()
    print(f"  Categories with < 100 samples : {rare}")

    # -- 3. Visualization: Class Distribution ------------------------------
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    fig, ax = plt.subplots(figsize=(13, 9))

    colors = [
        "#4C72B0" if cat.isdigit() else "#DD8452" for cat in cat_counts.index
    ]

    bars = ax.barh(
        range(len(cat_counts)),
        cat_counts.values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_yticks(range(len(cat_counts)))
    ax.set_yticklabels(
        [
            f"{cat} - {ICD10_CHAPTERS.get(cat, '?')}"
            for cat in cat_counts.index
        ],
        fontsize=9,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_title(
        "ICD-10 First-Character Category Distribution  (Class Imbalance)",
        fontsize=14,
        fontweight="bold",
    )

    # Count labels on bars
    for i, (count, bar) in enumerate(zip(cat_counts.values, bars)):
        pct = count / total * 100
        ax.text(
            count + max_count * 0.01,
            i,
            f" {count:,}  ({pct:.1f}%)",
            va="center",
            fontsize=8,
        )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#DD8452", label="Diagnoses (A-Z)"),
        Patch(facecolor="#4C72B0", label="Procedures (0-9)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    fig.tight_layout()
    out1 = PROJECT_DIR / "eda_class_distribution.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] Saved: {out1.name}")

    # -- 4. Visualization: Text Length Distribution ------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Word count histogram
    ax1 = axes[0]
    max_wc = min(int(df["word_count"].quantile(0.99)) + 2, df["word_count"].max() + 1)
    ax1.hist(
        df["word_count"].clip(upper=max_wc),
        bins=range(1, max_wc + 2),
        color="#4C72B0",
        edgecolor="white",
        alpha=0.85,
    )
    ax1.axvline(
        df["word_count"].mean(),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {df['word_count'].mean():.1f} words",
    )
    ax1.axvline(
        df["word_count"].median(),
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Median = {df['word_count'].median():.0f} words",
    )
    ax1.set_xlabel("Word Count", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Distribution of Literal Word Count", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)

    # Character length histogram
    ax2 = axes[1]
    ax2.hist(
        df["char_len"],
        bins=50,
        color="#DD8452",
        edgecolor="white",
        alpha=0.85,
    )
    ax2.axvline(
        df["char_len"].mean(),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {df['char_len'].mean():.1f} chars",
    )
    ax2.axvline(
        df["char_len"].median(),
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Median = {df['char_len'].median():.0f} chars",
    )
    ax2.set_xlabel("Character Length", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title(
        "Distribution of Literal Character Length", fontsize=13, fontweight="bold"
    )
    ax2.legend(fontsize=10)

    fig.tight_layout()
    out2 = PROJECT_DIR / "eda_text_length_distribution.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {out2.name}")

    # -- 5. Sample Literals ------------------------------------------------
    print(f"\n{'-' * 60}")
    print("  3. SAMPLE LITERALS  (top-3 & bottom-3 categories)")
    print(f"{'-' * 60}")

    top3 = cat_counts.head(3).index.tolist()
    bottom3 = cat_counts.tail(3).index.tolist()

    for cat in top3 + bottom3:
        group = df[df["y_category"] == cat]
        samples = group.sample(min(5, len(group)), random_state=42)
        desc = ICD10_CHAPTERS.get(cat, "?")
        print(f"\n  [{cat}] {desc}  ({len(group):,} samples):")
        for _, row in samples.iterrows():
            print(f"    • {row['Literal'][:90]}")

    # ── Done ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("EDA COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
