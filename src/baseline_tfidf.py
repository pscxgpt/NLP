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
    python -m src.baseline_tfidf
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
    """Find codification_data.csv in data/ or project root."""
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


if __name__ == "__main__":
    print("\nNote: Script moved to src/ folder. Update imports and paths accordingly.")
