#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ICD-10 First-Character Classification Pipeline
================================================
End-to-end PyTorch deep learning pipeline for single-label multiclass
NLP classification of ICD-10 codes (36 categories: 0-9, A-Z).

Architecture: RoBERTa-base-biomedical-clinical-es + Attention Pooling
              + Multi-Sample Dropout (5x, p=0.1) Regularization.

Usage:
    python -m src.icd10_pipeline
"""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parent.parent

print("\nNote: Script moved to src/ folder. Update imports and paths accordingly.")
