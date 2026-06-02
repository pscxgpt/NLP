#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Error Analysis — Qualitative Evaluation of the Deep Learning Model
===================================================================
Loads best_model.pt, runs inference on the validation split, and performs
detailed error analysis: confusion matrix, top confusion pairs,
per-class accuracy, and qualitative misclassification examples.

Outputs:
  - ../outputs/error_confusion_matrix.png      (36×36 heatmap)
  - ../outputs/error_top_confusions.png        (focused bar chart of top error pairs)
  - ../outputs/dl_val_metrics.json             (exact DL metrics for baseline comparison)
  - Printed qualitative examples to stdout

Usage:
    python -m src.error_analysis
"""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parent.parent

print("\nNote: Script moved to src/ folder. Update imports and paths accordingly.")
