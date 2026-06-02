#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis (EDA) -- ICD-10 Codification Dataset
=============================================================
Generates dataset statistics and visualizations for the project report.

Outputs:
  - ../outputs/eda_class_distribution.png    (horizontal bar chart)
  - ../outputs/eda_text_length_distribution.png  (word & character histograms)
  - Printed summary statistics to stdout

Usage:
    python -m src.eda_analysis
"""

import re
import string
import sys
from pathlib import Path

# Force UTF-8 output on Windows (cp1252 chokes on box-drawing chars)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parent.parent

print("\nNote: Script moved to src/ folder. Update imports and paths accordingly.")
