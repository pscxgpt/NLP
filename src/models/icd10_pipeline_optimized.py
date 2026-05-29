#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ICD-10 First-Character Classification Pipeline (Optimized for low-VRAM GPU)
============================================================================
End-to-end PyTorch deep learning pipeline for single-label multiclass
NLP classification of ICD-10 codes (36 categories: 0-9, A-Z).

Architecture: RoBERTa-base-biomedical-clinical-es + Attention Pooling
              + Multi-Sample Dropout (5x, p=0.1) Regularization.
Optimized for RTX 3050 6GB Laptop GPU.
"""

import os
import re
import gc
import time
import warnings
import string
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================
class CFG:
    # Paths
    project_dir = Path(__file__).resolve().parent
    train_file = project_dir / "codification_data.csv"
    icd_pairs_file = project_dir / "icd_d_p_pairs.csv"
    leaderboard_file = project_dir / "leaderboard_data.csv"
    output_dir = project_dir
    model_save_path = project_dir / "best_model_optimized.pt"

    # Model
    backbone = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
    max_len = 64  # Optimal for literals (max tokens = 49) to save memory and speed up
    hidden_size = 768
    num_classes = 36  # 0-9, A-Z

    # Training
    # Real batch = 32 per GPU step, accumulate gradients to simulate 128
    batch_size = 32
    accumulation_steps = 4  # effective batch = 32 * 4 = 128
    max_aug_samples_per_class = 1000  # Cap to reduce redundancy and speed up training
    freeze_layers = 6  # Number of lower transformer layers to freeze
    epochs = 50
    patience = 10
    lr = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1

    # Multi-sample dropout
    num_dropouts = 5
    dropout_p = 0.1

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.float16

    # Seed
    seed = 42
