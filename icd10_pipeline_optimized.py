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


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Label Mapping: 36 categories [0-9, A-Z]
# =============================================================================
CATEGORIES = list(string.digits + string.ascii_uppercase)  # 0-9 then A-Z
LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


# =============================================================================
# Text Preprocessing
# =============================================================================
def preprocess_text(text):
    """
    Apply the required preprocessing filter:
    - Cast to string
    - Strip leading and trailing whitespace
    - Reduce multiple whitespaces to a single space
    - Retain original casing, accents, and punctuation
    """
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text


# =============================================================================
# Data Engineering
# =============================================================================
def load_and_prepare_data():
    """
    Load primary training data and augment with ICD description-pairs.
    Returns augmented DataFrame with 'Literal', 'Code', and 'y_category' columns.
    """
    print("=" * 70)
    print("DATA ENGINEERING & PREPARATION")
    print("=" * 70)

    # 1. Load primary training data
    print("\n[1] Loading codification_data.csv ...")
    df_train = pd.read_csv(CFG.train_file)
    print(f"    Primary training samples: {len(df_train)}")
    print(f"    Columns: {df_train.columns.tolist()}")

    # Ensure correct column order: Code, Literal
    df_train = df_train[["Literal", "Code"]].copy()

    # Generate y_category from Code
    df_train["y_category"] = df_train["Code"].astype(str).str[0].str.upper()

    # 2. Load ICD description-procedure pairs for augmentation
    print("\n[2] Loading icd_d_p_pairs.csv for augmentation ...")
    df_icd = pd.read_csv(CFG.icd_pairs_file)
    print(f"    ICD pairs loaded: {len(df_icd)}")
    print(f"    Columns: {df_icd.columns.tolist()}")

    # Extract Code and Description, transform to match format
    df_aug = df_icd[["Code", "Description"]].copy()
    df_aug.rename(columns={"Description": "Literal"}, inplace=True)
    df_aug["y_category"] = df_aug["Code"].astype(str).str[0].str.upper()

    # 3. Filter to valid categories only
    valid_cats = set(CATEGORIES)
    df_train = df_train[df_train["y_category"].isin(valid_cats)].copy()
    df_aug = df_aug[df_aug["y_category"].isin(valid_cats)].copy()

    # Sample from augmentation data to limit training set size and class imbalance
    print(f"    Sampling at most {CFG.max_aug_samples_per_class} augmentation samples per class...")
    df_aug = pd.concat([
        grp.sample(n=min(len(grp), CFG.max_aug_samples_per_class), random_state=CFG.seed)
        for name, grp in df_aug.groupby("y_category")
    ], ignore_index=True)

    print(f"\n    Primary after filtering: {len(df_train)}")
    print(f"    Augmentation after filtering: {len(df_aug)}")

    # 4. Combine datasets
    df_combined = pd.concat([df_train, df_aug], ignore_index=True)

    # 5. Apply text preprocessing
    print("\n[3] Applying text preprocessing ...")
    df_combined["Literal"] = df_combined["Literal"].apply(preprocess_text)

    # Drop rows with empty Literal
    df_combined = df_combined[df_combined["Literal"].str.len() > 0].reset_index(drop=True)

    # Encode labels
    df_combined["label"] = df_combined["y_category"].map(LABEL2ID)

    # Drop any rows that couldn't be mapped
    df_combined = df_combined.dropna(subset=["label"]).reset_index(drop=True)
    df_combined["label"] = df_combined["label"].astype(int)

    print(f"\n    Total combined samples: {len(df_combined)}")
    print(f"\n    Category distribution:")
    cat_counts = df_combined["y_category"].value_counts().sort_index()
    for cat, count in cat_counts.items():
        print(f"      {cat}: {count:>7d}")

    return df_combined


def compute_weights(labels):
    """Compute balanced class weights using sklearn."""
    classes = np.arange(CFG.num_classes)
    # Only consider classes that exist in labels
    existing_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=existing_classes,
        y=labels
    )
    # Map to full class array
    full_weights = np.ones(CFG.num_classes, dtype=np.float32)
    for cls, w in zip(existing_classes, weights):
        full_weights[cls] = w

    return torch.tensor(full_weights, dtype=torch.float32)


def split_data(df):
    """Perform stratified 80/20 train/validation split."""
    print("\n[4] Performing stratified 80/20 train/validation split ...")
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=CFG.seed,
        stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"    Train: {len(train_df)}  |  Validation: {len(val_df)}")
    return train_df, val_df


# =============================================================================
# Dataset
# =============================================================================
class ICD10Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# =============================================================================
# Model Architecture
# =============================================================================
class AttentionPooling(nn.Module):
    """
    Custom attention mechanism over transformer hidden states.
    Learns optimal weights for individual tokens across the sequence
    prior to aggregation, replacing standard CLS or mean pooling.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
        Returns:
            pooled: (batch, hidden_size)
        """
        # Compute attention scores
        scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)

        # Mask padding tokens with very negative value
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # Softmax over sequence dimension
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)

        # Weighted sum of hidden states
        pooled = torch.bmm(
            weights.unsqueeze(1), hidden_states
        ).squeeze(1)  # (batch, hidden_size)

        return pooled


class ICD10Classifier(nn.Module):
    """
    Custom PyTorch module for ICD-10 classification:
    - Backbone: PlanTL-GOB-ES/roberta-base-biomedical-clinical-es
    - Attention Pooling over hidden states
    - Multi-Sample Dropout (5 parallel dropouts, p=0.1)
    - Linear classifier head (768 -> 36)
    """
    def __init__(self, config):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            CFG.backbone, config=config
        )

        # Freeze bottom layers of backbone to speed up backward pass
        if CFG.freeze_layers > 0:
            print(f"    Freezing embeddings and bottom {CFG.freeze_layers} backbone layers...")
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False
            for i in range(CFG.freeze_layers):
                for param in self.backbone.encoder.layer[i].parameters():
                    param.requires_grad = False

        self.attention_pool = AttentionPooling(CFG.hidden_size)

        # Multi-sample dropout layers
        self.dropouts = nn.ModuleList([
            nn.Dropout(p=CFG.dropout_p) for _ in range(CFG.num_dropouts)
        ])

        # Classifier head
        self.classifier = nn.Linear(CFG.hidden_size, CFG.num_classes)

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        # Get transformer hidden states
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)

        # Attention pooling
        pooled = self.attention_pool(hidden_states, attention_mask)  # (batch, 768)

        # Multi-sample dropout: pass through 5 parallel dropout layers,
        # compute logits for each, and average
        logits_list = []
        for dropout in self.dropouts:
            dropped = dropout(pooled)
            logits = self.classifier(dropped)
            logits_list.append(logits)

        # Average raw logits
        avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)  # (batch, 36)

        return avg_logits


# =============================================================================
# Training & Evaluation Functions
# =============================================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, scaler):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    accum = CFG.accumulation_steps
    device_type = "cuda" if CFG.device.type == "cuda" else "cpu"

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(CFG.device)
        attention_mask = batch["attention_mask"].to(CFG.device)
        labels = batch["label"].to(CFG.device)

        if CFG.use_amp:
            with autocast(device_type=device_type, dtype=CFG.amp_dtype):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels) / accum
            scaler.scale(loss).backward()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / accum
            loss.backward()

        total_loss += loss.item() * accum  # recover unscaled loss for logging
        num_batches += 1

        if (step + 1) % accum == 0 or (step + 1) == len(dataloader):
            if CFG.use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, criterion):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(CFG.device)
        attention_mask = batch["attention_mask"].to(CFG.device)
        labels = batch["label"].to(CFG.device)

        device_type = "cuda" if CFG.device.type == "cuda" else "cpu"
        if CFG.use_amp:
            with autocast(device_type=device_type, dtype=CFG.amp_dtype):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        num_batches += 1

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / num_batches
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, macro_f1


# =============================================================================
# Visualization
# =============================================================================
def plot_metrics(history):
    """Generate and save training/evaluation plots."""
    sns.set_style("whitegrid")

    epochs_range = range(1, len(history["train_loss"]) + 1)

    # 1. Loss Evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_range, history["train_loss"], "b-o", label="Training Loss", markersize=4)
    ax.plot(epochs_range, history["val_loss"], "r-o", label="Validation Loss", markersize=4)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Loss Evolution (Optimized Model)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CFG.output_dir / "loss_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: loss_evolution.png")

    # 2. Performance Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_range, history["val_accuracy"], "g-o", label="Validation Accuracy", markersize=4)
    ax.plot(epochs_range, history["val_macro_f1"], "m-o", label="Validation Macro-F1", markersize=4)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Performance Metrics (Optimized Model)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(CFG.output_dir / "performance_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: performance_metrics.png")


# =============================================================================
# Inference
# =============================================================================
@torch.no_grad()
def run_inference(model, tokenizer):
    """Run inference on leaderboard data and generate submission."""
    print("\n" + "=" * 70)
    print("INFERENCE & SUBMISSION")
    print("=" * 70)

    # Load leaderboard data
    df_test = pd.read_csv(CFG.leaderboard_file)
    print(f"\n  Leaderboard samples: {len(df_test)}")

    # Apply same preprocessing
    df_test["Literal_clean"] = df_test["Literal"].apply(preprocess_text)

    # Create dataset and dataloader
    test_dataset = ICD10Dataset(
        texts=df_test["Literal_clean"].values,
        labels=np.zeros(len(df_test), dtype=int),  # dummy labels
        tokenizer=tokenizer,
        max_len=CFG.max_len,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CFG.batch_size * 2,
        shuffle=False, num_workers=0, pin_memory=True
    )

    # Run inference
    model.eval()
    all_preds = []
    for batch in test_loader:
        input_ids = batch["input_ids"].to(CFG.device)
        attention_mask = batch["attention_mask"].to(CFG.device)

        device_type = "cuda" if CFG.device.type == "cuda" else "cpu"
        if CFG.use_amp:
            with autocast(device_type=device_type, dtype=CFG.amp_dtype):
                logits = model(input_ids, attention_mask)
        else:
            logits = model(input_ids, attention_mask)

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())

    # Map predictions back to category labels
    pred_categories = [ID2LABEL[p] for p in all_preds]

    # Build submission DataFrame
    submission = pd.DataFrame({
        "id": df_test["id"],
        "Literal": df_test["Literal"],
        "y_category": pred_categories,
    })

    # Save
    submission_path = CFG.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\n  Submission saved to: {submission_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  Columns: {submission.columns.tolist()}")
    print(f"\n  Preview:")
    print(submission.head(10).to_string(index=False))
    print(f"\n  Prediction distribution:")
    print(submission["y_category"].value_counts().sort_index().to_string())

    return submission


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("ICD-10 FIRST-CHARACTER CLASSIFICATION PIPELINE (OPTIMIZED)")
    print("=" * 70)
    print(f"\n  Device: {CFG.device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  AMP: {CFG.use_amp}")
    print(f"  Seed: {CFG.seed}")

    set_seed(CFG.seed)

    # =========================================================================
    # Step 1: Data Loading & Preparation
    # =========================================================================
    df = load_and_prepare_data()
    train_df, val_df = split_data(df)

    # Compute class weights
    print("\n[5] Computing balanced class weights ...")
    class_weights = compute_weights(train_df["label"].values)
    class_weights = class_weights.to(CFG.device)
    print(f"    Weights computed for {CFG.num_classes} classes")
    print(f"    Min weight: {class_weights.min().item():.4f}")
    print(f"    Max weight: {class_weights.max().item():.4f}")

    # =========================================================================
    # Step 2: Tokenizer & Datasets
    # =========================================================================
    print("\n[6] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.backbone)
    print(f"    Tokenizer: {CFG.backbone}")

    print("\n[7] Creating datasets ...")
    train_dataset = ICD10Dataset(
        texts=train_df["Literal"].values,
        labels=train_df["label"].values,
        tokenizer=tokenizer,
        max_len=CFG.max_len,
    )
    val_dataset = ICD10Dataset(
        texts=val_df["Literal"].values,
        labels=val_df["label"].values,
        tokenizer=tokenizer,
        max_len=CFG.max_len,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.batch_size,
        shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG.batch_size * 2,
        shuffle=False, num_workers=0, pin_memory=True,
    )

    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches: {len(val_loader)}")

    # =========================================================================
    # Step 3: Model
    # =========================================================================
    print("\n[8] Building model ...")
    config = AutoConfig.from_pretrained(CFG.backbone)
    model = ICD10Classifier(config)
    model.to(CFG.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")

    # =========================================================================
    # Step 4: Optimizer, Scheduler, Loss
    # =========================================================================
    print("\n[9] Setting up optimizer, scheduler, loss ...")

    # Filter out frozen parameters from optimizer
    trainable_model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_model_params,
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    # Cosine schedule with warmup
    total_steps = len(train_loader) * CFG.epochs
    warmup_steps = int(total_steps * CFG.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # AMP scaler
    scaler = GradScaler(device="cuda") if CFG.use_amp else None

    print(f"    Optimizer: AdamW (LR={CFG.lr}, WD={CFG.weight_decay})")
    print(f"    Scheduler: Cosine with warmup ({warmup_steps} warmup / {total_steps} total steps)")
    print(f"    Loss: CrossEntropyLoss with balanced class weights")

    # =========================================================================
    # Step 5: Training Loop
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, CFG.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler
        )

        # Evaluate
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)

        epoch_time = time.time() - epoch_start

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        # Print epoch results
        improved = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), CFG.model_save_path)
            improved = " ★ BEST"
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch:02d}/{CFG.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Time: {epoch_time:.1f}s{improved}"
        )

        # Early stopping
        if patience_counter >= CFG.patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={CFG.patience})")
            break

    print(f"\n  Best Validation Macro-F1: {best_f1:.4f}")

    # =========================================================================
    # Step 6: Visualization
    # =========================================================================
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    plot_metrics(history)

    # =========================================================================
    # Step 7: Restore Best Model & Run Inference
    # =========================================================================
    print("\n[10] Restoring best model weights ...")
    model.load_state_dict(torch.load(CFG.model_save_path, map_location=CFG.device))
    model.to(CFG.device)
    print(f"     Best model restored from: {CFG.model_save_path}")

    submission = run_inference(model, tokenizer)

    # =========================================================================
    # Done
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Output files:")
    print(f"    - best_model_optimized.pt")
    print(f"    - loss_evolution.png")
    print(f"    - performance_metrics.png")
    print(f"    - submission.csv ({len(submission)} rows)")
    print()


if __name__ == "__main__":
    main()
