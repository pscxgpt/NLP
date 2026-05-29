# ICD-10 Medical NLP Classification

This repository contains a comprehensive implementation for automated ICD-10 coding using deep learning, specifically focused on first-character classification of ICD-10 codes.

## Repository Structure

```
.
├── src/
│   ├── models/              # Model implementations
│   │   ├── icd10_pipeline.py
│   │   └── icd10_pipeline_optimized.py
│   └── utils/               # Utility functions
├── data/
│   ├── raw/                 # Original datasets
│   │   ├── codification_data.csv
│   │   └── icd_d_p_pairs.csv
│   └── processed/           # Processed datasets
│       └── leaderboard_data.csv
├── notebooks/               # Jupyter notebooks for analysis
│   └── analysis.ipynb
├── docs/                    # Documentation
│   ├── NLP-I_Task__ICD-10_Codification_DL_Baselines.md
│   └── survey_icd_coding.md
└── README.md
```

## Overview

This project implements an end-to-end PyTorch deep learning pipeline for single-label multiclass NLP classification of ICD-10 codes (36 categories: 0-9, A-Z).

### Architecture
- **Backbone**: RoBERTa-base-biomedical-clinical-es
- **Pooling**: Custom Attention Pooling
- **Regularization**: Multi-Sample Dropout (5x, p=0.1)
- **Optimizer**: AdamW with cosine schedule and warmup

## Features

- Two pipeline implementations:
  - Standard: Balanced for performance and memory usage
  - Optimized: Fine-tuned for low-VRAM GPUs (RTX 3050 6GB)
- Stratified train/validation split (80/20)
- Balanced class weights for imbalanced data
- Data augmentation from ICD description pairs
- Early stopping with patience
- Comprehensive evaluation metrics

## Dataset

- **Training**: codification_data.csv + icd_d_p_pairs.csv
- **Test**: leaderboard_data.csv
- **ICD Version**: ICD-10
- **Categories**: 36 (0-9, A-Z)

## Running the Pipeline

```bash
# Standard pipeline
python src/models/icd10_pipeline.py

# Optimized pipeline
python src/models/icd10_pipeline_optimized.py
```

## Output

- `best_model.pt` - Trained model weights
- `loss_evolution.png` - Training loss visualization
- `performance_metrics.png` - Validation metrics visualization
- `submission.csv` - Final predictions

## References

See `docs/survey_icd_coding.md` for comprehensive background on automated ICD coding methods.
