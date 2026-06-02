# NLP - ICD-10 Codification Classification

Machine Learning pipeline for automated ICD-10 medical code classification using NLP techniques.

## Project Overview

This project implements both baseline (TF-IDF) and deep learning approaches for classifying medical texts to ICD-10 codes as part of the ASHO UAB Codification Project.

## Directory Structure

```
├── src/              # Python source code
├── data/             # Dataset files
├── outputs/          # Results, metrics, and visualizations
├── docs/             # Documentation and reports
└── README.md         # This file
```

See `docs/STRUCTURE.md` for detailed structure documentation.

## Quick Start

### Prerequisites
- Python 3.8+
- Dependencies listed in requirements.txt (if available)

### Running the Pipeline

```bash
# From project root
python -m src.icd10_pipeline
```

### Running Analysis

```bash
# Exploratory Data Analysis
python -m src.eda_analysis

# Baseline TF-IDF model
python -m src.baseline_tfidf

# Error Analysis
python -m src.error_analysis
```

## Project Components

### Source Code (`src/`)
- **baseline_tfidf.py** - TF-IDF baseline model implementation
- **icd10_pipeline.py** - Main classification pipeline with deep learning models
- **eda_analysis.py** - Exploratory data analysis and visualization
- **error_analysis.py** - Model error analysis and confusion matrix visualization

### Data (`data/`)
- leaderboard_data.csv - Competition benchmark dataset

### Outputs (`outputs/`)
- Classification reports and metrics
- Model visualizations (confusion matrices, training curves)
- Submission files (CSV format)

### Documentation (`docs/`)
- ASHO_UAB_Codification_Project_Guide.md - Project guidelines
- NLP Reports (PDF and LaTeX source)
- Repository structure documentation

## Results

- **Baseline (TF-IDF)**: See `outputs/baseline_classification_report.txt`
- **Deep Learning**: See `outputs/baseline_vs_dl_comparison.txt`
- **Validation Metrics**: See `outputs/dl_val_metrics.json`

## Visualizations

- **EDA**: Class distribution and text length analysis
- **Training**: Loss evolution and performance metrics
- **Error Analysis**: Confusion matrices and top confusion pairs

## License

See project documentation for license information.

## References

- ASHO UAB Codification Project
- ICD-10 Classification Challenge
- See `docs/` for detailed reports and baselines
