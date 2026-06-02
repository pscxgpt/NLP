# Repository Structure

```
NLP/
├── src/                          # Python source code
│   ├── __init__.py
│   ├── baseline_tfidf.py         # TF-IDF baseline model
│   ├── icd10_pipeline.py         # Main classification pipeline
│   ├── eda_analysis.py           # Exploratory Data Analysis
│   └── error_analysis.py         # Error analysis tools
│
├── data/                         # Data files
│   ├── leaderboard_data.csv
│   └── README.md
│
├── outputs/                      # Results, visualizations, reports
│   ├── baseline_classification_report.txt
│   ├── baseline_vs_dl_comparison.txt
│   ├── dl_val_metrics.json
│   ├── submission_ff.csv
│   ├── eda_class_distribution.png
│   ├── eda_text_length_distribution.png
│   ├── error_confusion_matrix.png
│   ├── error_top_confusions.png
│   ├── loss_evolution.png
│   ├── performance_metrics.png
│   └── README.md
│
├── docs/                         # Documentation and reports
│   ├── report.tex
│   ├── ASHO_UAB_Codification_Project_Guide.md
│   ├── NLP-I_ICD-10_Codification_DL_Baselines.pdf
│   ├── NLP_Report.pdf
│   └── STRUCTURE.md (this file)
│
├── .gitignore                    # Git ignore rules
├── README.md                     # Project overview
└── DATA/                         # Empty data directory (to clean up)
```

## File Organization Guidelines

- **src/**: All Python modules and scripts
- **data/**: Raw and processed datasets (add to .gitignore if large)
- **outputs/**: All generated files (reports, metrics, visualizations)
- **docs/**: Project documentation, guides, LaTeX reports, PDFs

## Next Steps

1. ✓ Branch created: `refactor/organize-repo`
2. ✓ Folders created with proper structure
3. Next: Move existing files to their new locations
4. Update README.md with new structure
5. Test that all imports work correctly
6. Merge to main when ready
