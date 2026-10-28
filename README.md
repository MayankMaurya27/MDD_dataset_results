# MDD Detection Using SPWVD Time-Frequency Images & Deep Learning Features

End-to-end research reproduction, experimental validation, and manuscript revision for **"Automated Detection of Major Depressive Disorder Using SPWVD-Based EEG Time-Frequency Images and Deep Learning Features"**.

---

## 📖 Overview

This repository contains the complete experimental code, verification pipelines, statistical reports, confusion matrices, and the revised research manuscript for automated MDD diagnosis from multi-channel EEG recordings using Smoothed Pseudo Wigner-Ville Distribution (SPWVD) time-frequency representations and transfer learning backbones (ResNet18, AlexNet, ResNet50, VGG16).

For the full audit and synthesis, see [FINAL_EXPERIMENT_AND_REVISION_REPORT.md](FINAL_EXPERIMENT_AND_REVISION_REPORT.md).

---

## 🗂️ Repository Structure

```text
├── FINAL_EXPERIMENT_AND_REVISION_REPORT.md  # Comprehensive reproduction & revision report
├── README.md                                # Repository guide & documentation
├── requirements.txt                         # Python dependencies
├── manuscript/
│   └── Jha_MDD_Revised.docx                 # Fully revised, publication-ready manuscript
├── reports/
│   ├── final_baseline_verification.md       # Certified frozen baseline metrics report
│   ├── results_reconciliation.md            # Resolution of participant counts (58 vs 61)
│   ├── model_improvement_experiments.md     # Ablations & nested cross-validation report
│   ├── dataset_audit.md                     # BIDS filesystem and channel audit
│   ├── data_quality_report.md               # Signal quality & artifact evaluation
│   ├── experimental_protocol.md             # Leakage prevention & 10-fold CV design
│   ├── manuscript_claims_audit.md           # Claim-by-claim audit of original draft
│   ├── manuscript_revision_log.md           # Comprehensive edit & correction log
│   ├── dataset_inventory.csv                # Complete inventory of recordings & subjects
│   └── citation_audit.csv                   # Audit of all 21 references
├── results/
│   ├── baseline/                            # Frozen baseline CSVs, fold metrics, configs
│   ├── confusion_matrices/                  # High-resolution confusion matrices
│   ├── experiments/                         # Logs & metrics for Experiments 1–4
│   ├── features_cache/                      # Extracted deep-learning feature arrays (.npy)
│   └── metrics_summary.csv                  # Verified pooled summary metrics
├── src/
│   ├── audit_dataset.py                     # Dataset discovery & BIDS integrity verification
│   ├── preprocessing.py                     # EEG bandpass filtering & segmentation
│   ├── spwvd.py                             # SPWVD computation & RGB colormap rendering
│   ├── feature_extractor.py                 # PyTorch pre-trained backbone feature extraction
│   ├── extract_features.py                  # Batch feature caching runner
│   ├── classifiers.py                       # Supervised classifiers (SVM, KNN, MLP, Bagging)
│   ├── evaluation.py                        # Participant-level aggregation & performance metrics
│   ├── generate_folds.py                    # Stratified group 10-fold split generator
│   ├── run_experiments.py                   # Main 10-fold CV experiment runner
│   ├── run_improvement_experiments.py       # Ablations & nested CV runner
│   └── generate_revised_manuscript.py       # Automated DOCX manuscript generator
└── tests/
    └── test_pipeline.py                     # Pytest suite validating pipeline integrity
```

---

## 📊 Summary of Verified Results

Experiments were conducted with strict subject-level isolation to prevent data leakage:

| Condition | Cohort Size | Best Classifier | Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Eyes-Closed (EC)** | $N=58$ (28 HC, 30 MDD) | Linear SVM (Nested CV) | **91.38%** | 93.33% | 89.29% | 91.80% | 94.76% |
| **Eyes-Closed (EC)** | $N=58$ (28 HC, 30 MDD) | Linear SVM (Fixed Baseline) | **89.66%** | 93.33% | 85.71% | 90.32% | 95.36% |
| **Eyes-Open (EO)** | $N=61$ (29 HC, 32 MDD) | Medium Gaussian SVM | **91.80%** | 90.62% | 93.10% | 92.06% | 95.91% |

---

## 🛠️ Setup & Reproducibility

### 1. Requirements & Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset
The dataset utilized is OpenNeuro / NEMAR Accession **`nm000114_v1.0.0`** (DOI: `10.82901/nemar.nm000114`):
- Download and unpack the BIDS dataset into the root directory as `nm000114_v1.0.0/`.
- Verify dataset integrity:
```bash
python src/audit_dataset.py
```

### 3. Run Pipeline Tests
```bash
PYTHONPATH=. pytest tests/test_pipeline.py -v
```

### 4. Run Reproduction & Improvement Experiments
```bash
# Main baseline experiments across all backbones & classifiers
python src/run_experiments.py

# Model improvement experiments & nested cross-validation
python src/run_improvement_experiments.py
```

---

## 📜 Citation & License
Dataset is licensed under [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/). Code and analysis are released under the MIT License.

