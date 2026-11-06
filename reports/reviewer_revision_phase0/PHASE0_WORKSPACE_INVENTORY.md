# PHASE 0 — WORKSPACE AND ENVIRONMENT INVENTORY
## Jha MDD EEG SPWVD Research Paper — Forensic Reviewer Revision

**Auditor:** Senior Biomedical Signal Processing & Machine Learning Reproducibility Auditor  
**Date:** November 7, 2026  
**Project Root:** `/Users/mayankmaurya/Desktop/experiment`  
**Host Platform:** macOS (Darwin 24.x, Apple Silicon M-series)  
**Python Runtime:** Python 3.12.9 (Virtual Environment: `/Users/mayankmaurya/Desktop/experiment/.venv`)  
**Core Toolchain:** PyTorch 2.6.0, torchvision 0.21.0, MNE-Python 1.9.0, Scikit-Learn 1.6.1, SciPy 1.15.2, Pandas 2.2.3, NumPy 2.2.3, python-docx 1.1.2, pytest 8.3.4  
**Audit Policy:** Strict READ-ONLY inspection of all source code, datasets, caches, experiment outputs, and manuscripts.

---

## 1. Project Directory Structure and High-Level Layout

The project workspace is organized into self-contained modules covering raw data, preprocessing, feature caching, classification, experiment logs, manuscripts, and reports:

```
/Users/mayankmaurya/Desktop/experiment/
├── .venv/                                # Local Python 3.12 virtual environment
├── manuscript/                           # Primary and mirror Microsoft Word deliverables
│   ├── Jha_MDD_Revised_Final.docx        # Master designated manuscript (563,523 bytes)
│   └── Jha_MDD_Revised.docx              # Mirrored manuscript (563,523 bytes)
├── nm000114_v1.0.0/                      # HUSM BIDS derivative EEG dataset (181 EDF files)
│   ├── dataset_description.json          # BIDS 1.7.0 metadata (CC-BY-4.0)
│   ├── participants.tsv                  # 64 participant rows (sub-HS1..30, sub-MDDS1..34)
│   ├── sub-HS1/ ... sub-HS30/            # 30 Healthy Control participant directories
│   └── sub-MDDS1/ ... sub-MDDS34/        # 34 MDD patient participant directories
├── src/                                  # Pipeline source code
│   ├── preprocessing.py                  # MNE loading, 10-20 channel selection, bandpass, slicing
│   ├── spwvd.py                          # Analytic Hilbert, Kaiser/Hamming SPWVD, RGB normalization
│   ├── feature_extractor.py              # PyTorch backbones (ResNet-18/50, AlexNet, VGG-16)
│   ├── extract_features.py               # Cache generator for SPWVD features
│   ├── classifiers.py                    # 8 downstream scikit-learn classifiers
│   ├── evaluation.py                     # Metrics, Wilson CIs, bootstrap ROC/AUC, McNemar tests
│   ├── generate_folds.py                 # Stratified Group 10-Fold CV partition generator
│   ├── run_experiments.py                # Main baseline 10-fold CV pipeline runner
│   ├── run_improvement_experiments.py    # Ablations, channel weighting, nested CV
│   ├── generate_pipeline_diagram.py      # Matplotlib publication pipeline diagram generator
│   └── generate_revised_manuscript.py    # Master DOCX generation script (guarded)
├── results/                              # Empirical outputs, metrics, and cached features
│   ├── features_cache/                   # Cached .npy embeddings & metadata
│   ├── baseline/                         # Baseline run metrics, predictions, and confusion matrices
│   ├── experiments/                      # Four ablation and improvement experiment folders
│   ├── fold_assignments.csv              # Master 64-subject fold assignments
│   ├── out_of_fold_predictions_*.csv     # Segment-level out-of-fold probability files
│   ├── fold_metrics_*.csv                # Per-fold sensitivity, specificity, accuracy, F1, AUC
│   └── metrics_summary_*.csv             # Pooled and fold-mean summary tables
├── reports/                              # Audit reports and evidence logs
│   ├── FINAL_SUBMISSION_READINESS_AUDIT.md
│   ├── FINAL_HANDOFF_SUMMARY.md
│   ├── FINAL_CITATION_AUDIT.csv
│   ├── verified_evidence_results_ledger.csv
│   └── reviewer_revision_phase0/         # Phase 0 forensic deliverables
├── backups/                              # Checkpoint backups of manuscripts and reports
└── tests/                                # Automated verification test suite
    ├── test_pipeline.py                  # Leakage, determinism, and metric verification
    └── test_reproducibility.py           # Reproduction validation suite
```

---

## 2. Comprehensive Inventory of Workspace Artifacts

| Relative Path | Size | Format | Purpose & Description | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| `manuscript/Jha_MDD_Revised_Final.docx` | 563,523 B | DOCX | Master manuscript deliverable. SHA-256: `aa1acf63...` | Authoritative |
| `manuscript/Jha_MDD_Revised.docx` | 563,523 B | DOCX | Mirror copy matching master byte-for-byte. SHA-256: `aa1acf63...` | Authoritative |
| `nm000114_v1.0.0/participants.tsv` | 1,885 B | TSV | BIDS participant table: 64 rows, demographics set to `n/a`. | Authoritative |
| `nm000114_v1.0.0/dataset_description.json` | 536 B | JSON | BIDS dataset metadata (CC-BY-4.0 license, Mumtaz citation). | Authoritative |
| `nm000114_v1.0.0/sub-*/eeg/*.edf` (181 files) | ~1.4 GB | Binary EDF | Continuous 19-24 channel clinical EEG recordings at 256.0 Hz. | Authoritative |
| `src/preprocessing.py` | 3,813 B | Python | Raw EDF load, 19-lead extraction, 0.5–45 Hz filter, 2.0s windowing. | Audited |
| `src/spwvd.py` | 4,265 B | Python | Hilbert transform, Kaiser lag ($\beta=5.0$), Hamming time ($L=31$). | Audited |
| `src/feature_extractor.py` | 3,551 B | Python | PyTorch pretrained backbones, frozen weights, feature extraction. | Audited |
| `src/extract_features.py` | 5,190 B | Python | Loops recordings, computes SPWVD, saves `.npy` caches and metadata. | Audited |
| `src/classifiers.py` | 2,685 B | Python | 8 scikit-learn estimators: Linear/Quad/Cubic/RBF SVM, KNN, RF, MLP. | Audited |
| `src/evaluation.py` | 2,822 B | Python | Participant posterior pooling, fold metrics, Wilson CIs, AUC. | Audited |
| `src/generate_folds.py` | 3,245 B | Python | Stratified Group 10-Fold partition generator (`sub-HS15` isolated). | Audited |
| `src/run_experiments.py` | 13,456 B | Python | Execution harness for 10-fold CV across 8 classifiers and 4 CNNs. | Audited |
| `src/run_improvement_experiments.py` | 31,779 B | Python | Ablation suite: aggregation, channel weighting, smoothing, nested CV. | Audited |
| `src/generate_revised_manuscript.py` | 71,355 B | Python | Programmatic generator for docx with guard against overwriting. | Audited |
| `results/fold_assignments.csv` | 2,667 B | CSV | 64 participant assignments to Folds 0–9. | Authoritative |
| `results/features_cache/metadata_eyesClosed.csv` | 40,060 B | CSV | 870 segments (58 subjects $\times$ 15), sample start/end indices. | Authoritative |
| `results/features_cache/metadata_eyesOpen.csv` | 40,970 B | CSV | 930 segments (62 scans $\times$ 15), sample start/end indices. | Authoritative |
| `results/features_cache/features_resnet18_eyesClosed.npy` | 1,781,888 B | NumPy | 870 $\times$ 512 float32 ResNet-18 embeddings for Eyes-Closed. | Authoritative |
| `results/features_cache/features_resnet18_eyesOpen.npy` | 1,904,768 B | NumPy | 930 $\times$ 512 float32 ResNet-18 embeddings for Eyes-Open. | Authoritative |
| `results/features_cache/features_resnet50_eyesClosed.npy` | 7,127,168 B | NumPy | 870 $\times$ 2,048 float32 ResNet-50 embeddings for Eyes-Closed. | Authoritative |
| `results/features_cache/features_resnet50_eyesOpen.npy` | 7,618,688 B | NumPy | 930 $\times$ 2,048 float32 ResNet-50 embeddings for Eyes-Open. | Authoritative |
| `results/features_cache/features_alexnet_eyesClosed.npy` | 14,254,208 B | NumPy | 870 $\times$ 4,096 float32 AlexNet embeddings for Eyes-Closed. | Authoritative |
| `results/features_cache/features_alexnet_eyesOpen.npy` | 15,237,248 B | NumPy | 930 $\times$ 4,096 float32 AlexNet embeddings for Eyes-Open. | Authoritative |
| `results/features_cache/features_vgg16_eyesClosed.npy` | 14,254,208 B | NumPy | 870 $\times$ 4,096 float32 VGG-16 embeddings for Eyes-Closed. | Authoritative |
| `results/features_cache/features_vgg16_eyesOpen.npy` | 15,237,248 B | NumPy | 930 $\times$ 4,096 float32 VGG-16 embeddings for Eyes-Open. | Authoritative |
| `results/out_of_fold_predictions_eyesClosed.csv` | 488,241 B | CSV | 6,960 segment predictions (870 seg $\times$ 8 classifiers) with probabilities. | Authoritative |
| `results/out_of_fold_predictions_eyesOpen.csv` | 509,345 B | CSV | 7,440 segment predictions (930 seg $\times$ 8 classifiers) with probabilities. | Authoritative |
| `results/metrics_summary_eyesClosed.csv` | 1,898 B | CSV | Pooled and fold mean $\pm$ SD metrics across 8 classifiers (EC). | Authoritative |
| `results/metrics_summary_eyesOpen.csv` | 1,892 B | CSV | Pooled and fold mean $\pm$ SD metrics across 8 classifiers (EO). | Authoritative |
| `results/fold_metrics_eyesClosed.csv` | 7,502 B | CSV | 80 rows (10 folds $\times$ 8 classifiers) detailed performance metrics (EC). | Authoritative |
| `results/fold_metrics_eyesOpen.csv` | 7,352 B | CSV | 80 rows (10 folds $\times$ 8 classifiers) detailed performance metrics (EO). | Authoritative |
| `results/pipeline_architecture_diagram.png` | 370,587 B | PNG | High-resolution publication diagram of 6-stage methodology. | Authoritative |
| `results/confusion_matrices/figure2_composite_confusion_matrices.png` | 153,023 B | PNG | High-resolution composite confusion matrix for EC and EO. | Authoritative |
| `reports/FINAL_SUBMISSION_READINESS_AUDIT.md` | 27,570 B | Markdown | Pre-revision audit of nine core criteria and layout checks. | Audited |
| `reports/FINAL_CITATION_AUDIT.csv` | 13,044 B | CSV | Comprehensive audit of in-text citations [1]–[22] and retracted papers. | Authoritative |
| `reports/verified_evidence_results_ledger.csv` | 26,973 B | CSV | 108 verified numerical experimental result rows with exact provenance. | Authoritative |

---

## 3. Dependency Environment and Runtime Versions

The execution environment is locked and verified with the following exact package versions:

```
python:             3.12.9
torch:              2.6.0
torchvision:        0.21.0
mne:                1.9.0
mne-bids:           0.17.0
scikit-learn:       1.6.1
scipy:              1.15.2
pandas:             2.2.3
numpy:              2.2.3
python-docx:        1.1.2
matplotlib:         3.10.0
pytest:             8.3.4
```

All unit tests pass within `.venv` with zero dependency conflicts.

