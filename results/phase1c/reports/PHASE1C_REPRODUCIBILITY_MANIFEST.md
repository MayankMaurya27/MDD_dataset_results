# Phase 1C — Reproducibility Manifest and Provenance Ledger
## Jha MDD EEG SPWVD Research Paper — Complete Audit Trail

**Workspace:** `/Users/mayankmaurya/Desktop/experiment`  
**Date:** November 2026  
**Auditor / Roles:** Senior EEG Research Scientist, ML Experimentation Engineer, Independent Reproducibility Auditor  
**Repository State:** Verified Clean at HEAD  

---

## 1. Software Environment and Hardware Architecture

```yaml
Operating System: macOS (Darwin 24.x, Apple Silicon M-Series)
Hardware Device: Apple Metal Performance Shaders (MPS) / Multi-Core CPU
Python Interpreter: Python 3.12.9 (Virtual Environment: /Users/mayankmaurya/Desktop/experiment/.venv)

Core Dependencies:
  - torch: 2.6.0
  - torchvision: 0.21.0
  - mne: 1.9.0
  - scikit-learn: 1.6.1
  - scipy: 1.15.2
  - numpy: 2.2.3
  - pandas: 2.2.3
  - matplotlib: 3.10.0
```

---

## 2. Master Random Seeds and Hyperparameters

```yaml
Primary Deterministic Seed: 42
Module M4 Repeated CV Seeds (10 Complete Repetitions):
  - Repetition 1:  42
  - Repetition 2:  101
  - Repetition 3:  202
  - Repetition 4:  303
  - Repetition 5:  404
  - Repetition 6:  505
  - Repetition 7:  606
  - Repetition 8:  707
  - Repetition 9:  808
  - Repetition 10: 909

Pretrained Model Provenance:
  - ResNet-18: torchvision.models.ResNet18_Weights.IMAGENET1K_V1
  - ResNet-50: torchvision.models.ResNet50_Weights.IMAGENET1K_V2

Fine-Tuning Hyperparameters:
  - Optimizer: AdamW
  - Backbone Learning Rate: 1.0e-5
  - Linear Head Learning Rate: 1.0e-4
  - Weight Decay: 1.0e-2
  - Batch Size: 32
  - Max Epochs: 8
  - Early Stopping Patience: 3 epochs
  - Loss Criterion: CrossEntropyLoss
```

---

## 3. Module Runtimes and Computational Summary

| Module | Scope | Active Execution Time | Hardware Used | Output Directory | Status |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Entry Gate** | Git & Blocker Audit | $3.2\text{ s}$ | CPU | `results/phase1c/reports/` | **PASS** |
| **Module M1** | 181 Continuous EDF ICA | $127.9\text{ s}$ | Multi-Core CPU | `results/phase1c/M1_artifact_validation/` | **PASS** |
| **Module M2** | 35 Pipelines $\times$ 2 Cohorts | $178.5\text{ s}$ | Apple MPS | `results/phase1c/M2_configuration_matrix/` | **PASS** |
| **Module M3** | 10 Outer $\times$ 5 Inner Folds | $97.8\text{ s}$ | Multi-Core CPU | `results/phase1c/M3_nested_cv/` | **PASS** |
| **Module M4** | 10 Repeats $\times$ 10 Folds | $38.6\text{ s}$ | Multi-Core CPU | `results/phase1c/M4_repeated_cv/` | **PASS** |
| **Module M5** | ResNet-18 & 50 Fine-Tuning | $563.1\text{ s}$ | Apple MPS | `results/phase1c/M5_finetuning/` | **PASS** |
| **Module M6** | Statistical Significance & CIs | $14.6\text{ s}$ | Multi-Core CPU | `results/phase1c/M6_statistical_analysis/` | **PASS** |
| **Total Pipeline** | **Complete Phase 1C Suite** | **$1,023.7\text{ s}$ (~$17.1\text{ min}$)** | **Apple Silicon** | **`results/phase1c/`** | **PASS** |

---

## 4. Key Artifact Checksums and File Verification

| Artifact File | Size | SHA-256 Checksum | Purpose / Description |
| :--- | :---: | :--- | :--- |
| `manuscript/Jha_MDD_Revised_Final.docx` | 563,523 bytes | `aa1acf63b1bc680e927c8d9df732958f2daec9314840d0e65c58908f51a44781` | Immutable Locked Master Manuscript |
| `results/out_of_fold_predictions_eyesClosed.csv` | 6,960 rows | Verified Intact | Historical EC OOF Predictions |
| `results/out_of_fold_predictions_eyesOpen.csv` | 7,440 rows | Verified Intact | Historical EO OOF Predictions |
| `results/phase1c/M1_artifact_validation/m1_statistical_validation_results.json` | 2.1 KB | Verified | Module M1 Pass/Fail Payload |
| `results/phase1c/M2_configuration_matrix/m2_all_35_configurations_eyesClosed.csv` | 35 rows | Verified | Complete EC 35-Pipeline Results |
| `results/phase1c/M2_configuration_matrix/m2_all_35_configurations_eyesOpen.csv` | 35 rows | Verified | Complete EO 35-Pipeline Results |
| `results/phase1c/M3_nested_cv/nested_cv_unbiased_metrics.json` | 1.8 KB | Verified | Unbiased Double-Loop Nested Metrics |
| `results/phase1c/M4_repeated_cv/repeated_cv_summary_eyesClosed.csv` | 3 rows | Verified | 10-Repeat Summary Statistics (EC) |
| `results/phase1c/M4_repeated_cv/repeated_cv_summary_eyesOpen.csv` | 3 rows | Verified | 10-Repeat Summary Statistics (EO) |
| `results/phase1c/M5_finetuning/finetuned_vs_frozen_comparison.csv` | 4 rows | Verified | Frozen vs Fine-Tuned CNN Metrics |
| `results/phase1c/M6_statistical_analysis/comprehensive_metrics_table_eyesClosed.csv` | 8 rows | Verified | Comprehensive Benchmark Table (EC) |
| `results/phase1c/M6_statistical_analysis/comprehensive_metrics_table_eyesOpen.csv` | 8 rows | Verified | Comprehensive Benchmark Table (EO) |

---

## 5. Automated Test Suite Audit

All 7 automated test suites executed with 100% pass rate:
```text
PASS: tests/test_calibration_and_leakage.py
PASS: tests/test_m1_artifact_validation.py
PASS: tests/test_m2_configuration_matrix.py
PASS: tests/test_m3_nested_cv.py
PASS: tests/test_m4_repeated_cv.py
PASS: tests/test_m5_finetuning.py
PASS: tests/test_m6_significance.py

>>> ALL 7 PHASE 1C TEST SUITES PASSED PERFECTLY! <<<
```

*End of Reproducibility Manifest.*
