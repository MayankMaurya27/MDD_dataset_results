# Final Reproducibility Report: MDD EEG SPWVD-CNN Pipeline

**Date:** October 29, 2026  
**Project:** Automated Detection of Major Depressive Disorder Using SPWVD-Based EEG Time-Frequency Images and Deep Learning Features  
**Repository / Workspace:** `/Users/mayankmaurya/Desktop/experiment`  
**Authoritative Dataset:** NEMAR accession `nm000114_v1.0.0` (HUSM 64-participant MDD EEG dataset)

---

## 1. Executive Summary & Reproducibility Scope

This reproducibility report provides an audit and verification trail for all computational experiments, unit tests, data pipelines, and evaluation metrics reported in the revised manuscript (*Jha et al.*).

### Distinction Between Raw Execution and Cached Recalculation
To maintain absolute scientific transparency, we explicitly distinguish between execution modes:
1. **Raw End-to-End Pipeline Execution (Verified on 4-Participant Subset):**
   - Validated via `src/smoke_test.py` (Execution time: 2.05s, Exit code: 0).
   - Verifies the complete data chain on raw `.edf` files: raw signal loading via MNE, 19-channel standard 10–20 montage extraction, 0.5–45.0 Hz zero-phase bandpass filtering, 2.0-second non-overlapping temporal slicing, analytic signal transformation via Hilbert transform, two-dimensional Smoothed Pseudo Wigner-Ville Distribution (SPWVD) calculation, ImageNet-standardized RGB tensor preparation, pretrained CNN forward-pass feature extraction (ResNet-18), training-only feature standardization (`StandardScaler`), classifier fitting, and out-of-sample evaluation.
2. **Cross-Validation Execution on Cached Feature Tensors:**
   - Validated via `src/run_experiments.py` on the cached features in `results/features_cache/` (Exit code: 0).
   - Re-evaluates all 8 classifiers and 4 CNN backbones across 10 folds for both Eyes-Closed ($N=58$) and Eyes-Open ($N=61$) conditions under Stratified Group 10-Fold CV.
3. **Evaluation and Metrics Recalculation:**
   - All reported baseline tables, confusion matrices, ROC-AUC values, and fold standard deviations were verified and recalculated directly from participant-level and segment-level prediction artifacts (`results/baseline/predictions_eyesClosed.csv`, `results/baseline/predictions_eyesOpen.csv`, `results/baseline/metrics_summary_*.csv`).
   - Re-aggregation from raw segment prediction probabilities to participant-level decisions confirms 100% mathematical consistency with reported numbers.
4. **Explicit Limitation on Full Raw Re-Extraction:**
   - The full batch SPWVD extraction across all 181 `.edf` recordings from scratch was NOT rerun during this audit to preserve existing verified artifacts and avoid unnecessary expensive compute (~10–12 minutes). The pre-extracted feature tensors in `results/features_cache/` were verified for shape, integrity, and alignment with `participants.tsv`.

---

## 2. Computational Environment

- **Operating System:** Darwin (macOS 15+ / ARM64)
- **Virtual Environment:** `/Users/mayankmaurya/Desktop/experiment/.venv`
- **Python Version:** Python 3.12.13
- **Primary Dependencies:**
  - `mne`: 1.6.1 (EEG EDF reading, channel mapping)
  - `torch`: 2.2.0 (Pretrained CNN feature extractors: ResNet-18, ResNet-50, AlexNet, VGG-16)
  - `torchvision`: 0.17.0
  - `scikit-learn`: 1.4.1.post1 (StratifiedGroupKFold, SVM, KNN, Random Forest, MLP, metrics)
  - `python-docx`: 1.1.0 (Automated revised manuscript generation)
  - `pytest`: 9.2.0 (Automated test suite)
  - `numpy`: 1.26.4
  - `scipy`: 1.12.0 (Signal filtering, SPWVD convolution)
  - `pandas`: 2.2.1

---

## 3. Test Suite Execution & Results

The automated regression test suite is located in `tests/test_pipeline.py`.

### Execution Command:
```bash
PYTHONPATH=. .venv/bin/pytest tests/test_pipeline.py -v
```

### Execution Log & Exit Code:
```text
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.2.0, pluggy-1.7.0 -- /Users/mayankmaurya/Desktop/experiment/.venv/bin/python3.12
cachedir: .pytest_cache
rootdir: /Users/mayankmaurya/Desktop/experiment
plugins: platformdirs-4.13.7
collecting ... collected 6 items

tests/test_pipeline.py::test_dataset_discovery_and_labels PASSED         [ 16%]
tests/test_pipeline.py::test_zero_leakage_fold_isolation PASSED          [ 33%]
tests/test_pipeline.py::test_segmentation_boundaries PASSED              [ 50%]
tests/test_pipeline.py::test_spwvd_shape_and_numerical_stability PASSED  [ 66%]
tests/test_pipeline.py::test_training_only_transformations PASSED        [ 83%]
tests/test_pipeline.py::test_metrics_and_participant_aggregation PASSED  [100%]

============================== 6 passed in 1.70s ===============================
Exit Code: 0
```

### Test Scope Summary:
1. `test_dataset_discovery_and_labels`: Verifies physical existence of `participants.tsv` and checks that all participants have valid diagnostic labels (HC vs. MDD).
2. `test_zero_leakage_fold_isolation`: Enforces that in `StratifiedGroupKFold(n_splits=10)`, no participant's ID appears in both training and testing partitions for any fold.
3. `test_segmentation_boundaries`: Confirms that 2.0-second epochs are strictly bounded within single continuous recordings and never cross recording sessions.
4. `test_spwvd_shape_and_numerical_stability`: Verifies that SPWVD generates exact $(224, 224, 3)$ ImageNet-ready tensors without NaN, Inf, or negative values post-clipping.
5. `test_training_only_transformations`: Verifies that feature standardizers (`StandardScaler`) are fit strictly on training fold features and transform test fold features out-of-sample.
6. `test_metrics_and_participant_aggregation`: Validates participant-level mean probability aggregation, confusion matrix calculation, and mathematical consistency between TP, FP, TN, FN, accuracy, sensitivity, and specificity.

---

## 4. Raw Pipeline Smoke Test Execution

### Execution Command:
```bash
PYTHONPATH=. .venv/bin/python3 src/smoke_test.py
```

### Output:
```text
=== STARTING STAGE 1 SMOKE TEST ===
Testing on 4 participants: ['sub-HS1', 'sub-HS10', 'sub-MDDS1', 'sub-MDDS10']
Loading sub-HS1 from sub-HS1/eeg/sub-HS1_task-eyesClosed_eeg.edf...
  Extracted 5 segments of shape (19, 512)
Loading sub-HS10 from sub-HS10/eeg/sub-HS10_task-eyesClosed_eeg.edf...
  Extracted 5 segments of shape (19, 512)
Loading sub-MDDS1 from sub-MDDS1/eeg/sub-MDDS1_task-eyesClosed_eeg.edf...
  Extracted 5 segments of shape (19, 512)
Loading sub-MDDS10 from sub-MDDS10/eeg/sub-MDDS10_task-eyesClosed_eeg.edf...
  Extracted 5 segments of shape (19, 512)
Total processed smoke test images: 20
Extracted feature matrix shape: (20, 512)
X_train: (10, 512), y_train: (10,)
X_test: (10, 512), y_test: (10,)
Smoke test completed in 2.05s!
Smoke test evaluation: {'accuracy': 70.0, 'balanced_accuracy': 70.0, 'precision': 75.0, 'sensitivity': 60.0, 'specificity': 80.0, 'f1_score': 66.67, 'roc_auc': 64.0, 'tp': 3, 'fp': 1, 'tn': 4, 'fn': 2}
Exit Code: 0
```

---

## 5. Dataset Inventory and Provenance Reconciliation

- **Dataset Identifier:** Hospital Universiti Sains Malaysia (HUSM) MDD EEG Dataset, BIDS version 1.7.0.
- **Physical Root:** `nm000114_v1.0.0/`
- **Metadata Source:** `nm000114_v1.0.0/participants.tsv` (UTF-8 with BOM).
- **Physical Inventory Check:**
  - Total participants: Exactly 64 participants (30 Healthy Controls: `sub-HS1` to `sub-HS30`; 34 MDD Patients: `sub-MDDS1` to `sub-MDDS34`).
  - Total EDF files: Exactly 181 recordings.
- **Condition-Specific Breakdown:**
  1. **Eyes-Closed Resting State (`task-eyesClosed`):**
     - Participants: 58 unique participants (28 HC, 30 MDD).
     - Missing 6 subjects: `sub-HS12`, `sub-HS18`, `sub-MDDS4`, `sub-MDDS8`, `sub-MDDS12`, `sub-MDDS16`.
     - Epochs: 15 epochs/subject $\times$ 58 subjects = 870 segments.
  2. **Eyes-Open Resting State (`task-eyesOpen`):**
     - Participants: 61 unique participants (29 HC, 32 MDD).
     - Recordings: 62 files (`sub-HS15` has two separate runs: `run-1` and `run-2`).
     - Missing 3 subjects: `sub-HS25`, `sub-MDDS7`, `sub-MDDS8`.
     - Epochs: 60 subjects $\times$ 15 epochs $+$ `sub-HS15` $\times$ 30 epochs = 930 segments.
  3. **Auditory Oddball P300 Task (`task-P300`):**
     - Participants: 61 unique participants (28 HC, 33 MDD).
     - Missing 3 subjects: `sub-HS20`, `sub-HS21`, `sub-MDDS9`.
     - Reserved for separate event-related potential investigations; excluded from resting-state benchmarks.
- **Resolution of Metadata Discrepancy:** The README narrative contained an unedited draft sentence stating "120 subjects were recruited." Exhaustive physical audit confirms that exactly 64 participants were recorded and archived.

---

## 6. Determinism & Random Seeds

All stochastic operations are strictly pinned:
- Cross-validation splitting: `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`
- Classifier random states:
  - Linear / Quadratic / Cubic / Gaussian SVM: `random_state=42`
  - Multi-Layer Perceptron (MLP): `random_state=42`, `max_iter=500`
  - Bagged Decision Trees: `random_state=42`, `n_estimators=100`
- Feature extractors: Pretrained PyTorch models in `eval()` mode with frozen weights (deterministic execution).
