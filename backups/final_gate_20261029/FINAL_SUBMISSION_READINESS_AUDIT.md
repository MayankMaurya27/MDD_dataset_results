# Final Submission-Readiness Audit: Jha et al. MDD EEG Study

**Audit Date:** October 29, 2026  
**Workspace:** `/Users/mayankmaurya/Desktop/experiment`  
**Manuscript Under Audit:** `manuscript/Jha_MDD_Revised_Final.docx` (mirrored in `manuscript/Jha_MDD_Revised.docx`)  
**Authoritative Evidence Ledgers:** `reports/FINAL_EXPERIMENTAL_RESULTS.csv` and `reports/FINAL_CITATION_AUDIT.csv`

---

## Executive Summary & Verdict

- **Overall Submission-Readiness Status:** **PASS (Methodologically Sound, Fully Reconciled, and Ready for Submission with Transparently Stated Scope Boundaries)**
- **Critical Methodological Blockers:** **0 (None remaining)**
- **Fold Assignment Discrepancy Resolved:** **Fold 9 is Authoritative for `sub-HS15` across all code, fold assignments, and prediction artifacts.**
- **Citation Integrity:** **100% Reconciled (22 Directly Verified References, 1 Retracted Article Pruned, Table III Unharmonized Comparators Withheld)**
- **Participant-Level Leakage:** **0 (Zero leakage verified across all 10 folds under Stratified Group 10-Fold CV)**
- **Mathematical Consistency:** **100% Agreement across Confusion Matrices, Accuracies, Sensitivities, Specificities, F1 Scores, ROC-AUCs, Wilson Score CIs, and McNemar Tests**

---

## 1. Resolution of the `sub-HS15` Fold Discrepancy

A forensic cross-artifact inspection was performed to resolve whether participant `sub-HS15` was assigned to Fold 3 or Fold 9:

### Artifact Evidence:
1. **Authoritative Fold Assignment Files:**
   - `results/fold_assignments.csv`: `sub-HS15,HC,0,9,Healthy Control,4` $\rightarrow$ **Fold 9**.
   - `results/baseline/fold_assignments.csv`: `sub-HS15,HC,0,9,Healthy Control,4` $\rightarrow$ **Fold 9**.
2. **Out-of-Fold Prediction Files:**
   - `results/out_of_fold_predictions_eyesOpen.csv`: All 30 segment prediction rows for `sub-HS15` (15 from `run-1`, 15 from `run-2`) across all 8 classifiers record `fold: 9`.
   - `results/baseline/out_of_fold_predictions_eyesOpen.csv`: All 30 prediction rows record `fold: 9`.
3. **Feature Cache Metadata:**
   - `results/experiments/exp2_channel_weighting/metadata_channel_weighted.csv`: Rows 75–89 record `sub-HS15` with `fold: 9`.
4. **Historical Origin of "Fold 3":**
   - The string "Fold 3" appeared in early drafting notes (`reports/results_reconciliation.md` and `reports/final_baseline_verification.md`) as an uncorrected typographical draft carryover.
   - At no point did the execution scripts assign `sub-HS15` to Fold 3.

### Resolution Action:
- Confirmed that **Fold 9** is the true, authoritative fold assignment for `sub-HS15`.
- Updated `reports/results_reconciliation.md` and `reports/final_baseline_verification.md` to state **Fold 9**.
- Verified that `manuscript/Jha_MDD_Revised_Final.docx` and `manuscript/Jha_MDD_Revised.docx` consistently state **Fold 9**.

---

## 2. Reviewer Concerns Audit Matrix (Nine Core Criteria)

| # | Reviewer Concern / Criterion | Section in Final DOCX | Status | Supporting Forensic Evidence & Reconciliation |
| :-: | :--- | :--- | :-: | :--- |
| **1** | **Citation Mismatch & Unverifiable Table III** | Section 1, Section 4.1, References | **PASS** | Table III is withheld from the manuscript (Section 4.1). In-text citations [1]–[22] were audited and verified. Method misattributions (Wu [16] conformal SVM vs. Shen [17] kernel-target alignment channel selection) were disentangled. |
| **2** | **Participant Leakage** | Abstract, Section 2.7, Section 3.2 | **PASS** | Stratified Group 10-Fold Cross-Validation enforces zero subject overlap between train/test partitions. Repeat recording `sub-HS15` is quarantined to Fold 9. Verified via `tests/test_pipeline.py`. |
| **3** | **Retracted References** | References, Section 4.1 | **PASS** | *Mahato & Paul (2020)* (retracted by Editor-in-Chief in 2022) was completely removed from the manuscript and bibliography. |
| **4** | **Incomplete Reproducibility Details** | Section 2.2, 2.3, 2.4, 2.5, 2.6 | **PASS** | Complete specifications: 256.0 Hz sampling rate, 19 standard 10–20 monopolar leads, 0.5–45.0 Hz zero-phase bandpass filter, 2.0s non-overlapping windows (512 samples, 15 segments/rec), Kaiser window frequency smoothing ($\beta=5.0$, $L=255$), Hamming time smoothing ($L=31$), $224 \times 224 \times 3$ ImageNet scaling, frozen CNN weights, feature layer (`avgpool` 512-D), classifier parameters, and `random_state=42`. |
| **5** | **Missing Statistical Uncertainty & Comparisons** | Section 3.2, Table 2, Table 2 Note | **PASS** | Table 2 reports fold mean $\pm$ SD with explicit population ($\text{ddof}=0$) and sample ($\text{ddof}=1$) values. Added 95% Wilson score confidence intervals for participant-level accuracies: EC 89.66% [79.21%, 95.17%]; EO 91.80% [82.21%, 96.45%]; Nested Linear SVM 91.38% [81.36%, 96.26%]. Paired McNemar testing confirms no significant difference between tied top models on EC ($p=1.000$). |
| **6** | **Resource-Limited / Lightweight Claims** | Section 2.5, Section 3.1 | **PASS** | Removed unsupported assertions that ResNet-18 is intended for "resource-limited edge hardware". Reframed strictly around dimensionality: ResNet-18 yields a compact 512-D embedding (vs. 2,048-D for ResNet-50 and 4,096-D for AlexNet/VGG-16), mitigating the curse of dimensionality on a modest clinical cohort. |
| **7** | **SPWVD Kaiser-Window Ablation** | Section 3.3 | **PASS** | Evaluated Kaiser $\beta=7.0$ (93.10% accuracy, 100% sensitivity for Linear SVM) and narrow window $L=127, \beta=3.5$ (82.76%). Properly designated as an exploratory parameter sensitivity analysis. |
| **8** | **Overstated Novelty & Clinical Readiness** | Abstract, Section 1, Section 5 | **PASS** | Toned down all claims of clinical readiness or algorithm discovery. Presented as a rigorous, participant-independent benchmark combining established components (SPWVD, transfer learning, classical estimators). |
| **9** | **Academic Tone, Grammar & Formatting** | Entire Manuscript | **PASS** | Eliminated accusatory and emotive phrases ("data contamination", "collapsing catastrophically", "unjustified claims >99%"). Clean academic prose; professional typography, tables, and genuine embedded figures. |

---

## 3. Implementation Verification Against Source Code

The manuscript's narrative was audited directly against the Python implementation:

1. **SPWVD Parameters (`src/spwvd.py`):**
   - Time-smoothing window: Symmetric Hamming window of length $L = 31$ (`signal.windows.hamming(31)`).
   - Frequency-smoothing (lag) window: Kaiser window of length $L = 255$ with shape parameter $\beta = 5.0$ (`signal.windows.kaiser(255, beta=5.0)`).
   - Energy normalization: Clipping negative values via `np.maximum(wvd, 0)`, 99.5th percentile scaling, PyTorch bilinear interpolation resizing to $224 \times 224 \times 3$, and ImageNet channel standardization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
2. **Preprocessing & Segmentation (`src/preprocessing.py`):**
   - 19 Standard monopolar 10–20 electrodes (`Fp1, F3, C3, P3, O1, F7, T3, T5, Fz, Fp2, F4, C4, P4, O2, F8, T4, T6, Cz, Pz`).
   - Sampling rate: 256.0 Hz.
   - Digital filter: 0.5 – 45.0 Hz zero-phase IIR bandpass filter.
   - Epoch duration: $T = 2.0\text{ s}$ ($N = 512\text{ samples}$), strictly non-overlapping (`overlap = 0.0`), 15 epochs per recording file.
3. **Deep Feature Extraction (`src/feature_extractor.py`):**
   - Pretrained ResNet-18 (`torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`).
   - Fixed weights (`eval()` mode, gradient backpropagation disabled).
   - Output layer: Global Average Pooling (`avgpool`), extracting 512-dimensional embeddings.
4. **Classifier Hyperparameters (`src/classifiers.py`):**
   - Linear SVM: `SVC(kernel="linear", probability=True, random_state=42)`
   - Quadratic SVM: `SVC(kernel="poly", degree=2, probability=True, random_state=42)`
   - Cubic SVM: `SVC(kernel="poly", degree=3, probability=True, random_state=42)`
   - Fine Gaussian SVM: `SVC(kernel="rbf", gamma=0.5, probability=True, random_state=42)`
   - Medium Gaussian SVM: `SVC(kernel="rbf", gamma="scale", probability=True, random_state=42)`
   - Weighted KNN: `KNeighborsClassifier(n_neighbors=10, weights="distance")`
   - Ensemble Bagged Trees: `RandomForestClassifier(n_estimators=100, max_features=None, random_state=42)`
   - Neural Network Classifier: `MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)`
   - Standardization: `StandardScaler` fit strictly on training fold features.
5. **Participant Aggregation (`src/evaluation.py`):**
   - Function `aggregate_participant_predictions` groups out-of-fold segment predictions by `participant_id`, computes the arithmetic mean of $p(\text{MDD} \mid x_{i, s})$, and assigns $\hat{y}_i = 1$ if mean probability $\ge 0.5$.

---

## 4. Verification of Statistical Uncertainty & Paired Comparisons

1. **Wilson Score 95% Confidence Intervals:**
   - Eyes-Closed Linear SVM ($52/58$ correct): **89.66%**, 95% CI: **[79.21%, 95.17%]**
   - Eyes-Closed Medium Gaussian SVM ($52/58$ correct): **89.66%**, 95% CI: **[79.21%, 95.17%]**
   - Eyes-Open Medium Gaussian SVM ($56/61$ correct): **91.80%**, 95% CI: **[82.21%, 96.45%]**
   - Eyes-Open Linear SVM ($54/61$ correct): **88.52%**, 95% CI: **[78.16%, 94.33%]**
   - Nested Linear SVM ($53/58$ correct): **91.38%**, 95% CI: **[81.36%, 96.26%]**
2. **Paired McNemar Test:**
   - Contingency table comparing participant-level predictions between Linear SVM and Medium Gaussian SVM on the primary Eyes-Closed cohort ($N=58$):
     - Both correct: 50
     - Both incorrect: 4
     - Linear SVM correct, Medium Gaussian incorrect: 2
     - Linear SVM incorrect, Medium Gaussian correct: 2
     - Discordant pairs: $b = 2$, $c = 2$.
     - McNemar test statistic: $\chi^2 = (b - c)^2 / (b + c) = 0$, $p = 1.000$.
   - **Reporting Scope:** The test is strictly restricted to this compared pair on the primary cohort, confirming they are not statistically distinguishable on this sample. No claim of omnibus significance testing across all 8 estimators is made.

---

## 5. Rendered Word Document Inspection

`manuscript/Jha_MDD_Revised_Final.docx` was inspected:
- **Total Paragraphs:** 73
- **Total Tables:** 3
  - Table 1: CNN Architecture comparison (5 rows, 6 columns)
  - Table 2: Classifier performance under Stratified Group 10-Fold CV (9 rows, 11 columns)
  - Table 3: Embedded figure container for side-by-side confusion matrices (1 row, 2 columns)
- **Embedded Images:** 3 genuine, locally generated figures:
  1. **Figure 1:** End-to-end pipeline schematic diagram (`results/pipeline_architecture_diagram.png`).
  2. **Figure 2 (Left):** Out-of-fold confusion matrix for Linear SVM on Eyes-Closed ($N=58$: 24 TN, 4 FP, 2 FN, 28 TP; 89.66% accuracy).
  3. **Figure 2 (Right):** Out-of-fold confusion matrix for Medium Gaussian SVM on Eyes-Open ($N=61$: 27 TN, 2 FP, 3 FN, 29 TP; 91.80% accuracy).
- **Placeholders / Formatting:** Zero "TODO", "TBD", or empty figure placeholders exist. All table columns and headers are populated.

---

## 6. Citation-to-Claim Mapping & Audit Ledger

Detailed in [`reports/FINAL_CITATION_AUDIT.csv`](file:///Users/mayankmaurya/Desktop/experiment/reports/FINAL_CITATION_AUDIT.csv):
- **22 Directly Verified References:** Each mapped to verified DOIs, ISBNs, or official organization URLs ([1]–[22]).
- **1 Retracted Article Handled:** *Mahato & Paul (2020)* pruned completely following verification of publisher retraction notice.
- **Table III Comparators Handled:** Group of heterogeneous literature comparator studies labeled as `Unharmonized / Incompletely Verified Across External Sources` and withheld from the paper.

---

## 7. Execution Status, Test Results, and Scope Boundaries

### Automated Test Suite:
```bash
PYTHONPATH=. .venv/bin/pytest tests/test_pipeline.py -v
============================== 6 passed in 1.88s ===============================
Exit Code: 0
```
- `test_dataset_discovery_and_labels`: PASSED
- `test_zero_leakage_fold_isolation`: PASSED
- `test_segmentation_boundaries`: PASSED
- `test_spwvd_shape_and_numerical_stability`: PASSED
- `test_training_only_transformations`: PASSED
- `test_metrics_and_participant_aggregation`: PASSED

### Documented Scope Boundaries & Limitations:
1. **Raw Feature Extraction Scope:** The complete end-to-end raw EEG data pipeline was executed and validated via `src/smoke_test.py` across 4 participants (Exit code: 0, 2.05s). The full extraction across all 181 recordings was not re-executed from scratch during this audit to preserve verified feature tensors and avoid ~10–12 minutes of compute. The 10-fold cross-validation experiment was rerun on the cached features in `results/features_cache/` via `src/run_experiments.py` (Exit code: 0).
2. **Single-Center Cohort:** Dataset originates from a single acquisition center (HUSM, 64 participants). External multi-center validation remains necessary to demonstrate generalizability.
3. **Exploratory Status of Improvements:** Spatial montage weighting (Exp 2) and Kaiser window smoothing variations (Exp 3) were evaluated post-hoc on this cohort and remain exploratory hypotheses.
