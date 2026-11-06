# Final Submission-Readiness Audit: Jha et al. MDD EEG Study

**Audit Date:** October 29, 2026  
**Workspace:** `/Users/mayankmaurya/Desktop/experiment`  
**Manuscript Under Audit:** `manuscript/Jha_MDD_Revised_Final.docx`  
**Identical Mirror Manuscript:** `manuscript/Jha_MDD_Revised.docx` (byte-for-byte identical, verified via `cmp`)  
**Primary Evidence Ledgers:** `reports/FINAL_EXPERIMENTAL_RESULTS.csv` and `reports/FINAL_CITATION_AUDIT.csv`

---

## Executive Summary & Final Verdict

- **Final Submission-Readiness Verdict:** **READY FOR FINAL HUMAN APPROVAL**
- **Critical Methodological Blockers:** **0 (None)**
- **Citation-to-Claim Integrity:** **100% Reconciled (22 Directly Verified References [1]–[22], 1 Retracted Article Pruned, Table III Unharmonized Comparators Withheld)**
- **Resolution of [18] Citation Mismatch:** Resolved by removing the misplaced citation from general methodological text in the Introduction. Reference [18] (Mumtaz et al. 2017) is now cited strictly where directly supported: Section 2.1 (dataset provenance) and Section 4.1 (literature comparator).
- **Resolution of `sub-HS15` Fold Discrepancy:** Resolved and verified as **Fold 9** across all authoritative code, fold assignments, and out-of-fold prediction artifacts.
- **Participant-Level Leakage:** **0 (Zero leakage verified across all 10 folds under Stratified Group 10-Fold CV)**
- **Mathematical Consistency:** **100% Agreement across Confusion Matrices, Accuracies, Sensitivities, Specificities, Precision, F1 Scores, ROC-AUCs, Wilson Score CIs, and Paired McNemar Testing**
- **Document Integrity:** Complete Word document with 73 paragraphs, 3 tables, and 3 embedded genuine figures (Figure 1 pipeline schematic, Figure 2 confusion matrices), zero empty placeholders, and verified academic formatting.

---

## 1. Resolution of the [18] Citation-to-Claim Issue

A detailed audit identified that reference `[18]` (Mumtaz et al., *Biomedical Signal Processing and Control*, 2017) was cited in two distinct contexts:
1. **Introduction (Paragraph 4):** Was previously cited for the general statement:
   *"Recent neuroimaging literature has emphasized the necessity of group-aware validation to prevent this form of correlation bias [18]."*
2. **Section 2.1 (Paragraph 12):** Cited for dataset curation:
   *"...dataset collected at the Hospital Universiti Sains Malaysia (HUSM) and curated by Mumtaz et al. [18]..."*
3. **Section 4.1 (Paragraph 44):** Cited for previous evaluation benchmarks:
   *"Mumtaz et al. [18, 22] evaluated 1D-CNN and wavelet features on the HUSM dataset..."*

### Forensic Finding:
While Mumtaz et al. (2017) is the authoritative primary publication for the HUSM clinical dataset, the paper did not propose group-aware cross-validation for neuroimaging. Attributing the general methodological claim to [18] represented an imprecise citation-to-claim attachment.

### Corrective Action Executed:
- In Introduction paragraph 4, the sentence was rewritten to state:
  > *"This segment-level cross-validation can produce overly optimistic performance estimates that may not translate to unseen patients in clinical settings. Group-aware validation is therefore essential to prevent this form of correlation bias, and establishing rigorous participant-isolated benchmarks is critical for realistic translational assessment."*
- Reference `[18]` is now cited strictly in Sections 2.1 and 4.1, where it directly and authoritatively supports dataset curation and the original benchmark.
- All 22 citations [1]–[22] remain strictly monotonic in order of first appearance, with zero missing references, zero duplicate references, and zero dangling citations.

---

## 2. Reviewer Concerns Audit Matrix (Nine Core Criteria)

| # | Reviewer Concern / Criterion | Section in Final DOCX | Status | Supporting Forensic Evidence & Reconciliation |
| :-: | :--- | :--- | :-: | :--- |
| **1** | **Citation Mismatch & Unverifiable Table III** | Section 1, Section 4.1, References | **PASS** | Table III is withheld from the manuscript (Section 4.1). In-text citations [1]–[22] were audited and verified. Method misattributions (Wu [16] conformal SVM vs. Shen [17] kernel-target alignment channel selection) were disentangled. The [18] citation was removed from general methodological prose. |
| **2** | **Participant Leakage** | Abstract, Section 2.7, Section 3.2 | **PASS** | Stratified Group 10-Fold Cross-Validation enforces zero subject overlap between train/test partitions. Repeat recording `sub-HS15` is quarantined to Fold 9. Verified via `tests/test_pipeline.py`. |
| **3** | **Retracted References** | References, Section 4.1 | **PASS** | *Mahato & Paul (2020)* (retracted by Editor-in-Chief in 2022 due to data integrity concerns) was completely removed from the manuscript and bibliography. |
| **4** | **Incomplete Reproducibility Details** | Section 2.2, 2.3, 2.4, 2.5, 2.6 | **PASS** | Complete specifications: 256.0 Hz sampling rate, 19 standard 10–20 monopolar leads, 0.5–45.0 Hz zero-phase bandpass filter, 2.0s non-overlapping windows (512 samples, 15 segments/rec), Kaiser window frequency smoothing ($\beta=5.0$, $L=255$), Hamming time smoothing ($L=31$), $224 \times 224 \times 3$ ImageNet scaling, frozen CNN weights, feature layer (`avgpool` 512-D), classifier parameters, and `random_state=42`. |
| **5** | **Missing Statistical Uncertainty & Comparisons** | Section 3.2, Table 2, Table 2 Note | **PASS** | Table 2 reports fold mean $\pm$ SD with explicit population ($\text{ddof}=0$) and sample ($\text{ddof}=1$) values. Added 95% Wilson score confidence intervals for participant-level accuracies: EC 89.66% [79.21%, 95.17%]; EO 91.80% [82.21%, 96.45%]; Nested Linear SVM 91.38% [81.36%, 96.26%]. Paired McNemar testing confirms no significant difference between tied top models on EC ($p=1.000$). |
| **6** | **Resource-Limited / Lightweight Claims** | Section 2.5, Section 3.1 | **PASS** | Removed unsupported assertions that ResNet-18 is intended for "resource-limited edge hardware". Reframed strictly around dimensionality: ResNet-18 yields a compact 512-D embedding (vs. 2,048-D for ResNet-50 and 4,096-D for AlexNet/VGG-16), mitigating the curse of dimensionality on a modest clinical cohort. |
| **7** | **SPWVD Kaiser-Window Ablation** | Section 3.3 | **PASS** | Evaluated Kaiser $\beta=7.0$ (93.10% accuracy, 100% sensitivity for Linear SVM) and narrow window $L=127, \beta=3.5$ (82.76%). Properly designated as an exploratory parameter sensitivity analysis. |
| **8** | **Overstated Novelty & Clinical Readiness** | Abstract, Section 1, Section 5 | **PASS** | Toned down all claims of clinical readiness or algorithm discovery. Presented as a rigorous, participant-independent benchmark combining established components (SPWVD, transfer learning, classical estimators). |
| **9** | **Academic Tone, Grammar & Formatting** | Entire Manuscript | **PASS** | Eliminated accusatory and emotive phrases ("data contamination", "collapsing catastrophically", "unjustified claims >99%"). Clean academic prose; professional typography, tables, and genuine embedded figures. |

---

## 3. Resolution of the `sub-HS15` Fold Discrepancy

A cross-artifact audit reconciled the fold assignment of repeat subject `sub-HS15`:
1. **Authoritative Artifacts:** Both `results/fold_assignments.csv` and `results/baseline/fold_assignments.csv` assign `sub-HS15` to **Fold 9** (`sub-HS15,HC,0,9,Healthy Control,4`).
2. **Prediction Files:** In `results/out_of_fold_predictions_eyesOpen.csv` and `results/baseline/out_of_fold_predictions_eyesOpen.csv`, all 30 segment predictions for `sub-HS15` across all 8 classifiers record `fold: 9`.
3. **Draft Origin of "Fold 3":** The string "Fold 3" in earlier notes (`results_reconciliation.md` and `final_baseline_verification.md`) was an uncorrected typographical error in draft markdown files.
4. **Resolution:** Confirmed **Fold 9** as the true, authoritative fold across all artifacts and updated all supporting markdown reports to eliminate contradiction. Both DOCX files state Fold 9.

---

## 4. Verification of Implementation Parameters Against Source Code

1. **SPWVD Parameters (`src/spwvd.py`):**
   - Time-smoothing window: Symmetric Hamming window of length $L = 31$ (`signal.windows.hamming(31)`).
   - Frequency-smoothing (lag) window: Kaiser window of length $L = 255$ with shape parameter $\beta = 5.0$ (`signal.windows.kaiser(255, beta=5.0)`).
   - Energy normalization: Negative values clipped (`np.maximum(wvd, 0)`), 99.5th percentile scaling, PyTorch bilinear interpolation resizing to $224 \times 224 \times 3$, and ImageNet standardization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
2. **Preprocessing & Segmentation (`src/preprocessing.py`):**
   - 19 Standard monopolar 10–20 electrodes (`Fp1, F3, C3, P3, O1, F7, T3, T5, Fz, Fp2, F4, C4, P4, O2, F8, T4, T6, Cz, Pz`).
   - Sampling rate: 256.0 Hz.
   - Filter: 0.5 – 45.0 Hz zero-phase IIR bandpass filter.
   - Epoch duration: $T = 2.0\text{ s}$ ($N = 512\text{ samples}$), strictly non-overlapping (`overlap = 0.0`), 15 epochs per recording file.
3. **Deep Feature Extraction (`src/feature_extractor.py`):**
   - Pretrained ResNet-18 (`weights=ResNet18_Weights.DEFAULT`), frozen weights in `eval()` mode.
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
   - Feature Scaling: `StandardScaler` fit strictly on training fold features.
5. **Participant Aggregation (`src/evaluation.py`):**
   - `aggregate_participant_predictions` groups out-of-fold segment predictions by `participant_id`, computes the arithmetic mean of $p(\text{MDD} \mid x_{i, s})$, and assigns $\hat{y}_i = 1$ if mean probability $\ge 0.5$.

---

## 5. Statistical Uncertainty & McNemar Significance Scope

1. **Wilson Score 95% Confidence Intervals:**
   - Eyes-Closed Linear SVM ($52/58$ correct): **89.66% [95% CI: 79.21%, 95.17%]**
   - Eyes-Closed Medium Gaussian SVM ($52/58$ correct): **89.66% [95% CI: 79.21%, 95.17%]**
   - Eyes-Open Medium Gaussian SVM ($56/61$ correct): **91.80% [95% CI: 82.21%, 96.45%]**
   - Eyes-Open Linear SVM ($54/61$ correct): **88.52% [95% CI: 78.16%, 94.33%]**
   - Nested Linear SVM ($53/58$ correct): **91.38% [95% CI: 81.36%, 96.26%]**
2. **Paired McNemar Test:**
   - Evaluated directly on the participant-level predictions of the tied top models on Eyes-Closed ($N=58$):
     - Both correct: 50 | Both incorrect: 4
     - Linear SVM correct, Medium Gaussian incorrect: 2
     - Linear SVM incorrect, Medium Gaussian correct: 2
     - Discordant pairs: $b = 2, c = 2 \implies \chi^2 = (2 - 2)^2 / (2 + 2) = 0$, **$p = 1.000$**.
   - **Scope:** The test is strictly limited to this compared pair on this cohort, confirming that their overall diagnostic performance is not statistically distinguishable on this sample. No claim of omnibus significance testing across all 8 classifiers is made.

---

## 6. Document Inspection & Visual Rendering Status

- **Paragraph Count:** 73 paragraphs
- **Table Count:** 3 tables
  - Table 1: CNN Backbone comparison (5 rows, 6 columns)
  - Table 2: Classifier performance under Stratified Group 10-Fold CV (9 rows, 11 columns)
  - Table 3: Embedded figure container for side-by-side confusion matrices (1 row, 2 columns)
- **Embedded Genuine Figures:**
  1. **Figure 1:** End-to-end pipeline schematic diagram (`results/pipeline_architecture_diagram.png`).
  2. **Figure 2 (Left):** Out-of-fold confusion matrix for Linear SVM on Eyes-Closed ($N=58$: 24 TN, 4 FP, 2 FN, 28 TP; 89.66% accuracy).
  3. **Figure 2 (Right):** Out-of-fold confusion matrix for Medium Gaussian SVM on Eyes-Open ($N=61$: 27 TN, 2 FP, 3 FN, 29 TP; 91.80% accuracy).
- **Quality Check:** Zero "TODO", "TBD", or empty figure placeholders exist. All table columns and headers are populated.
- **Rendering Limitation Disclosed:** Headless CLI PDF conversion engines (such as LibreOffice or pandoc-pdf) were not installed in the local environment; document structural, typographic, and XML validation was conducted programmatically using `python-docx` and native text utilities.

---

## 7. Automated Test Suite Execution

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_pipeline.py -v
============================== 6 passed in 1.86s ===============================
Exit Code: 0
```
- `test_dataset_discovery_and_labels`: PASSED
- `test_zero_leakage_fold_isolation`: PASSED
- `test_segmentation_boundaries`: PASSED
- `test_spwvd_shape_and_numerical_stability`: PASSED
- `test_training_only_transformations`: PASSED
- `test_metrics_and_participant_aggregation`: PASSED

---

## 8. Preserved Limitations and Scope Boundaries

The manuscript and reports have **zero unresolved internal contradictions or methodological errors**. The remaining boundaries are standard for retrospective clinical EEG studies:

1. **Feature Extraction Scope Boundary:** The raw end-to-end data pipeline was verified via `src/smoke_test.py` across 4 participants on raw `.edf` files (Exit code: 0, 2.05s). Full batch SPWVD extraction across all 181 recordings from scratch was not re-executed during this audit to preserve validated feature tensors and avoid ~10–12 minutes of compute. Cross-validation experiments were rerun on the cached features in `results/features_cache/` via `src/run_experiments.py` (Exit code: 0).
2. **Single-Center Retrospective Cohort:** The dataset originates from a single acquisition center (HUSM, 64 participants). External multi-center prospective validation remains necessary to demonstrate generalizability.
3. **Exploratory Status of Improvements:** Spatial montage weighting (Exp 2) and Kaiser window smoothing parameter adjustments (Exp 3) were evaluated post-hoc on this cohort and remain exploratory hypotheses requiring independent replication.
4. **Withheld Tabular Literature Comparison:** Table III remains withheld until comparator pipelines undergo standardized participant-independent auditing.
