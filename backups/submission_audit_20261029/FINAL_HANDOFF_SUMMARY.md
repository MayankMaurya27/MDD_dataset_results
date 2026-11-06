# Final Project Handoff Summary: Jha et al. MDD EEG Study

**Date:** October 29, 2026  
**Project:** Automated Detection of Major Depressive Disorder Using SPWVD-Based EEG Time-Frequency Images and Deep Learning Features: A Reproducible, Participant-Independent Benchmark  
**Working Root:** `/Users/mayankmaurya/Desktop/experiment`  
**Status:** **Complete, Scientifically Reconciled, and Verified**

---

## 1. Executive Summary

This project has completed a full forensic audit, metric reconciliation, scientific toning, and manuscript revision. All scientific claims, confusion matrix counts, ROC-AUC values, and validation protocols have been validated against local raw and cached artifacts. 

The revised manuscript has been generated as a standard Word document (`manuscript/Jha_MDD_Revised_Final.docx` and `manuscript/Jha_MDD_Revised.docx`) containing verified data tables, neutral scientific phrasing, and transparent methodological qualifications.

---

## 2. Deliverables Inventory

| Deliverable | Path | Description |
| :--- | :--- | :--- |
| **1. Revised Manuscript** | `manuscript/Jha_MDD_Revised_Final.docx`<br>`manuscript/Jha_MDD_Revised.docx` | Full revised research paper formatted in docx with updated Tables 1 & 2, neutralized tone, and corrected citations. |
| **2. Final Results Ledger** | `reports/FINAL_EXPERIMENTAL_RESULTS.csv` | Machine-readable CSV containing 108 verified experimental rows, exact confusion matrix counts, AUCs, fold means, and ddof=0/ddof=1 standard deviations. |
| **3. Reproducibility Report** | `reports/FINAL_REPRODUCIBILITY_REPORT.md` | Verification record detailing environment, test suite commands (6 passed), smoke test execution, dataset inventory, and determinism. |
| **4. Scientific Validation Report** | `reports/FINAL_SCIENTIFIC_VALIDATION.md` | In-depth technical report reconciling all metrics, backbone selection rationale, nested CV qualifications, and citation status. |
| **5. Executive Handoff Summary** | `reports/FINAL_HANDOFF_SUMMARY.md` | This executive summary document. |

---

## 3. Key Scientific and Editorial Corrections Executed

1. **Neutralized Language Regarding Prior Work:**
   - *Removed:* Unsupported assertions that published studies "frequently suffer from inter-subject data leakage", make claims that "collapse catastrophically", or represent "widespread data contamination".
   - *Added:* Objective methodological discussion explaining how segment-level splitting can allow correlated temporal windows from the same subject into training and testing folds, producing optimistic estimates compared to participant-isolated evaluation.
2. **Backbone Architecture Selection Rationale (Table 1):**
   - ResNet-18 was selected based on the primary Eyes-Closed resting benchmark (86.21% vs. 84.48% for ResNet-50) and its 4-fold lower feature dimensionality (512-D vs. 2,048-D), reducing overfitting risks.
   - The manuscript transparently notes that ResNet-50 scored higher on the secondary Eyes-Open condition (90.16% vs. 88.52%).
3. **Reconciled Table 2 with Full ROC-AUC & Standard Deviations:**
   - Table 2 now features 11 side-by-side columns presenting both Eyes-Closed ($N=58$) and Eyes-Open ($N=61$) metrics.
   - Eyes-Open ROC-AUC values have been reconciled from raw out-of-fold probability vectors: **93.32%** (Medium Gaussian SVM), **96.01%** (Neural Network), **95.69%** (Linear SVM).
   - Table notes explicitly document both population standard deviation ($\text{ddof}=0$) and sample standard deviation ($\text{ddof}=1$).
4. **Separate Reporting of Evaluation Levels (Segment vs. Participant):**
   - Removed unqualified claims asserting a numerical percentage point "improvement" across differing denominators.
   - Segment-level accuracies (77.70%–81.84% on EC; 80.65%–85.27% on EO) and participant-level accuracies (89.66% on EC; 91.80% on EO) are reported separately with explicit denominators ($N=870/930$ segments vs. $N=58/61$ participants).
   - Clarified that averaging segment probabilities dampens isolated segment-level classification errors without claiming physiological artifact cancellation.
5. **Qualified Nested Cross-Validation (Section 3.3):**
   - Titled: *"Nested Group-Aware Hyperparameter Tuning on a Pre-selected Feature Backbone"*.
   - Explains that while $C \in [0.01, 0.1]$ was tuned strictly within inner folds without test-fold contamination, the preceding selection of ResNet-18 features and Linear SVM was informed by prior cohort evaluation.
   - Clarifies that 91.38% demonstrates the benefit of within-model regularization tuning rather than an end-to-end unbiased selection across all pipeline dimensions.
6. **Exploratory Status of Improvements:**
   - Experiments 1 (Aggregation rules), 2 (Frontal montage weighting), and 3 (Kaiser smoothing variations) are explicitly designated as exploratory hypotheses.
7. **Withholding of Literature Comparison Table (Table III):**
   - Withheld due to unverified DOIs, unstandardized cross-validation protocols, and incompatible metrics across published comparator papers. The rationale is stated transparently in Section 4.1.

---

## 4. Key Verified Results At-a-Glance

### Primary Baseline Classifiers (ResNet-18 Features, Stratified Group 10-Fold CV)

| Metric | Primary Eyes-Closed ($N=58$) | Secondary Eyes-Open ($N=61$) |
| :--- | :--- | :--- |
| **Top Performing Models** | Linear SVM, Med Gaussian SVM, MLP | Med Gaussian SVM, MLP |
| **Participant Accuracy** | **89.66%** (52/58 correct) | **91.80%** (56/61 correct) |
| **Sensitivity (MDD Detection)** | **93.33%** (28/30 detected) [Linear SVM]<br>90.00% (27/30) [Med Gaussian SVM, MLP] | **90.62%** (29/32 detected) |
| **Specificity (HC Identification)** | **85.71%** (24/28) [Linear SVM]<br>**89.29%** (25/28) [Med Gaussian SVM, MLP] | **93.10%** (27/29 controls) |
| **F1-Score** | **90.32%** [Linear SVM], 90.00% [Med Gaussian] | **92.06%** [Med Gaussian SVM, MLP] |
| **ROC-AUC** | **95.36%** [Linear SVM], 94.40% [MLP] | **96.01%** [MLP], 95.69% [Linear SVM], 93.32% [Med Gaussian] |
| **Fold Mean ± SD (ddof=0 / ddof=1)** | 89.90 ± 8.34% / ±8.79% [Linear SVM]<br>89.90 ± 11.19% / ±11.79% [Med Gaussian] | 91.48 ± 12.49% / ±13.16% [Med Gaussian]<br>88.14 ± 11.98% / ±12.63% [Linear SVM] |
| **Segment-Level Accuracy** | **80.34%** (Linear SVM), **81.72%** (Med Gaussian) | **80.65%** (Linear SVM), **84.19%** (Med Gaussian) |
| **Evaluation Units** | Segment: $N=870$ windows; Participant: $N=58$ subjects | Segment: $N=930$ windows; Participant: $N=61$ subjects |

---

## 5. Verification Status Summary

| Pipeline Component | Status | Verification Detail |
| :--- | :---: | :--- |
| Raw EDF Ingestion & SPWVD | **Verified** | `src/smoke_test.py` completed on raw EDFs in 2.05s (4 subjects) |
| Regression Test Suite | **Verified** | 6 of 6 unit tests passing in 1.70s (`tests/test_pipeline.py`) |
| Cached Feature Cross-Validation | **Verified** | `src/run_experiments.py` executed across 10 folds for EC & EO |
| Participant Isolation | **Verified** | Zero leakage confirmed across all 10 folds |
| Dataset Counts | **Verified** | Authoritative 64 participants (30 HC, 34 MDD), 181 recordings |
| Reconciled Prediction Metrics | **Verified** | 100% agreement with `results/baseline/` artifacts |
| Word Manuscript Generation | **Verified** | Cleanly rendered via `src/generate_revised_manuscript.py` |

---

## 6. Documented Scope Boundaries & Limitations

1. **Raw Feature Extraction Scope:**
   The full batch SPWVD extraction from scratch across all 181 `.edf` files was not rerun, as existing tensors in `results/features_cache/` are verified and intact. The full data-ingestion code was verified via `src/smoke_test.py`.
2. **Single-Center Cohort:**
   Data originates from HUSM (64 participants). Independent multi-center validation remains necessary to evaluate external clinical generalizability.
3. **Exploratory Enhancements:**
   Montage channel weighting and Kaiser smoothing parameter variations remain exploratory findings evaluated post-hoc on this cohort.
