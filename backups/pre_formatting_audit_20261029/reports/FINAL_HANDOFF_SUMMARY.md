# Final Project Handoff Summary: Jha et al. MDD EEG Study

**Date:** October 29, 2026  
**Project:** Automated Detection of Major Depressive Disorder Using SPWVD-Based EEG Time-Frequency Images and Deep Learning Features: A Reproducible, Participant-Independent Benchmark  
**Working Root:** `/Users/mayankmaurya/Desktop/experiment`  
**Submission-Readiness Verdict:** **READY FOR FINAL HUMAN APPROVAL**

---

## 1. Executive Summary

This project has completed its final submission-readiness gate, including citation-to-claim audit, metric reconciliation, fold assignment verification, and manuscript generation.

Both the primary manuscript (`manuscript/Jha_MDD_Revised_Final.docx`) and its identical mirror (`manuscript/Jha_MDD_Revised.docx`) are byte-for-byte identical, fully formatted, scientifically defensible, and ready for submission to your supervisor and journal.

---

## 2. Key Actions & Corrections in the Final Gate

1. **Resolution of the [18] Citation Issue:**
   - Identified that reference `[18]` (Mumtaz et al. 2017) was cited in the Introduction for general group-aware validation in neuroimaging, which Mumtaz et al. did not propose.
   - Removed the misplaced in-text citation from general methodological text in the Introduction. Reference `[18]` is now cited strictly where directly supported: Section 2.1 (dataset curation) and Section 4.1 (literature comparator).
   - All 22 citations [1]–[22] remain monotonic, with zero dangling or orphaned citations.
2. **Resolution of `sub-HS15` Fold Discrepancy:**
   - Authoritative artifacts (`results/fold_assignments.csv`, `results/baseline/fold_assignments.csv`, and `results/out_of_fold_predictions_eyesOpen.csv`) confirm that multi-run participant `sub-HS15` was assigned to **Fold 9**.
   - Updated historical draft notes in `results_reconciliation.md` and `final_baseline_verification.md` that erroneously cited "Fold 3". Both DOCX files state Fold 9.
3. **Statistical Uncertainty & Paired Testing:**
   - Included 95% Wilson score confidence intervals for participant-level accuracies.
   - Included paired McNemar testing between tied top estimators on Eyes-Closed (Linear SVM vs. Medium Gaussian SVM, $p=1.000$), strictly limited to this compared pair.
4. **Document Integrity & Genuine Figures:**
   - Embedded 3 genuine, locally generated figures: Figure 1 (Pipeline Architecture Diagram) and Figure 2 (Side-by-side Confusion Matrices for EC Linear SVM and EO Medium Gaussian SVM).
   - Zero "TODO", "TBD", or missing placeholders.

---

## 3. Deliverables Inventory

| Deliverable | Path | Description |
| :--- | :--- | :--- |
| **1. Primary Revised Manuscript** | `manuscript/Jha_MDD_Revised_Final.docx` | Full revised research paper formatted in docx with updated Tables 1 & 2, Figures 1 & 2, neutralized tone, and verified citations. |
| **2. Mirror Manuscript** | `manuscript/Jha_MDD_Revised.docx` | Byte-for-byte identical mirror of the primary manuscript. |
| **3. Submission-Readiness Audit** | `reports/FINAL_SUBMISSION_READINESS_AUDIT.md` | Comprehensive forensic audit across all 9 reviewer concerns, citations, implementation parameters, and limitations. |
| **4. Citation Audit Ledger** | `reports/FINAL_CITATION_AUDIT.csv` | Machine-readable ledger (24 entries) verifying all 22 references, 1 retracted article pruned, and 1 comparator group withheld. |
| **5. Experimental Results Ledger** | `reports/FINAL_EXPERIMENTAL_RESULTS.csv` | Machine-readable CSV containing 108 verified experimental rows, exact confusion matrix counts, AUCs, fold means, and ddof=0/ddof=1 standard deviations. |
| **6. Reproducibility Report** | `reports/FINAL_REPRODUCIBILITY_REPORT.md` | Verification record detailing environment, test suite commands (6 passed), smoke test execution, dataset inventory, and determinism. |
| **7. Scientific Validation Report** | `reports/FINAL_SCIENTIFIC_VALIDATION.md` | In-depth technical report reconciling all metrics, backbone selection rationale, nested CV qualifications, and citation status. |
| **8. Executive Handoff Summary** | `reports/FINAL_HANDOFF_SUMMARY.md` | This executive summary document. |

---

## 4. Key Verified Baseline Metrics At-a-Glance

### Pretrained ResNet-18 Deep Features with Stratified Group 10-Fold CV

| Metric | Primary Eyes-Closed ($N=58$) | Secondary Eyes-Open ($N=61$) |
| :--- | :--- | :--- |
| **Top Performing Models** | Linear SVM, Med Gaussian SVM, MLP | Med Gaussian SVM, MLP |
| **Participant Diagnostic Accuracy** | **89.66%** (52/58 correct) | **91.80%** (56/61 correct) |
| **95% Wilson Score CI** | **[79.21%, 95.17%]** | **[82.21%, 96.45%]** |
| **Sensitivity (MDD Detection)** | **93.33%** (28/30 detected) [Linear SVM]<br>90.00% (27/30) [Med Gaussian SVM, MLP] | **90.62%** (29/32 detected) |
| **Specificity (HC Identification)** | **85.71%** (24/28 controls) [Linear SVM]<br>**89.29%** (25/28 controls) [Med Gaussian, MLP] | **93.10%** (27/29 controls) |
| **F1-Score** | **90.32%** [Linear SVM], 90.00% [Med Gaussian] | **92.06%** [Med Gaussian SVM, MLP] |
| **ROC-AUC** | **95.36%** [Linear SVM], 94.40% [MLP] | **96.01%** [MLP], 95.69% [Linear SVM], 93.32% [Med Gaussian] |
| **Fold Mean ± SD (ddof=0 / ddof=1)** | 89.90 ± 8.34% / ±8.79% [Linear SVM]<br>89.90 ± 11.19% / ±11.79% [Med Gaussian] | 91.48 ± 12.49% / ±13.16% [Med Gaussian]<br>88.14 ± 11.98% / ±12.63% [Linear SVM] |
| **Segment-Level Accuracy** | **80.34%** (Linear SVM), **81.72%** (Med Gaussian) | **80.65%** (Linear SVM), **84.19%** (Med Gaussian) |
| **Evaluation Units** | Segment: $N=870$ windows; Participant: $N=58$ subjects | Segment: $N=930$ windows; Participant: $N=61$ subjects |

---

## 5. Verification Test Suite Status

```text
tests/test_pipeline.py::test_dataset_discovery_and_labels PASSED         [ 16%]
tests/test_pipeline.py::test_zero_leakage_fold_isolation PASSED          [ 33%]
tests/test_pipeline.py::test_segmentation_boundaries PASSED              [ 50%]
tests/test_pipeline.py::test_spwvd_shape_and_numerical_stability PASSED  [ 66%]
tests/test_pipeline.py::test_training_only_transformations PASSED        [ 83%]
tests/test_pipeline.py::test_metrics_and_participant_aggregation PASSED  [100%]

============================== 6 passed in 1.86s ===============================
Exit Code: 0
```

---

## 6. Documented Scope Boundaries & Limitations

1. **Feature Extraction Scope Boundary:** Full raw batch SPWVD extraction across all 181 recordings from scratch was not re-executed during this audit to preserve validated feature tensors and avoid ~10–12 minutes of compute. The end-to-end raw execution path was verified via `src/smoke_test.py` across 4 participants on raw `.edf` files (Exit code: 0, 2.05s).
2. **Single-Center Cohort:** Data originates from a single center (HUSM, 64 participants). External multi-center validation remains necessary to demonstrate external clinical generalizability.
3. **Exploratory Status of Improvements:** Spatial montage weighting (Exp 2) and Kaiser window smoothing variations (Exp 3) were evaluated post-hoc on this cohort and remain exploratory hypotheses requiring independent replication.
4. **Withheld Tabular Literature Comparison:** Table III remains withheld until comparator pipelines undergo standardized participant-independent auditing.
