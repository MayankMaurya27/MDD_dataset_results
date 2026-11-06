# PHASE 0 — CRITICAL BLOCKERS, METHODOLOGICAL RISKS, AND EXECUTION PLAN
## Jha MDD EEG SPWVD Research Paper — Forensic Reviewer Revision

**Auditor:** Senior Biomedical Signal Processing & Machine Learning Reproducibility Auditor  
**Date:** November 7, 2026  
**Deliverable Directory:** `reports/reviewer_revision_phase0/`  
**Execution Policy:** Strict Phase 0 completion. Wait for explicit user authorization before launching Phase 1.

---

## 1. Identification of Critical Blockers and Methodological Risks

### Blocker 1: Complete Absence of Demographic and Clinical Severity Covariates in BIDS Metadata
- **Severity**: Critical / Data Limitation.
- **Artifact Evidence**: In `nm000114_v1.0.0/participants.tsv`, all columns for `age`, `sex`, `hand`, `weight`, and `height` contain the string `n/a`. Furthermore, no individual HAM-D, BDI, or medication regimen records are provided in any BIDS file.
- **Impact on Revision**: Reviewers requesting sub-group analyses (e.g., classification accuracy stratified by age bracket, sex, or depression severity) cannot be satisfied by querying local files. Any attempt to synthesize or estimate these values would violate scientific integrity.
- **Resolution Strategy**: 
  1. Transparently report this data limitation in the manuscript Methods and Limitations sections.
  2. Ground overall cohort demographics in the peer-reviewed foundational literature (Mumtaz et al., 2017: 34 MDD patients, mean age $40.3 \pm 12.9$; 30 HC, mean age $38.3 \pm 15.6$).
  3. Explain that the OpenNeuro/NEMAR BIDS derivative release (`nm000114_v1.0.0`) de-identified and stripped per-subject demographic keys to protect participant anonymity.

### Blocker 2: Absence of Conventional Baseline Implementations (Welch PSD, STFT, CWT)
- **Severity**: High / Reviewer Expectation.
- **Artifact Evidence**: The workspace currently contains four CNN backbones and eight downstream classifiers on SPWVD representations, but lacks comparative code or results for:
  - Welch Power Spectral Density (PSD) band-power + SVM.
  - Short-Time Fourier Transform (STFT) spectrograms + ResNet-18.
  - Continuous Wavelet Transform (CWT) scalograms + ResNet-18.
- **Impact on Revision**: Without these baselines, reviewers can question whether the complex SPWVD + ResNet-18 pipeline provides genuine value over standard clinical EEG features.
- **Resolution Strategy**: In Phase 1, implement modular baseline scripts (`src/baselines_psd.py`, `src/baselines_tfr.py`) and evaluate them under identical Stratified Group 10-Fold CV partitions.

### Blocker 3: Lack of Automated Artifact Rejection / ICA in Preprocessing Pipeline
- **Severity**: Medium / Methodological Defense.
- **Artifact Evidence**: `src/preprocessing.py` performs only zero-phase 0.5–45.0 Hz bandpass filtering. No peak-to-peak amplitude thresholding or ICA ocular component rejection is executed.
- **Impact on Revision**: Reviewers in clinical neurophysiology expect discussion of blink and muscle artifact handling.
- **Resolution Strategy**: 
  1. Build a diagnostic script to measure the percentage of epochs exceeding standard amplitude thresholds ($\pm 100\,\mu V$ and $\pm 150\,\mu V$).
  2. In the manuscript, defend the robustness of multi-channel spatial averaging and deep feature representation against localized transients, while transparently noting the omission of ICA as a design choice.

### Blocker 4: Discrepancy in the "Exactly Three Controls per Fold" Claim
- **Severity**: Low / Text Precision.
- **Artifact Evidence**: The master 64 partition allocates exactly 3 HC to each fold. However, in the Eyes-Closed cohort ($N=58$), Folds 1 and 6 contain only 2 controls (due to missing recordings for `sub-HS12` and `sub-HS18`). In the Eyes-Open cohort ($N=61$), Fold 1 contains only 2 controls (missing `sub-HS25`).
- **Resolution Strategy**: Update Section 2.7 of the manuscript to state clearly that the 3-control balance applies strictly to the master 64 partition, and report the exact condition-specific fold counts.

---

## 2. Chronological Order of Execution for Phase 1 Onward

Following user approval of this Phase 0 audit, work should proceed in the following strict chronological sequence:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   PHASE 1: BASELINES & ARTIFACTS                       │
│  - Implement Welch PSD band-power baseline (delta/theta/alpha/beta)    │
│  - Implement STFT spectrogram baseline + ResNet-18                     │
│  - Implement CWT Morlet scalogram baseline + ResNet-18                 │
│  - Run epoch-level peak-to-peak amplitude artifact diagnostic          │
│  - Output: results/baseline_comparisons/                               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              PHASE 2: STATISTICAL CALIBRATION & ROC CURVES             │
│  - Generate high-resolution publication ROC curves from continuous     │
│    out-of-fold posterior probabilities (EC and EO)                     │
│  - Compute 1,000-sample bootstrap 95% AUC confidence intervals         │
│  - Generate McNemar contingency tables & p-values for tied classifiers │
│  - Output: results/roc_curves/, results/statistical_tests/             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              PHASE 3: P300 ODDBALL ERP BENCHMARK (OPTIONAL)            │
│  - Extract 15 segments across 61 available P300 ERP recordings         │
│  - Evaluate under identical Group 10-Fold CV                           │
│  - Provide tri-condition comparison (Eyes-Closed vs Eyes-Open vs P300) │
│  - Output: results/features_cache/metadata_P300.csv                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              PHASE 4: MANUSCRIPT INTEGRATION & DELIVERABLE LOCK        │
│  - Update manuscript text to incorporate verified baseline results     │
│  - Clarify fold composition and Pierre Guetschel docstring origin     │
│  - Embed new continuous ROC curve figures                              │
│  - Regenerate primary DOCX and mirror DOCX                             │
│  - Perform visual layout audit (inspect rendered pages)                │
│  - Produce final submission audit report                               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Decision Gates Requiring User Approval

Before initiating Phase 1, the user must approve:
1. **Approval of Phase 0 Findings**: Concurrence with the 64-subject cohort resolution, the explanation of tied classifier metrics, and the 14-point revision matrix.
2. **Execution of Conventional Baselines**: Authorization to execute the three missing comparative baselines (Welch PSD, STFT, CWT) on the local machine.
3. **P300 ERP Condition Inclusion**: Decision on whether to extract and evaluate the 61 P300 ERP recordings as an additional experimental dimension, or retain the manuscript's focus strictly on resting-state EEG (EC and EO).

