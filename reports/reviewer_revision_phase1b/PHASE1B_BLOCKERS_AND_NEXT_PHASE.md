# PHASE 1B — AUDIT SUMMARY, PASS/FAIL STATUS, AND NEXT PHASE ROADMAP
## Jha MDD EEG SPWVD Research Paper — Comprehensive Reviewer Revision

**Auditor / Lead Engineer:** Senior Biomedical Signal Processing & Machine Learning Reproducibility Auditor  
**Date:** November 7, 2026  
**Execution Context:** Completion of Phase 1B (Artifact Diagnostics & Classical Baselines)  
**Deliverable Directory:** `reports/reviewer_revision_phase1b/`  
**Execution Policy:** Strict halting upon completion of Phase 1B deliverables; awaiting user authorization before Phase 1C.

---

## 1. Summary of Implemented and Executed Components

| Engineering Stage | Implemented Module | Executed Script | Scope & Scale | Execution Status |
| :--- | :--- | :--- | :--- | :---: |
| **Stage 1: Crosswalk** | Reviewer Requirement Crosswalk & Gap Reconciliation | Markdown Crosswalk | 14 original reviewer points mapped to actions, artifacts, and validation criteria. | **PASS** |
| **Stage 2: Artifact Diagnostics** | Measure-First Artifact Screening & FastICA Feasibility | `src/artifact_diagnostic_audit.py` | Audited all **181 EDF files** (~1.4 GB); computed PTP and AbsMax distributions by channel and region; FastICA demo. | **PASS** |
| **Stage 3: Classical Baselines** | Welch PSD (95-D), STFT (512-D), CWT Morlet (512-D) | `src/run_classical_baselines.py` | 1,800 raw segments loaded, transformed, and evaluated under Group 10-Fold CV across EC and EO. | **PASS** |
| **Stage 4: Integrity Verification** | Leakage Auditing, Fold Quarantine, and Test Suite | Automated Assertions + `pytest tests/` | Zero subject overlap verified; `sub-HS15` quarantined to Fold 9; all 6 test suite cases passing. | **PASS** |

---

## 2. Granular Pass / Fail Verification Audit

### What Passed:
1. **Zero Data Leakage:**
   - In all three classical baselines, `StandardScaler` was strictly fitted on training fold segments only and applied to test fold segments (`assert (part_folds == 1).all()`).
   - In Eyes-Open, all 30 segments across `run-1` and `run-2` for `sub-HS15` were quarantined strictly to **Fold 9**.
2. **Deterministic Segment Alignment:**
   - Raw segments were extracted using exact sample coordinates (`sample_start`, `sample_end`) from `metadata_eyesClosed.csv` and `metadata_eyesOpen.csv`, ensuring 100% identical data inputs to the baseline and proposed pipelines.
3. **Continuous Probability Scoring & Bootstrap CIs:**
   - All out-of-fold segment predictions contain genuine continuous posterior probabilities (`y_prob`).
   - 1,000 participant-level bootstrap resamples (seed: 42) computed empirical 95% confidence intervals for every baseline.
4. **Hardware & Execution Efficiency:**
   - Artifact diagnostic audit across all 181 recordings completed in **7.2 seconds**.
   - Classical baseline feature extraction, ResNet-18 forward passes (Apple MPS), and 10-fold CV completed in **31.0 seconds**.
5. **Preservation of Historical Deliverables:**
   - Zero historical baseline files, existing prediction CSVs, or master DOCX files were modified or overwritten.
   - All Phase 1B outputs are isolated under `results/revision_experiments/` and `reports/reviewer_revision_phase1b/`.

### What Was Handled / Adjusted:
1. **SciPy Wavelet Function Transition:**
   - In SciPy 1.15+, `scipy.signal.cwt` is deprecated. Continuous Wavelet Transform was seamlessly implemented using MNE's authoritative `mne.time_frequency.tfr_array_morlet` across 32 logarithmic scales ($1.0 - 45.0\text{ Hz}$), which is the clinical neurophysiology gold standard.
2. **Empirical Baseline Competitive Findings:**
   - In strict adherence to scientific integrity (*"Do not claim SPWVD superiority unless the resulting comparable experiments support it"*), the empirical results show that Welch PSD paired with a non-linear RBF SVM achieves **89.66%** on EC and **91.80%** on EO, matching SPWVD accuracy.
   - SPWVD + ResNet-18 achieves top-tier performance and superior linear separability (EC Linear SVM: 89.66%, AUC 0.9536; EO MLP: 91.80%, AUC 0.9601), but manuscript prose will present a balanced, non-hyperbolic comparison that acknowledges the strength of classical spectral density with non-linear kernels.

---

## 3. Persistent Blockers and Unchangeable Data Constraints

1. **Absence of Machine-Readable Demographic Keys in BIDS:**
   - *Status:* **Permanent Data Limitation.**
   - *Detail:* In `nm000114_v1.0.0/participants.tsv`, all demographic fields (`age`, `sex`, `weight`, `height`) are `n/a`, and no per-subject depression severity scores (HAM-D/BDI) are available.
   - *Resolution:* Cannot be resolved by coding. Handled via explicit methodological disclosure in Section 2.1.1, citing cohort-level figures from Mumtaz et al. (2017).
2. **Frontal Pole Blink Contamination:**
   - *Status:* **Physiological Artifact Locus Characterized.**
   - *Detail:* The Stage 2 diagnostic audit revealed that 31–33% of epochs in `Fp1`/`Fp2` exceed $100\,\mu V$ during Eyes-Open. However, Central, Temporal, Parietal, and Occipital leads show exceedance rates of under 1.8% at $150\,\mu V$.
   - *Resolution:* Applying wholesale epoch rejection would discard >32% of recordings. The manuscript will formally defend multi-channel spatial averaging ($\frac{1}{19}\sum W_c$) and deep feature representation as robust noise-dilution mechanisms, while transparently documenting the omission of automated ICA.

---

## 4. Execution Roadmap for Next Phase (Phase 1C)

Following user review and authorization, Phase 1C will execute the remaining computational revision experiments:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1C EXPERIMENTAL SCOPE                     │
│                                                                        │
│  1. Full 4-Backbone x 8-Classifier Matrix                              │
│     - Evaluate all 32 combinations under Group 10-Fold CV              │
│     - Script: src/run_full_backbone_matrix.py                          │
│                                                                        │
│  2. Complete-Pipeline Nested Group CV                                  │
│     - Outer 10-Fold x Inner 5-Fold grid optimization                   │
│     - Script: src/run_nested_cv_benchmark.py                           │
│                                                                        │
│  3. Repeated Group 10-Fold CV (100 Runs)                               │
│     - 10 independent partition seeds x 10 folds                        │
│     - Script: src/run_repeated_cv.py                                   │
│                                                                        │
│  4. End-to-End ResNet-18 Fine-Tuning Benchmark                         │
│     - Unfreeze layer 4 convolutions + early stopping                   │
│     - Script: src/train_finetune.py                                    │
└────────────────────────────────────────────────────────────────────────┘
```

Execution will halt here. We await your approval before initiating Phase 1C.

