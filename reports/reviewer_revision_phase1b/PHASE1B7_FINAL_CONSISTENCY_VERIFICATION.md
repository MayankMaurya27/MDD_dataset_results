# Phase 1B.7 — Final Consistency Verification and Execution Authorization Gate
## Jha MDD EEG SPWVD Research Paper — Pre-Execution Audit Report

**Workspace:** `/Users/mayankmaurya/Desktop/experiment`  
**Date:** November 2026  
**Audited Commit:** `c831f80` (with Phase 1B.7 documentation updates)  
**Verification Harness:** `src/verify_phase1b7_consistency.py`  
**Status:** All 7 Blockers Resolved — Pre-Flight Automated Verification Passed  

---

## 1. Executive Summary

This audit report documents the resolution of the seven blockers identified during the review of the Phase 1B.6 documentation against the physical codebase, BIDS dataset metadata, and historical experiment outputs.

All seven issues have been resolved without altering historical baseline data or modifying the master manuscript. The automated verification suite (`src/verify_phase1b7_consistency.py`) was executed and passed all test suites with zero errors.

---

## 2. Resolution of the Seven Blockers

### Blocker 1: Epoch Protocol Verification
- **Historical Ground Truth:** The historical baseline pipeline (`src/preprocessing.py`, `src/extract_features.py`, `src/run_experiments.py`, `src/generate_revised_manuscript.py`) uses **2.0-second epochs** ($N = 512$ samples at $fs = 256\text{ Hz}$), NOT 5.0-second epochs.
- **Selection Formula:** Exactly **15 segments per recording** are selected via deterministic uniform subsampling:
  $$\text{step} = \lfloor N_{\text{total\_epochs}} / 15 \rfloor, \quad \text{selected\_indices} = [i \times \text{step} \text{ for } i \in \{0, \dots, 14\}]$$
- **Exact Sample Coordinates:** For each segment $k \in \{0, \dots, 14\}$, `sample_start = k * step * 512`, `sample_end = sample_start + 512`. Preserved in `results/features_cache/metadata_eyesClosed.csv` and `metadata_eyesOpen.csv`.
- **Resolution:** All erroneous references to "5.0-second epochs" in Phase 1B.6 documents have been corrected to 2.0-second epochs. The historical baseline is preserved 100% intact.

### Blocker 2: Cohort Counts and Duplicate Recording Quarantining
- **Verified Cohort Counts:**
  - **Eyes Closed (EC):** 58 unique participants (30 MDD, 28 HC), 58 recordings, $58 \times 15 = \mathbf{870}$ segments (450 MDD, 420 HC).
  - **Eyes Open (EO):** 61 unique participants (**32 MDD, 29 HC**), 62 recordings, $62 \times 15 = \mathbf{930}$ segments (480 MDD, 450 HC).
- **Duplicate Recording Verification:** Participant `sub-HS15` (Healthy Control) has two recording files in EO (`run-01` and `run-02`), contributing $15 \times 2 = 30$ segments. Both runs and all 30 segments are strictly quarantined in **Fold 9** (`fold = 9`), ensuring zero participant leakage across folds.
- **Resolution:** Prior documentation mentioning "31 MDD, 30 HC" for EO has been updated to the verified ground truth of **32 MDD, 29 HC**.

### Blocker 3: Reviewer Crosswalk (14 Original Points vs. 15 Internal Tasks)
- **Original Reviewer Document:** Exactly **14 review points** exist in the authoritative reviewer matrix (`reports/reviewer_revision_phase0/PHASE0_REVIEWER_14_POINT_MATRIX.md`).
- **Internal Tracking Expansion:** During Phase 1B.5, internal tracking expanded Reviewer Point 9 into two engineering sub-tasks:
  - Task 9A: Mathematical resolution of Fine Gaussian SVM degeneracy ($K \approx I$ due to fixed $\gamma=10.0$ in 512-D space).
  - Task 9B: Analysis of segment probability divergence converging under posterior mean aggregation ($p=1.000$ McNemar test).
- **Resolution:** The documentation now explicitly identifies the 15 items as an **internal engineering tracking breakdown** and maps them directly to the 14 original reviewer comments:

| Original Reviewer Point | Description | Internal Tracking Item | Phase 1C Module |
| :-: | :--- | :-: | :-: |
| **Point 1** | Dataset Sample Size Discrepancy (34 vs 64) | Task 1 | Handled in Phase 0 / Manuscript |
| **Point 2** | Demographic & Clinical Covariate Reporting | Task 2 | Handled in Phase 0 / Manuscript |
| **Point 3** | Artifact Rejection & ICA Omission | Task 3 | Module M1 |
| **Point 4** | 15-Epoch Selection Determinism & Coordinates | Task 4 | Handled in Phase 0 / M1 |
| **Point 5** | SPWVD Formulation & Cross-Term Mitigation | Task 5 | Module M2 |
| **Point 6** | CNN Backbone Selection (ResNet-18 vs ResNet-50) | Task 6 | Module M2, M5 |
| **Point 7** | Subject Leakage & Repeat Run Quarantining (`sub-HS15`) | Task 7 | Module M1–M6 |
| **Point 8** | Control Counts per Fold Clarification | Task 8 | Handled in Phase 0 / Manuscript |
| **Point 9** | Tied Classifier Accuracy & Continuous Scores | **Task 9A & Task 9B** | Module M6 |
| **Point 10** | Continuous ROC Curves & Bootstrap AUC 95% CIs | Task 10 | Module M6 |
| **Point 11** | Classical Baselines (Welch PSD, STFT, CWT) | Task 11 | Module M2 |
| **Point 12** | Hyperparameter Optimization & Nested CV | Task 12 | Module M3 |
| **Point 13** | Literature Comparison Harmonization (Table III) | Task 13 | Handled in Manuscript |
| **Point 14** | Citation Integrity & Retracted Paper Pruning | Task 14 | Handled in Phase 0 / Manuscript |

### Blocker 4: ICA Consistency and Biophysical Clarification
- **Unified Algorithmic Parameters:**
  - Frontolar correlation: $|\rho(\mathbf{s}_k, \mathbf{y}_{\text{ocular}})| \ge 0.40$, where $\mathbf{y}_{\text{ocular}} = \frac{1}{2}(x_{F_{p1}} + x_{F_{p2}})$.
  - Frontal spatial energy ratio: $\mathcal{R}_{\text{frontal}}(k) \ge 0.50$.
  - Temporal kurtosis: $\kappa(\mathbf{s}_k) \ge 4.5$.
  - Rejection cap: Maximum 2 components out of 19 zeroed per recording.
  - Ranking hierarchy: Descending order by $|\rho(\mathbf{s}_k, \mathbf{y}_{\text{ocular}})|$.
  - Solver fallback: FastICA ($\tanh$) $\to$ Picard / Extended Infomax $\to$ uncorrected fallback (Condition A).
  - Validation metrics: Frontal PTP reduction $\ge 35\%$ ($p < 10^{-10}$); posterior alpha power retention $\ge 85\%$ across $O_1, O_2, P_z$.
- **Biophysical Reality Clarification:** Retaining 17 independent components preserves 17 linear degrees of freedom in the sensor space, but does **not** guarantee 17 pure, uncorrupted neural sources. Due to scalp volume conduction, hundreds of cortical dipoles project onto the 19 scalp sensors. The cap is a safeguard against over-cleaning, not a claim of complete source isolation.

### Blocker 5: Model Search Space Correction (2D TFR vs. 1D Welch PSD)
- **Error in Prior Formulation:** Previous drafts described passing Welch PSD through ResNet image backbones. Welch PSD produces 1D spectral density vectors, which cannot be processed by 2D image CNNs.
- **Corrected Search Space:**
  - **2D Time-Frequency Representations (SPWVD, STFT, CWT):** 3 representations $\times$ 2 image backbones (ResNet-18 [$D=512$], ResNet-50 [$D=2048$]) $\times$ 5 downstream classifiers (Linear SVM, RBF SVM, MLP, Random Forest, Logistic Regression) = **30 valid pipelines**.
  - **1D Classical Spectral Representation (Welch PSD):** 1 representation $\times$ Direct Spectral Features ($D=95$ or $D=855$, NO CNN) $\times$ 5 downstream classifiers = **5 valid pipelines**.
  - **Total Valid Pipelines:** Exactly **35 strictly valid pipelines**. Invalid combinations (e.g. Welch PSD $\to$ ResNet) are explicitly prohibited.

### Blocker 6: Leakage-Free Probability Calibration for SVMs
- **Problem:** Support Vector Machines output uncalibrated signed distances to the hyperplane $f(\mathbf{x}) \in (-\infty, +\infty)$. Participant posterior pooling requires calibrated probabilities $p \in [0, 1]$.
- **Calibration Protocol:** Continuous probabilities are obtained via Platt scaling:
  $$p(y = 1 \mid \mathbf{x}) = \frac{1}{1 + \exp(A \cdot f(\mathbf{x}) + B)}$$
- **Anti-Leakage Safeguard:** Sigmoid parameters $(A, B)$ must be fitted on an internal **participant-grouped calibration split** within the training set. Outer validation and test participants are strictly excluded from calibration fitting.

### Blocker 7: Git Provenance Reconciliation
- **Lineage:**
  - `00a08df`: Phase 0 audit and Phase 1A OOF verification.
  - `d5dac42`: Phase 1B implementation and baseline results.
  - `baeb462`: Phase 1B.5 scientific audit reports.
  - `c831f80`: Initial Phase 1B.6 protocol freeze.
- **Active Working Baseline:** All Phase 1B.6 documents have been updated, synchronized, and verified against `c831f80`.

---

## 3. Automated Pre-Flight Verification Results

The automated script [`src/verify_phase1b7_consistency.py`](file:///Users/mayankmaurya/Desktop/experiment/src/verify_phase1b7_consistency.py) was executed to verify all parameters against disk artifacts:

```text
======================================================================
         PHASE 1B.7 AUTOMATED CONSISTENCY VERIFICATION SUITE         
======================================================================
--- [Check 1/5] Epoch Protocol & Sample Coordinates ---
  ✓ Epoch duration is exactly 2.0s (512 samples at 256 Hz).
  ✓ Exactly 15 segments selected per recording file.
  ✓ Total segments: EC = 870, EO = 930.
--- [Check 2/5] Cohort Counts & Duplicate Recording ---
  ✓ EC: 58 unique participants (30 MDD, 28 HC), 870 segments.
  ✓ EO: 61 unique participants (32 MDD, 29 HC), 930 segments.
  ✓ sub-HS15 has 2 recordings (30 segments) strictly quarantined in Fold 9.
--- [Check 3/5] Participant Isolation & Zero Leakage Across Folds ---
  ✓ Zero participant overlap across folds (100% strict grouping).
  ✓ Verified against both feature cache metadata and fold_assignments.csv.
--- [Check 4/5] Valid Model Search Space Combinations ---
  ✓ 2D TFR combos: 3 reps × 2 backbones × 5 classifiers = 30 models.
  ✓ 1D Welch PSD combos: 1 rep × direct features × 5 classifiers = 5 models.
  ✓ Total valid representation-backbone-classifier combinations: 35.
  ✓ Physically invalid combos (e.g. Welch PSD -> ResNet) explicitly prohibited.
--- [Check 5/5] Prediction Aggregation & Historical AUC Verification ---
  ✓ EC Linear SVM: 58 participants, Acc = 89.66%, AUC = 0.9536.
  ✓ EO MLP: 61 participants, Acc = 91.80%, AUC = 0.9601.
  ✓ Continuous segment posterior averaging reproducing historical benchmarks.
======================================================================
  ALL 5 CHECKS PASSED PERFECTLY. ZERO INCONSISTENCIES FOUND.
======================================================================
```

---

## 4. Final Execution Readiness Verdict

```
══════════════════════════════════════════════════════════════════════════
                      PHASE 1B.7 READINESS GATE VERDICT
══════════════════════════════════════════════════════════════════════════

  VERDICT:                >>> READY FOR PHASE 1C EXECUTION <<<

  BLOCKERS RESOLVED:      ALL 7 METHODOLOGICAL BLOCKERS FORMALLY RESOLVED
  AUTOMATED CHECKS:       5 / 5 AUTOMATED VERIFICATION TESTS PASSED
  DATA INTEGRITY:         EPOCHS (2.0s), COHORTS (58 EC, 61 EO) 100% VERIFIED
  LEAKAGE GUARDS:         PARTICIPANT GROUPING & CALIBRATION ISOLATION LOCKED
  SEARCH SPACE:           35 VALID PIPELINES (WELCH DIRECT FEATURES CORRECTED)
  ACTIVE REPO STATE:      CLEAN WORKING TREE READY FOR EXECUTION COMMIT

══════════════════════════════════════════════════════════════════════════
```

**Next Action:** Await user authorization to launch Phase 1C experimental execution (Modules M1–M6).

*End of Final Consistency Verification Report.*
