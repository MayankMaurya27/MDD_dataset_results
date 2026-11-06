# PHASE 1B — REVIEWER REQUIREMENT CROSSWALK AND REVISED EXECUTION ARCHITECTURE
## Jha MDD EEG SPWVD Research Paper — Comprehensive Reviewer Revision

**Auditor / Lead Engineer:** Senior Biomedical Signal Processing & Machine Learning Reproducibility Auditor  
**Date:** November 7, 2026  
**Execution Stage:** Stage 1 — Crosswalk & Corrected Execution Architecture  
**Working Directory:** `/Users/mayankmaurya/Desktop/experiment`  
**Revision Policy:** Strict preservation of historical baseline files and existing OOF predictions; versioned outputs sequestered under `results/revision_experiments/`.

---

## 1. Executive Reconciliation of Phase 1A Gaps

Following the completion of the Phase 1A baseline recalculation, four specific methodological gaps in the revision roadmap were identified and are here formally resolved:

1. **Complete-Pipeline Nested Model Selection**:
   - *Phase 1A Limitation:* The initial nested CV plan focused primarily on classifier hyperparameters ($C$, $\gamma$) for a fixed backbone.
   - *Phase 1B Correction:* Nested selection must encompass the **entire model-selection pipeline**—including the choice of feature representation/backbone (SPWVD vs STFT vs CWT, ResNet-18 vs ResNet-50), scaler fitting, and classifier architecture—evaluated strictly on inner folds without information leakage into outer test folds.
2. **Empirical Artifact Handling vs. Premature Automated Cleaning**:
   - *Phase 1A Limitation:* Risked prematurely implementing automated ICA or thresholding without measuring true baseline contamination rates.
   - *Phase 1B Correction:* Mandates a **measure-first diagnostic protocol**: quantitatively compute peak-to-peak and absolute voltage distributions across all 181 recordings at $\pm 100\,\mu V$ and $\pm 150\,\mu V$, stratify by electrode region, evaluate ICA component topology, and only apply artifact rejection if empirical evidence justifies it.
3. **Reproducibility & Execution Ledgering**:
   - *Phase 1A Limitation:* Lacked unified machine-readable execution configuration YAMLs linking software dependencies, random seeds, hardware specs, and runtime profiling.
   - *Phase 1B Correction:* All experimental scripts will generate companion `.yaml` configuration files and execution logs recording execution timestamps, peak RSS memory, elapsed wall-clock time, and exact library versions.
4. **Manuscript Academic Language Cleanup**:
   - *Phase 1A Limitation:* Focused heavily on code and numbers without detailing the line-by-line language decontamination.
   - *Phase 1B Correction:* Formally schedules the elimination of defensive, hyperbolic, or speculative phrasing (e.g., edge-hardware deployment claims, overstating clinical diagnostic readiness, or informal critique of prior literature) in favor of neutral, high-rigor academic prose.

---

## 2. Complete 14-Point Reviewer Requirement Crosswalk

The matrix below maps each of the 14 reviewer comments to its concrete engineering action, experimental design, output artifacts, validation criteria, and dependency chain:

| # | Reviewer Requirement Point | Required Action | Planned Experiment or Manuscript Correction | Expected Output Artifact | Validation Criteria | Dependencies & Blockers |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Dataset Sample Size Contradiction (34 vs. 64)** | Reconcile Pierre Guetschel docstring draft error in `code/mumtaz2016.py` against 64 physical BIDS folders. | Manuscript Section 2.1 amendment; formal documentation of 181 `.edf` files. | `reports/reviewer_revision_phase0/PHASE0_DATASET_AND_SUBJECT_AUDIT.md`, updated DOCX Section 2.1. | 100% verification across all 64 subject directories (`sub-HS1`..`30`, `sub-MDDS1`..`34`) and `participants.tsv`. | None. Fully resolved in Phase 0. |
| **2** | **Demographics & Clinical Covariates** | Transparently disclose omission of per-subject demographic keys in OpenNeuro/NEMAR BIDS v1.0.0. | Manuscript Section 2.1.1 addition: report cohort-level figures from Mumtaz (2017) ($40.3 \pm 12.9$ vs $38.3 \pm 15.6$ yrs) and discuss medication washout. | `reports/reviewer_revision_phase1a/PHASE1A_CORRECTED_EXECUTION_PLAN.md`, updated DOCX Section 2.1.1. | Zero fabricated demographic fields; clear disclosure that `participants.tsv` contains `n/a` for age/sex/weight/height. | Blocker: Patient-level demographic sub-analyses are impossible without unreleased keys. |
| **3** | **Artifact Rejection & ICA Evaluation** | Measure epoch voltage distributions across all 181 recordings and evaluate ICA component separation. | Stage 2 implementation: `src/artifact_diagnostic_audit.py`. Compute peak-to-peak and absolute voltage at $\pm 100\,\mu V$ and $\pm 150\,\mu V$ by channel/region; run FastICA feasibility on representative traces. | `results/revision_experiments/artifacts/epoch_amplitude_metrics.csv`, `results/revision_experiments/artifacts/regional_artifact_summary.csv`, `reports/reviewer_revision_phase1b/PHASE1B_ARTIFACT_DIAGNOSTICS.md`. | Complete quantification across 181 EDFs; regional breakdown (Frontal, Central, Temporal, Parietal, Occipital); documented blink/muscle components. | None. Implemented in Phase 1B Stage 2. |
| **4** | **15-Epoch Selection Determinism** | Document the exact mathematical rule selecting 15 epochs per recording and verify sample timestamps. | Manuscript Section 2.3 precision update: document formula $\text{step} = \lfloor N / 15 \rfloor$ and verify `sample_start`/`sample_end` alignment. | `results/features_cache/metadata_*.csv`, updated DOCX Section 2.3. | Exact reproduction of selected indices from EDF sample lengths; zero stochastic sampling. | None. Verified in Phase 0 & Phase 1A. |
| **5** | **SPWVD Parameter Rationale & Cross-Terms** | Detail the analytic Hilbert signal formulation and Kaiser window frequency smoothing parameter ($\beta=5.0$). | Manuscript Section 2.4 and Results Section 3.3 update incorporating Exp 3 empirical ablation data ($\beta=3.5, 5.0, 7.0$). | `results/experiments/exp3_spwvd_smoothing/smoothing_comparison.csv`, updated DOCX Section 2.4 & 3.3. | Mathematical justification for cross-term suppression; empirical proof of narrow window degradation ($\beta=3.5 \to 82.76\%$). | None. Historical ablation verified. |
| **6** | **CNN Backbone Selection & Dimensionality** | Evaluate 4 backbones (ResNet-18/50, AlexNet, VGG-16) and justify ResNet-18 via dimensionality mitigation. | Phase 1C experiment: `src/run_full_backbone_matrix.py` evaluating all $4 \times 8 = 32$ models; update Table 1 and Section 3.1. | `results/revision_experiments/backbone_matrix/full_32_model_matrix_*.csv`. | ResNet-18 (512-D) outperforms 2,048-D and 4,096-D representations under identical Group 10-Fold CV; avoids curse of dimensionality on modest $N$. | None. Scheduled for Phase 1C. |
| **7** | **Subject Leakage & Cross-Validation Isolation** | Enforce zero subject overlap across folds; quarantine repeat subject `sub-HS15` multi-runs. | Verification of `results/fold_assignments.csv` and evaluation harness; unit test validation in `tests/test_pipeline.py`. | `tests/test_pipeline.py` passing log, Section 2.7. | Automated assertion: `len(set(train_subs).intersection(test_subs)) == 0` for all 10 folds; `sub-HS15` run-1 & run-2 strictly in Fold 9. | None. Verified in Phase 0 & Phase 1A. |
| **8** | **Clarification of "3 Controls per Fold" Claim** | Reconcile master partition balance (3 HC/fold) with actual condition-specific omissions. | Manuscript Section 2.7 text update: state master partition has 3 HC/fold, but EC has 2 HC in Folds 1 & 6, and EO has 2 HC in Fold 1. | Updated DOCX Section 2.7, `reports/reviewer_revision_phase0/PHASE0_FOLD_AND_PREDICTION_AUDIT.md`. | Exact alignment between manuscript text and condition-specific fold counts in `metadata_*.csv`. | None. Pure manuscript precision correction. |
| **9** | **Tied Metrics & Identical Prediction Audit** | Explain why Medium Gaussian SVM and MLP report identical discrete accuracy and fold mean $\pm$ SD. | Manuscript Table 2 Note and Section 3.2: document segment probability divergence converging under posterior mean aggregation; McNemar test $p=1.000$. | `reports/reviewer_revision_phase1a/PHASE1A_OOF_VERIFICATION.md`, updated DOCX Table 2. | Empirical proof: 0/870 (EC) and 0/930 (EO) probabilities identical; 43 (EC) and 90 (EO) segment predictions disagree; McNemar $p=1.000$. | None. Verified in Phase 1A. |
| **10** | **Continuous ROC Curves & Bootstrap AUC CIs** | Replace discrete approximations with continuous participant ROC curves and 1,000-sample bootstrap 95% CIs. | Embed publication ROC curves in DOCX; report continuous participant AUCs (Linear SVM: 0.9536 [0.8909, 0.9952] EC; MLP: 0.9601 [0.9073, 0.9956] EO). | `results/phase1a_verification/figure_roc_curves_verified.png`, `reports/reviewer_revision_phase1a/PHASE1A_BOOTSTRAP_AUC_RESULTS.csv`. | Continuous trapezoidal AUC integration from unthresholded mean posteriors; participant-level resampling with fixed seed (`seed=42`). | None. Completed in Phase 1A. |
| **11** | **Classical Baselines (Welch PSD, STFT, CWT)** | Benchmark SPWVD + ResNet-18 against standard spectral power and classical time-frequency representations. | Stage 3 implementation: `src/baselines_psd.py`, `src/baselines_stft.py`, `src/baselines_cwt.py` under identical Group 10-Fold CV. | `results/revision_experiments/baselines/*_results.csv`, `reports/reviewer_revision_phase1b/PHASE1B_BASELINE_RESULTS.md`. | Identical subject folds, identical 15-epoch extraction, identical standard scaler fitted on training folds only; honest reporting of comparative performance. | None. Implemented in Phase 1B Stage 3. |
| **12** | **Nested Group CV Across Full Pipeline** | Perform hyperparameter and model selection strictly within training folds using inner cross-validation. | Phase 1C implementation: `src/run_nested_cv_benchmark.py` (Outer 10-Fold $\times$ Inner 5-Fold) evaluating grid across scalers, models, and regularization. | `results/revision_experiments/nested_cv/nested_cv_benchmark_summary.csv`. | Zero test fold leakage during hyperparameter selection; comparison of nested generalization vs. fixed baseline. | Dependent on baseline implementations in Phase 1B. |
| **13** | **Literature Comparison Harmonization** | Resolve withheld Table III comparators; avoid misleading comparisons with unharmonized/flawed studies. | Manuscript Section 4.1 synthesis: explain why cross-dataset comparisons with segment-shuffled splits (>99%) are invalid; provide audited criteria. | `reports/FINAL_CITATION_AUDIT.csv`, updated DOCX Section 4.1. | Retain withholding of raw Table III; present transparent methodological contrast focusing on leakage prevention. | None. Audited in Phase 0. |
| **14** | **Citation Integrity & Retracted Article Pruning** | Ensure every citation supports its claim; permanently excise retracted article *Mahato & Paul (2020)*. | Complete audit of in-text citations [1]–[22]; removal of Mahato (2020); correction of ImageNet citation to Deng et al. (2009). | `reports/FINAL_CITATION_AUDIT.csv`, updated DOCX References. | 100% verified DOI records; zero citations to retracted works; disentangled method attribution (Wu [16] vs Shen [17]). | None. Verified in Phase 0. |

---

## 3. Strict Boundary and Phase 1B Execution Sequence

In accordance with user instructions, execution in Phase 1B is strictly partitioned:
- **Execute Now (Phase 1B):**
  1. **Stage 1:** Reviewer Requirement Crosswalk (this document).
  2. **Stage 2:** Artifact Diagnostic Audit across all 181 EDF files (`src/artifact_diagnostic_audit.py`).
  3. **Stage 3:** Classical Baselines (`src/baselines_psd.py`, `src/baselines_stft.py`, `src/baselines_cwt.py`).
  4. **Stage 4:** Integrity verification, runtime profiling, and deliverable reporting.
- **Defer to Subsequent Phases:**
  - Full 4-backbone matrix ($4 \times 8 = 32$ evaluations) $\to$ **Phase 1C**.
  - Repeated Group 10-Fold CV (10 repeats $\times$ 10 folds = 100 runs) $\to$ **Phase 1C**.
  - Complete-pipeline nested CV $\to$ **Phase 1C**.
  - ResNet-18 fine-tuning $\to$ **Phase 1C**.
  - Manuscript DOCX modification $\to$ **Phase 2**.
