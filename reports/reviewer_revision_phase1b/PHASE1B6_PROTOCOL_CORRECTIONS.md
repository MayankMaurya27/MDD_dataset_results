# Phase 1B.6 — Protocol Corrections and Methodological Safeguards
## Jha MDD EEG SPWVD Research Paper — Final Protocol Freeze Before Phase 1C

**Workspace:** `/Users/mayankmaurya/Desktop/experiment`  
**Date:** November 2026  
**Audited Baseline Commit:** `c831f80` (`docs(revision): complete Phase 1B.6 protocol corrections and execution checklist`)  
**Status:** Protocol Frozen — Awaiting Phase 1C Execution Authorization  

---

## 1. Executive Summary and Scope

Following the Phase 1B.5 scientific audit and Phase 1B.7 consistency review, this protocol-correction document formalizes all essential procedural, mathematical, and statistical safeguards required before launching the Phase 1C experimental campaign (Modules M1–M6).

The purpose of Phase 1B.6 is strictly non-destructive:
1. **No experimental code is executed in this phase.**
2. **No historical results or out-of-fold prediction files (`results/out_of_fold_predictions_*.csv`) are modified or overwritten.**
3. **The master manuscript (`manuscript/Jha_MDD_Revised_Final.docx`) remains completely locked.**
4. **All methodological ambiguities identified across the 14 original reviewer requirements (mapped to 15 internal tracking tasks) are formally resolved and frozen.**

This document details five vital protocol corrections:
- **Correction 1:** Rigorous framing and validation of the artifact-controlled experiment (Conditions A, B, and C), with biophysical clarification on ICA degrees of freedom vs neural sources.
- **Correction 2:** Complete-pipeline nested cross-validation architecture, probability calibration, and separation of exploratory sweeps from unbiased generalization estimates.
- **Correction 3:** Correct statistical aggregation for repeated group cross-validation (repeat-level pooling vs. fold-level independence).
- **Correction 4:** Leakage-free fine-tuning safeguards, inner validation partitioning, and provenance tracking.
- **Correction 5:** Repository commit ledger reconciliation and documentation synchronization.

---

## 2. Protocol Correction 1: Artifact-Controlled Experiment Refinement

Reviewer critiques regarding ocular contamination and artifact mitigation (Point 3, Point 7, and Point 10) require an unassailable experimental design that avoids both under-cleaning (leaving blink artifacts to drive spurious classification) and over-cleaning (destroying genuine neural dynamics).

### 2.1 The Three Experimental Conditions Defined
The Phase 1C experimental pipeline shall evaluate exactly three pre-specified conditions:

| Condition | Role | Definition & Pipeline |
| :--- | :--- | :--- |
| **Condition A** | **Historical Baseline (Uncorrected)** | Continuous 19-channel EEG bandpass filtered at $0.5 - 45.0\text{ Hz}$ (zero-phase 4th-order Butterworth). Non-notch filtered. Segmented into **2.0 s non-overlapping epochs** (512 samples at 256 Hz). Exactly **15 segments per recording** selected via deterministic spacing (`step = len(segs) // 15`). No independent component removal. Identical to original Jha et al. pipeline. |
| **Condition B** | **Primary Artifact-Corrected Pipeline** | Continuous 19-channel EEG bandpass filtered at $0.5 - 45.0\text{ Hz}$. Independent Component Analysis (Extended Infomax / FastICA) fitted per continuous 181 recording. Ocular artifact components identified and zeroed using frozen algorithmic criteria (max 2 components). Cleaned signals reconstructed via inverse mixing matrix $\mathbf{S}_{\text{clean}} = \mathbf{A} \mathbf{W}_{\text{clean}} \mathbf{X}$. Segmented into identical **2.0 s non-overlapping epochs** with identical 15-segment spacing. |
| **Condition C** | **Sensitivity Analysis (Epoch Replacement)** | **Explicitly designated as a sensitivity check, NOT an alternative primary preprocessing condition.** Segments in Condition A exhibiting extreme high-voltage contamination ($>150\,\mu\text{V}$ peak-to-peak amplitude across $>3$ channels) are replaced with the nearest temporally adjacent clean segment from the same participant and condition. Evaluates whether extreme transient excursions alter participant-level majority posteriors. |

### 2.2 Clarification of Frontal Electrodes as Ocular Proxies
The 19-channel montage used in the Russian MDD dataset follows the International 10–20 System:
$$\mathcal{M} = \{F_{p1}, F_{p2}, F_7, F_3, F_z, F_4, F_8, T_3, C_3, C_z, C_4, T_4, T_5, P_3, P_z, P_4, T_6, O_1, O_2\}$$

> [!IMPORTANT]
> The dataset **does not possess dedicated bipolar electro-oculogram (EOG) channels** (neither vertical VEOG nor horizontal HEOG). The leads $F_{p1}$ and $F_{p2}$ are scalp EEG electrodes placed on the forehead. While electro-ocular volume conduction dominates these leads during blinks and saccades, they simultaneously record genuine frontopolar cortical activity. 
> Therefore, in all scientific reports and manuscript revisions, $F_{p1}$ and $F_{p2}$ must be explicitly referred to as **"EEG-derived ocular proxies"**, and never mischaracterized as dedicated bipolar EOG sensors.

### 2.3 Objective Component Rejection and Fallback Rules
To prevent observer bias, ICA component identification is completely automated and frozen prior to training downstream classifiers:
1. **Ocular Component Matching:** A synthetic ocular reference signal is constructed as $\mathbf{y}_{\text{blink}}(t) = \frac{1}{2}\left(x_{F_{p1}}(t) + x_{F_{p2}}(t)\right)$.
2. **Selection Thresholds:** An independent component $k$ is flagged as ocular if:
   - Absolute Pearson correlation $|\rho(\mathbf{s}_k, \mathbf{y}_{\text{blink}})| \ge 0.40$.
   - Frontal spatial projection ratio $\frac{\mathbf{A}_{F_{p1}, k}^2 + \mathbf{A}_{F_{p2}, k}^2}{\sum_{i=1}^{19} \mathbf{A}_{i, k}^2} \ge 0.50$.
   - Component sample kurtosis $\kappa(\mathbf{s}_k) \ge 4.5$.
3. **Component Rejection Cap and Biophysical Clarification:** At most **two (2) components** out of 19 may be rejected per recording. If $>2$ components meet criteria, only the 2 with the highest frontal spatial power are removed, guaranteeing that $\ge 17$ spatial degrees of freedom are preserved for neural electrodynamics.
   > [!IMPORTANT]
   > **Biophysical Reality:** Retaining 17 independent components in a 19-channel decomposition guarantees preserving 17 linear degrees of freedom in the sensor space, but does **not** guarantee 17 pure, uncorrupted neural sources. Due to scalp volume conduction, hundreds of cortical dipolar generators linearly superimpose across the 19 scalp sensors. The cap is a safety measure to prevent over-cleaning, not a claim of complete neural source separation.
4. **Zero-Component Rule:** If no component meets the joint threshold, exactly 0 components are rejected; the recording is logged as artifact-quiescent, and the raw continuous signal is retained.
5. **Convergence Failure Fallback:** If FastICA fails to converge within 500 iterations, the algorithm automatically falls back to Picard / Infomax. If convergence still fails, the recording automatically defaults to Condition A (uncorrected) with an explicit warning entry in the audit ledger (`reports/reviewer_revision_phase1c/M1_artifact_audit.log`).

### 2.4 Multi-Modal Artifact vs. Neural Preservation Validation
The success of Condition B must **never** be claimed solely based on the reduction of frontal amplitude (which could merely indicate signal erasure). Validation must simultaneously establish:
- **Ocular Attenuation:** Statistically significant reduction in $F_{p1}/F_{p2}$ peak-to-peak amplitude during blink events ($p < 0.001$).
- **Posterior Alpha Preservation:** Preservation of power spectral density in posterior occipital/parietal channels ($O_1, O_2, P_3, P_4, P_z$) in the alpha band ($8.0 - 12.0\text{ Hz}$), with post-ICA power retention required to be $\ge 85\%$ of uncorrected power.
- **Topographic Specificity:** Preservation of central beta ($13 - 30\text{ Hz}$) and sensorimotor mu rhythms.

---

## 3. Protocol Correction 2: Complete-Pipeline Nested Model Selection

Reviewer Point 3, Point 5, and Point 8 demand an unbiased performance estimate free of hyperparameter tuning optimism, model selection bias, and feature scaling leakage.

### 3.1 Distinction: Exploratory Matrix vs. Nested-CV Unbiased Estimate
Phase 1C explicitly decouples two distinct analytical outputs to avoid confounding:
1. **Module M2 (Empirical 35-Pipeline Factorial Sweep):**
   - Evaluates a factorial grid of valid combinations:
     - 2D Time-Frequency: 3 representations (SPWVD, STFT, CWT) $\times$ 2 backbones (ResNet-18, ResNet-50) $\times$ 5 downstream classifiers (Linear SVM, RBF SVM, MLP, LogReg, RF) = 30 pipelines.
     - 1D Spectral: 1 representation (Welch PSD) $\times$ direct spectral features (NO CNN backbone) $\times$ 5 downstream classifiers = 5 pipelines.
     - Total: 35 strictly valid pipelines (rejecting invalid combinations such as Welch PSD passed into ResNet).
   - Evaluated across identical, frozen 10-fold Stratified Group CV splits.
   - Purpose: Direct side-by-side empirical comparison across representations and backbones under standardized conditions.
2. **Module M3 (Unbiased Autonomous Nested Cross-Validation):**
   - Implements a true double-loop nested CV: Outer 10-Fold Stratified Group CV $\times$ Inner 5-Fold Stratified Group CV.
   - Purpose: To report a single, completely unbiased generalization metric (Accuracy, AUC, F1) for the autonomous algorithm without any researcher-degree-of-freedom in selecting the best model.

### 3.2 Participant-Level Posterior Model Selection & Probability Calibration
Model selection in the inner loop must optimize the actual clinical deployment metric:
- **Leakage-Free Probability Calibration:** For margin classifiers (Linear SVM, RBF SVM) producing uncalibrated decision values $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$, continuous posteriors are obtained via Platt scaling:
  $$p(y = 1 \mid \mathbf{x}) = \frac{1}{1 + \exp(A \cdot f(\mathbf{x}) + B)}$$
  Calibration parameters $(A, B)$ must be fitted on an inner **participant-grouped calibration split** (or via nested group CV within the training folds). Under no circumstances are outer validation or test participants used for calibration.
- In each inner fold $j$, calibrated segment posterior probabilities $p_{i, s}$ are pooled per participant:
  $$P_i = \frac{1}{S_i} \sum_{s=1}^{S_i} p_{i, s}$$
- Selection criterion is the mean **Participant-Level Area Under the ROC Curve (AUC)** across the 5 inner validation folds.
- **Deterministic Tie-Breaking:** If two candidate pipelines achieve identical inner participant AUC (within $10^{-5}$):
  1. Higher inner Participant Balanced Accuracy.
  2. Higher inner Participant Sensitivity.
  3. Lower parametric complexity (Linear SVM > Logistic Regression > RBF SVM > MLP > Random Forest).
  4. Lower alphanumeric model index.

---

## 4. Protocol Correction 3: Repeated Group Cross-Validation Statistical Reporting

Reviewer Point 4 raised concern regarding the stability of cross-validation splits and potential variance across fold assignments.

### 4.1 Statistical Unit of Observation
A critical methodological trap in repeated cross-validation is the improper pooling of fold-level metrics:
- A $10 \times 10$-fold CV produces $100$ individual fold accuracy numbers.
- **Treating these 100 fold scores as 100 independent observations severely violates statistical independence** because participants are reused across repeats, yielding artificially narrow confidence intervals and inflated $t$-statistics.

### 4.2 Correct Repeat-Level Reporting Standard
In Module M4, the evaluation unit is strictly defined at the **Repeat Level**:
1. **One Repeat ($r \in \{1, \dots, 10\}$):** A full, complete 10-fold Stratified Group partition of the entire participant cohort (e.g., all 58 participants in EC, all 61 participants in EO).
2. **Repeat-Level Metric ($M_r$):** The out-of-fold predictions from all 10 folds within repeat $r$ are assembled to yield a single participant-level posterior probability vector for every subject in the cohort. The cohort metric $M_r$ (Accuracy, Sensitivity, Specificity, AUC) is computed once on this full set.
3. **Reported Distribution:**
   - Mean metric: $\bar{M} = \frac{1}{10} \sum_{r=1}^{10} M_r$.
   - Standard deviation across repeats: $\sigma_{\text{repeat}} = \sqrt{\frac{1}{9} \sum_{r=1}^{10} (M_r - \bar{M})^2}$.
   - Full range: $[\min(M_r), \max(M_r)]$.
   - 95% Confidence Interval computed via 1,000-sample participant-level bootstrap within each repeat, or through Nadeau-Bengio corrected repeated-CV $t$-test.

---

## 5. Protocol Correction 4: Fine-Tuning Safeguards and Reproducibility

Reviewer Point 5 requested validation with deeper convolutional backbones (ResNet-50) and fine-tuning. Deep fine-tuning introduces severe risks of test leakage if validation splits are improperly constructed.

### 5.1 Participant-Grouped Inner Validation for Fine-Tuning
When fine-tuning ResNet-18 or ResNet-50 on time-frequency representations:
1. **Zero Exposure of Outer Test Set:** Under no circumstances shall segments from outer test participants be present in the training set, validation set, early-stopping monitor, or learning-rate scheduler.
2. **Inner Validation Partition:** The outer training set (e.g., ~52 participants) is partitioned into an inner train set (~46 participants) and an inner validation set (~6 participants) using `StratifiedGroupKFold(n_splits=8, shuffle=True, seed=42)`.
3. **Early Stopping:** Monitored strictly on the participant-pooled loss or participant AUC of the inner validation set, with a patience of 5 epochs and max epochs capped at 25.
4. **Checkpoint Selection:** The model weights $\mathbf{W}^*$ restored for outer-test evaluation are the weights that achieved the minimum inner-validation loss, completely blind to the outer test set.

### 5.2 Deep Learning Provenance and Reproducibility Ledger
Every fine-tuning and backbone evaluation run must log the exact hardware and software execution signature to `results/reviewer_revision_phase1c/provenance_ledger.json`:
- **Pretrained Weights Provenance:** Official PyTorch Torchvision models (`ResNet18_Weights.IMAGENET1K_V1`, `ResNet50_Weights.IMAGENET1K_V2`).
- **Input Preprocessing:** Time-frequency matrices interpolated to $224 \times 224 \times 3$, normalized with standard ImageNet mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$.
- **Optimization Parameters:** AdamW optimizer, base learning rate $\eta = 10^{-4}$ for newly initialized classification head, $\eta = 10^{-5}$ for backbone layers; weight decay $\lambda = 10^{-2}$.
- **Random Seeds:** Master random seed frozen at `42`, torch seed `42`, numpy seed `42`, deterministic CUDA/MPS algorithms enabled where available.
- **Runtimes & Hardware:** Compute device recorded (`Apple M-Series Metal / MPS` or `CPU`), total wall-clock seconds per epoch and per fold.

---

## 6. Protocol Correction 5: Repository & Report Consistency Audit

### 6.1 Git Commit Reconciliation
The lineage of commits leading to Phase 1C execution is formally documented:
- **Commit `00a08df`:** Completed Phase 0 audit and Phase 1A OOF verification.
- **Commit `d5dac42`:** Committed Phase 1B implementation scripts (`src/artifact_diagnostic_audit.py`, `src/run_classical_baselines.py`) and generated baseline results.
- **Commit `baeb462`:** Committed Phase 1B.5 read-only scientific audit reports (`PHASE1B_CROSSWALK_CORRECTION.md`, `PHASE1B_ARTIFACT_HANDLING_DECISION.md`, etc.).
- **Commit `c831f80`:** Committed initial Phase 1B.6 protocol correction documents.
- **Authoritative Working Baseline:** The active git HEAD is verified with zero uncommitted changes. All future Phase 1C commits branch directly from this verified baseline.

### 6.2 Authoritative Source of Truth
The definitive reference for reviewer critiques is the set of **14 original reviewer comments** detailed in `reports/reviewer_revision_phase0/PHASE0_REVIEWER_14_POINT_MATRIX.md`. The 15 items tracked in Phase 1B.5/1B.6 represent an **internal engineering tracking breakdown** (specifically dividing Reviewer Point 9 into Fine Gaussian degeneracy and tied-accuracy posterior convergence). Internal tracking numbers must never be misrepresented as additional original reviewer comments.

---

## 7. Protocol Freeze Sign-Off

All five protocol corrections detailed above are mathematically formalized and locked into the repository. The experimental execution plan for Phase 1C is completely specified.

*End of Protocol Corrections Document.*

