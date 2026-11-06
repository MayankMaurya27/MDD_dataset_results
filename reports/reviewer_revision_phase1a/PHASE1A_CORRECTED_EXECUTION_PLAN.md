# PHASE 1A — COMPREHENSIVE EXPERIMENTAL REVISION & EXECUTION PLAN
## Jha MDD EEG SPWVD Research Paper — Complete Reviewer Revision

**Auditor / Lead Engineer:** Senior Biomedical Signal Processing & Machine Learning Reproducibility Auditor  
**Date:** November 7, 2026  
**Project Workspace:** `/Users/mayankmaurya/Desktop/experiment`  
**Revision Policy:** Complete, evidence-backed implementation of ALL reviewer requirements. Versioned outputs; zero overwriting of baseline deliverables.

---

## 1. Scope and Architectural Principles

This execution plan operationalizes the complete set of reviewer requirements across signal processing, machine learning validation, baseline benchmarking, and clinical transparency. All new experiments will be sequestered into versioned directories (`results/revision_experiments/`) to preserve the provenance of historical outputs.

### Core Integrity Safeguards:
1. **Strict Participant Independence**: Every train/validation/test split will strictly enforce zero subject overlap.
2. **Inner Cross-Validation for Model Selection**: No hyperparameter tuning, feature selection, or backbone selection will access held-out outer test folds.
3. **Reproducibility Locking**: All scripts will log seeds, hardware specs, software versions, and configuration YAMLs.
4. **Honest Baseline Reporting**: If classical spectral baselines (e.g., Welch PSD) perform competitively, report it directly without selective filtering.

---

## 2. Detailed Implementation Specifications Across All Reviewer Requirements

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        REVIEWER REVISION IMPLEMENTATION ARCHITECTURE                   │
│                                                                                        │
│  [Module 1: Preprocessing & Artifact Rejection]                                        │
│     ├── Artifact Diagnostic Script (Peak-to-Peak Amplitude Thresholding)               │
│     └── Picard / FastICA Comparative Module & Ocular Artifact Analysis                 │
│                                                                                        │
│  [Module 2: Classical Time-Frequency & Spectral Baselines]                             │
│     ├── Welch Power Spectral Density (PSD: Delta, Theta, Alpha, Beta, Gamma) + SVM    │
│     ├── Short-Time Fourier Transform (STFT Spectrograms) + ResNet-18                   │
│     └── Continuous Wavelet Transform (CWT Morlet Scalograms) + ResNet-18               │
│                                                                                        │
│  [Module 3: Deep Transfer Learning & Fine-Tuning Benchmarks]                           │
│     ├── Full 4-Backbone (ResNet-18/50, AlexNet, VGG-16) x 8-Classifier Matrix          │
│     └── End-to-End Fine-Tuned ResNet-18 (Unfrozen Convolutions + Early Stopping)       │
│                                                                                        │
│  [Module 4: Leakage-Free Optimization & Rigorous Validation]                           │
│     ├── Full Nested Group 10-Fold CV (Outer 10-Fold x Inner 5-Fold)                    │
│     └── Repeated Group 10-Fold CV (10 independent seeds x 10 folds = 100 runs)         │
│                                                                                        │
│  [Module 5: Clinical Metadata, Confounders & Reference Rectification]                 │
│     ├── Demographic Transparency & Confound Analysis (Age/Sex/Medication Limitations) │
│     └── Citation Harmonization & Reference Matrix (Excluding Mahato 2020)              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Requirement 1: Artifact Rejection, ICA Evaluation, and Preprocessing Diagnostics
- **Rationale:** Reviewers require empirical quantification of ocular and muscular artifacts and justification for the filtering protocol.
- **Implementation File:** `src/artifact_audit.py`
- **Algorithmic Specifications:**
  1. Compute peak-to-peak amplitude distributions per epoch across all 19 monopolar channels.
  2. Measure the percentage of epochs exceeding standard clinical artifact thresholds ($\pm 100\,\mu V$ and $\pm 150\,\mu V$), stratified by electrode region (Frontal: Fp1, Fp2, F7, F8; Central: C3, C4, Cz; Occipital: O1, O2).
  3. Implement Picard/FastICA decomposition on sample raw recordings to evaluate blink/saccade component isolation.
  4. Compare classification accuracy on raw bandpassed segments vs. amplitude-cleaned segments.
- **Output Artifacts:** `results/revision_experiments/artifacts/amplitude_distribution_summary.csv`, `artifact_rejection_comparison.csv`.
- **Manuscript Target:** Section 2.2 and Discussion (Clinical Translation).

---

### Requirement 2: Exact Deterministic Epoch Selection
- **Rationale:** Reviewers requested confirmation that epoch extraction is deterministic and temporal coordinates are traceable.
- **Implementation:** Formalized in `src/preprocessing.py` and `metadata_*.csv`.
- **Algorithmic Specifications:**
  - Formula: $\text{step} = \lfloor N_{\text{total\_epochs}} / 15 \rfloor$; $\text{index}_k = k \cdot \text{step}$ for $k = 0, \dots, 14$.
  - Exact time ranges: $\text{time\_start} = \text{sample\_start} / 256.0$, $\text{time\_end} = \text{sample\_end} / 256.0$.
- **Output Artifacts:** Manifest table `results/revision_experiments/preprocessing/epoch_selection_manifest.csv`.
- **Manuscript Target:** Section 2.3.

---

### Requirement 3: Full 4-Backbone $\times$ 8-Classifier Evaluation Matrix
- **Rationale:** Reviewers requested either evaluating all 8 classifiers on all 4 backbones ($4 \times 8 = 32$ model combinations) or providing a rigorous scientific defense for focusing on ResNet-18.
- **Implementation File:** `src/run_full_backbone_matrix.py`
- **Algorithmic Specifications:**
  - Ingest cached features for ResNet-18 (512-D), ResNet-50 (2,048-D), AlexNet (4,096-D), and VGG-16 (4,096-D).
  - Train and evaluate all 8 classifiers (Linear SVM, Quadratic SVM, Cubic SVM, Fine Gaussian SVM, Medium Gaussian SVM, Weighted KNN, Ensemble Bagged Tree, MLP) across the identical Stratified Group 10-Fold CV partition.
  - Compute segment-level and participant-level pooled accuracy, sensitivity, specificity, F1, and continuous AUC.
- **Output Artifacts:** `results/revision_experiments/backbone_matrix/full_32_model_matrix_EC.csv`, `full_32_model_matrix_EO.csv`.
- **Manuscript Target:** Table 1 and Section 3.1.

---

### Requirement 4: Nested Group Cross-Validation Across Model Selection
- **Rationale:** Prevent hyperparameter selection bias from contaminating the outer cross-validation evaluation.
- **Implementation File:** `src/run_nested_cv_benchmark.py`
- **Algorithmic Specifications:**
  - Outer loop: Stratified Group 10-Fold CV ($N=58$ for EC; $N=61$ for EO).
  - Inner loop: Stratified Group 5-Fold CV on training folds only (zero validation subject leakage).
  - Hyperparameter grid:
    - Linear SVM: $C \in \{0.001, 0.01, 0.1, 1.0, 10.0, 100.0\}$.
    - Medium Gaussian SVM: $C \in \{0.1, 1.0, 10.0\}$, $\gamma \in \{\text{'scale'}, 0.001, 0.01, 0.1\}$.
    - MLP: Hidden sizes $\in \{(50,), (100,), (100, 50)\}$, $\alpha \in \{10^{-4}, 10^{-3}, 10^{-2}\}$, learning rate $\in \{10^{-3}, 10^{-4}\}$.
  - Outer test folds evaluated strictly using the best inner-CV hyperparameter combination.
- **Output Artifacts:** `results/revision_experiments/nested_cv/nested_vs_fixed_hyperparameters.csv`, `best_params_per_fold.csv`.
- **Manuscript Target:** Section 3.4.

---

### Requirement 5: Repeated Group 10-Fold Cross-Validation (10 Repeats $\times$ 10 Folds)
- **Rationale:** Eliminate seed selection bias and assess cross-validation variance across partition seeds.
- **Implementation File:** `src/run_repeated_cv.py`
- **Algorithmic Specifications:**
  - Generate 10 independent stratified group partitions using random seeds $S \in \{42, 101, 202, 303, 404, 505, 606, 707, 808, 909\}$.
  - In every repeat, enforce participant grouping and isolation of `sub-HS15` multi-runs.
  - Run top classifiers (Linear SVM, Medium Gaussian SVM, MLP) across all 100 fold iterations ($10 \times 10$).
  - Calculate empirical mean, 95% confidence intervals across repeats, and partition stability metrics.
- **Output Artifacts:** `results/revision_experiments/repeated_cv/repeated_10x10cv_summary.csv`.
- **Manuscript Target:** Section 3.2.

---

### Requirement 6: Classical Baseline 1 — Welch Power Spectral Density (PSD) + SVM
- **Rationale:** Standard neurophysiology benchmark to prove SPWVD 2D representations provide added value over classic spectral bands.
- **Implementation File:** `src/baselines_psd.py`
- **Algorithmic Specifications:**
  - Compute Welch PSD across 19 channels with 1.0s Hamming windows and 50% overlap.
  - Extract absolute and relative band power across 5 physiological bands:
    - Delta: $0.5 - 4.0\text{ Hz}$
    - Theta: $4.0 - 8.0\text{ Hz}$
    - Alpha: $8.0 - 13.0\text{ Hz}$
    - Beta: $13.0 - 30.0\text{ Hz}$
    - Gamma: $30.0 - 45.0\text{ Hz}$
  - Feature vector: $19 \times 5 = 95\text{-D}$ feature vector per 2.0s epoch.
  - Train Linear SVM and RBF SVM under the identical Group 10-Fold CV partition.
- **Output Artifacts:** `results/revision_experiments/baselines/welch_psd_results.csv`.
- **Manuscript Target:** Section 4.1.

---

### Requirement 7: Classical Baseline 2 — Short-Time Fourier Transform (STFT) + ResNet-18
- **Rationale:** Benchmark SPWVD against the standard fixed-window time-frequency representation.
- **Implementation File:** `src/baselines_stft.py`
- **Algorithmic Specifications:**
  - Compute STFT for each 2.0s epoch across 19 channels using Hann window of length 64 samples ($\approx 250\text{ ms}$) with 50% overlap.
  - Spatially average the 19 spectrogram matrices into a single $(F, T)$ energy map.
  - Normalize and resize to $(3, 224, 224)$ using identical ImageNet standardization.
  - Extract 512-D features using frozen ResNet-18; evaluate with downstream Linear SVM and MLP under Group 10-Fold CV.
- **Output Artifacts:** `results/revision_experiments/baselines/stft_resnet18_results.csv`.
- **Manuscript Target:** Section 4.1.

---

### Requirement 8: Classical Baseline 3 — Continuous Wavelet Transform (CWT) + ResNet-18
- **Rationale:** Benchmark SPWVD against multi-scale wavelet time-frequency representations.
- **Implementation File:** `src/baselines_cwt.py`
- **Algorithmic Specifications:**
  - Compute CWT using complex Morlet wavelets across 64 pseudo-frequencies spanning $0.5 - 45.0\text{ Hz}$.
  - Spatially average across 19 channels into a $(64, 512)$ scalogram.
  - Normalize and resize to $(3, 224, 224)$; extract 512-D features via frozen ResNet-18; evaluate with Linear SVM under Group 10-Fold CV.
- **Output Artifacts:** `results/revision_experiments/baselines/cwt_resnet18_results.csv`.
- **Manuscript Target:** Section 4.1.

---

### Requirement 9: End-to-End Fine-Tuned ResNet-18 Baseline
- **Rationale:** Address whether task-specific fine-tuning outperforms frozen transfer learning.
- **Implementation File:** `src/train_finetune.py`
- **Algorithmic Specifications:**
  - Architecture: Pretrained ResNet-18 with final linear layer modified for binary classification ($512 \to 2$).
  - Training Protocol: Unfreeze layer 4 and linear head; freeze layers 1–3 to prevent overfitting on 870/930 training images.
  - Optimizer: AdamW, initial learning rate $10^{-4}$ with cosine annealing, batch size 32, weight decay $10^{-2}$.
  - Early stopping based on inner-fold validation loss (patience 5 epochs, maximum 20 epochs).
  - Evaluate out-of-fold predictions at segment and participant levels.
- **Output Artifacts:** `results/revision_experiments/finetune/finetuned_resnet18_results.csv`.
- **Manuscript Target:** Section 3.1.

---

### Requirement 10: Clinical Covariates, Confounders, and Demographic Limitations
- **Rationale:** Transparently report data limitations and prevent ungrounded claims regarding age/sex stratification.
- **Actionable Execution:**
  - Audit verified: `nm000114_v1.0.0/participants.tsv` has `n/a` for age, sex, weight, height.
  - Dedicated Manuscript Section: Add Subsection 2.1.1 *"Data De-Identification, Covariate Limitations, and Confound Analysis"*.
  - Detail clinical cohort age distributions cited from Mumtaz et al. (2017) (MDD: $40.3 \pm 12.9$ yrs; HC: $38.3 \pm 15.6$ yrs) while explicitly confirming that individual participant keys are uncoupled in OpenNeuro/NEMAR BIDS v1.0.0.
  - Discuss medication wash-out protocols and potential pharmacological confounds (antidepressants altering frontal alpha/theta power).
- **Manuscript Target:** Section 2.1, Section 5 (Discussion & Limitations).

---

### Requirement 11: Ethics Approval, Consent, and Licensing Verification
- **Rationale:** Fully document clinical compliance and open-access rights.
- **Actionable Execution:**
  - Acquisition Center: Hospital Universiti Sains Malaysia (HUSM).
  - Ethics Committee: Human Research Ethics Committee of Universiti Sains Malaysia (JEPeM Code: `USM/JEPeM/1401016`).
  - Consent: Written informed consent from all participants or guardians.
  - License: CC-BY-4.0 explicitly documented in `dataset_description.json`.
- **Manuscript Target:** Declarations / Ethics Statement.

---

### Requirement 12: Related Work, Citation Integrity, and Literature Harmonization
- **Rationale:** Remove unverified claims, disentangle misattributions, and prune retracted articles.
- **Actionable Execution:**
  - Keep retracted paper *Mahato & Paul (2020)* completely excised from text and references.
  - Keep Table III comparators withheld; provide a methodological synthesis of why cross-dataset comparisons with unharmonized cross-validation schemes (e.g., segment-shuffled splits reporting >99%) are scientifically invalid.
  - Verify every in-text citation [1]–[22] against primary DOI records.
- **Manuscript Target:** Section 1 (Introduction), Section 4.1, References.

---

## 3. Recommended Phased Execution Strategy

To execute this plan systematically without system overload or unverified shortcuts, the remaining revision work is organized into three sequential sub-phases:

| Phase | Milestone Name | Scope of Work | Deliverables |
| :--- | :--- | :--- | :--- |
| **Phase 1B** | **Baselines & Artifact Diagnostics** | 1. Implement Welch PSD, STFT, and CWT baselines.<br>2. Run peak-to-peak amplitude artifact audit.<br>3. Benchmark all baselines under identical 10-fold CV. | Baseline comparison tables, artifact distribution CSV, verification plots. |
| **Phase 1C** | **Backbone Matrix, Fine-Tuning & Nested CV** | 1. Run full 4-backbone $\times$ 8-classifier matrix.<br>2. Execute nested CV hyperparameter tuning.<br>3. Run repeated 10-fold CV (10 repeats $\times$ 10 folds).<br>4. Run ResNet-18 fine-tuning benchmark. | Extended experimental ledger, nested CV comparison, repeated CV variance metrics. |
| **Phase 2** | **Manuscript Synthesis & Final Submission Lock** | 1. Update text, tables, and figures in DOCX.<br>2. Embed verified continuous ROC figures.<br>3. Regenerate and visually audit final Word deliverables.<br>4. Git commit and push complete audit repository. | `Jha_MDD_Revised_Final.docx`, complete audit reports, GitHub commit push. |

