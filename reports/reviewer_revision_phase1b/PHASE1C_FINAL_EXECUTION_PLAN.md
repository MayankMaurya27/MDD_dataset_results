# PHASE 1C — COMPREHENSIVE EXPERIMENTAL REVISION ARCHITECTURE & EXECUTION PLAN
## Jha MDD EEG SPWVD Research Paper — Forensic Reviewer Revision

**Auditor / Lead Engineer:** Senior Biomedical Signal Processing & Machine Learning Reproducibility Auditor  
**Date:** November 7, 2026  
**Execution Context:** Final Phase 1C Experimental Architecture and Engineering Specifications  
**Deliverable Directory:** `reports/reviewer_revision_phase1b/`  
**Output Target Directory for Experiments:** `results/revision_experiments/phase1c/`  
**Precondition:** Strictly awaiting explicit user authorization before launching any script.

---

## 1. Scope of Phase 1C Revision Experiments

Phase 1C operationalizes the remaining computational experiments required to satisfy all 15 reviewer requirements:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1C EXPERIMENTAL SUITE ARCHITECTURE                        │
│                                                                                        │
│  [Module 1: Full 4-Backbone x 8-Classifier Matrix]                                     │
│     ├── Evaluates ResNet-18 (512-D), ResNet-50 (2,048-D), AlexNet (4,096-D), VGG-16    │
│     ├── Across all 8 downstream classifiers on both EC (N=58) and EO (N=61)            │
│     └── 32 distinct models evaluated under identical Group 10-Fold CV                  │
│                                                                                        │
│  [Module 2: Complete-Pipeline Nested Group Cross-Validation]                           │
│     ├── Outer Loop: Stratified Group 10-Fold CV (Outer Test Folds Held Out)           │
│     ├── Inner Loop: Stratified Group 5-Fold CV on Training Folds Only                  │
│     └── Joint Selection: Feature Representation (SPWVD vs STFT vs CWT vs Welch) +     │
│         Backbone + Scaler + Downstream Classifier + Hyperparameters                    │
│                                                                                        │
│  [Module 3: Repeated Group 10-Fold Cross-Validation (100 Runs)]                        │
│     ├── 10 Independent Partition Seeds (S in {42, 101, 202, ..., 909})                 │
│     ├── 10 Folds per Seed = 100 CV Iterations on Top Classifiers                       │
│     └── Quantifies Partition Variance, Mean Accuracy, and 95% Confidence Bounds        │
│                                                                                        │
│  [Module 4: End-to-End ResNet-18 Fine-Tuning Benchmark]                                │
│     ├── Unfreezes Layer 4 Convolutions + New Linear Head                               │
│     ├── AdamW Optimizer, Cosine Annealing, Inner-Validation Early Stopping             │
│     └── Tests whether Task-Specific Gradients Outperform Frozen Feature Extraction     │
│                                                                                        │
│  [Module 5: Fine Gaussian SVM Mathematical Rectification]                              │
│     ├── Formal Diagnosis: Fixed gamma=10.0 forces Dirac-delta identity kernel (K ~ I)  │
│     └── Solution: Replace gamma=10.0 with dimensionally scaled gamma = 1 / (P * Var(X)) │
│                                                                                        │
│  [Module 6: Artifact-Controlled Benchmark Reruns]                                      │
│     ├── Runs Condition A (Raw Uncorrected) vs. Condition B (FastICA Ocular Corrected)  │
│     └── Evaluates SPWVD + ResNet-18, Welch PSD, STFT, and CWT under Clean EEG          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Engineering Specifications for Each Module

### Module 1: Full 4-Backbone $\times$ 8-Classifier Matrix (32 Models)
- **Script Path:** `src/run_full_backbone_matrix.py`
- **Inputs:** Cached feature arrays in `results/features_cache/`:
  - `features_resnet18_*.npy` (512-D)
  - `features_resnet50_*.npy` (2,048-D)
  - `features_alexnet_*.npy` (4,096-D)
  - `features_vgg16_*.npy` (4,096-D)
- **Classifiers:** Linear SVM, Quadratic SVM, Cubic SVM, Medium Gaussian SVM, Weighted KNN, Ensemble Bagged Tree, MLP, and Corrected Fine Gaussian SVM.
- **Protocol:** Stratified Group 10-Fold CV; StandardScaler fit on train folds only; mean posterior participant aggregation.
- **Output Files:**
  - `results/revision_experiments/phase1c/backbone_matrix/oof_predictions_32models_eyesClosed.csv`
  - `results/revision_experiments/phase1c/backbone_matrix/oof_predictions_32models_eyesOpen.csv`
  - `results/revision_experiments/phase1c/backbone_matrix/matrix_summary_table.csv`

---

### Module 2: Complete-Pipeline Nested Group Cross-Validation
- **Script Path:** `src/run_nested_cv_benchmark.py`
- **Scientific Requirement:** Every model-selection decision must be made within inner training folds. Outer test folds are touched only once for unbiased performance estimation.
- **Inner Loop Grid ($5\text{-Fold}$ on Outer Train Folds):**
  - **Representation Level:** Candidate feature sets: Welch PSD (95-D), STFT (512-D), CWT Morlet (512-D), SPWVD ResNet-18 (512-D), SPWVD ResNet-50 (2,048-D).
  - **Classifier Level:** Linear SVM ($C \in \{0.01, 0.1, 1.0, 10.0\}$), RBF SVM ($C \in \{0.1, 1.0, 10.0\}, \gamma \in \{\text{'scale'}, 0.01, 0.1\}$), MLP (Hidden $\in \{(50,), (100,), (100, 50)\}$, $\alpha \in \{10^{-4}, 10^{-3}\}$).
- **Execution Mechanism:** For each outer fold $k \in \{0, \dots, 9\}$:
  1. Inner 5-fold CV evaluates each pipeline combination on the training folds.
  2. The single pipeline achieving the highest mean inner validation accuracy is selected.
  3. The selected pipeline is retrained on all training fold data and evaluated on outer held-out fold $k$.
- **Output Files:**
  - `results/revision_experiments/phase1c/nested_cv/nested_outer_oof_predictions.csv`
  - `results/revision_experiments/phase1c/nested_cv/nested_selected_pipeline_per_fold.csv`
  - `results/revision_experiments/phase1c/nested_cv/nested_cv_vs_fixed_baseline.csv`

---

### Module 3: Repeated Group 10-Fold Cross-Validation (100 Runs)
- **Script Path:** `src/run_repeated_cv.py`
- **Scientific Requirement:** Disprove that baseline results depend on a lucky seed.
- **Protocol:**
  - 10 independent random partition seeds: $S \in \{42, 101, 202, 303, 404, 505, 606, 707, 808, 909\}$.
  - For each seed, generate a Stratified Group 10-Fold partition strictly isolating participants and quarantining `sub-HS15` multi-runs.
  - Run top 3 classifiers (Linear SVM, Medium Gaussian SVM, MLP) across all 100 fold iterations.
  - Compute global mean, standard deviation, and empirical 95% confidence intervals across all 10 partition seeds.
- **Output Files:**
  - `results/revision_experiments/phase1c/repeated_cv/repeated_10x10cv_run_metrics.csv`
  - `results/revision_experiments/phase1c/repeated_cv/repeated_10x10cv_summary_statistics.csv`

---

### Module 4: End-to-End ResNet-18 Fine-Tuning Benchmark
- **Script Path:** `src/train_finetune.py`
- **Scientific Requirement:** Test whether task-specific convolutional fine-tuning outperforms frozen transfer learning.
- **Protocol:**
  - Pretrained ResNet-18 modified with a 2-class linear output head.
  - Unfreeze `layer4` convolutional blocks (residual blocks 7 & 8) while keeping `conv1` through `layer3` frozen to prevent overfitting.
  - Optimizer: AdamW (learning rate $10^{-4}$, weight decay $10^{-2}$) with cosine annealing schedule.
  - Evaluated under Stratified Group 10-Fold CV; early stopping monitored on inner-fold validation loss (patience 5 epochs, maximum 20 epochs).
- **Output Files:**
  - `results/revision_experiments/phase1c/finetune/oof_predictions_finetuned_resnet18.csv`
  - `results/revision_experiments/phase1c/finetune/finetuning_metrics_comparison.csv`

---

### Module 5: Fine Gaussian SVM Mathematical Rectification
- **Mathematical Analysis:**
  - In a $D = 512$ dimensional standardized space, the expected squared Euclidean distance between two independent Gaussian random vectors is:
    $$\mathbb{E}[\|\mathbf{x}_i - \mathbf{x}_j\|^2] \approx 2D = 1,024$$
  - With the hardcoded parameter $\gamma = 10.0$:
    $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-10 \times 1024) = \exp(-10240) \approx 0.0$$
  - Off-diagonal kernel elements vanish, forcing the gram matrix to the identity $K \approx I$. The decision boundary cannot interpolate across the space, collapsing to the class prior ($50.0\%$ in EC, $52.46\%$ in EO with $100\%$ sensitivity and $0\%$ specificity).
- **Rectification in Phase 1C:**
  - Implement **Auto-Scaled Fine Gaussian SVM**: $\gamma = \frac{\sqrt{D}}{4} \times \frac{1}{D \cdot \text{Var}(X)}$ (matching the original intended local RBF radius).
  - Compare both the unscaled degenerate model and the corrected model to document the mathematical root cause transparently in the manuscript.

---

### Module 6: Artifact-Controlled Benchmark Reruns (Condition B: ICA)
- **Script Path:** `src/run_artifact_controlled_baselines.py`
- **Protocol:**
  - Apply automated 19-component FastICA across all continuous recordings.
  - Remove components meeting the 3-point quantitative criteria (correlation with $Fp1+Fp2 \ge 0.75$, frontal-to-posterior gradient $\ge 2.0$, kurtosis $\ge 3.5$).
  - Reconstruct clean continuous EEG; re-extract the identical 15 deterministic epochs.
  - Recompute SPWVD + ResNet-18, Welch PSD, STFT, and CWT on the ICA-cleaned signals under Group 10-Fold CV.
- **Output Files:**
  - `results/revision_experiments/phase1c/artifact_controlled/ica_cleaned_metrics_summary.csv`
  - `results/revision_experiments/phase1c/artifact_controlled/clean_vs_raw_comparison.csv`
