# Final Scientific Validation Report: MDD EEG Research Project

**Date:** October 29, 2026  
**Project:** Automated Detection of Major Depressive Disorder Using SPWVD-Based EEG Time-Frequency Images and Deep Learning Features  
**Target Manuscript:** `manuscript/Jha_MDD_Revised_Final.docx`  
**Primary Evidence Ledger:** `reports/FINAL_EXPERIMENTAL_RESULTS.csv`

---

## 1. Metric Reconciliation and Mathematical Consistency

All experimental metrics were re-audited and reconciled against raw prediction artifacts, fold summaries, and out-of-fold probability distributions.

### A. Primary Eyes-Closed Resting-State Cohort ($N = 58$)
- **Participant Counts:** 28 Healthy Controls (HC), 30 MDD Patients ($N = 58$).
- **Segments:** 15 epochs per subject = 870 total epochs.
- **Aggregation Protocol:** Mean segment posterior probability thresholded at $0.5$.
- **Reconciled Results Table:**

| Classifier | TN | FP | FN | TP | Total | Acc (%) | 95% Wilson CI | Sens (%) | Spec (%) | Prec (%) | F1 (%) | ROC-AUC (%) | Fold Mean (%) | SD (ddof=0) | SD (ddof=1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | 24 | 4 | 2 | 28 | 58 | **89.66** | [79.21%, 95.17%] | 93.33 | 85.71 | 87.50 | 90.32 | **95.36** | 89.90 | ±8.34 | ±8.79 |
| **Medium Gaussian SVM** | 25 | 3 | 3 | 27 | 58 | **89.66** | [79.21%, 95.17%] | 90.00 | 89.29 | 90.00 | 90.00 | **93.57** | 89.90 | ±11.19 | ±11.79 |
| **Neural Network (MLP)** | 25 | 3 | 3 | 27 | 58 | **89.66** | [79.21%, 95.17%] | 90.00 | 89.29 | 90.00 | 90.00 | **94.40** | 89.90 | ±11.19 | ±11.79 |
| **Cubic SVM** | 24 | 4 | 2 | 28 | 58 | 89.66 | [79.21%, 95.17%] | 93.33 | 85.71 | 87.50 | 90.32 | 92.74 | 89.90 | ±11.19 | ±11.79 |
| **Quadratic SVM** | 23 | 5 | 2 | 28 | 58 | 87.93 | [77.12%, 94.03%] | 93.33 | 82.14 | 84.85 | 88.89 | 90.95 | 88.24 | ±10.79 | ±11.37 |
| **Weighted KNN** | 24 | 4 | 4 | 26 | 58 | 86.21 | [75.07%, 92.84%] | 86.67 | 85.71 | 86.67 | 86.67 | 92.14 | 86.57 | ±10.11 | ±10.66 |
| **Ensemble Bagged Tree** | 22 | 6 | 2 | 28 | 58 | 86.21 | [75.07%, 92.84%] | 93.33 | 78.57 | 82.35 | 87.50 | 91.43 | 86.81 | ±14.06 | ±14.82 |
| **Fine Gaussian SVM** | 3 | 25 | 4 | 26 | 58 | 50.00 | [37.54%, 62.46%] | 86.67 | 10.71 | 50.98 | 64.20 | 42.74 | 49.71 | ±9.52 | ±10.04 |

**Mathematical Proof of Consistency (Linear SVM):**
- $\text{Total} = 24 + 4 + 2 + 28 = 58$ (Matches enrolled cohort).
- $\text{Accuracy} = (24 + 28) / 58 = 52 / 58 = 89.6552\% \approx 89.66\%$.
- $\text{Sensitivity} = 28 / (28 + 2) = 28 / 30 = 93.3333\% \approx 93.33\%$.
- $\text{Specificity} = 24 / (24 + 4) = 24 / 28 = 85.7143\% \approx 85.71\%$.
- $\text{Precision} = 28 / (28 + 4) = 28 / 32 = 87.50\%$.
- $\text{F1-Score} = 2 \times (0.875 \times 0.93333) / (0.875 + 0.93333) = 90.3226\% \approx 90.32\%$.
- $\text{ROC-AUC} = 95.36\%$ (Calculated via `roc_auc_score(y_true, y_prob)` across 58 participant probabilities).

**Paired Model Testing:**
McNemar's test comparing the top two estimators (Linear SVM vs. Medium Gaussian SVM, both achieving 52/58 correct) showed 4 discordant cases ($b=2, c=2$), giving $p = 1.000$. The models are not statistically distinguishable on this sample.

---

### B. Secondary Eyes-Open Resting-State Cohort ($N = 61$)
- **Participant Counts:** 29 Healthy Controls (HC), 32 MDD Patients ($N = 61$).
- **Recordings:** 62 files (`sub-HS15` contributed two runs; quarantined to Fold 9).
- **Segments:** 930 total epochs.
- **Reconciled Results Table:**

| Classifier | TN | FP | FN | TP | Total | Acc (%) | 95% Wilson CI | Sens (%) | Spec (%) | Prec (%) | F1 (%) | ROC-AUC (%) | Fold Mean (%) | SD (ddof=0) | SD (ddof=1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Medium Gaussian SVM** | 27 | 2 | 3 | 29 | 61 | **91.80** | [82.21%, 96.45%] | 90.62 | 93.10 | 93.55 | 92.06 | **93.32** | 91.48 | ±12.49 | ±13.16 |
| **Neural Network (MLP)** | 27 | 2 | 3 | 29 | 61 | **91.80** | [82.21%, 96.45%] | 90.62 | 93.10 | 93.55 | 92.06 | **96.01** | 91.48 | ±12.49 | ±13.16 |
| **Linear SVM** | 25 | 4 | 3 | 29 | 61 | 88.52 | [78.16%, 94.33%] | 90.62 | 86.21 | 87.88 | 89.23 | **95.69** | 88.14 | ±11.98 | ±12.63 |
| **Quadratic SVM** | 25 | 4 | 3 | 29 | 61 | 88.52 | [78.16%, 94.33%] | 90.62 | 86.21 | 87.88 | 89.23 | 95.26 | 88.14 | ±11.98 | ±12.63 |
| **Cubic SVM** | 25 | 4 | 3 | 29 | 61 | 88.52 | [78.16%, 94.33%] | 90.62 | 86.21 | 87.88 | 89.23 | 92.56 | 88.14 | ±14.11 | ±14.87 |
| **Weighted KNN** | 25 | 4 | 3 | 29 | 61 | 88.52 | [78.16%, 94.33%] | 90.62 | 86.21 | 87.88 | 89.23 | 94.61 | 88.14 | ±11.98 | ±12.63 |
| **Ensemble Bagged Tree** | 24 | 5 | 3 | 29 | 61 | 86.89 | [76.20%, 93.20%] | 90.62 | 82.76 | 85.29 | 87.88 | 93.43 | 86.48 | ±13.59 | ±14.33 |
| **Fine Gaussian SVM** | 1 | 28 | 1 | 31 | 61 | 52.46 | [40.16%, 64.47%] | 96.88 | 3.45 | 52.54 | 68.13 | 44.07 | 51.86 | ±10.64 | ±11.22 |

---

## 2. Evaluation Levels: Segment-Level vs. Participant-Level

To prevent misleading claims of an "unqualified percentage point improvement" across different sample sizes and denominators, the two evaluation levels are reported separately:

### Eyes-Closed ($N = 58$ participants, $N = 870$ segments):
- **Linear SVM:** Segment-level accuracy = **80.34%** ($N=870$) | Participant-level accuracy = **89.66%** ($N=58$)
- **Medium Gaussian SVM:** Segment-level accuracy = **81.72%** ($N=870$) | Participant-level accuracy = **89.66%** ($N=58$)
- **Neural Network (MLP):** Segment-level accuracy = **81.84%** ($N=870$) | Participant-level accuracy = **89.66%** ($N=58$)
- **Weighted KNN:** Segment-level accuracy = **78.16%** ($N=870$) | Participant-level accuracy = **86.21%** ($N=58$)
- **Ensemble Bagged Tree:** Segment-level accuracy = **78.97%** ($N=870$) | Participant-level accuracy = **86.21%** ($N=58$)

### Eyes-Open ($N = 61$ participants, $N = 930$ segments):
- **Medium Gaussian SVM:** Segment-level accuracy = **84.19%** ($N=930$) | Participant-level accuracy = **91.80%** ($N=61$)
- **Neural Network (MLP):** Segment-level accuracy = **85.27%** ($N=930$) | Participant-level accuracy = **91.80%** ($N=61$)
- **Linear SVM:** Segment-level accuracy = **80.65%** ($N=930$) | Participant-level accuracy = **88.52%** ($N=61$)
- **Weighted KNN:** Segment-level accuracy = **81.94%** ($N=930$) | Participant-level accuracy = **88.52%** ($N=61$)
- **Ensemble Bagged Tree:** Segment-level accuracy = **83.01%** ($N=930$) | Participant-level accuracy = **86.89%** ($N=61$)

**Methodological Interpretation:**
Averaging posterior probabilities across multiple segments from each participant yields a consensus diagnostic decision that reduces the influence of isolated segment-level classification errors. In clinical psychiatric practice, diagnoses are rendered per patient rather than per 2-second snippet; reporting both levels preserves transparency without conflating distinct statistical units.

---

## 3. CNN Backbone Architecture Selection Rationale

Four pretrained convolutional neural networks were evaluated under identical Stratified Group 10-Fold CV using Weighted KNN:

| Backbone | Feature Dimension | EC Pooled Acc. (%) | EC Fold Mean ± SD (%) | EO Pooled Acc. (%) | EO Fold Mean ± SD (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | **512** | **86.21%** | 86.57 ± 10.11% | 88.52% | 88.14 ± 11.98% |
| **ResNet-50** | **2,048** | 84.48% | 84.81 ± 11.19% | **90.16%** | 89.81 ± 12.35% |
| **AlexNet** | 4,096 | 81.03% | 81.81 ± 15.34% | 85.25% | 85.14 ± 15.30% |
| **VGG-16** | 4,096 | 79.31% | 78.90 ± 11.00% | 83.61% | 83.71 ± 12.43% |

### Selection Rationale:
1. **Primary Evaluation Alignment:** The primary clinical benchmark pre-specified in the protocol was the Eyes-Closed resting condition. On EC, ResNet-18 outperformed ResNet-50 (86.21% vs. 84.48%), AlexNet (81.03%), and VGG-16 (79.31%).
2. **Dimensional Parsimony & Overfitting Protection:** ResNet-18 extracts a 512-dimensional embedding, whereas ResNet-50 yields 2,048 dimensions and AlexNet/VGG-16 yield 4,096 dimensions. For a clinical cohort of 58–61 participants, smaller feature spaces reduce parameter estimation variance and guard against the curse of dimensionality.
3. **Transparent Acknowledgment:** While ResNet-50 achieved higher performance in the secondary Eyes-Open condition (90.16% vs. 88.52%), ResNet-18 was favored for the primary benchmark due to its superior eyes-closed performance and dimensional parsimony.

---

## 4. Nested Cross-Validation and Model Selection Status

### Evaluation of Experiment 4 (Nested Linear SVM on EC):
- **Protocol:** Outer 10-fold Stratified Group CV reserved for evaluation; inner 5-fold Stratified Group CV conducted strictly on training folds to tune $C \in [0.01, 0.1, 1.0, 10.0]$.
- **Reconciliation with Output Files:** Inspected `results/experiments/exp4_nested_tuning/nested_cv_results.csv`:
  - Outer-test performance: **Accuracy = 91.38%** (95% Wilson CI: [81.36%, 96.26%]; 53/58 correct), **Sensitivity = 93.33%** (28/30), **Specificity = 89.29%** (25/28), **Precision = 90.32%**, **F1-Score = 91.80%**, **ROC-AUC = 94.76%** (TN=25, FP=3, FN=2, TP=28).
  - Selected hyperparameters across inner folds (`best_hyperparams_per_fold.csv`): $C = 0.01$ (folds 0, 1, 2, 5, 6, 8) and $C = 0.1$ (folds 3, 4, 7, 9).

### Methodological Qualification & Nuance:
- **What is strictly unbiased:** The choice of regularization penalty $C$ was tuned strictly within inner training folds without exposure to outer test folds.
- **What is NOT strictly unbiased (Macro-Selection):** The upstream decisions to select the ResNet-18 feature representation (over ResNet-50, AlexNet, VGG-16) and to choose the Linear SVM classifier family (over KNN, Tree, or MLP) were informed by baseline comparisons on the same cohort.
- **Reporting Decision:** In the revised manuscript, this experiment is explicitly titled:
  `"Nested Group-Aware Hyperparameter Tuning on a Pre-selected Feature Backbone"`.
  The text explicitly clarifies that 91.38% reflects within-model tuning on a pre-selected representation, not an end-to-end unbiased discovery across all pipeline hyperparameters.

### Status of Experiments 1, 2, and 3:
- **Experiment 1 (Aggregation Rules):** Median aggregation achieved 91.38% on EC Linear SVM, but dropped to 82.76% on KNN. Arithmetic Mean was retained as the standard confirmatory default. Labeled as **Exploratory**.
- **Experiment 2 (Spatial Montage Weighting):** Frontal emphasis improved Medium Gaussian SVM to 91.38% (+1.72%). Suggestive of frontal neurophysiology, but evaluated post-hoc. Labeled as **Exploratory**.
- **Experiment 3 (Kaiser Smoothing Variations):** Kaiser $\beta = 7.0$ achieved 93.10% accuracy (100% sensitivity) for Linear SVM, while narrow window degraded to 82.76%. Labeled as **Exploratory**.

---

## 5. Citation Audit and Literature Comparison Status

### A. Correction of Problematic Citations:
- **Pruned Retracted Paper:** Preliminary drafts cited *Mahato & Paul (2020)*, which was formally retracted by the publisher. It was removed and replaced with verified peer-reviewed studies (*Mumtaz et al. 2017, 2019; Wu et al. 2020*).
- **Verified Core Literature:**
  - Acharya et al. (2018): *CMPB*, 13-layer CNN on 30 subjects.
  - Mumtaz et al. (2017, 2019): Original HUSM cohort curators, wavelet + CNN pipelines.
  - Wu et al. (2020): *IEEE TNSRE*, conformal kernel SVM on 55 subjects.
  - Shen et al. (2020): *IEEE JBHI*, kernel-target alignment channel selection.

### B. Tabular Withholding Decision (Table III):
- Preliminary manuscript versions included a comparative table benchmarking reported accuracy figures across 8+ published studies.
- Audit established that published studies utilized heterogeneous validation schemes (e.g., segment-level random splitting vs. subject-isolated splitting) and varying participant cohorts.
- Directly comparing our group-isolated participant-level metrics against unverified or segment-level figures creates an unscientific comparison.
- **Action:** Table III is deliberately withheld from the revised manuscript. Section 4.1 transparently details the reasons for withholding.

---

## 6. Scientific Tone and Language Corrections

The revised manuscript (`manuscript/Jha_MDD_Revised_Final.docx`) removes all accusatory language:
1. **Abstract:** Removed claims that published pipelines "frequently suffer from inter-subject data leakage" and make "artificially inflated claims exceeding 99%". Replaced with neutral methodological prose explaining how segment-level splitting can allow correlated temporal windows into training and test partitions.
2. **Introduction:** Replaced assertions that prior literature produces accuracy that "collapses catastrophically" with a measured discussion of the difference between segment-level and participant-independent cross-validation.
3. **Discussion:** Removed allegations that high scores in literature "almost invariably" arise from bad practice. Replaced with objective analysis of participant-independent validation and a separate description of segment-level vs. participant-level evaluation.
