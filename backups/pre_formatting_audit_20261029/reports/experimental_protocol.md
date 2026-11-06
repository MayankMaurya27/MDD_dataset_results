# Scientific Experimental Protocol and Leakage-Prevention Design

**Study**: Automated Detection of Major Depressive Disorder Using SPWVD-Based EEG Time-Frequency Images and Deep Learning Features  
**Author**: Senior EEG Signal-Processing & Machine-Learning Experimental Auditor  
**Date**: October 28, 2026  
**Status**: Approved Protocol  

---

## 1. Protocol Architecture & Objectives

The primary objective of this protocol is to conduct an uncompromising, scientifically rigorous, and fully reproducible independent validation of the machine-learning pipeline for Major Depressive Disorder (MDD) detection from electroencephalographic (EEG) data.

Past literature and preliminary drafts in automated EEG diagnosis frequently suffer from subtle data leakage (e.g., segment-level shuffling across folds, global normalization, or subject contamination), leading to artificially inflated classification metrics (such as >99% accuracy). This protocol establishes a **Participant-Independent, Group-Stratified Cross-Validation framework** with strict isolation of training and testing distributions.

---

## 2. Participant-Grouped Stratified Cross-Validation

### 2.1 Grouping Unit
The fundamental experimental unit is the **individual participant** (`participant_id`).
- All recordings (Eyes Closed, Eyes Open, P300), runs (`run-1`, `run-2`), time windows, and derived Smoothed Pseudo Wigner-Ville Distribution (SPWVD) time-frequency representations from participant $i$ are **strictly bound to a single fold**.
- Under no circumstance may any segment of participant $i$ appear in the training partition if participant $i$ is assigned to the test partition of that fold.

### 2.2 Fold Structure (10-Fold Group Stratification)
The dataset comprises $N=64$ unique participants:
- **30 Healthy Controls (HC)** (`sub-HS1` to `sub-HS30`, Class 0)
- **34 MDD Patients** (`sub-MDDS1` to `sub-MDDS34`, Class 1)

To preserve class balance across 10 folds:
- Each fold receives **exactly 3 Healthy Controls** ($10 \times 3 = 30\text{ HC}$).
- Folds 0 through 3 receive **4 MDD Patients** ($4 \times 4 = 16\text{ MDD}$).
- Folds 4 through 9 receive **3 MDD Patients** ($6 \times 3 = 18\text{ MDD}$).
- Total MDD allocated: $16 + 18 = 34\text{ MDD}$.

| Fold | Total Participants | HC Count | MDD Count | Total Recordings |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 7 | 3 | 4 | 20 |
| **Fold 1** | 7 | 3 | 4 | 19 |
| **Fold 2** | 7 | 3 | 4 | 17 |
| **Fold 3** | 7 | 3 | 4 | 21 |
| **Fold 4** | 6 | 3 | 3 | 16 |
| **Fold 5** | 6 | 3 | 3 | 17 |
| **Fold 6** | 6 | 3 | 3 | 17 |
| **Fold 7** | 6 | 3 | 3 | 18 |
| **Fold 8** | 6 | 3 | 3 | 17 |
| **Fold 9** | 6 | 3 | 3 | 19 |
| **Total** | **64** | **30** | **34** | **181** |

Deterministic fold assignments are saved in `results/fold_assignments.csv` using fixed random seed `SEED = 42`.

---

## 3. Resolution of the Manuscript's Cross-Validation Contradiction

### The Contradiction in the Draft
Section *"Subject-Wise Grouped Cross-Validation"* of the draft states:
> *"The sample comprised 64 individuals, categorized into 10 distinct folds, each including about equal numbers of MDD sufferers and healthy controls... Every iteration of the investigation, totaling 10, included a single test subject."*

### Scientific Resolution
This statement contains a direct mathematical impossibility:
- If each fold evaluates a single test subject across 10 iterations, the total evaluation cohort would be only 10 subjects (or conversely, testing 64 subjects individually would require a 64-fold Leave-One-Subject-Out (LOSO) scheme).
- In a valid 10-fold cross-validation of 64 subjects, **each test fold must evaluate 6 or 7 distinct subjects**, while training on the remaining 57 or 58 subjects.
- In our revised protocol and revised manuscript, this contradiction is resolved: the evaluation is formalized as **Stratified Group 10-Fold Cross-Validation**, with 6–7 test subjects per fold.

---

## 4. Evaluation Units and Aggregation Strategy

EEG recordings are divided into discrete temporal segments for time-frequency analysis. Evaluating performance solely at the segment level introduces two severe issues:
1. **Clinical Irrelevance**: Clinicians diagnose patients, not arbitrary 5-second or 10-second snippets.
2. **Sample Weight Bias**: Participants with longer recordings or additional tasks contribute more segments, skewing overall metrics.

### Dual-Level Metric Framework:
1. **Primary Evaluation Unit: Participant-Level Evaluation**
   For test subject $i$ with $S_i$ valid segments, the model produces segment-level posterior probability estimates $p_{i, s} = P(y=1 \mid \mathbf{x}_{i, s})$.
   The participant-level diagnostic probability $\hat{P}_i$ is computed via mean probability aggregation:
   $$\hat{P}_i = \frac{1}{S_i} \sum_{s=1}^{S_i} p_{i, s}$$
   The predicted binary diagnostic classification is:
   $$\hat{y}_i = \begin{cases} 1 & \text{if } \hat{P}_i \ge 0.5 \\ 0 & \text{otherwise} \end{cases}$$
   All primary metrics (Accuracy, Balanced Accuracy, Sensitivity, Specificity, Precision, F1-Score, and ROC-AUC) are computed on the pooled out-of-fold participant predictions $(\hat{y}_i, y_i)_{i=1}^{64}$ and fold-wise participant distributions.

2. **Secondary Evaluation Unit: Segment-Level Evaluation**
   Reported for methodological transparency to assess raw time-frequency segment discriminability, but clearly designated as secondary.

---

## 5. Comprehensive Leakage-Prevention Checklist

| Stage | Leakage Vector | Mitigation Measure | Verification Check |
| :--- | :--- | :--- | :--- |
| **Segmentation** | Segment crossing recording boundaries | Segment windows strictly within individual files. | Segment timestamps $\le$ file duration. |
| **Train/Test Split**| Subject contamination across partitions | All segments of participant $i$ mapped exclusively to fold $k$. | $\text{TrainParticipants} \cap \text{TestParticipants} = \emptyset$. |
| **Filtering / Reference**| Global continuous filtering | Filter applied independently per recording file. | No cross-file filtering operations. |
| **Feature Scaling** | Test-set contamination in z-score / min-max | StandardScalers fit strictly on $\mathbf{X}_{\text{train}}$, applied to $\mathbf{X}_{\text{test}}$. | Scaler `mean_` and `scale_` computed only on train indices. |
| **Feature Selection**| Supervised selection using full cohort | ANOVA/F-statistic or Lasso fit strictly inside the training fold. | Feature mask derived strictly from fold training split. |
| **Classifier Fitting**| Contamination of test features in model fit | Estimators (`fit()`) execute strictly on training fold data. | Zero test indices passed to `fit()`. |
| **Hyperparameter Tuning**| Leaking validation folds into training | Inner cross-validation or nested cross-validation within training fold only. | No hyperparameter selection using test fold scores. |
| **Model Weight Provenance**| Pretraining on the target EEG dataset | Pretrained weights sourced strictly from ImageNet (frozen backbones). | Backbone weights are official PyTorch ImageNet checkpoints. |
| **Evaluation Bias** | Reporting only the best fold | Average $\pm$ standard deviation across all 10 folds, plus pooled out-of-fold metrics. | Full out-of-fold table published in `results/out_of_fold_predictions.csv`. |

---

## 6. Documented Random Seeds and Software Versions

- **Primary Random Seed**: `SEED = 42` for all NumPy, PyTorch, and Scikit-Learn components.
- **Python Version**: 3.12.x
- **Core Dependencies**: MNE-Python, PyTorch 2.14+, Torchvision, Scipy 1.18+, Scikit-Learn 1.9+, Pandas 3.1+.
- **Hardware Architecture**: Apple Silicon (arm64, M-series).
