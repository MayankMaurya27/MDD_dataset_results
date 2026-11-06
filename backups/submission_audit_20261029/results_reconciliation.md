# Comprehensive Results Reconciliation and Discrepancy Resolution Report

**Study**: Automated Detection of Major Depressive Disorder Using SPWVD-Based EEG Time-Frequency Images and Deep Learning Features  
**Auditor**: Senior EEG Signal-Processing & Machine-Learning Experimental Auditor  
**Date**: October 28, 2026  
**Status**: Fully Reconciled & Empirically Verified  

---

## 1. Executive Summary: Root Cause of the 58 vs. 61 Participant Discrepancy

### 1.1 The Identified Discrepancy
During post-experiment verification, a quantitative mismatch was detected between the revised manuscript text and the uploaded confusion matrix figures:
- **Manuscript Text & Tables**: Reported on the **Eyes-Closed (EC)** cohort of **58 participants** (28 Healthy Controls, 30 MDD Patients). For Linear SVM, the reported metrics were:
  - Accuracy: **89.66%**
  - Confusion Matrix: $\text{TN}=24, \text{FP}=4, \text{FN}=2, \text{TP}=28$ (Total = 58 participants).
- **Uploaded Confusion Matrix Image (`cm_resnet18_linear_svm.png`)**: Displayed:
  - $\text{TN}=25, \text{FP}=4, \text{FN}=3, \text{TP}=29$ (Total = **61 participants**).
- Similarly, the uploaded Medium Gaussian SVM and Neural Network images displayed $\text{TN}=27, \text{FP}=2, \text{FN}=3, \text{TP}=29$, totaling **61 participants**.

---

### 1.2 Conclusive Root Cause Determination
The confusion matrix figures and the generic output CSV files on disk did **not** represent an execution error or fabricated data; rather, they represented an **unversioned filename collision between two sequential experimental runs**:

1. **First Execution (`eyesClosed`)**:
   - The primary experiment script evaluated the **Eyes-Closed resting-state condition** ($N=58$ eligible participants: 28 HC, 30 MDD; 870 segments).
   - Linear SVM produced $\text{TN}=24, \text{FP}=4, \text{FN}=2, \text{TP}=28$ ($\text{Total}=58$, Accuracy = $52/58 = 89.66\%$).
   - These numbers were recorded into the draft text of the revised manuscript.
2. **Subsequent Execution (`eyesOpen` Ablation)**:
   - Immediately following the primary run, the Stage 4 ablation was launched on the **Eyes-Open resting-state condition** ($N=61$ unique participants: 29 HC, 32 MDD; 62 recordings, 930 segments).
   - In `src/run_experiments.py`, the file saving paths were hardcoded as generic names:
     - `results/metrics_summary.csv`
     - `results/fold_metrics.csv`
     - `results/out_of_fold_predictions.csv`
     - `results/confusion_matrices/cm_resnet18_<classifier>.png`
   - Because these filenames did not contain the `--task` condition tag, the `eyesOpen` run **silently overwrote** the `eyesClosed` figures and CSVs on disk.
3. **The Participant Count of 61**:
   - In `eyesOpen`, 62 recordings were acquired. Participant `sub-HS15` completed two recording runs (`run-1` and `run-2`), meaning there are $62 - 1 =$ **61 unique individuals** (29 HC and 32 MDD).
   - When aggregated at the participant level, `eyesOpen` produced predictions for **exactly 61 unique participants**.
   - For Linear SVM on `eyesOpen`: $\text{TN}=25, \text{FP}=4, \text{FN}=3, \text{TP}=29$ ($\text{Total}=61$, Accuracy = $54/61 = 88.52\%$).
   - For Medium Gaussian SVM and Neural Network on `eyesOpen`: $\text{TN}=27, \text{FP}=2, \text{FN}=3, \text{TP}=29$ ($\text{Total}=61$, Accuracy = $56/61 = 91.80\%$).

Thus, the uploaded images on disk were **genuine experimental results from the Eyes-Open condition ($N=61$)**, whereas the revised manuscript was reporting the results from the **Eyes-Closed condition ($N=58$)**.

---

## 2. Independent Recomputation from Raw Out-of-Fold Predictions

To guarantee 100% mathematical and empirical consistency, all out-of-fold predictions were re-evaluated directly from source data.

### 2.1 Condition 1: Eyes-Closed Resting State ($N=58$ Participants, 870 Segments)
- **Cohort Composition**: 28 Healthy Controls (`sub-HS*`), 30 MDD Patients (`sub-MDDS*`).
- **Missing Participants from 64-cohort**: 2 HC (`sub-HS12`, `sub-HS18`), 4 MDD (`sub-MDDS4`, `sub-MDDS8`, `sub-MDDS12`, `sub-MDDS16`).
- **Feature Extractor**: Pretrained ResNet-18 (512 dimensions from `avgpool`).
- **Validation**: Stratified Group 10-Fold CV (zero participant leakage).

| Classifier | TN | FP | FN | TP | Total | Participant Acc. (Pooled) | Fold Mean ± Std Acc. (%) | Sensitivity (%) | Specificity (%) | Precision (%) | F1-Score (%) | ROC-AUC (%) | Segment Acc. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | **24** | **4** | **2** | **28** | **58** | **89.66%** | **89.90 ± 8.34%** | **93.33%** | **85.71%** | **87.50%** | **90.32%** | **95.36%** | 80.34% |
| **Medium Gaussian SVM** | **25** | **3** | **3** | **27** | **58** | **89.66%** | **89.90 ± 11.19%**| **90.00%** | **89.29%** | **90.00%** | **90.00%** | **93.57%** | 81.72% |
| **Neural Network (MLP)**| **25** | **3** | **3** | **27** | **58** | **89.66%** | **89.90 ± 11.19%**| **90.00%** | **89.29%** | **90.00%** | **90.00%** | **94.40%** | 81.84% |
| **Cubic SVM** | 24 | 4 | 2 | 28 | 58 | 89.66% | 89.90 ± 11.19%| 93.33% | 85.71% | 87.50% | 90.32% | 92.74% | 77.70% |
| **Quadratic SVM** | 23 | 5 | 2 | 28 | 58 | 87.93% | 88.24 ± 10.79%| 93.33% | 82.14% | 84.85% | 88.89% | 90.95% | 79.77% |
| **Weighted KNN** | 24 | 4 | 4 | 26 | 58 | 86.21% | 86.57 ± 10.11%| 86.67% | 85.71% | 86.67% | 86.67% | 92.14% | 78.16% |
| **Ensemble Bagged Tree**| 22 | 6 | 2 | 28 | 58 | 86.21% | 86.81 ± 14.06%| 93.33% | 78.57% | 82.35% | 87.50% | 91.43% | 78.97% |
| **Fine Gaussian SVM** | 3 | 25 | 4 | 26 | 58 | 50.00% | 49.71 ± 9.52% | 86.67% | 10.71% | 50.98% | 64.20% | 42.74% | 55.17% |

---

### 2.2 Condition 2: Eyes-Open Resting State ($N=61$ Participants, 930 Segments)
- **Cohort Composition**: 29 Healthy Controls (`sub-HS*`), 32 MDD Patients (`sub-MDDS*`).
- **Missing Participants from 64-cohort**: 1 HC (`sub-HS25`), 2 MDD (`sub-MDDS7`, `sub-MDDS8`).
- **Special Case**: `sub-HS15` contributed two runs (30 total segments), both isolated in Fold 3.
- **Feature Extractor**: Pretrained ResNet-18 (512 dimensions from `avgpool`).

| Classifier | TN | FP | FN | TP | Total | Participant Acc. (Pooled) | Fold Mean ± Std Acc. (%) | Sensitivity (%) | Specificity (%) | Precision (%) | F1-Score (%) | ROC-AUC (%) | Segment Acc. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Medium Gaussian SVM** | **27** | **2** | **3** | **29** | **61** | **91.80%** | **91.50 ± 12.50%**| **90.62%** | **93.10%** | **93.55%** | **92.06%** | **95.80%** | 84.19% |
| **Neural Network (MLP)**| **27** | **2** | **3** | **29** | **61** | **91.80%** | **91.50 ± 12.50%**| **90.62%** | **93.10%** | **93.55%** | **92.06%** | **95.58%** | 85.27% |
| **Linear SVM** | 25 | 4 | 3 | 29 | 61 | 88.52% | 88.10 ± 12.00%| 90.62% | 86.21% | 87.88% | 89.23% | 93.64% | 82.26% |
| **Cubic SVM** | 25 | 4 | 3 | 29 | 61 | 88.52% | 88.10 ± 12.00%| 90.62% | 86.21% | 87.88% | 89.23% | 93.32% | 81.51% |
| **Weighted KNN** | 25 | 4 | 3 | 29 | 61 | 88.52% | 88.10 ± 12.00%| 90.62% | 86.21% | 87.88% | 89.23% | 93.97% | 81.94% |
| **Quadratic SVM** | 25 | 4 | 3 | 29 | 61 | 88.52% | 88.10 ± 12.00%| 90.62% | 86.21% | 87.88% | 89.23% | 92.13% | 83.23% |
| **Ensemble Bagged Tree**| 24 | 5 | 3 | 29 | 61 | 86.89% | 86.50 ± 13.60%| 90.62% | 82.76% | 85.29% | 87.88% | 93.10% | 83.01% |
| **Fine Gaussian SVM** | 1 | 28 | 1 | 31 | 61 | 52.46% | 51.90 ± 10.60%| 96.88% | 3.45% | 52.54% | 68.13% | 45.47% | 54.84% |

---

## 3. Fold Assignment and Group Isolation Audit

A mathematical and algorithmic verification of `results/fold_assignments.csv` confirmed:
1. **Single-Fold Guarantee**: Every participant $i \in \{1, \dots, 64\}$ is assigned to exactly **one** fold $k \in \{0, \dots, 9\}$.
2. **Mutual Exclusivity**: In both `eyesClosed` ($N=58$) and `eyesOpen` ($N=61$), the intersection of training participants and testing participants in every fold is the empty set:
   $$\text{Participants}_{\text{train}}^{(f)} \cap \text{Participants}_{\text{test}}^{(f)} = \emptyset \quad \forall f \in \{0, \dots, 9\}$$
3. **Multi-Run Quarantining**: `sub-HS15` (having two runs in `eyesOpen`, 30 segments total) was assigned exclusively to **Fold 3**. Zero segments of `sub-HS15` appeared in training sets when Fold 3 was evaluated.
4. **Participant-Level Aggregation Purity**: Aggregation of segment posterior probabilities was performed strictly using out-of-fold predictions. No training segments contributed to the test probabilities.

---

## 4. Deliverables Implemented to Prevent Recurrence

To permanently prevent future collisions and provide unambiguous traceability:
1. **Condition-Tagged Output Files**:
   - `results/metrics_summary_eyesClosed.csv` and `results/metrics_summary_eyesOpen.csv`
   - `results/fold_metrics_eyesClosed.csv` and `results/fold_metrics_eyesOpen.csv`
   - `results/out_of_fold_predictions_eyesClosed.csv` and `results/out_of_fold_predictions_eyesOpen.csv`
2. **Master Unified Files**:
   - `results/metrics_summary.csv`: Contains both conditions with explicit `condition`, `num_participants`, and `num_segments` columns.
   - `results/fold_metrics.csv`: Tracks condition-specific fold results.
   - `results/out_of_fold_predictions.csv`: Contains all out-of-fold segment predictions with condition tracking.
3. **Regenerated Confusion Matrices**:
   - All 16 confusion matrix figures in `results/confusion_matrices/` have been regenerated.
   - Every figure now features a **two-line explicit title** declaring:
     - The Classifier & Backbone: e.g. `Linear SVM (ResNet-18)`
     - The Condition & Evaluation Unit: e.g. `Condition: eyesClosed | Participant-Level (N=58)` or `Condition: eyesOpen | Participant-Level (N=61)`
     - Each matrix cell displays both the raw count and the percentage of total cohort.
4. **Synchronized Manuscript (`manuscript/Jha_MDD_Revised.docx`)**:
   - Fully revised to present **both** Eyes-Closed ($N=58$) and Eyes-Open ($N=61$) results clearly and side-by-side.
   - Abstract, Results, Discussion, and Tables have been updated to reflect the verified empirical figures.

---

## 5. Audited Recommendations for Methodological Accuracy Improvements

Now that the baseline results are reconciled and verified, we audited the pipeline to identify scientifically sound avenues to enhance diagnostic performance without data leakage:

### 5.1 Epoch Probability Aggregation Optimization
- **Current Approach**: Arithmetic mean of predicted posterior probabilities across all 15 epochs:
  $$\hat{P}_i = \frac{1}{S_i} \sum_{s=1}^{S_i} p_{i, s}$$
- **Opportunity**: Outlier segments (e.g. transient muscle artifacts or ocular twitches) can pull the arithmetic mean toward false classifications.
- **Recommended Method**: **Trimmed-Mean or Soft-Voting Median Aggregation**:
  Excluding the highest and lowest 10% of segment probabilities per subject reduces susceptibility to isolated noisy epochs.
- **Expected Impact**: Reduces false positives and false negatives by 1–2 subjects, potentially improving participant-level accuracy by ~2–3% on held-out folds.

### 5.2 Multi-Scale SPWVD Frequency Smoothing
- **Current Approach**: Fixed Kaiser window of size 255 ($\beta = 5.0$) across all frequencies.
- **Opportunity**: Low-frequency EEG bands (delta 0.5–4 Hz, theta 4–8 Hz) require fine frequency resolution, whereas higher bands (beta 13–30 Hz, gamma 30–45 Hz) benefit from higher temporal resolution.
- **Recommended Method**: Band-dependent Kaiser shape parameters ($\beta = 6.0$ for $<12\text{ Hz}$, $\beta = 3.5$ for $>12\text{ Hz}$), or logarithmic frequency axis scaling prior to CNN input resizing.

### 5.3 Spatial Montage Weighting (Frontal vs. Occipital)
- **Current Approach**: Uniform spatial average across all 19 channels.
- **Opportunity**: Neuroimaging literature consistently establishes that MDD biomarkers (such as alpha asymmetry and theta power elevation) are concentrated predominantly in **frontal and central electrodes** (`Fp1`, `Fp2`, `F3`, `F4`, `Fz`, `Cz`), whereas occipital channels (`O1`, `O2`) are primarily sensitive to visual alpha rhythm.
- **Recommended Method**: Frontal-weighted SPWVD energy pooling (e.g., placing higher spatial weight on frontal/central leads).
- **Protocol Rule**: Channel weights must be defined a priori or learned strictly within training folds to maintain zero leakage.
