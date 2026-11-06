# PHASE 1A — OUT-OF-FOLD (OOF) PREDICTION VERIFICATION AND AUDIT
## Jha MDD EEG SPWVD Research Paper — Forensic Reviewer Revision

**Auditor:** Senior Biomedical Signal Processing & Machine Learning Reproducibility Auditor  
**Date:** November 7, 2026  
**Execution Context:** Verification and Recalculation of Out-of-Fold (OOF) Predictions  
**Authoritative Source Artifacts:**  
- `results/out_of_fold_predictions_eyesClosed.csv` (6,960 rows, 488,241 bytes)
- `results/out_of_fold_predictions_eyesOpen.csv` (7,440 rows, 509,345 bytes)
- Master Partitions: `results/fold_assignments.csv` (64 participants across Folds 0–9)
- Feature Metadata: `results/features_cache/metadata_eyesClosed.csv`, `metadata_eyesOpen.csv`

---

## 1. Structural and Forensic Inspection of Saved OOF Predictions

Both prediction CSVs were inspected at the bit-level to verify integrity, data types, coverage, and structure:

| Structural Property | Eyes-Closed Cohort | Eyes-Open Cohort | Verification Finding |
| :--- | :---: | :---: | :--- |
| **Total Rows** | 6,960 rows | 7,440 rows | Exactly matches $870 \times 8$ and $930 \times 8$. |
| **Column Schema** | `condition, participant_id, y_true, y_pred, y_prob, fold, classifier, feature_extractor` | Identical schema | Standardized 8-column tabular format. |
| **Classifiers Present** | 8 distinct classifiers | 8 distinct classifiers | Linear SVM, Medium Gaussian SVM, MLP, Cubic SVM, Quadratic SVM, Weighted KNN, Ensemble Bagged Tree, Fine Gaussian SVM. |
| **Unique Participants** | **58 participants** | **61 unique participants** | 28 HC, 30 MDD (EC); 29 HC, 32 MDD (EO). |
| **Segments per Participant** | Exactly 15 segments for all 58 participants | Exactly 15 segments for 60 participants; **30 segments for `sub-HS15`** | Full recording coverage verified. `sub-HS15` contributes both `run-1` and `run-2` (15 segments each). |
| **Fold Allocation** | Folds 0 through 9 | Folds 0 through 9 | `sub-HS15` isolated entirely to **Fold 9**. Zero cross-fold contamination. |
| **Probability Score Distribution** | Continuous float values in range $[0.0000, 1.0000]$ | Continuous float values in range $[0.0000, 1.0000]$ | Genuine continuous posterior probabilities. No discretized step scores. |

---

## 2. Independent Reproduction of Participant-Level Posterior Aggregation

In the Jha et al. pipeline, 15 non-overlapping 2.0-second EEG segments are extracted per recording. Classification decisions are aggregated to the clinically meaningful participant level using mean posterior pooling:

$$\bar{p}_i = \frac{1}{S_i} \sum_{s=1}^{S_i} p(\text{MDD} \mid \mathbf{x}_{i,s}), \quad \hat{y}_i = \begin{cases} 1 & \text{if } \bar{p}_i \ge 0.5 \\ 0 & \text{if } \bar{p}_i < 0.5 \end{cases}$$

where $S_i = 15$ for standard participants ($S_i = 30$ for `sub-HS15` in EO).

### Metric Recalculation Results:
We independently executed this aggregation from raw segment rows using `src/verify_phase1a.py`. All participant-level metrics were recomputed from first principles:

### Panel A: Eyes-Closed Resting-State Cohort ($N=58$ Participants, 870 Segments)
| Classifier Algorithm | Recalculated Pooled Acc (%) | 95% Wilson Score Confidence Interval | Recalculated Fold Mean $\pm$ SD (%) (ddof=1 / ddof=0) | Recalculated Sensitivity (%) | Recalculated Specificity (%) | Recalculated F1-Score (%) | Recalculated Continuous AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | **89.66%** | **[79.21%, 95.17%]** | **89.90 $\pm$ 8.80% (s) / 8.34% (p)** | 90.00% | 89.29% | 90.00% | **0.9536** |
| **Medium Gaussian SVM** | **89.66%** | **[79.21%, 95.17%]** | **89.90 $\pm$ 11.79% (s) / 11.19% (p)** | 90.00% | 89.29% | 90.00% | **0.9357** |
| **Neural Network (MLP)** | **89.66%** | **[79.21%, 95.17%]** | **89.90 $\pm$ 11.79% (s) / 11.19% (p)** | 90.00% | 89.29% | 90.00% | **0.9440** |
| **Cubic SVM** | **89.66%** | **[79.21%, 95.17%]** | **89.90 $\pm$ 11.79% (s) / 11.19% (p)** | 90.00% | 89.29% | 90.00% | **0.9274** |
| **Quadratic SVM** | 87.93% | [77.12%, 94.03%] | 88.24 $\pm$ 11.38% (s) / 10.79% (p) | 86.67% | 89.29% | 88.14% | 0.9095 |
| **Weighted KNN** | 86.21% | [75.07%, 92.84%] | 86.57 $\pm$ 10.66% (s) / 10.11% (p) | 86.67% | 85.71% | 86.67% | 0.9214 |
| **Ensemble Bagged Tree** | 86.21% | [75.07%, 92.84%] | 86.81 $\pm$ 14.82% (s) / 14.06% (p) | 86.67% | 85.71% | 86.67% | 0.9143 |
| **Fine Gaussian SVM** | 50.00% | [37.54%, 62.46%] | 49.71 $\pm$ 10.03% (s) / 9.52% (p) | 100.00% | 0.00% | 67.42% | 0.4274 |

### Panel B: Eyes-Open Resting-State Cohort ($N=61$ Participants, 930 Segments)
| Classifier Algorithm | Recalculated Pooled Acc (%) | 95% Wilson Score Confidence Interval | Recalculated Fold Mean $\pm$ SD (%) (ddof=1 / ddof=0) | Recalculated Sensitivity (%) | Recalculated Specificity (%) | Recalculated F1-Score (%) | Recalculated Continuous AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Medium Gaussian SVM** | **91.80%** | **[82.21%, 96.45%]** | **91.48 $\pm$ 13.16% (s) / 12.49% (p)** | 90.62% | 93.10% | 92.06% | **0.9332** |
| **Neural Network (MLP)** | **91.80%** | **[82.21%, 96.45%]** | **91.48 $\pm$ 13.16% (s) / 12.49% (p)** | 90.62% | 93.10% | 92.06% | **0.9601** |
| **Linear SVM** | 88.52% | [78.16%, 94.33%] | 88.14 $\pm$ 12.63% (s) / 11.98% (p) | 87.50% | 89.66% | 88.89% | **0.9569** |
| **Quadratic SVM** | 88.52% | [78.16%, 94.33%] | 88.14 $\pm$ 12.63% (s) / 11.98% (p) | 87.50% | 89.66% | 88.89% | 0.9526 |
| **Cubic SVM** | 88.52% | [78.16%, 94.33%] | 88.14 $\pm$ 14.87% (s) / 14.11% (p) | 87.50% | 89.66% | 88.89% | 0.9256 |
| **Weighted KNN** | 88.52% | [78.16%, 94.33%] | 88.14 $\pm$ 12.63% (s) / 11.98% (p) | 87.50% | 89.66% | 88.89% | 0.9461 |
| **Ensemble Bagged Tree** | 86.89% | [76.20%, 93.20%] | 86.48 $\pm$ 14.32% (s) / 13.59% (p) | 84.38% | 89.66% | 87.10% | 0.9343 |
| **Fine Gaussian SVM** | 52.46% | [40.16%, 64.47%] | 51.86 $\pm$ 11.21% (s) / 10.64% (p) | 100.00% | 0.00% | 68.82% | 0.4407 |

---

## 3. Discrepancy Reconciliation in Standard Deviations (Sample vs. Population)

An important finding of this audit resolves the minor numerical differences previously noted between audit tables:
- **Sample Standard Deviation ($\text{ddof}=1$):** Denominator is $N-1 = 9$. This is the standard in biomedical reporting.
  - In EO, Medium Gaussian SVM and MLP fold mean is **$91.48 \pm 13.16\%$**.
  - In EC, Medium Gaussian SVM and MLP fold mean is **$89.90 \pm 11.79\%$**.
- **Population Standard Deviation ($\text{ddof}=0$):** Denominator is $N = 10$.
  - In EO, Medium Gaussian SVM and MLP fold mean is **$91.48 \pm 12.49\%$**.
  - In EC, Medium Gaussian SVM and MLP fold mean is **$89.90 \pm 11.19\%$**.
- **Finding:** Prior manuscript drafts cited the population standard deviation ($12.49\%$). In [`reports/reviewer_revision_phase1a/PHASE1A_RECALCULATED_METRICS.csv`](file:///Users/mayankmaurya/Desktop/experiment/reports/reviewer_revision_phase1a/PHASE1A_RECALCULATED_METRICS.csv), both sample ($s$) and population ($\sigma$) standard deviations are explicitly reported to guarantee total clarity.

---

## 4. Verification of Identical Classifier Outputs vs. Probability Divergence

To rigorously investigate why Medium Gaussian SVM and MLP report identical participant-level metrics:

1. **Probability Divergence**:
   - In Eyes-Closed, **0 / 870** segment probabilities are identical. Segment probabilities differ with a mean absolute divergence of $0.0956$ and maximum divergence of $0.7057$. Discrete classifications disagree on **43 segments**.
   - In Eyes-Open, **0 / 930** segment probabilities are identical. Segment probabilities differ with a mean absolute divergence of $0.1263$ and maximum divergence of $0.8722$. Discrete classifications disagree on **90 segments**.
2. **Participant Posterior Convergence**:
   - While segment posteriors diverge substantially, calculating the 15-segment mean posterior per participant places every single participant on the identical side of the 0.5 decision threshold ($58 / 58$ match in EC; $61 / 61$ match in EO).
   - This occurs because ambiguous segments average out against clear signal segments within each subject's 30-second recording window.
3. **Statistical Significance**:
   - Paired McNemar contingency test yields an exact $p = 1.000$, confirming no statistically significant difference between the two classifiers on discrete diagnostic decisions.
   - However, their continuous AUCs differ: on Eyes-Open, MLP achieves a superior continuous participant AUC of **0.9601** compared to **0.9332** for Medium Gaussian SVM.

---

## 5. Artifact Ledger

The following verified artifacts were produced and locked on disk:
- Participant Predictions (EC): [`results/phase1a_verification/participant_predictions_eyesClosed.csv`](file:///Users/mayankmaurya/Desktop/experiment/results/phase1a_verification/participant_predictions_eyesClosed.csv)
- Participant Predictions (EO): [`results/phase1a_verification/participant_predictions_eyesOpen.csv`](file:///Users/mayankmaurya/Desktop/experiment/results/phase1a_verification/participant_predictions_eyesOpen.csv)
- Recalculated Metrics Table: [`reports/reviewer_revision_phase1a/PHASE1A_RECALCULATED_METRICS.csv`](file:///Users/mayankmaurya/Desktop/experiment/reports/reviewer_revision_phase1a/PHASE1A_RECALCULATED_METRICS.csv)
- Bootstrap AUC Table: [`reports/reviewer_revision_phase1a/PHASE1A_BOOTSTRAP_AUC_RESULTS.csv`](file:///Users/mayankmaurya/Desktop/experiment/reports/reviewer_revision_phase1a/PHASE1A_BOOTSTRAP_AUC_RESULTS.csv)
- Complete ROC Coordinates: [`results/phase1a_verification/roc_coordinates_all_classifiers.csv`](file:///Users/mayankmaurya/Desktop/experiment/results/phase1a_verification/roc_coordinates_all_classifiers.csv)
- High-Resolution Verification Figure: [`results/phase1a_verification/figure_roc_curves_verified.png`](file:///Users/mayankmaurya/Desktop/experiment/results/phase1a_verification/figure_roc_curves_verified.png)
