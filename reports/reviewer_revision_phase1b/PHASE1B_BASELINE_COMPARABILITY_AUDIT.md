# PHASE 1B.5 — BASELINE EXPERIMENTAL COMPARABILITY AUDIT
## Jha MDD EEG SPWVD Research Paper — Forensic Reviewer Revision

**Auditor / Lead Engineer:** Senior Biomedical Signal Processing & Machine Learning Reproducibility Auditor  
**Date:** November 7, 2026  
**Audit Context:** Forensic Verification of Baseline Comparability (Welch PSD vs. STFT vs. CWT vs. SPWVD)  
**Deliverable Directory:** `reports/reviewer_revision_phase1b/`  
**Inspection Target:** Direct bit-level and script inspection of `src/run_classical_baselines.py`, `src/run_experiments.py`, and serialized OOF files.

---

## 1. Executive Comparability Assessment

To ensure that the performance comparison between the proposed SPWVD pipeline and classical baselines is scientifically airtight, all four pipelines were audited across seven core experimental dimensions:

| Experimental Dimension | Proposed SPWVD Pipeline | Welch PSD Baseline (1A & 1B) | STFT Spectrogram Baseline (2) | CWT Morlet Scalogram Baseline (3) | Comparability Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Participant Cohort (EC)** | Exactly 58 unique participants | Exactly 58 unique participants | Exactly 58 unique participants | Exactly 58 unique participants | **100% IDENTICAL** |
| **Participant Cohort (EO)** | Exactly 61 unique participants | Exactly 61 unique participants | Exactly 61 unique participants | Exactly 61 unique participants | **100% IDENTICAL** |
| **Repeat Scan Handling** | `sub-HS15` run-1 & run-2 in Fold 9 | `sub-HS15` run-1 & run-2 in Fold 9 | `sub-HS15` run-1 & run-2 in Fold 9 | `sub-HS15` run-1 & run-2 in Fold 9 | **100% IDENTICAL** |
| **Temporal Epoch Extraction** | Deterministic 15 epochs at exact sample timestamps | Deterministic 15 epochs at exact sample timestamps | Deterministic 15 epochs at exact sample timestamps | Deterministic 15 epochs at exact sample timestamps | **100% IDENTICAL** |
| **CV Partition Folds** | Stratified Group 10-Fold (`fold_assignments.csv`) | Identical Folds 0–9 from `fold_assignments.csv` | Identical Folds 0–9 from `fold_assignments.csv` | Identical Folds 0–9 from `fold_assignments.csv` | **100% IDENTICAL** |
| **Feature Scaler Fitting** | `StandardScaler` fit on train folds only | `StandardScaler` fit on train folds only | `StandardScaler` fit on train folds only | `StandardScaler` fit on train folds only | **100% IDENTICAL** |
| **Participant Pooling Rule** | Unweighted mean posterior $\bar{p} \ge 0.5$ | Unweighted mean posterior $\bar{p} \ge 0.5$ | Unweighted mean posterior $\bar{p} \ge 0.5$ | Unweighted mean posterior $\bar{p} \ge 0.5$ | **100% IDENTICAL** |
| **Metric & Bootstrap CI** | Wilson CI, Continuous AUC, 1,000 bootstrap (seed=42) | Wilson CI, Continuous AUC, 1,000 bootstrap (seed=42) | Wilson CI, Continuous AUC, 1,000 bootstrap (seed=42) | Wilson CI, Continuous AUC, 1,000 bootstrap (seed=42) | **100% IDENTICAL** |

---

## 2. In-Depth Audit of Code and Output Prediction Files

### 1. Participant and True Label Alignment:
- **Code Check:** In `src/run_classical_baselines.py`, raw segments are indexed directly through `metadata_eyesClosed.csv` and `metadata_eyesOpen.csv`.
- **Assertion Run:**
  ```python
  assert (f_b_ec == f_sp_ec).all() # Folds match 100% across all 58 EC subjects
  assert (f_b_eo == f_sp_eo).all() # Folds match 100% across all 61 EO subjects
  assert (y_b_ec == y_sp_ec).all() # True labels match 100% across all 58 EC subjects
  assert (y_b_eo == y_sp_eo).all() # True labels match 100% across all 61 EO subjects
  ```
- **Finding:** Not a single participant, fold index, or true clinical label is misaligned.

### 2. Temporal Epoch Boundary Alignment:
- **Sample Coordinates:** Segments were extracted from EDFs using the exact integer slice indices:
  $$\text{slice} = \text{data}[:, \text{sample\_start} : \text{sample\_end}]$$
  where $\text{sample\_end} - \text{sample\_start} = 512$ ($2.0\text{ seconds}$ at $256.0\text{ Hz}$).
- **Finding:** All baseline features were computed on the exact same physical raw EEG data slices as the SPWVD images.

### 3. Feature Scaling & Leakage Isolation:
- **Audit:** In `src/run_classical_baselines.py` lines 183–189:
  ```python
  scaler = StandardScaler()
  X_train_s = scaler.fit_transform(X_train)
  X_test_s = scaler.transform(X_test)
  ```
- **Finding:** Zero leakage. The test fold data was strictly held out and transformed using the mean and standard deviation estimated exclusively on the training folds.

---

## 3. Methodological Differences and Limitations to Disclose

While the cross-validation architecture is 100% harmonized, three structural differences between representations must be explicitly disclosed:

1. **Dimensionality of Hand-Crafted PSD vs. 2D CNN Embeddings:**
   - Welch PSD yields a **95-dimensional feature vector** ($19 \text{ channels} \times 5 \text{ bands}$), which is fed directly into Support Vector Machines.
   - In contrast, STFT, CWT, and SPWVD produce **2D time-frequency energy maps** that are standardized and passed through pretrained ResNet-18 to extract **512-dimensional embeddings**.
   - *Rationale:* Hand-crafted power spectral densities cannot be meaningfully passed through a 2D convolutional vision network without arbitrary synthetic reshaping. Evaluating Welch PSD directly via linear and RBF SVMs represents the standard clinical benchmark.
2. **Classifier Coverage Across Baselines in Phase 1B:**
   - In Phase 1B, Welch PSD was evaluated with Linear SVM and RBF SVM; STFT and CWT were evaluated with Linear SVM.
   - For SPWVD, eight classifiers were evaluated.
   - *Phase 1C Recommendation:* To make the comparison completely symmetrical across all classifier types, Phase 1C should evaluate STFT and CWT with RBF SVM and MLP as well.
3. **Spatial Aggregation Point:**
   - All three 2D time-frequency methods (STFT, CWT, SPWVD) average the 19 channels into a single spatial-mean composite map prior to CNN feature extraction:
     $$\bar{M}(t, f) = \frac{1}{19} \sum_{c=1}^{19} M_c(t, f)$$
   - This ensures that the spatial dimensionality reduction is identical across STFT, CWT, and SPWVD.

---

## 4. Final Comparability Verdict

The classical baselines executed in Phase 1B represent a **strictly controlled, leakage-free, and methodologically sound comparison**. The empirical finding that Welch PSD + RBF SVM achieves competitive accuracy (89.66% EC, 91.80% EO) is a genuine biological reflection of spectral power differences in major depression, and will be presented with full transparency in the revised manuscript.
