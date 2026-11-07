# Phase 1C — Entry Gate Verification and Blocker Resolution Report
## Jha MDD EEG SPWVD Research Paper — Controlled Experimental Execution

**Workspace:** `/Users/mayankmaurya/Desktop/experiment`  
**Date:** November 2026  
**Auditor / Roles:** Senior EEG Research Scientist, ML Experimentation Engineer, Independent Reproducibility Auditor  
**Verification Suite:** [`tests/test_calibration_and_leakage.py`](file:///Users/mayankmaurya/Desktop/experiment/tests/test_calibration_and_leakage.py)  
**Status:** ALL 3 ENTRY BLOCKERS RESOLVED — GATE STATUS: PASS  

---

## 1. Repository and Workspace Baseline Verification

Before initiating Phase 1C experimentation, a forensic inspection of the physical repository was conducted:

| Item | Expected Baseline | Actual Verified State | Verification Method | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Git Branch** | `main` | `main` | `git status` | **PASS** |
| **Active HEAD Commit** | `7dad01f` | `7dad01f` | `git rev-parse HEAD` | **PASS** |
| **Direct Parent Commit** | `c831f80` | `c831f80` | `git log -n 2 --oneline` | **PASS** |
| **Working Tree** | Clean (0 uncommitted changes) | Clean (0 uncommitted changes) | `git status --porcelain` | **PASS** |
| **Commit Lineage** | `00a08df` $\to$ `d5dac42` $\to$ `baeb462` $\to$ `c831f80` $\to$ `7dad01f` | Confirmed linear history | `git log --graph --oneline` | **PASS** |
| **Locked Manuscript** | `manuscript/Jha_MDD_Revised_Final.docx` | 563,523 bytes; SHA-256: `aa1acf63b1bc680e...` | `hashlib.sha256` | **LOCKED** |
| **Mirror Manuscript** | `manuscript/Jha_MDD_Revised.docx` | 563,523 bytes; SHA-256: `aa1acf63b1bc680e...` | `hashlib.sha256` | **LOCKED** |
| **Historical EC OOF** | `results/out_of_fold_predictions_eyesClosed.csv` | 6,960 rows (870 segs $\times$ 8 clfs) | `pandas` row count | **PASS** |
| **Historical EO OOF** | `results/out_of_fold_predictions_eyesOpen.csv` | 7,440 rows (930 segs $\times$ 8 clfs) | `pandas` row count | **PASS** |

---

## 2. Resolution of the Three Entry Blockers

### Blocker A — Git Provenance and Active Baseline
- **Audit Findings:** Commit `7dad01f` (`docs(revision): complete Phase 1B.7 final consistency verification and automated test suite`) is verified via Git as the authoritative HEAD of branch `main`.
- **Predecessor Relationship:** Commit `c831f80` is the direct ancestor and parent commit of `7dad01f`. Prior commits `baeb462` (Phase 1B.5 audit), `d5dac42` (Phase 1B implementation), and `00a08df` (Phase 0/1A audit) form an unbroken linear commit chain on `origin/main`.
- **Verdict on Blocker A:** **RESOLVED (PASS).**

---

### Blocker B — Welch Feature Dimensionality Resolution
Earlier documentation contained an ambiguity regarding the Welch Power Spectral Density (PSD) feature dimension, mentioning `D = 95 or D = 855`.

A forensic inspection of the reference implementation in [`src/run_classical_baselines.py`](file:///Users/mayankmaurya/Desktop/experiment/src/run_classical_baselines.py#L81-L101) resolves this ambiguity definitively:
- **Number of Channels:** 19 channels (Standard International 10–20: $F_{p1}, F_3, C_3, P_3, O_1, F_7, T_3, T_5, F_z, F_{p2}, F_4, C_4, P_4, O_2, F_8, T_4, T_6, C_z, P_z$).
- **Frequency Range and Windowing:** Welch periodogram computed with `nperseg = 256` samples ($1.0\text{ s}$ window at $fs = 256\text{ Hz}$) and $50\%$ overlap (`noverlap = 128`), producing 129 frequency bins ($0.0 - 128.0\text{ Hz}$ at $1.0\text{ Hz}$ spacing).
- **Canonical Frequency Bands:**
  - $\delta$ (Delta): $0.5 - 4.0\text{ Hz}$
  - $\theta$ (Theta): $4.0 - 8.0\text{ Hz}$
  - $\alpha$ (Alpha): $8.0 - 13.0\text{ Hz}$
  - $\beta$ (Beta): $13.0 - 30.0\text{ Hz}$
  - $\gamma$ (Gamma): $30.0 - 45.0\text{ Hz}$
- **Feature Aggregation:** For each of the 19 channels, spectral power is integrated across each band using trapezoidal integration (`np.trapezoid(psd[:, :, idx_band], f[idx_band], axis=-1)`), yielding 5 scalar band-power values per channel.
- **Log Transformation:** Channel features are concatenated across the 5 bands ($5 \times 19 = 95$) and log-transformed: $\mathbf{x}_{\text{welch}} = \log_{10}(\max(\mathbf{p}_{\text{bands}}, 10^{-12}))$.
- **Exact Input Dimension:** The input dimension passed to every classifier is **$D = 95$**.
- *(Note on $D = 855$: The alternative $D = 855$ corresponds to $19 \times 45$ un-banded 1-Hz bin densities, which was an exploratory consideration in drafting, but the actual validated codebase strictly utilizes the 5 physiological bands $D = 95$.)*
- **Verdict on Blocker B:** **RESOLVED (PASS). Frozen at $D = 95$.**

---

### Blocker C — Leakage-Free Probability Calibration for SVMs
- **Vulnerability Identified in Default Scikit-Learn:**
  Standard `sklearn.svm.SVC(probability=True)` uses Platt scaling by running an internal 5-fold cross-validation on the rows of the training set. However, scikit-learn's internal Platt CV uses standard `StratifiedKFold`, which splits samples randomly. When multiple epochs (15 per recording) originate from the same participant, standard `SVC(probability=True)` leaks segments from the same participant across internal calibration folds.
- **Engineered Solution:**
  Implemented [`src/participant_calibrated_svm.py`](file:///Users/mayankmaurya/Desktop/experiment/src/participant_calibrated_svm.py) with class `ParticipantCalibratedSVM`.
  1. Base estimator `SVC(probability=False)` trains hyperplane parameters $(\mathbf{w}, b)$ without row-level calibration.
  2. Platt scaling calibration is wrapped via `CalibratedClassifierCV(estimator=base_estimator, method='sigmoid', cv=cv_splits)` where `cv_splits` is explicitly generated by `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` using participant IDs as `groups`.
  3. Every calibration fold guarantees that $\text{train\_participants} \cap \text{calibration\_participants} = \emptyset$.
- **Nested CV Anti-Leakage Safeguards:**
  - **Inner Validation Loops:** `StandardScaler` and `ParticipantCalibratedSVM` are fitted strictly on inner-training participants. Inner-validation participants are held out and evaluated only after fitting is complete.
  - **Outer Fold Loops:** When the winning pipeline is retrained on the full outer training set, calibration splits are generated strictly within outer-training participants. Outer-test participants remain untouched until final out-of-fold scoring.
- **Automated Verification:** Verified in `tests/test_calibration_and_leakage.py` (Test 1 and Test 2).
- **Verdict on Blocker C:** **RESOLVED (PASS).**

---

## 3. Automated Entry Gate Test Results

The automated test script [`tests/test_calibration_and_leakage.py`](file:///Users/mayankmaurya/Desktop/experiment/tests/test_calibration_and_leakage.py) was executed with the following verified output:

```text
======================================================================
          PHASE 1C ENTRY GATE AUTOMATED TEST SUITE                   
======================================================================
--- [Test 1/5] ParticipantCalibratedSVM Calibration Group Isolation ---
  ✓ ParticipantCalibratedSVM fits and outputs valid calibrated probabilities.
  ✓ Zero intra-participant calibration leakage confirmed.
--- [Test 2/5] Nested CV Inner and Outer Participant Isolation ---
  ✓ Inner CV: Preprocessing, scaling, training, and calibration strictly use inner train.
  ✓ Outer CV: Outer test participants strictly quarantined until final evaluation.
--- [Test 3/5] Welch PSD Feature Dimensionality Verification ---
  ✓ Channels: 19 (Standard 10-20).
  ✓ Frequency bands: 5 (['delta', 'theta', 'alpha', 'beta', 'gamma']).
  ✓ Integration: Trapezoidal band-power integration + log10 transform.
  ✓ Exact classifier input dimensionality confirmed: D = 95 (concatenated).
--- [Test 4/5] Historical Prediction Files Invariance ---
  ✓ EC OOF predictions intact: 6960 rows across 8 classifiers.
  ✓ EO OOF predictions intact: 7440 rows across 8 classifiers.
--- [Test 5/5] Locked Manuscript File Invariance ---
  ✓ manuscript/Jha_MDD_Revised_Final.docx: 563523 bytes, SHA-256: aa1acf63b1bc680e...
  ✓ manuscript/Jha_MDD_Revised.docx: 563523 bytes, SHA-256: aa1acf63b1bc680e...
  ✓ Master manuscript is locked and completely unaltered.
======================================================================
  ALL 5 ENTRY GATE TESTS PASSED WITH ZERO LEAKAGE DETECTED.
======================================================================
```

---

## 4. Entry Gate Decision

```
══════════════════════════════════════════════════════════════════════════
                     PHASE 1C ENTRY GATE VERDICT: PASS
══════════════════════════════════════════════════════════════════════════

  BLOCKER A (GIT PROVENANCE):      RESOLVED — HEAD AT 7dad01f, CLEAN TREE
  BLOCKER B (WELCH DIMENSION):     RESOLVED — FROZEN AT D = 95
  BLOCKER C (SVM CALIBRATION):     RESOLVED — ParticipantCalibratedSVM LOCKED
  INVARIANCE AUDIT:                HISTORICAL OOF AND MANUSCRIPT LOCKED
  AUTOMATED TESTS:                 5 / 5 TESTS PASSED WITH ZERO ERRORS

  AUTHORIZATION:                   >>> PROCEED TO MODULE M1 <<<

══════════════════════════════════════════════════════════════════════════
```

*Proceeding to Module M1 (Signal-Quality and Artifact-Handling Validation).*

