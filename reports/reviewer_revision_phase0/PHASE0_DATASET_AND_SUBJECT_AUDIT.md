# PHASE 0 — DATASET AND SUBJECT IDENTITY AUDIT
## Jha MDD EEG SPWVD Research Paper — Forensic Reviewer Revision

**Auditor:** Senior Biomedical Signal Processing & Machine Learning Reproducibility Auditor  
**Date:** November 7, 2026  
**Target Dataset:** HUSM MDD EEG OpenNeuro / NEMAR BIDS Archive (`nm000114_v1.0.0`)  
**Persistent DOIs:** Dataset DOI: [`10.82901/nemar.nm000114`](https://doi.org/10.82901/nemar.nm000114) | Figshare Source: [`10.6084/m9.figshare.4244171.v2`](https://doi.org/10.6084/m9.figshare.4244171.v2)  
**Primary Reference:** Mumtaz et al. (2017), *Biomedical Signal Processing and Control*, vol. 31, pp. 108–115.

---

## 1. Executive Findings on Dataset Provenance & Cohort Size

1. **Definitive Subject Count**: The authoritative filesystem and BIDS metadata confirm exactly **64 unique human participants**:
   - **30 Healthy Controls (HC)**: `sub-HS1` through `sub-HS30` (Class 0 / Negative).
   - **34 Major Depressive Disorder (MDD) Patients**: `sub-MDDS1` through `sub-MDDS34` (Class 1 / Positive).
2. **Resolution of the 34 vs. 64 Subject Contradiction**:
   - The introductory `README.md` text authored by BIDS curator Pierre Guetschel asserts: *"The dataset includes 34 participants (19 healthy controls and 15 MDD patients)"*.
   - Forensic tracing of the repository code reveals that this text was copied directly from line 3 of `code/mumtaz2016.py` (`README_CONTENT = ...`), which described an early 34-patient pilot extraction ($19 + 15 = 34$).
   - However, the actual BIDS metadata table (`participants.tsv`) contains exactly 64 rows, the filesystem contains 64 individual subject folders (`sub-HS*` and `sub-MDDS*`), and the automated MNE-BIDS summary at the footer of the README correctly states: *"The dataset consists of 64 participants... There were 181 scans in total."*
   - Therefore, the cohort size of 64 is verified from physical artifacts, and the "34 participants" string is a superseded docstring artifact.

---

## 2. Condition-Specific Recording and Participant Breakdown

The dataset comprises continuous European Data Format (`.edf`) recordings across three acquisition protocols. The available recordings vary by task:

| Condition / Task | Unique Subjects | HC Subjects | MDD Subjects | Total Recordings | Segments Extracted (15/rec) | Missing Participants (Not Recorded) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Eyes-Closed (EC)** | **58** | 28 | 30 | **58** | **870** | HC: `sub-HS12`, `sub-HS18`<br>MDD: `sub-MDDS4`, `sub-MDDS8`, `sub-MDDS12`, `sub-MDDS16` |
| **Eyes-Open (EO)** | **61** | 29 | 32 | **62** | **930** | HC: `sub-HS25`<br>MDD: `sub-MDDS7`, `sub-MDDS8` |
| **P300 Oddball (ERP)** | **61** | 28 | 33 | **61** | N/A (Not in baseline) | HC: `sub-HS20`, `sub-HS21`<br>MDD: `sub-MDDS9` |
| **Total Across Cohort** | **64** | **30** | **34** | **181** | **1,800** | Full dataset scan inventory |

### Notes on Acquisition Discrepancies:
- **Only one subject (`sub-MDDS8`) is missing from both resting-state cohorts (EC and EO).**
- All 64 participants have at least one valid EEG recording in the archive.
- The 15 segments extracted per recording yield exactly **870 segments for Eyes-Closed** ($58 \times 15$) and **930 segments for Eyes-Open** ($62 \times 15$).

---

## 3. Repeat-Session Audit: `sub-HS15` vs. README "MDD S15" Typo

The dataset documentation contains a documented naming confusion regarding duplicate recordings:
1. `README.md` states: *"One participant (MDD S15) had two separate recording sessions, resulting in duplicate recordings for this subject."*
2. **Forensic Filesystem Inspection**:
   - `sub-MDDS15/eeg/` contains only standard single recordings:
     - `sub-MDDS15_task-eyesClosed_eeg.edf`
     - `sub-MDDS15_task-eyesOpen_eeg.edf`
     - `sub-MDDS15_task-P300_eeg.edf`
   - In contrast, `sub-HS15/eeg/` contains duplicate acquisition runs:
     - `sub-HS15_task-eyesOpen_run-1_eeg.edf`
     - `sub-HS15_task-eyesOpen_run-2_eeg.edf`
     - `sub-HS15_task-eyesClosed_eeg.edf`
     - `sub-HS15_task-P300_eeg.edf`
3. **Finding**: The curator mistakenly wrote "MDD S15" in the narrative README, whereas the actual subject with duplicate runs is **Healthy Subject 15 (`sub-HS15`)** in the Eyes-Open condition.
4. **Data Isolation**: Both `run-1` and `run-2` of `sub-HS15` generate 15 segments each (total 30 segments). In the cross-validation partition (`results/fold_assignments.csv`), `sub-HS15` is strictly assigned to **Fold 9**, preventing any train/test leakage across the two runs.

---

## 4. Subject Identity Key and Collision Analysis

A critical risk in multi-cohort biomedical datasets is subject ID collision:
- **Raw Identifier Collisions**: Both classes use numeric suffixes from 1 to 30 (`1` through `30`). If prefixes are stripped (e.g., using integer `15`), healthy participant `15` collides with MDD participant `15`.
- **Global Key Enforcement**: In all project scripts (`preprocessing.py`, `extract_features.py`, `generate_folds.py`, `run_experiments.py`), the globally unique key is strictly enforced as the **full BIDS participant identifier** (`sub-HS1`..`sub-HS30` and `sub-MDDS1`..`sub-MDDS34`).
- **Audit Verification**: No collision exists in any table or metadata file. `fold_assignments.csv`, `metadata_eyesClosed.csv`, `metadata_eyesOpen.csv`, and `out_of_fold_predictions_*.csv` all use string-matched full identifiers.

---

## 5. Demographics, Ethics, and Clinical Covariate Completeness

1. **Authoritative Table (`participants.tsv`) Audit**:
   - Columns present: `participant_id`, `age`, `sex`, `hand`, `weight`, `height`.
   - **Value Audit**: Every single data cell for `age`, `sex`, `hand`, `weight`, and `height` across all 64 rows contains the string **`n/a`**.
   - **Finding**: The BIDS distribution does not provide machine-readable participant-level demographic or physical covariates.
2. **Clinical Severity & Medication**:
   - No Hamilton Depression Rating Scale (HAM-D), Beck Depression Inventory (BDI), or medication status (washout vs. medicated) is included in BIDS metadata files.
3. **Literature Grounding for Demographics**:
   - The original publication (Mumtaz et al., 2017) reports that the clinical cohort comprised 34 MDD outpatients (mean age $40.3 \pm 12.9$ years; 17 males, 17 females) diagnosed according to DSM-IV criteria, and 30 healthy controls (mean age $38.3 \pm 15.6$ years; 15 males, 15 females).
   - In the Jha manuscript, general demographic statements ("aged 12 to 77 years") were derived from historical Mumtaz notes. The manuscript correctly notes that patient-level demographic stratification cannot be executed due to the absence of per-subject demographic keys in `participants.tsv`.
4. **Ethics Approval and Data-Use Licensing**:
   - **Acquisition Center**: Hospital Universiti Sains Malaysia (HUSM), Kota Bharu, Kelantan, Malaysia.
   - **Ethics Committee**: Approved by the Human Research Ethics Committee of Universiti Sains Malaysia (USM) (Ethics protocol code: `USM/JEPeM/1401016`).
   - **Consent**: Written informed consent was obtained from all participants (or legal guardians for minors).
   - **Open License**: Creative Commons Attribution 4.0 International (CC-BY-4.0), permitting open academic reuse with appropriate attribution.

