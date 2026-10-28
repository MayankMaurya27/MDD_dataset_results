# Data Quality and Integrity Report

**Dataset**: MDD Patients and Healthy Controls EEG Data (`nm000114_v1.0.0`)  
**Auditor**: Senior EEG Signal-Processing & Experimental Auditor  
**Date**: October 28, 2026  

---

## 1. Overview of Data Quality Assessment

A rigorous data-quality audit was performed on all 181 European Data Format (EDF) EEG files across all 64 participants. The audit evaluated:
1. File format compliance and binary integrity.
2. Channel consistency, labeling standardization, and montage uniformity.
3. Temporal properties: sampling rate consistency, record duration, and epoch completeness.
4. Missing data, missing conditions, and repeated recordings.
5. Inherent signal filtering, noise characteristics, and artifact considerations.

---

## 2. Integrity and Binary Health

| Quality Dimension | Test Performed | Result | Details |
| :--- | :--- | :--- | :--- |
| **Header Validity** | Binary parsing of 256-byte main EDF header | **100% Pass** (181/181) | All records comply with standard EDF specifications. |
| **Channel Header Validity** | Binary parsing of channel sub-headers | **100% Pass** (181/181) | Physical dimensions ($\mu\text{V}$), digital/physical min/max match expectations. |
| **Data Record Continuity** | Data record block size verification | **100% Pass** (181/181) | No truncated records or missing blocks detected. |
| **Corrupt Files** | Check for unreadable/zero-byte files | **0 Corrupt Files** | Smallest file is 7,858,688 bytes (~7.86 MB). |

---

## 3. Channel Montage and Homogeneity Analysis

A critical requirement for automated machine learning and deep learning pipelines is spatial uniformity across participants and sessions.

### Findings:
1. **Core 19-Channel Montage**:
   The standard 10–20 international system 19 EEG electrodes are present in all 181 recordings:
   - Frontal: `Fp1`, `Fp2`, `F7`, `F3`, `Fz`, `F4`, `F8`
   - Central: `C3`, `Cz`, `C4`
   - Temporal: `T3`, `T4`, `T5`, `T6`
   - Parietal: `P3`, `Pz`, `P4`
   - Occipital: `O1`, `O2`
2. **Reference Channel**:
   - `EEG A2-A1` is present in all 181 recordings as a linked-ear derivation.
3. **Auxiliary / Non-standard Channels**:
   - `EEG 23A-23R` and `EEG 24A-24R` appear in **112 recordings** (Config 1) and are **absent in 69 recordings** (Config 0).
   - In accordance with BIDS conversion notes, these were peripheral bipolar electrooculographic / electromyographic sensors.
   - **Quality Recommendation**: Exclude `23A-23R`, `24A-24R`, `A2-A1`, and `EDF Annotations` from the machine-learning pipeline, restricting the spatial feature representation strictly to the 19 standard EEG electrodes. This guarantees 100% montage parity across all subjects and conditions without requiring synthetic channel interpolation.

---

## 4. Sampling Frequency and Temporal Stability

- **Sampling Rate ($f_s$)**: Exactly **256.0 Hz** for every channel in every recording ($1\text{ sample} = 3.90625\text{ ms}$).
- **Record Duration**:
  - Minimum: 180.0 seconds (3.0 minutes).
  - Maximum: 686.0 seconds (11.43 minutes).
  - Mean: 408.84 seconds (~6.81 minutes).
  - Standard Deviation: 155.23 seconds.
- **Implication for Segmentation**:
  Every file contains at least 180 seconds ($46,080$ time samples per channel). Segmenting into 5-second, 10-second, or 20-second non-overlapping or overlapping windows will yield consistent sample counts across all participants.

---

## 5. Completeness of Experimental Conditions

Not every participant completed all three experimental tasks:
- **Eyes Open (EO)**: 62 recordings (30 HC, 32 MDD).
  - Missing in MDD: `sub-MDDS7`, `sub-MDDS8`.
  - Present in all 30 HC.
- **Eyes Closed (EC)**: 58 recordings (28 HC, 30 MDD).
  - Missing in HC: `sub-HS12`, `sub-HS18`.
  - Missing in MDD: `sub-MDDS4`, `sub-MDDS8`, `sub-MDDS12`, `sub-MDDS16`.
- **P300 Task (Auditory Oddball)**: 61 recordings (28 HC, 33 MDD).
  - Missing in HC: `sub-HS20`, `sub-HS21`.
  - Missing in MDD: `sub-MDDS9`.
- **Special Case (Repeated Runs)**:
  - `sub-HS15`: Contains two runs of Eyes Open (`run-1` and `run-2`), recorded in the same session. In participant-grouped validation, both runs belong to `sub-HS15` and must remain strictly within `sub-HS15`'s fold.

---

## 6. Preprocessing and Artifact Considerations

According to the original acquisition protocol (Mumtaz et al., 2017) and BIDS curator documentation:
1. **Hardware Filtering**:
   - High-pass analog filter: 0.5 Hz (suppresses DC drift and sweat potentials).
   - Low-pass analog filter: 70.0 Hz (anti-aliasing).
   - Power-line notch filter: 50.0 Hz (attenuates Malaysian mains electricity interference).
2. **Acquisition Quality**:
   - All electrode impedances were maintained below 5 k$\Omega$ during recording.
   - Recordings deposited in Figshare were selected segments where excessive gross oculographic and myographic movements were already pruned during initial acquisition.
3. **Pipeline Recommendations**:
   - To prevent edge artifacts and spectral leakage in time-frequency analysis (SPWVD), continuous recordings should be filtered if additional bandpass filtering is applied before window segmentation.
   - Re-referencing: Common Average Reference (CAR) or preservation of the original linked-ears (`LE`) reference should be explicitly documented.
   - Segment boundaries must never cross recording files.

---

## 7. Usable Dataset Definition for Machine Learning

| Cohort | Total Enrolled | Verified Usable | Exclusion Reason |
| :--- | :---: | :---: | :--- |
| **Healthy Controls (HC)** | 30 | **30** | None. All 30 have at least 2 usable conditions. |
| **MDD Patients** | 34 | **34** | None. All 34 have at least 1 usable condition (`sub-MDDS8` has P300; 33 have $\ge 2$ conditions). |
| **Total Participants** | **64** | **64** | **0 Excluded** |

### Condition-Specific Subsets:
- If evaluating **Eyes-Closed (EC)** only: 58 participants (28 HC, 30 MDD).
- If evaluating **Eyes-Open (EO)** only: 62 participants (30 HC, 32 MDD).
- If evaluating **All Conditions Combined**: 64 participants (30 HC, 34 MDD; 181 recordings).
