# Comprehensive Dataset Audit Report

**Dataset**: MDD Patients and Healthy Controls EEG Data (NEMAR `nm000114_v1.0.0`)  
**Auditor**: Senior EEG Signal-Processing & Machine-Learning Research Auditor  
**Date**: October 28, 2026  
**Status**: Verified & Audited  

---

## 1. Executive Summary

This audit rigorously evaluates the electroencephalogram (EEG) dataset used in the manuscript *"Automated Detection of Major Depressive Disorder Using SPWVD-Based EEG Time-Frequency Images and Deep Learning Features"*. The dataset is stored in BIDS format (`nm000114_v1.0.0`) derived from the original Figshare repository published by Mumtaz et al. (2016).

All 181 European Data Format (EDF) files across 64 participant directories were inspected for binary header integrity, channel consistency, sampling rate uniformity, recording durations, task allocations, and diagnostic metadata. 

### Key Audit Findings:
1. **Total Usable Participants**: **64 unique individuals** (30 Healthy Controls, 34 MDD Patients).
2. **Total Recordings**: **181 complete, uncorrupted EDF scans**.
3. **Sampling Frequency**: **256.0 Hz** uniformly across all 181 recordings.
4. **Usable EEG Montage**: The 19 standard 10–20 international system EEG channels (`Fp1`, `F3`, `C3`, `P3`, `O1`, `F7`, `T3`, `T5`, `Fz`, `Fp2`, `F4`, `C4`, `P4`, `O2`, `F8`, `T4`, `T6`, `Cz`, `Pz`) are present in **100% of the recordings**.
5. **Data Quality**: **Zero corrupt files**; all 181 EDF headers and signal arrays parse successfully without data truncation.

---

## 2. Resolution of the Participant Count Discrepancy (34 vs. 64)

### The Reported Contradiction
- In `nm000114_v1.0.0/README.md` (and the conversion script `code/mumtaz2016.py` line 3), the introductory text states:
  > *"The dataset includes 34 participants (19 healthy controls and 15 MDD patients) with recordings during eyes-closed rest, eyes-open rest, and an auditory oddball P300 task."*
- In contrast, the Jha et al. manuscript claims:
  > *"The data were obtained from the EEG of 34 depressed and 30 healthy subjects aged 12 to 77 years at the Hospital Universiti Sains Malaysia (HUSM)."* (Total: 64 participants).
- In addition, the automated MNE-BIDS report at the footer of `README.md` states:
  > *"The dataset consists of 64 participants... There were 181 scans in total."*

### Verified Finding
The discrepancy is definitively caused by a **typographical error in the hardcoded introduction string written by the BIDS conversion curator (Pierre Guetschel)** in `code/mumtaz2016.py`:
1. When Pierre Guetschel authored the BIDS conversion wrapper script (`code/mumtaz2016.py`), he hardcoded a docstring (`README_CONTENT`) containing the phrase *"34 participants (19 healthy controls and 15 MDD patients)"*. Notice that $19 + 15 = 34$, which was likely copied from an earlier pilot study or subgroup analysis in the literature.
2. However, the actual raw source dataset supplied to the converter contained 64 subjects:
   - 30 healthy control directories: `sub-HS1` through `sub-HS30`
   - 34 MDD patient directories: `sub-MDDS1` through `sub-MDDS34`
3. The dataset's authoritative `participants.tsv` file lists **exactly 64 rows** (plus header): 30 `sub-HS*` and 34 `sub-MDDS*`.
4. The automated report generated directly from filesystem inspection by `mne_bids.make_report()` confirms:
   > *"The dataset consists of 64 participants... There were 181 scans in total."*
5. Therefore, the manuscript's claim of **64 participants (34 MDD and 30 Healthy Controls)** is **correct** and matches the actual data on disk. The text in the README introduction was an uncorrected curator draft note.

---

## 3. Dataset Provenance, Version, and License

| Property | Verified Value | Source / Traceability |
| :--- | :--- | :--- |
| **Dataset Title** | MDD Patients and Healthy Controls EEG Data | `dataset_description.json` |
| **Dataset Version** | 1.0.0 (BIDS 1.7.0 derivative) | `dataset_description.json` |
| **Dataset DOI** | `10.82901/nemar.nm000114` | NEMAR / OpenNeuro derivative |
| **Source Data DOI** | `10.6084/m9.figshare.4244171.v2` | Figshare (Mumtaz, 2016) |
| **License** | Creative Commons Attribution 4.0 International (CC-BY-4.0) | `dataset_description.json` |
| **Original Acquisition Site**| Hospital Universiti Sains Malaysia (HUSM) | Mumtaz et al. (2017) |
| **Acquisition Authors** | Wajid Mumtaz, Likun Xia, Syed Saad Azhar Ali, Mohd Azhar Mohd Yasin, Mazhar Hussain, Aamir Saeed Malik | Biomedical Signal Processing and Control (2017) |
| **BIDS Conversion Curator**| Pierre Guetschel | MNE-BIDS 0.17.0 / NEMAR |

---

## 4. Participant & Diagnostic Label Verification

Diagnostic labels were verified directly from authoritative BIDS metadata (`participants.tsv` and `*_scans.tsv`) rather than inferred ad-hoc:

| Group | BIDS ID Pattern | Total Participants | Numeric Class Label |
| :--- | :--- | :--- | :--- |
| **Healthy Control (HC)** | `sub-HS1` to `sub-HS30` | 30 | `0` (Negative) |
| **Major Depressive Disorder (MDD)** | `sub-MDDS1` to `sub-MDDS34` | 34 | `1` (Positive) |
| **Total** | — | **64** | — |

*Note on demographics*: In `participants.tsv`, columns for `age`, `sex`, `hand`, `weight`, and `height` are recorded as `n/a` due to de-identification during BIDS deposition. The manuscript's assertion of "aged 12 to 77 years" derives from the parent cohort description in Mumtaz et al. (2017).

---

## 5. Recording Conditions, Tasks, and Scan Inventory

Each participant underwent up to three recording conditions:
1. `eyesClosed` (EC): Resting state with eyes closed (5 minutes protocol).
2. `eyesOpen` (EO): Resting state with eyes open (5 minutes protocol).
3. `P300`: Auditory oddball paradigm continuous recording (standard vs infrequent target tones).

### Scan Distribution Across Groups:
| Condition / Task | Healthy Controls (HC) | MDD Patients | Total Scans |
| :--- | :---: | :---: | :---: |
| **eyesOpen (EO)** | 30 | 32 | 62 |
| **P300 (Oddball)** | 28 | 33 | 61 |
| **eyesClosed (EC)** | 28 | 30 | 58 |
| **Total Scans** | **86** | **95** | **181** |

### Scan Completeness Per Participant:
- **52 Participants** have all 3 recording conditions complete ($52 \times 3 = 156$ scans).
- **10 Participants** have 2 recording conditions available ($10 \times 2 = 20$ scans):
  - Missing `eyesClosed`: `sub-HS12`, `sub-HS18`, `sub-MDDS12`, `sub-MDDS16`, `sub-MDDS4` (5 participants).
  - Missing `eyesOpen`: `sub-HS25`, `sub-MDDS7` (2 participants).
  - Missing `P300`: `sub-HS20`, `sub-HS21`, `sub-MDDS9` (3 participants).
- **1 Participant** has 1 recording condition available:
  - `sub-MDDS8`: has only `P300` (missing `eyesClosed` and `eyesOpen`).
- **1 Participant** has 4 recording scans:
  - `sub-HS15`: has `P300`, `eyesClosed`, `eyesOpen_run-1`, and `eyesOpen_run-2`. (Note: The BIDS README incorrectly stated "MDD S15 had two sessions", whereas the filesystem explicitly shows `sub-HS15` contains `run-1` and `run-2`).

---

## 6. Channel Layout and Hardware Specifications

- **Acquisition Sampling Rate**: 256.0 Hz ($T_s = 3.90625\text{ ms}$).
- **Hardware Pre-filtering**: High-pass filter at 0.5 Hz, low-pass filter at 70.0 Hz, 50.0 Hz notch filter.
- **Reference**: Monopolar acquisition referenced to linked ears (`A1+A2` / `LE`).
- **Channels**:
  - **19 Standard EEG Channels** present across all 181 recordings:  
    `Fp1`, `F3`, `C3`, `P3`, `O1`, `F7`, `T3`, `T5`, `Fz`, `Fp2`, `F4`, `C4`, `P4`, `O2`, `F8`, `T4`, `T6`, `Cz`, `Pz`.
  - **1 Bipolar Reference Channel**: `A2-A1` present across all 181 recordings.
  - **1 Annotation Track**: `EDF Annotations` present across all 181 recordings.
  - **2 Auxiliary Bipolar Channels**: `23A-23R` and `24A-24R` present in 112 of the 181 recordings (absent in 69 recordings).
- **Electrode Policy**: For uniform, reproducible cross-subject analysis, the pipeline must utilize the **19 standard 10–20 channels** present in all recordings. Auxiliary bipolar channels must be excluded from the EEG tensor.

---

## 7. Recording Durations

Across all 181 recordings:
- **Minimum Duration**: 180.0 seconds (3.0 minutes).
- **Maximum Duration**: 686.0 seconds (11.43 minutes).
- **Mean Duration**: 408.84 seconds (~6.81 minutes).
- **Standard Deviation**: 155.23 seconds.
- **Total Recorded Time**: 74,000.29 seconds (~20.56 hours of multi-channel EEG data).

Every recording comfortably exceeds standard short-epoch segmentation requirements (e.g. 5–30 second windows).

---

## 8. Final Audit Verification Checklist

- [x] Dataset ZIP and BIDS directory located and validated.
- [x] All 181 EDF files verified uncorrupted via binary header decoding.
- [x] Authoritative participant counts confirmed (64 total: 30 HC, 34 MDD).
- [x] Source of 34 vs 64 contradiction documented (BIDS curator docstring typo).
- [x] 19-channel standard 10-20 montage confirmed universal across all scans.
- [x] Sampling rate verified at exactly 256 Hz for all scans.
- [x] Full scan inventory exported to `reports/dataset_inventory.csv`.
- [x] Ready to formulate Phase 2 experimental protocol.
