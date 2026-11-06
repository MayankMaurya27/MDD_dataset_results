# PHASE 0 — PREPROCESSING AND SIGNAL PROCESSING AUDIT
## Jha MDD EEG SPWVD Research Paper — Forensic Reviewer Revision

**Auditor:** Senior Biomedical Signal Processing & Machine Learning Reproducibility Auditor  
**Date:** November 7, 2026  
**Source Code Modules Audited:**  
- `src/preprocessing.py` (3,813 bytes)
- `src/spwvd.py` (4,265 bytes)
- `src/extract_features.py` (5,190 bytes)
- Feature Metadata: `results/features_cache/metadata_eyesClosed.csv`, `metadata_eyesOpen.csv`

---

## 1. Continuous EDF Ingestion and Channel Selection

1. **EDF Parser**: Ingested via MNE-Python (`mne.io.read_raw_edf(..., preload=True)`).
2. **Channel Mapping and Standardization**:
   - Raw recordings at HUSM contain between 22 and 24 channels, including reference leads (`LE`, `A1`, `A2`) and auxiliary bipolar derivations.
   - `src/preprocessing.py` extracts exactly the **19 standard International 10–20 monopolar EEG leads**:
     `['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fz', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'Cz', 'Pz']`
   - Naming variations (e.g., `'EEG Fp1-LE'`, `'EEG Cz-LE'`) are cleaned via string stripping. Non-10–20 auxiliary leads are discarded.
3. **Sampling Frequency**:
   - Authoritative sampling rate is exactly **256.0 Hz** across all 181 recordings.
   - Verified via `raw.info['sfreq']`.

---

## 2. Temporal Filtering and Referencing

1. **Bandpass Filtering**:
   - Zero-phase IIR bandpass filter applied between **0.5 Hz** (high-pass cutoff) and **45.0 Hz** (low-pass cutoff).
   - Eliminates low-frequency baseline drift (<0.5 Hz) and high-frequency EMG / line noise (>45 Hz).
2. **Notch Filtering**:
   - No explicit 50 Hz notch filter is applied because the 45.0 Hz low-pass filter operates below the 50 Hz Malaysian electrical mains frequency, attenuating power-line contamination.
3. **Common Average Referencing (CAR)**:
   - Implemented in `src/preprocessing.py` as an optional flag (`apply_car=False` by default). The baseline feature extraction ran with default monopolar linked-earlobe referencing as recorded.
4. **Signal Scaling**:
   - MNE stores data internally in Volts ($V$). `src/preprocessing.py` converts signals to microvolts ($\mu V$, factor $10^6$) before segmentation.

---

## 3. Epoch Segmentation and 15-Epoch Selection Determinism

1. **Epoch Geometry**:
   - Fixed window duration: **$T = 2.0\text{ seconds}$** ($N = 512\text{ samples}$ at 256.0 Hz).
   - Overlap: **0.0%** (strictly non-overlapping contiguous slices).
2. **15-Epoch Selection Algorithm**:
   - In `src/extract_features.py`:
     ```python
     segs, indices = segment_recording(data, fs, window_sec=2.0)
     if len(segs) > max_segments_per_file:
         step = len(segs) // max_segments_per_file
         selected_idx = [i * step for i in range(max_segments_per_file)]
         segs = segs[selected_idx]
         indices = [indices[i] for i in selected_idx]
     ```
   - **Determinism**: The algorithm is **100% deterministic**. It does not use pseudo-random sampling.
   - **Temporal Distribution**: Because `step = len(segs) // 15`, the 15 epochs are evenly spaced across the full duration of each continuous recording (e.g., if a file has 150 available 2-second epochs, it selects epochs 0, 10, 20, ..., 140).
3. **Traceability of Epoch Time Indices**:
   - Both `metadata_eyesClosed.csv` and `metadata_eyesOpen.csv` store the exact sample boundaries: `sample_start` and `sample_end`.
   - For example, `sub-HS1` recording has `sample_start = [0, 5120, 10240, ...]`, confirming exact preservation of temporal coordinates.

---

## 4. Artifact Rejection and ICA Audit (Current Gaps)

1. **Current Implementation**:
   - **No Automated Artifact Rejection**: There is currently no automated epoch rejection based on peak-to-peak amplitude thresholds (e.g., rejecting epochs with voltage exceeding $\pm 100\,\mu V$ or $\pm 150\,\mu V$).
   - **No Ocular/Cardiac Artifact Removal**: No Independent Component Analysis (ICA, such as FastICA or Infomax) or EOG regression is implemented in `src/preprocessing.py`.
2. **Reviewer Vulnerability**:
   - Reviewers in clinical EEG journals frequently request details on artifact handling (blinks, saccades, myogenic activity).
   - Relying solely on a 0.5–45.0 Hz bandpass filter means that high-amplitude blink artifacts in frontal leads (`Fp1`, `Fp2`) and temporal muscle tension remain present in the raw slices.
3. **Required Phase 1 Scope**:
   - Develop an artifact inspection script to quantify peak-to-peak amplitude distributions and identify artifact-saturated segments.
   - Formulate a clear defense in the manuscript: explain that deep transfer learning from multi-channel averaged SPWVD representations exhibits inherent robustness to localized transient artifacts, while acknowledging the lack of ICA as an explicit methodological parameter.

---

## 5. Smoothed Pseudo Wigner-Ville Distribution (SPWVD) Specifications

The SPWVD implementation in `src/spwvd.py` balances time-frequency resolution against cross-term interference:

1. **Analytic Signal Formulation**:
   - The input signal $x(t)$ is converted to the analytic signal $z(t) = x(t) + j \mathcal{H}\{x(t)\}$ via the Hilbert transform.
   - **Theoretical Purpose**: Eliminates negative frequency components, suppressing cross-terms between positive and negative frequency spectral peaks.
2. **Smoothing Windows**:
   - **Time-Smoothing Window $g(t)$**:
     - Hamming window of length $L_g = 31$ samples ($\approx 121\text{ ms}$).
     - Normalized to unit area: $g = g / \sum g$.
     - Smoothes along the time axis (convolution along axis 1 via `ndimage.convolve1d`).
   - **Frequency-Smoothing (Lag) Window $h(\tau)$**:
     - Kaiser window of length $L_h = 255$ samples ($\approx 996\text{ ms}$) with shape parameter $\beta = 5.0$.
     - Normalized to peak value 1.0: $h = h / h[M_h]$.
     - Attenuates outer lag products along the delay axis $\tau$ prior to the Fourier transform.
3. **FFT Grid**:
   - $N_{\text{FFT}} = 256$ points along the lag axis, yielding 128 positive frequency bins between 0.0 Hz and 128.0 Hz (Nyquist).
4. **Channel Aggregation**:
   - In `src/extract_features.py`, the SPWVD is computed independently for each of the 19 standard leads:
     $$\bar{W}(t, f) = \frac{1}{19} \sum_{c=1}^{19} W_c(t, f)$$
   - The spatial average collapses the 19 channels into a single joint spectral-temporal energy map of shape $(128, 512)$.

---

## 6. Image Normalization and Transfer Learning Preprocessing

1. **Clipping and Outlier Suppression**:
   - Negative cross-term ripples (inherent to Wigner distributions) are rectified: $W_{\text{clipped}} = \max(W, 0)$.
   - Upper dynamic range outliers are capped at the **99.5th percentile**: $W_{\text{norm}} = \text{clip}(W_{\text{clipped}} / p_{99.5}, 0.0, 1.0)$.
2. **Interpolation & Resizing**:
   - Resized from $(128, 512)$ to **$224 \times 224$** via PyTorch bilinear interpolation (`align_corners=False`).
3. **Color Replication & ImageNet Standardization**:
   - Single-channel energy map replicated across 3 RGB channels: shape $(3, 224, 224)$.
   - Standardized using ImageNet channel statistics:
     $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
   - Matches the input distribution required by PyTorch pretrained convolutional backbones.

