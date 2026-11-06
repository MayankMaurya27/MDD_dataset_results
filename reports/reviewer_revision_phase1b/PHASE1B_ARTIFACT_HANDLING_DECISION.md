# PHASE 1B.5 — INDEPENDENT ARTIFACT DIAGNOSTIC AUDIT & CONTROLLED EXPERIMENT DESIGN
## Jha MDD EEG SPWVD Research Paper — Forensic Reviewer Revision

**Auditor / Lead Engineer:** Senior Biomedical Signal Processing & Machine Learning Reproducibility Auditor  
**Date:** November 7, 2026  
**Audit Context:** Independent Scientific Evaluation of Stage 2 Artifact Diagnostics & Formulation of Artifact-Controlled Protocol  
**Deliverable Directory:** `reports/reviewer_revision_phase1b/`  
**Governing Principle:** Preprocessing decisions must be grounded in physical signal properties, leakage-free cross-validation architectures, and objective noise metrics—never selected post-hoc to inflate classification accuracy.

---

## 1. Independent Audit of Phase 1B Artifact Diagnostics

A forensic inspection of `src/artifact_diagnostic_audit.py` and the outputs in `results/revision_experiments/artifacts/` was conducted to address seven specific methodological questions:

### 1. Did the audit genuinely process all 181 recordings?
- **Finding:** **CONFIRMED GENUINE.**
- **Artifact Evidence:** In [`recording_level_artifact_metrics.csv`](file:///Users/mayankmaurya/Desktop/experiment/results/revision_experiments/artifacts/recording_level_artifact_metrics.csv), exactly **181 individual recording rows** are populated.
- **Distribution:** Exactly 62 Eyes-Open recordings (including `sub-HS15` run-1 and run-2), 61 P300 Oddball recordings, and 58 Eyes-Closed recordings across all 64 unique participants.
- **Total Epochs:** Exactly $181 \times 15 = 2,715$ study epochs were individually parsed, filtered, and analyzed across all 19 standard channels ($2,715 \times 19 = 51,585$ channel-epoch traces).

### 2. Are Peak-to-Peak (PTP) and Absolute Maximum (AbsMax) mathematically and dimensionally correct?
- **Finding:** **MATHEMATICALLY CORRECT, BUT DIMENSIONALLY DISTINCT.**
- **Code Audit:** In `src/artifact_diagnostic_audit.py`:
  $$\text{PTP} = \max_{t \in [0, T]} v(t) - \min_{t \in [0, T]} v(t), \quad \text{AbsMax} = \max_{t \in [0, T]} |v(t)|$$
- **Dimensional Clarification:**
  - For a zero-mean, symmetric oscillating signal (such as an occipital alpha burst of $\pm 50\,\mu V$), the amplitude envelope is $50\,\mu V$ ($\text{AbsMax} = 50\,\mu V$), but its peak-to-peak span is $100\,\mu V$ ($\text{PTP} = 100\,\mu V$).
  - Therefore, a threshold of $\text{PTP} > 100\,\mu V$ is mathematically twice as sensitive as $\text{AbsMax} > 100\,\mu V$.
  - In our empirical audit, **44.31% of study epochs have at least one channel exceeding $\text{PTP} > 100\,\mu V$**, whereas only **15.2% exceed $\text{AbsMax} > 100\,\mu V$**.
  - *Critical Caveat:* A threshold exceedance of $\text{PTP} > 100\,\mu V$ does **NOT** constitute definitive proof of an artifact; high-amplitude physiological delta or slow-wave oscillations naturally produce peak-to-peak spans of $80 - 120\,\mu V$.

### 3. Was FastICA applied to representative recordings or the entire archive?
- **Finding:** **REPRESENTATIVE ONLY (By Design).**
- In Phase 1B Stage 2, FastICA was executed on representative continuous recordings (`sub-HS1` Eyes-Open) to demonstrate mathematical feasibility and ocular component separation. It was **not** applied globally across the 181 recordings.
- Full-dataset ICA was intentionally withheld in accordance with the strict Phase 1B boundary (*"Do not apply automatic ICA rejection to the complete dataset until the diagnostic method is justified"*).

### 4. Was component selection validated quantitatively?
- **Finding:** **SEMI-QUANTITATIVE (Correlation-Based).**
- In the demonstration script, the blink component was identified by calculating Pearson correlation with the anterior frontal lead ($|r| > 0.82$ with `Fp1`).
- For a production artifact correction pipeline, component selection must be automated and validated against explicit quantitative multi-modal criteria (spatial topography, spectral slope, and kurtosis).

### 5. Is the spatial averaging argument scientifically sufficient on its own?
- **Finding:** **PARTIALLY SUFFICIENT, BUT SCIENTIFICALLY VULNERABLE TO REVIEWERS.**
- The spatial average $\bar{W}(t, f) = \frac{1}{19} \sum_{c=1}^{19} W_c(t, f)$ dilutes frontal blink energy by $\approx 10\times$ across the 17 non-polar sensors.
- However, because energy in Wigner-Ville distributions scales quadratically with voltage ($W \propto v^2$), an uncorrected blink of $300\,\mu V$ produces $9\times$ the spectral energy of a $100\,\mu V$ signal. Even after attenuation by $1/19$, low-frequency (0.5–3 Hz) energy leakage can contaminate the composite spectral map.
- Relying solely on spatial averaging without a controlled comparative benchmark will not satisfy a rigorous clinical EEG reviewer.

---

## 2. Controlled Artifact Experiment Design: Three-Condition Architecture

To resolve Reviewer Concern R03 definitively, we propose an **Artifact-Controlled Benchmark Experiment** comparing three distinct, clinically grounded preprocessing regimes:

```
┌────────────────────────────────────────────────────────────────────────┐
│               THREE-CONDITION ARTIFACT EXPERIMENTAL MATRIX             │
│                                                                        │
│  [Condition A: Baseline Uncorrected]                                   │
│     ├── Zero-phase 0.5 - 45.0 Hz bandpass filter                       │
│     ├── 15 non-overlapping 2.0s epochs (exact sample coordinates)      │
│     └── Spatial-mean SPWVD / Baselines (Historical baseline)           │
│                                                                        │
│  [Condition B: Semi-Automated ICA Ocular Decontamination]              │
│     ├── Fitted strictly per recording on continuous 19-channel EEG     │
│     ├── FastICA / Picard decomposition (19 components)                 │
│     ├── Automated ocular component rejection:                          │
│     │     - Correlation with (Fp1 + Fp2) > 0.75                        │
│     │     - Spatial dipole gradient (Frontal weight > 2x Posterior)    │
│     │     - Kurtosis > 5.0 (intermittent sharp blink deflections)      │
│     ├── Zeroing artifact components & continuous inverse projection    │
│     └── Re-extract identical 15 epochs at exact sample timestamps      │
│                                                                        │
│  [Condition C: Amplitude-Thresholded Epoch Replacement]                │
│     ├── Conservative PTP threshold on non-frontal leads (> 150 µV)     │
│     ├── If epoch corrupted by movement/amplifier saturation:           │
│     │     - Replace corrupted epoch with nearest clean 2.0s slice      │
│     │     - Maintain exactly 15 epochs per subject                     │
│     └── Protects participant posterior pooling sample size             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Algorithmic Specifications for Condition B (ICA)

### 1. Leakage-Free Fitting Scope:
- **Rule:** ICA will be fitted **strictly per recording** on the continuous multi-channel raw EEG, prior to epoch segmentation.
- **Rationale:** ICA is an unsupervised blind source separation algorithm that estimates the linear spatial unmixing matrix $\mathbf{W} \in \mathbb{R}^{19 \times 19}$ solely from the covariance structure of the multi-channel time series. Because it uses zero diagnostic class labels and operates within a single recording, it introduces **zero cross-subject or cross-fold data leakage**.

### 2. Multi-Modal Quantitative Component Selection Criteria:
A component $k \in \{1, \dots, 19\}$ is designated an ocular artifact component if and only if it satisfies all three quantitative criteria:
1. **Frontal Pole Correlation:** Absolute Pearson correlation with the bipolar vertical EOG proxy ($v_{\text{Fp1}} + v_{\text{Fp2}}$) exceeds threshold:
   $$\rho_k = |\text{corr}(\mathbf{s}_k, v_{\text{Fp1}} + v_{\text{Fp2}})| \ge 0.75$$
2. **Anterior-to-Posterior Spatial Gradient:** In the spatial unmixing column $\mathbf{a}_k$, the mean absolute weight of frontal polar leads exceeds posterior leads by at least a factor of two:
   $$\frac{\frac{1}{2}(|a_{k, \text{Fp1}}| + |a_{k, \text{Fp2}}|)}{\frac{1}{5}(|a_{k, \text{P3}}| + |a_{k, \text{P4}}| + |a_{k, \text{Pz}}| + |a_{k, \text{O1}}| + |a_{k, \text{O2}}|)} \ge 2.0$$
3. **Temporal Kurtosis:** Blink activity manifests as sparse, high-amplitude excursions resulting in high positive kurtosis:
   $$\text{Kurt}(\mathbf{s}_k) = \frac{\mathbb{E}[(\mathbf{s}_k - \mu)^4]}{\sigma^4} - 3 \ge 3.5$$

### 3. Reconstruction and Epoch Extraction:
- Components meeting all criteria (typically 1 or 2 components per recording) have their activation zeroed: $\mathbf{s}_k(t) \leftarrow 0$.
- The continuous cleaned EEG is reconstructed via inverse projection: $\hat{\mathbf{X}} = \mathbf{A}_{\text{clean}} \mathbf{S}$.
- The identical 15 deterministic epochs are extracted from $\hat{\mathbf{X}}$ at the exact sample timestamps (`sample_start`, `sample_end`) recorded in `metadata_*.csv`.

---

## 4. Objective Before/After Quality Metrics

To ensure the artifact cleaning method is evaluated objectively (rather than selecting a strategy solely based on downstream classification accuracy), the following physical signal metrics will be logged:

1. **Frontal Voltage Reduction:** Pre- vs. post-ICA peak-to-peak amplitude in `Fp1` and `Fp2`.
2. **Delta Band Power Attenuation:** Reduction of 0.5–3.0 Hz spectral power in frontal channels (blinks concentrate energy below 3 Hz).
3. **Posterior Signal Preservation:** Relative change in occipital alpha (8–13 Hz) and central mu rhythms (must change by $< 5\%$ to confirm genuine neural rhythms were not attenuated).
4. **Signal-to-Artifact Ratio (SAR):** Ratio of variance outside blink periods to variance during detected blink intervals.

---

## 5. Scope of Required Experimental Reruns

Under this protocol, all comparison pipelines will be benchmarked under both **Condition A (Uncorrected)** and **Condition B (ICA-Corrected)**:
1. **SPWVD + ResNet-18** (Linear SVM, Medium Gaussian SVM, MLP).
2. **Welch PSD + SVM** (Linear and RBF).
3. **STFT + ResNet-18** (Linear SVM).
4. **CWT Morlet + ResNet-18** (Linear SVM).

This 2-way factorial comparison ($2 \text{ Preprocessing} \times 4 \text{ Representations}$) provides the exact evidence reviewers demanded: proving whether the classification differences between methods persist after ocular artifact removal.
