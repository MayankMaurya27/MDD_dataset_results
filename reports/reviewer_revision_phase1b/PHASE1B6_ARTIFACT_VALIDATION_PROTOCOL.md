# Phase 1B.6 — Artifact Validation Protocol: Ocular Attenuation and Neural Signal Preservation
## Jha MDD EEG SPWVD Research Paper — Methodological Specification

**Workspace:** `/Users/mayankmaurya/Desktop/experiment`  
**Date:** November 2026  
**Audited Baseline Commit:** `baeb462`  
**Status:** Protocol Frozen — Awaiting Phase 1C Execution Authorization  

---

## 1. Rationale and Reviewer Mandate

Reviewers raised substantive critiques regarding the handling of physiological artifacts in the Russian MDD EEG dataset:
1. **Ocular Contamination (Point 1, Point 7, Point 10):** Uncleaned blink excursions can inject massive, non-neural low-frequency power into frontal electrodes ($F_{p1}, F_{p2}$). If blinks are unevenly distributed between MDD patients and healthy controls (due to psychomotor retardation, blunted affect, or anxiety), classifiers may learn to detect blink frequency rather than neurophysiological signatures of depression.
2. **The Over-Cleaning Fallback Trap:** Blindly applying aggressive thresholding or subtracting excessive components can decimate frontal theta ($4 - 8\text{ Hz}$) and anterior alpha rhythms, distorting the very biomarkers under investigation.
3. **The Unwarranted EOG Claim:** Prior draft revisions colloquially referred to $F_{p1}$ and $F_{p2}$ as "EOG channels." Methodologically, these are standard scalp electrodes; electro-oculogram signals derived from them are synthetic bipolar proxies.

This protocol establishes the rigorous, objective, and frozen procedures for executing Condition B (ICA Cleaning) and validating that **ocular artifacts are attenuated while authentic neurophysiological dynamics are preserved**.

---

## 2. Anatomical and Technical Clarification: The Ocular Proxy

The Russian MDD dataset was acquired using a 19-channel EEG system with electrodes positioned in accordance with the International 10–20 System:
- **Frontal / Prefrontal:** $F_{p1}, F_{p2}, F_7, F_3, F_z, F_4, F_8$
- **Central / Temporal:** $T_3, C_3, C_z, C_4, T_4$
- **Parietal / Posterior Temporal:** $T_5, P_3, P_z, P_4, T_6$
- **Occipital:** $O_1, O_2$
- **Reference:** Linked earlobes ($A_1 + A_2$).

```
                      (Nasion)
                     Fp1   Fp2      <-- Frontopolar EEG leads
                  F7   F3  Fz  F4   F8   (Synthetic Ocular Proxy:
                    T3  C3  Cz  C4  T4     y_blink = 0.5 * (Fp1 + Fp2))
                  T5   P3  Pz  P4   T6
                     O1     O2      <-- Primary Alpha Oscillations (8-12 Hz)
                      (Inion)
```

> [!IMPORTANT]
> **Strict Nomenclature Mandate:**
> At no point in the scientific paper, technical reports, or code docstrings shall $F_{p1}$ and $F_{p2}$ be described as dedicated EOG leads. They must be explicitly described as:
> *"Frontopolar scalp EEG channels ($F_{p1}$ and $F_{p2}$) utilized as a synthetic bipolar electro-ocular reference proxy."*

The ocular reference proxy $\mathbf{y}_{\text{ocular}}(t)$ is defined as:
$$\mathbf{y}_{\text{ocular}}(t) = \frac{1}{2} \left( x_{F_{p1}}(t) + x_{F_{p2}}(t) \right)$$

---

## 3. Definition of the Three Preprocessing Conditions

The Phase 1C experimental pipeline executes three controlled preprocessing conditions across all 181 recordings (64 unique participants, 58 EC, 61 EO):

### Condition A: Historical Baseline (Uncorrected)
- **Bandpass Filtering:** Zero-phase 4th-order Butterworth bandpass filter between $0.5\text{ Hz}$ and $45.0\text{ Hz}$.
- **Notch Filter:** No notch filter applied (mains noise at $50\text{ Hz}$ is outside the $45\text{ Hz}$ cutoff).
- **Segmentation:** Continuous recording divided into $5.0\text{ s}$ non-overlapping epochs ($1,280$ samples at $256\text{ Hz}$).
- **Artifact Handling:** Zero component removal, zero epoch rejection. This preserves the historical baseline published in Jha et al. to allow exact apples-to-apples comparison.

### Condition B: Primary Artifact-Corrected Pipeline (Per-Recording ICA)
- **Continuous Decomposition:** Independent Component Analysis (FastICA with $\tanh$ contrast function, max iterations = 500) is fitted on the continuous 19-channel recording ($T \approx 76,800$ samples across 5 minutes).
- **Component Decomposition:** Decomposes the observed scalp signals $\mathbf{X} \in \mathbb{R}^{19 \times T}$ into 19 independent time courses:
  $$\mathbf{S} = \mathbf{W} \mathbf{X}, \quad \mathbf{X} = \mathbf{A} \mathbf{S}$$
  where $\mathbf{W}$ is the unmixing matrix and $\mathbf{A} = \mathbf{W}^{-1}$ is the mixing (spatial topography) matrix.
- **Automated Component Rejection:** Components matching the frozen multi-criteria decision rule (§4) are zeroed: $\mathbf{S}_{\text{cleaned}} = \mathbf{M} \mathbf{S}$, where $M_{kk} = 0$ for rejected components and $1$ otherwise.
- **Signal Reconstruction:** Cleaned scalp signals are reconstructed via:
  $$\mathbf{X}_{\text{cleaned}} = \mathbf{A} \mathbf{S}_{\text{cleaned}} = \mathbf{A} \mathbf{M} \mathbf{W} \mathbf{X}$$
- **Segmentation:** The cleaned continuous record is segmented into identical $5.0\text{ s}$ non-overlapping epochs.

### Condition C: Sensitivity Analysis (Nearest-Clean-Epoch Replacement)
- **Explicit Classification:** **Condition C is strictly designated as a secondary sensitivity analysis, NOT an alternative primary preprocessing pipeline.**
- **Rationale:** Evaluates whether transient high-voltage ocular excursions (spikes $>150\,\mu\text{V}$) disproportionately skew participant posterior averages $\hat{P}_i = \frac{1}{S_i} \sum p_{i, s}$.
- **Replacement Mechanism:** Segments in Condition A where peak-to-peak amplitude exceeds $150\,\mu\text{V}$ in $>3$ channels are flagged as contaminated. Instead of discarding the segment (which would produce unequal epoch counts across participants and disrupt comparative alignment), each contaminated segment is replaced by the nearest temporally adjacent clean segment from the same participant session.
- **Reporting:** Reported in a dedicated sensitivity analysis section to confirm whether findings in Condition A and Condition B are robust to localized transient outliers.

---

## 4. Frozen Component Selection Criteria and Fallback Rules

All component selection decisions are executed by a deterministic, automated decision engine. **No manual visual inspection or post-hoc threshold adjustment is permitted after viewing downstream classification results.**

### 4.1 Quantitative Decision Thresholds
An independent component $\mathbf{s}_k = [s_k(1), \dots, s_k(T)]$ is marked for rejection if and only if it satisfies all three of the following criteria:

1. **Frontopolar Correlation:**
   $$|\rho(\mathbf{s}_k, \mathbf{y}_{\text{ocular}})| = \frac{|\sum_{t=1}^T (s_k(t) - \bar{s}_k)(y_{\text{ocular}}(t) - \bar{y}_{\text{ocular}})|}{\sqrt{\sum (s_k(t) - \bar{s}_k)^2 \sum (y_{\text{ocular}}(t) - \bar{y}_{\text{ocular}})^2}} \ge 0.40$$
2. **Spatial Energy Concentration:** The fraction of the component's back-projected scalp variance concentrated in the frontopolar leads ($F_{p1}, F_{p2}$) must exceed 50%:
   $$\mathcal{R}_{\text{frontal}}(k) = \frac{A_{F_{p1}, k}^2 + A_{F_{p2}, k}^2}{\sum_{i=1}^{19} A_{i, k}^2} \ge 0.50$$
3. **Temporal Kurtosis:** High kurtosis characteristic of intermittent, transient blink excursions:
   $$\kappa(\mathbf{s}_k) = \frac{\frac{1}{T} \sum_{t=1}^T (s_k(t) - \bar{s}_k)^4}{\left( \frac{1}{T} \sum_{t=1}^T (s_k(t) - \bar{s}_k)^2 \right)^2} \ge 4.5$$

### 4.2 Degree-of-Freedom Preservation Cap
- **Strict Cap:** Under no circumstances shall more than **two (2) components** be rejected from any single 19-channel recording.
- **Rank Ordering:** If 3 or more components meet the rejection criteria, they are ranked in descending order by their frontal correlation $|\rho(\mathbf{s}_k, \mathbf{y}_{\text{ocular}})|$, and **only the top 2 are zeroed**.
- **Guarantee:** This guarantees that for every participant, at least 17 independent spatial degrees of freedom ($\ge 89.5\%$ of channel dimensionality) are preserved strictly for brain electrophysiology.

### 4.3 Convergence Failures and Edge Cases
1. **Zero-Component Case:** If no component in a recording meets the joint rejection criteria, the recording is classified as artifact-quiescent. Exactly 0 components are rejected, and the uncorrected signal is passed through.
2. **Convergence Failure:** If FastICA fails to reach the tolerance threshold ($\Delta W < 10^{-4}$) within 500 iterations:
   - **Step 1:** Automatically restart with randomized initial orthogonal weights (up to 3 attempts).
   - **Step 2:** If still non-convergent, switch solver to Extended Infomax (Picard engine).
   - **Step 3 (Emergency Fallback):** If Infomax also fails, abort component removal for that specific recording, retain the uncorrected signal (Condition A), and log the recording ID in `reports/reviewer_revision_phase1c/M1_convergence_exceptions.log`.

---

## 5. Dual-Sided Quantitative Validation Protocol

The efficacy of artifact correction must be demonstrated through a dual-sided balance: demonstrating significant artifact attenuation while rigorously proving that genuine neural signals were not destroyed.

```
                    Validation Protocol Architecture
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
[Artifact Attenuation]                             [Neural Preservation]
- Frontal PTP reduction (>40%)                     - Posterior Alpha retention (>85%)
- Frontal peak correlation decay                   - Occipital PSD shape invariance
- Kurtosis normalization                           - Cross-channel coherence stability
```

### 5.1 Ocular Artifact Attenuation Metrics
For each recording, the following metrics are evaluated before and after Condition B cleaning:
1. **Frontopolar Peak-to-Peak (PTP) Amplitude:**
   $$\Delta \text{PTP}_{\text{frontal}} = \frac{\text{PTP}_{\text{raw}}(F_{p1}, F_{p2}) - \text{PTP}_{\text{clean}}(F_{p1}, F_{p2})}{\text{PTP}_{\text{raw}}(F_{p1}, F_{p2})} \times 100\%$$
   *Success Criterion:* Statistically significant reduction in mean frontal PTP amplitude ($\Delta \text{PTP} \ge 35\%$, paired Wilcoxon signed-rank test $p < 10^{-10}$).
2. **Excursion Rate Above Thresholds:**
   - Percentage of epochs with $|V| > 100\,\mu\text{V}$ in $F_{p1}/F_{p2}$ (must decrease from baseline $>31\%$ to $<5\%$).
   - Percentage of epochs with $|V| > 150\,\mu\text{V}$ in $F_{p1}/F_{p2}$ (must decrease from baseline $>18\%$ to $<1\%$).
3. **Correlation with Ocular Proxy:**
   - Maximum channel-level correlation with the blink template $\mathbf{y}_{\text{ocular}}$ must drop below $0.25$.

### 5.2 Neurophysiological Signal Preservation Metrics
To prove that brain activity was not compromised, posterior electrodes ($O_1, O_2, P_3, P_4, P_z$) are subjected to strict preservation standards:
1. **Posterior Alpha Power Retention Ratio ($\mathcal{R}_{\alpha}$):**
   Alpha power ($8.0 - 12.0\text{ Hz}$) is computed using Welch's periodogram in posterior channels where ocular volume conduction is minimal:
   $$\mathcal{R}_{\alpha} = \frac{\int_8^{12} P_{\text{clean}}(f; O_1, O_2, P_z) \, df}{\int_8^{12} P_{\text{raw}}(f; O_1, O_2, P_z) \, df}$$
   *Preservation Rule:* Mean alpha retention across the cohort must satisfy:
   $$\bar{\mathcal{R}}_{\alpha} \ge 0.85 \quad (85\%\text{ power retention})$$
   Any drop below 80% triggers an immediate pipeline halt and algorithmic audit.
2. **Occipital Spectral Profile Invariance:**
   The spectral correlation between pre- and post-cleaning PSDs across $0.5 - 45.0\text{ Hz}$ in occipital leads:
   $$\rho_{\text{PSD}} = \operatorname{corr}\left( \log P_{\text{raw}}(f), \log P_{\text{clean}}(f) \right) \ge 0.95$$
3. **Topographic Gradient Conservation:**
   The anterior-to-posterior power gradient (ratio of frontal theta to posterior alpha) must remain biologically consistent with standard resting-state norms.

---

## 6. Pre-Registration and Traceability Commitment

All validation results produced by Module M1 will be automatically logged to:
- `results/reviewer_revision_phase1c/artifact_audit_summary.csv`
- `reports/reviewer_revision_phase1c/M1_ARTIFACT_VALIDATION_REPORT.md`

Every recording will have its rejected component count, identified component indices, ocular correlation scores, frontal PTP deltas, and posterior alpha retention ratios published in full.

*End of Artifact Validation Protocol.*
