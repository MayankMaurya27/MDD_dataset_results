# Phase 1B.6 — Nested Cross-Validation Candidate Search Space and Model Selection Protocol
## Jha MDD EEG SPWVD Research Paper — Mathematical Specification

**Workspace:** `/Users/mayankmaurya/Desktop/experiment`  
**Date:** November 2026  
**Audited Baseline Commit:** `baeb462`  
**Status:** Protocol Frozen — Awaiting Phase 1C Execution Authorization  

---

## 1. Mathematical Framework of Nested Group Cross-Validation

Standard $K$-fold cross-validation yields optimistic performance estimates when hyperparameter tuning, feature selection, or model architecture comparisons are performed using the same cross-validation loop. In biomedical EEG applications where segments originate from a finite cohort of participants, naive cross-validation risks severe **data leakage** and **model selection bias**.

To provide a mathematically unbiased estimate of generalization performance on unseen participants, Phase 1C implements a strict **Nested Stratified Group Cross-Validation** protocol:
- **Outer Loop ($k = 1, \dots, K_{\text{outer}}$, with $K_{\text{outer}} = 10$):** Partitions the participant cohort $\mathcal{P} = \{p_1, \dots, p_N\}$ into 10 disjoint subsets. In each outer iteration $k$, fold $\mathcal{P}_{\text{test}}^{(k)}$ is held out completely as an unseen test cohort.
- **Inner Loop ($j = 1, \dots, K_{\text{inner}}$, with $K_{\text{inner}} = 5$):** The remaining outer training participants $\mathcal{P}_{\text{train}}^{(k)}$ are partitioned into 5 disjoint validation folds $\mathcal{P}_{\text{val}}^{(k, j)}$ to perform autonomous model selection across the candidate hypothesis space $\mathcal{H}$.

```
Data Hierarchy:
Cohort: N Participants (EC: 58, EO: 61)
  └─ Continuous EEG (19 channels, ~5 min)
       └─ S_i Epochs per participant (5.0 s non-overlapping, ~59-65 epochs)
            └─ Segment Features x_(i, s) in R^D
```

---

## 2. Distinction: Exploratory Sweep vs. Nested Unbiased Estimator

To satisfy the full breadth of reviewer requirements, Phase 1C maintains an explicit distinction between two experimental procedures:

### 2.1 The Exploratory 32-Model Factorial Sweep (Module M2)
- **Design:** Evaluates every combination of 4 representations $\times$ 2 deep backbones $\times$ 4 downstream classifiers across a fixed 10-fold Stratified Group partition.
- **Purpose:** Characterizes the empirical landscape, answering specific reviewer questions (e.g., "How does ResNet-50 compare to ResNet-18?", "Does STFT match SPWVD across identical classifiers?", "Why did Fine Gaussian SVM fail?").
- **Reporting:** Reported as an empirical comparison matrix. It provides transparent comparative evidence but does not constitute the single reported primary pipeline metric.

### 2.2 The Autonomous Nested-CV Pipeline (Module M3)
- **Design:** In each outer fold $k$, the complete candidate pipeline space $\mathcal{H}$ is searched exclusively using the inner validation folds. The best model $h_k^* \in \mathcal{H}$ is selected, refit on all outer training participants, and evaluated exactly once on the holdout outer test fold.
- **Purpose:** Produces a single, leak-free, unbiased generalization estimate of the end-to-end algorithmic framework.

---

## 3. Candidate Search Space Specification ($\mathcal{H}$)

The candidate hypothesis space $\mathcal{H}$ consists of pre-specified pipelines defined by the tuple:
$$h = (\mathcal{T}, \mathcal{B}, \mathcal{S}, \mathcal{C}, \boldsymbol{\theta})$$
where $\mathcal{T}$ is the time-frequency transformation, $\mathcal{B}$ is the deep backbone feature extractor, $\mathcal{S}$ is the normalization transform, $\mathcal{C}$ is the classification algorithm, and $\boldsymbol{\theta}$ represents hyperparameter configurations.

### 3.1 Candidate Representations ($\mathcal{T}$)
1. **SPWVD (Smoothed Pseudo Wigner-Ville Distribution):**
   - Temporal window: Hamming window $N_t = 63$ points.
   - Frequency window: Hamming window $N_f = 63$ points.
   - Frequency resolution: 256 frequency bins, cropped to $0.5 - 45.0\text{ Hz}$.
   - Time points: 256 samples across 5.0 s ($51.2\text{ Hz}$ effective sampling).
2. **STFT (Short-Time Fourier Transform):**
   - Window: Hann window, length 128 points ($0.5\text{ s}$).
   - Overlap: 50% ($64\text{ points}$).
   - FFT points: 256, mapped to frequency range $0.5 - 45.0\text{ Hz}$.
3. **CWT Morlet (Continuous Wavelet Transform):**
   - Wavelet family: Complex Morlet with center frequency parameter $\omega_0 = 6$.
   - Frequencies: 32 log-spaced scales from $1.0\text{ Hz}$ to $45.0\text{ Hz}$.
   - Temporal sampling: Interpolated to 256 time bins.
4. **Welch PSD (Classical Spectral Baseline):**
   - Window: Welch segment length 256 samples (1.0 s) with 50% overlap.
   - Frequency resolution: $1.0\text{ Hz}$ bins across $0.5 - 45.0\text{ Hz}$ (45 spectral features per channel $\times$ 19 channels = 855 features).

### 3.2 Feature Extractors & Backbones ($\mathcal{B}$)
1. **ResNet-18 (Torchvision `ResNet18_Weights.IMAGENET1K_V1`):**
   - Global average pooling output dimension: $D = 512$.
   - Pretrained frozen weights.
2. **ResNet-50 (Torchvision `ResNet50_Weights.IMAGENET1K_V2`):**
   - Global average pooling output dimension: $D = 2048$.
   - Pretrained frozen weights.
3. **Direct Feature Concatenation (for Welch PSD):**
   - Dimension: $D = 855$.

### 3.3 Normalization Transforms ($\mathcal{S}$)
- **StandardScaler (Z-score):**
  $$\mu_d = \frac{1}{|\mathcal{S}_{\text{train}}|} \sum_{s \in \mathcal{S}_{\text{train}}} x_{s, d}, \quad \sigma_d = \sqrt{\frac{1}{|\mathcal{S}_{\text{train}}|} \sum_{s \in \mathcal{S}_{\text{train}}} (x_{s, d} - \mu_d)^2 + \epsilon}$$
  *Mandatory Rule: $\mu_d$ and $\sigma_d$ are fitted strictly on the inner training segments and applied without modification to validation and test segments.*

### 3.4 Downstream Classifiers & Hyperparameter Grids ($\mathcal{C}, \boldsymbol{\theta}$)

| Classifier ($\mathcal{C}$) | Hyperparameter Search Grid ($\boldsymbol{\theta}$) | Total Variants |
| :--- | :--- | :---: |
| **Linear SVM** | $C \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0, 100.0\}$, loss='hinge' or 'squared_hinge' | 7 |
| **RBF / Gaussian SVM** | $C \in \{0.1, 1.0, 10.0, 100.0\}$, $\gamma \in \{\text{'scale'}, \text{'auto'}, 10^{-4}, 10^{-3}, 10^{-2}\}$ | 20 |
| **Multi-Layer Perceptron (MLP)** | Hidden layers: $[(100,), (100, 50)]$; $\alpha \in \{10^{-5}, 10^{-4}, 10^{-3}\}$; solver='adam', max_iter=500 | 6 |
| **Logistic Regression** | $C \in \{10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0\}$, penalty='l2', solver='lbfgs' | 5 |
| **Random Forest** | n_estimators $\in \{100, 300\}$, max_depth $\in \{5, 10, \text{None}\}$, max_features='sqrt' | 6 |

> [!NOTE]
> **Resolution of Fine Gaussian SVM Degeneracy:**
> In the historical MATLAB-derived baseline, "Fine Gaussian SVM" utilized a fixed bandwidth parameter equivalent to $\gamma = 10.0$. In a $D = 512$-dimensional normalized feature space, the pairwise Euclidean distance between arbitrary samples typically satisfies $\|\mathbf{x}_a - \mathbf{x}_b\|^2 \ge 200$. Consequently, the kernel evaluated to $K(\mathbf{x}_a, \mathbf{x}_b) = \exp(-10.0 \times 200) \approx 0$ for all distinct pairs $a \neq b$. The Gram matrix collapsed to the identity matrix $\mathbf{K} \approx \mathbf{I}$, producing identical, degenerate predictions ($58.62\%$ EC, $54.10\%$ EO, exactly matching majority class frequency).
> In the nested search space $\mathcal{H}$, $\gamma$ is explicitly tuned with data-driven scaling ($\gamma = \frac{1}{D \cdot \text{Var}(X)}$) and fine logarithmic grids ($10^{-4}$ to $10^{-2}$), restoring genuine non-linear margin classification.

---

## 4. Inner-Fold Model Selection Objective & Scoring

### 4.1 Segment-to-Participant Posterior Pooling
In standard machine learning, model selection optimizes segment-level log-loss or accuracy. However, clinical EEG diagnosis operates at the **individual participant level**. Optimizing for segment-level accuracy can favor models that perform well on noisy or outlier segments while degrading cohort consensus.

Therefore, the inner validation score for candidate pipeline $h$ is computed strictly after participant posterior pooling:
1. Candidate pipeline $h$ generates calibrated continuous probability estimates $\hat{p}(y = 1 \mid \mathbf{x}_{i, s})$ for all segments $s \in \{1, \dots, S_i\}$ belonging to inner validation participant $p_i$.
2. Segment posteriors are aggregated via arithmetic mean to form the participant-level posterior:
   $$\hat{P}_i = \frac{1}{S_i} \sum_{s=1}^{S_i} \hat{p}(y = 1 \mid \mathbf{x}_{i, s})$$
3. Across all participants in the inner validation fold, the **Participant-Level ROC AUC** ($\text{AUC}_{\text{part}}^{(j)}$) and **Participant-Level Balanced Accuracy** ($\text{BAcc}_{\text{part}}^{(j)}$) are calculated:
   $$\text{BAcc}_{\text{part}}^{(j)} = \frac{1}{2} \left( \frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}} \right)$$

### 4.2 Objective Selection Function and Deterministic Tie-Breaking
The selection criterion for outer fold $k$ is the mean participant AUC across the 5 inner folds:
$$\mathcal{S}(h) = \frac{1}{5} \sum_{j=1}^5 \text{AUC}_{\text{part}}^{(k, j)}(h)$$

To guarantee 100% deterministic reproducibility across execution environments, ties are broken using the following strict hierarchy:
1. **Primary Metric:** Highest Mean Inner Participant AUC ($\mathcal{S}(h)$).
2. **Secondary Metric:** Highest Mean Inner Participant Balanced Accuracy.
3. **Tertiary Metric:** Highest Mean Inner Participant Sensitivity (prioritizing MDD detection rate).
4. **Complexity Regularization:** Preference for simpler parametric models:
   $$\text{Linear SVM} > \text{Logistic Regression} > \text{RBF SVM} > \text{MLP} > \text{Random Forest}$$
5. **Alphabetical Tie-Break:** Stable sort based on configuration hash.

---

## 5. Outer-Fold Generalization Evaluation (Single-Pass Protocol)

Once the winning pipeline $h_k^* = \operatorname{argmax}_{h \in \mathcal{H}} \mathcal{S}(h)$ is selected for outer fold $k$:
1. The model $h_k^*$ is refit from scratch using **all outer training data** $\mathcal{P}_{\text{train}}^{(k)}$ (including fitting a fresh `StandardScaler`).
2. The outer test participants $\mathcal{P}_{\text{test}}^{(k)}$ are transformed using the outer scaler and evaluated **exactly once**.
3. Segment predictions are pooled to participant posteriors:
   $$\hat{P}_i^{\text{outer}} = \frac{1}{S_i} \sum_{s=1}^{S_i} \hat{p}_{h_k^*}(y = 1 \mid \mathbf{x}_{i, s}), \quad \forall p_i \in \mathcal{P}_{\text{test}}^{(k)}$$
4. The resulting out-of-fold participant posterior vector $\hat{\mathbf{P}} = [\hat{P}_1, \dots, \hat{P}_N]$ across all $K_{\text{outer}} = 10$ folds provides the definitive, unbiased test performance of the nested system.

---

## 6. Verification and Anti-Leakage Assertion Checklist

Prior to execution in Module M3, the pipeline asserts the following automated invariant checks:
- [x] $\mathcal{P}_{\text{train}}^{(k)} \cap \mathcal{P}_{\text{test}}^{(k)} = \emptyset$ for all $k \in \{1, \dots, 10\}$.
- [x] $\mathcal{P}_{\text{train}}^{(k, j)} \cap \mathcal{P}_{\text{val}}^{(k, j)} = \emptyset$ for all $j \in \{1, \dots, 5\}$.
- [x] Scaler parameters $(\boldsymbol{\mu}, \boldsymbol{\sigma})$ depend strictly on $\mathcal{P}_{\text{train}}$ and never on $\mathcal{P}_{\text{test}}$ or $\mathcal{P}_{\text{val}}$.
- [x] Downstream classifier hyperparameter grids are completely frozen prior to execution.
- [x] Outer test fold data are loaded only after $h_k^*$ has completed fitting and parameter checkpointing.

*End of Nested CV Search Space Specification.*
