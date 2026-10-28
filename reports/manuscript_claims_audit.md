# Comprehensive Manuscript Claims Audit

**Manuscript Title**: Automated Detection of Major Depressive Disorder Using SPWVD-Based EEG Time-Frequency Images and Deep Learning Features  
**Auditor**: Senior EEG Signal-Processing & Machine-Learning Experimental Auditor  
**Date**: October 28, 2026  
**Status**: Critical Scientific Audit  

---

## 1. Executive Summary

This audit evaluates all scientific, empirical, methodological, and bibliographic claims in the manuscript draft. Each claim was scrutinized against peer-reviewed literature, the underlying dataset (`nm000114_v1.0.0`), signal-processing theory, and rigorous machine learning standards.

The draft exhibits severe methodological and bibliographic vulnerabilities, including:
1. Citation of an officially retracted paper ([14]).
2. Citation mismatches (e.g. attributing depression symptoms to a biometric user identification paper [7], attributing delta waves to a gamma paper [6], and swapping Shen et al. [17] with Wu et al.).
3. Mathematically contradictory cross-validation descriptions (claiming 10 folds but only 1 test subject per iteration for 64 subjects).
4. Unsubstantiated claims of novelty, clinical readiness, and 99.7% diagnostic accuracy stemming from historical data leakage.
5. Technical confusion regarding transfer learning ("freezing the last three layers" and "extracting learned weight matrices").

---

## 2. Granular Audit of Manuscript Claims

### Claim 1: "To the best of our knowledge, no one has previously investigated this integrated technique..."
- **Draft Location**: Introduction, Paragraph 7.
- **Draft Statement**: *"To the best of our knowledge, no one has previously investigated this integrated technique for MDD identification using EEG data, which combines CNN and machine learning modules."*
- **Audit Verdict**: **UNSUPPORTED / OVERSTATED**.
- **Evidence**: Transfer learning feature extraction via deep CNNs coupled with classical machine learning classifiers (SVM, KNN, Random Forest) is a widely established, ubiquitous paradigm in biomedical signal processing and EEG analysis since at least 2017. Numerous studies cited in Table 2 and literature (e.g. Uyulan et al., Shahabi et al.) have investigated CNN representations with downstream classifiers.
- **Revision Mandate**: Remove the absolute novelty claim. Reframe the study as an empirical investigation comparing transfer learning representations with classical classifiers under rigorous, leakage-safe participant-grouped cross-validation.

---

### Claim 2: 99.7% Diagnostic Accuracy with Weighted KNN and Cubic SVM
- **Draft Location**: Abstract, Paragraph 3; Results, Paragraphs 43–46; Table 1.
- **Draft Statement**: *"With the use of this strict assessment system, K-nearest neighbor classifier achieves an impressive accuracy, precision, sensitivity and specificity of 99.7%."*
- **Audit Verdict**: **ARTIFACT OF HISTORICAL DATA LEAKAGE / METHODOLOGICAL CONFUSION**.
- **Evidence**: 
  - Achieving 99.7% accuracy across 64 subjects under strict subject-independent cross-validation is virtually unseen in psychiatric neuroimaging due to substantial inter-subject biological heterogeneity in resting-state EEG.
  - The text betrays the origin of the 99.7% figure: it reports segment-level results generated from either non-grouped cross-validation (where multiple epochs of the same subject were split between train and test sets) or random image-level splitting.
  - In our Phase 2 experimental protocol, true participant-independent group cross-validation isolates all epochs of test subjects completely.
- **Revision Mandate**: Replace historical 99.7% claims with the real, empirical results obtained from our verified, leakage-safe experimental pipeline. Clearly report both fold-wise participant-level and segment-level metrics with mean $\pm$ standard deviation.

---

### Claim 3: "Every iteration of the investigation, totaling 10, included a single test subject."
- **Draft Location**: Materials and Methods, Paragraph 30.
- **Draft Statement**: *"The sample comprised 64 individuals, categorized into 10 distinct folds... Every iteration of the investigation, totaling 10, included a single test subject."*
- **Audit Verdict**: **MATHEMATICALLY IMPOSSIBLE CONTRADICTION**.
- **Evidence**: In a cohort of 64 participants evaluated via 10-fold cross-validation, each fold must evaluate $64 / 10 \approx 6.4$ participants (specifically, 4 folds with 7 subjects and 6 folds with 6 subjects). Having 1 test subject per iteration across 10 iterations would evaluate only 10 participants total; conversely, testing 1 subject per fold across the entire cohort would require a 64-fold Leave-One-Subject-Out (LOSO) scheme.
- **Revision Mandate**: Eliminate the contradictory sentence. Explicitly define the stratified group 10-fold partition (Folds 0–3: 7 participants [3 HC, 4 MDD]; Folds 4–9: 6 participants [3 HC, 3 MDD]).

---

### Claim 4: "The last three layers of these models are frozen..."
- **Draft Location**: Materials and Methods, Paragraph 32.
- **Draft Statement**: *"The last three layers of these models are frozen, and the learned weight matrices are extracted as features for traditional machine learning classifiers."*
- **Audit Verdict**: **TERMINOLOGY CONFUSION / TECHNICALLY INACCURATE**.
- **Evidence**:
  - In transfer learning, "freezing" layers refers to locking parameter gradients during fine-tuning (backpropagation). When using a pretrained CNN as a feature extractor without fine-tuning, the **entire backbone is frozen**.
  - In MATLAB DAGNetworks, the final three layers are typically `FullyConnected`, `Softmax`, and `ClassificationOutput`. Feature extraction in transfer learning extracts the **activation embedding** (e.g. from the penultimate global average pooling layer `avgpool`, dimension 512 for ResNet18), NOT the "learned weight matrices".
- **Revision Mandate**: Correct the technical description: state that the pretrained ImageNet backbones were utilized as fixed feature extractors without retraining, extracting deep feature representations from the penultimate pooling/dense layers.

---

### Claim 5: Methodological Underspecification (Missing Window Size, Channels, and Normalization)
- **Draft Location**: Materials and Methods, Paragraphs 12, 16–21.
- **Audit Verdict**: **CRITICAL METHODOLOGICAL GAPS**.
- **Evidence**:
  - The draft never states the EEG segment length (e.g., 2 s, 4 s, or 5 s).
  - The draft never specifies which channels were used (out of the 19 standard channels, 2 bipolar channels, and 2 aux channels present in the EDF files).
  - The draft mentions a "Kaiser window of size 255" without defining the beta parameter ($\beta$), the time-smoothing window ($g$), or the image generation normalization details.
- **Revision Mandate**: Provide explicit, fully documented pipeline parameters:
  - 19 standard 10–20 EEG channels.
  - Specified segment duration and stride.
  - Exact mathematical definition of SPWVD smoothing windows.
  - Standardized resizing to $224 \times 224 \times 3$ with ImageNet z-score normalization.

---

### Claim 6: Clinical Readiness and "Revolutionizing Diagnostics"
- **Draft Location**: Abstract, Paragraph 3; Discussion, Paragraph 55; Conclusion, Paragraph 57.
- **Draft Statement**: *"aims to revolutionize the diagnostic process and improve patient outcomes... suitable for real-time clinical usage..."*
- **Audit Verdict**: **PREMATURE CLINICAL TRANSLATION CLAIM**.
- **Evidence**: Retrospective analysis of a de-identified research dataset collected under experimental conditions cannot support claims of immediate clinical utility or real-time diagnostic readiness without prospective clinical trials, blinded validation on comorbid cohorts (e.g. bipolar disorder, anxiety), and regulatory clearance.
- **Revision Mandate**: Temper claims to reflect an exploratory computational study demonstrating proof-of-concept. Discuss explicit clinical translation limitations in the Discussion.
