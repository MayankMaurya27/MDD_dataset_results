# Manuscript Revision Log and Action Plan

**Original Manuscript**: `Jha_et_al_MDD_SPWVD.docx`  
**Phase 1 Audit Reference**: `Jha_MDD_Phase1_Audit.docx`  
**Revised Manuscript Target**: `manuscript/Jha_MDD_Revised.docx`  
**Date**: October 28, 2026  

---

## 1. Section-by-Section Revision Plan

| Section in Manuscript | Issue in Original Draft | Action in Revised Manuscript | Rationale |
| :--- | :--- | :--- | :--- |
| **Title** | "SPWVD-Based EEG Time-Frequency Images and Deep Learning Features" | Retain or clarify as exploratory study under subject-independent validation. | Title accurately conveys the technical pipeline once leakage is resolved. |
| **Abstract** | Claims 99.7% accuracy with Weighted KNN under "strict assessment". | Replace 99.7% with verified participant-level and segment-level cross-validation results from experiments. | Accuracy of 99.7% was an artifact of data leakage; scientific integrity requires reporting real numbers. |
| **Introduction (Para 6)** | Cites [7] (Campisi: biometrics) for depressive cognitive impairment; cites [6] (gamma oscillations) for delta rhythm; cites [14] (retracted paper) for LSTM; cites [17] (Shen et al. KTA channel selection) for conformal SVM. | 1. Replace [7] with a verified depression cognitive impairment study.<br>2. Correct [6] to reference gamma rhythms.<br>3. Remove retracted [14] citation and discuss validated sequence models.<br>4. Re-attribute conformal SVM to Wu et al. and cite Shen et al. accurately. | Citations must directly support the in-text factual claims. Retracted articles cannot be cited. |
| **Introduction (Para 7)** | Claims absolute novelty ("to best of knowledge no one has investigated..."). | Reframe as a systematic comparative evaluation of transfer learning representations with multiple classifiers under strict participant-grouped validation. | Hybrid CNN-ML approaches on EEG spectrograms have extensive prior precedent in the literature. |
| **Materials & Methods (Dataset)** | Mentions "34 depressed and 30 healthy subjects aged 12 to 77 years at HUSM". | Clarify the BIDS provenance (`nm000114_v1.0.0`), confirm 64 participants, and note the resolution of the README 34-participant typo. Detail the 3 tasks (EC, EO, P300) and 181 recordings. | Explains provenance, resolves reader confusion over the README typo, and ensures complete reporting. |
| **Materials & Methods (SPWVD)** | Kaiser window size 255 mentioned, but missing segment duration, channel selection, and smoothing details. | Specify 19 standard channels, segment window duration, time smoothing window ($g$), frequency smoothing Kaiser window ($h$, size 255), and image resizing to $224 \times 224 \times 3$. | Eliminates critical ambiguity, ensuring full reproducibility. |
| **Materials & Methods (Cross-Val)** | Contradiction: 10 folds but "included a single test subject" per iteration. | Formalize as Stratified Group 10-Fold Cross-Validation (Folds 0–3: 7 subjects; Folds 4–9: 6 subjects; all segments grouped by subject). | Corrects mathematical impossibility and aligns with experimental protocol. |
| **Materials & Methods (Transfer Learning)**| "Last three layers are frozen and learned weight matrices are extracted". | Correct to: CNN backbones are used as fixed feature extractors (ImageNet weights frozen), extracting deep feature representations from penultimate layers. | Fixes technical terminology to match standard deep learning practices. |
| **Results (CNN Comparison)** | Draft Table 1 lists 92.5%, 91.6%, 90.4%, 91.3% for CNN models without protocol details. | Update with empirical results from verified participant-grouped 10-fold cross-validation. | Results must be grounded in verified execution. |
| **Results (Classifier Table)**| Table 2 reports 99.7% across multiple classifiers. | Replace Table 2 with actual empirical metrics (Acc, Prec, Sens, Spec, F1, Balanced Acc, ROC-AUC) at both participant-level and segment-level. | Provides true generalizability metrics without leakage. |
| **Discussion (Comparison Table)**| Table 3 comparator studies lacked verified citations, DOIs, and protocols. | Keep comparison table withheld or present verified comparators only (Acharya et al. 2018, Mumtaz et al. 2019, Wu et al. 2020) with explicit validation protocol distinctions. | Comparing non-grouped results with grouped results without caveats is scientifically misleading. |
| **Limitations & Conclusion** | Draft claims clinical readiness and real-time suitability. | Emphasize retrospective dataset limitations, lack of independent external clinical validation, and need for prospective physician trials. | Scientific modesty and adherence to regulatory realities in clinical AI. |
