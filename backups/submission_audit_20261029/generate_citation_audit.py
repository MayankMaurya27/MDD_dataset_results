"""Generate comprehensive citation audit CSV for the manuscript."""

import csv

citations = [
    {
        "ref_id": "[1]",
        "draft_attribution": "Depression is identified by ongoing sadness and decline in interest/enjoyment; affects 5% of people globally.",
        "reference_entry": "V. U. Wanniarachchi, C. Greenhalgh, A. Choi, and J. R. Warren, 'Personalization variables in digital mental health interventions for depression and anxiety in adolescents and youth: a scoping review,' Frontiers in Digital Health, vol. 7, p. 1500220, 2025.",
        "status": "Valid",
        "audit_finding": "Recent review on digital mental health interventions for depression. General background statement supported by clinical literature (WHO depression guidelines).",
        "action": "Keep"
    },
    {
        "ref_id": "[2]",
        "draft_attribution": "Leading cause of disability in the world is depression, adding greatly to burden of disease.",
        "reference_entry": "K. Gupta, 'A machine learning framework for automated detection of perceived stress in female students,' Journal of Data Science and Intelligent Systems, 2026.",
        "status": "Questionable / Self-cite",
        "audit_finding": "Paper focuses on stress detection in female students using ML, not global disability epidemiology. Global burden of disease should cite WHO or GBD 2019/Lancet study.",
        "action": "Replace with authoritative epidemiological citation (e.g., GBD/WHO)"
    },
    {
        "ref_id": "[3]",
        "draft_attribution": "When depression persists for two or more weeks or longer, it is referred to be MDD.",
        "reference_entry": "E. J. Lenze, J. F. Karp, M. A. Gebara, G. E. Nicol, and B. H. Mulsant, 'Pharmacotherapy of major depressive disorder in older adults: from an evidence-informed stepwise algorithm to precision medicine,' Neuropsychopharmacology, pp. 1–13, 2026.",
        "status": "Valid",
        "audit_finding": "Peer-reviewed precision medicine review on MDD pharmacotherapy; supports standard DSM-5 diagnostic definition (2-week duration).",
        "action": "Keep"
    },
    {
        "ref_id": "[4]",
        "draft_attribution": "After the coronavirus pandemic, sudden increase in patients with depression.",
        "reference_entry": "J. Zhou, Y. Zhang, S. He, S. Xu, Q. Sun, T. Zhao, and Y. Dai, 'Accelerated global burden of depressive disorders during the covid-19 pandemic from 2019 to 2021,' Scientific Reports, vol. 15, no. 1, p. 9529, 2025.",
        "status": "Valid",
        "audit_finding": "Direct empirical study from Scientific Reports quantifying the accelerated burden of depression during COVID-19 pandemic.",
        "action": "Keep"
    },
    {
        "ref_id": "[5]",
        "draft_attribution": "EEG represents electrical brain activity; more accurate/consistent results obtained through EEG than ECG/speech.",
        "reference_entry": "K. Gupta, 'Optimal decomposition of eeg sensor data using tqwt method for mind-wandering detection,' SPIE OWT 2025, vol. 14168, 2026, pp. 825–830.",
        "status": "Mismatched / Self-cite",
        "audit_finding": "Paper is about TQWT decomposition for mind-wandering detection, NOT MDD vs ECG/speech comparative accuracy. Overgeneralized attribution.",
        "action": "Revise statement to avoid unsupported comparative superiority claim, cite established EEG review"
    },
    {
        "ref_id": "[6]",
        "draft_attribution": "Visible change can be found in the delta rhythm of MDD patient compared to normal.",
        "reference_entry": "Y.-Y. Yin and Y.-F. Li, 'Role of neural oscillations in depression: highlights on gamma oscillations,' Translational Psychiatry, 2026.",
        "status": "Mismatched",
        "audit_finding": "Paper explicitly focuses on GAMMA oscillations in depression, but text claims it reports DELTA rhythm changes. Factually mismatched.",
        "action": "Correct text to refer to gamma oscillations or cite a paper investigating delta oscillations"
    },
    {
        "ref_id": "[7]",
        "draft_attribution": "Brain areas of people with depression had altered, reducing decision-making abilities.",
        "reference_entry": "P. Campisi and D. La Rocca, 'Brain waves for automatic biometric-based user recognition,' IEEE TIFS, vol. 9, no. 5, pp. 782–800, 2014.",
        "status": "Severe Mismatch",
        "audit_finding": "Paper is on BIOMETRIC USER RECOGNITION (identifying healthy subjects via brain waves for security authentication), not depression or decision-making impairment.",
        "action": "Replace with verified neuroimaging citation on depressive cognitive/decision impairment"
    },
    {
        "ref_id": "[8]",
        "draft_attribution": "Differentiating between bipolar depression patients and healthy controls using ANOVA.",
        "reference_entry": "S. Lechner and G. Northoff, 'Abnormal resting-state eeg phase dynamics distinguishes major depressive disorder and bipolar disorder,' Journal of Affective Disorders, vol. 359, pp. 269–276, 2024.",
        "status": "Valid",
        "audit_finding": "Direct study on resting-state EEG phase dynamics in MDD vs BD vs controls.",
        "action": "Keep"
    },
    {
        "ref_id": "[9]",
        "draft_attribution": "Genetic algorithm picking multidimensional EEG features with ANN classifier achieving 89.12% accuracy.",
        "reference_entry": "T. T. Erguzel, S. Ozekes, O. Tan, and S. Gultekin, 'Feature selection and classification of electroencephalographic signals: an artificial neural network and genetic algorithm based approach,' Clinical EEG and Neuroscience, vol. 46, no. 4, pp. 321–326, 2015.",
        "status": "Valid",
        "audit_finding": "Verified peer-reviewed publication matching stated methods and results.",
        "action": "Keep"
    },
    {
        "ref_id": "[10]",
        "draft_attribution": "Nonlinear features from EEGs of depressed and healthy people with feature selection.",
        "reference_entry": "B. Hosseinifard, M. H. Moradi, and R. Rostami, 'Classifying depression patients and normal subjects using machine learning techniques and nonlinear features from eeg signal,' CMPB, vol. 109, no. 3, pp. 339–345, 2013.",
        "status": "Valid",
        "audit_finding": "Highly cited landmark paper on nonlinear EEG features for depression detection.",
        "action": "Keep"
    },
    {
        "ref_id": "[11]",
        "draft_attribution": "Common spatial patterns and PCA on EEG filter banks achieving 81.23% accuracy.",
        "reference_entry": "S.-C. Liao, C.-T. Wu, H.-C. Huang, W.-T. Cheng, and Y.-H. Liu, 'Major depression detection from eeg signals using kernel eigen-filter-bank common spatial patterns,' Sensors, vol. 17, no. 6, p. 1385, 2017.",
        "status": "Valid",
        "audit_finding": "Verified open-access paper matching stated CSP methodology and accuracy.",
        "action": "Keep"
    },
    {
        "ref_id": "[12]",
        "draft_attribution": "Entropy of 16-channel EEG signals with fuzzy logic classifier achieving 87.5% accuracy.",
        "reference_entry": "Y. Mohammadi, M. Hajian, and M. H. Moradi, 'Discrimination of depression levels using machine learning methods on eeg signals,' IEEE ICEE 2019, pp. 1765–1769.",
        "status": "Valid",
        "audit_finding": "Verified IEEE conference proceedings paper matching stated entropy and fuzzy logic methods.",
        "action": "Keep"
    },
    {
        "ref_id": "[13]",
        "draft_attribution": "13-layered CNN model for depression diagnosis in 2018.",
        "reference_entry": "U. R. Acharya, S. L. Oh, Y. Hagiwara, J. H. Tan, H. Adeli, and D. P. Subha, 'Automated eeg-based screening of depression using deep convolutional neural network,' CMPB, vol. 161, pp. 103–113, 2018.",
        "status": "Valid",
        "audit_finding": "Foundational deep learning EEG depression study (13-layer CNN on left/right hemisphere).",
        "action": "Keep"
    },
    {
        "ref_id": "[14]",
        "draft_attribution": "Two-layered LSTM model retrieving quantitative parameters from input EEG signals.",
        "reference_entry": "P. Nagabushanam, S. Thomas George, and S. Radha, 'Retracted article: Eeg signal classification using lstm and improved neural network algorithms,' Soft Computing, vol. 24, no. 13, pp. 9981–10003, 2020.",
        "status": "RETRACTED ARTICLE",
        "audit_finding": "Officially RETRACTED by Springer Soft Computing due to compromised peer review and scientific integrity concerns. Cannot be used as scientific evidence.",
        "action": "REMOVE RETRACTED CITATION. Replace with a valid, peer-reviewed LSTM/RNN depression detection study or remove reference to this work."
    },
    {
        "ref_id": "[15]",
        "draft_attribution": "Maximum likelihood estimation for MDD channel selection.",
        "reference_entry": "A. K. Choudhary, A. Mishra, K. N. Mishra, R. K. Lal, and A. Kumari, 'Detection of major depressive disorder using maximum likelihood estimation for channel selection: A novel approach,' Springer ICICC 2025, pp. 375–396.",
        "status": "Valid",
        "audit_finding": "Recent conference publication on channel selection for MDD.",
        "action": "Keep"
    },
    {
        "ref_id": "[16]",
        "draft_attribution": "End-to-end features and CNN-based model for depression diagnosis.",
        "reference_entry": "U. R. Acharya, V. K. Sudarshan, H. Adeli, J. Santhosh, J. E. Koh, and A. Adeli, 'Computer-aided diagnosis of depression using eeg signals,' European Neurology, vol. 73, no. 5-6, pp. 329–336, 2015.",
        "status": "Mismatched Methodology",
        "audit_finding": "Paper was published in 2015 using nonlinear features (DFA, sample entropy) and decision trees, NOT end-to-end deep CNNs (which Acharya published in 2018 [13]).",
        "action": "Correct description to reflect classical nonlinear features or cite appropriate end-to-end CNN work"
    },
    {
        "ref_id": "[17]",
        "draft_attribution": "Power features of EEG signals and conformal kernel SVM for classification.",
        "reference_entry": "J. Shen, X. Zhang, X. Huang, M. Wu, J. Gao, D. Lu, Z. Ding, and B. Hu, 'An optimal channel selection for eeg-based depression detection via kernel-target alignment,' IEEE JBHI, vol. 25, no. 7, pp. 2545–2556, 2020.",
        "status": "Severe Mismatch",
        "audit_finding": "Reference [17] is Shen et al. on Kernel-Target Alignment (KTA) for channel selection. The conformal kernel SVM study is Wu et al. (Table 2 row 3 in draft).",
        "action": "Disentangle Shen et al. (channel selection) and Wu et al. (conformal kernel SVM); assign correct references."
    },
    {
        "ref_id": "[18]",
        "draft_attribution": "Publicly available dataset of 34 MDD and 30 healthy controls at HUSM; eyes-closed, eyes-open, and task-based.",
        "reference_entry": "W. Mumtaz, L. Xia, M. A. Mohd Yasin, S. S. Azhar Ali, and A. S. Malik, 'A wavelet-based technique to predict treatment outcome for major depressive disorder,' PLoS ONE, vol. 12, no. 2, p. e0171409, 2017.",
        "status": "Valid (Incomplete citation set)",
        "audit_finding": "Mumtaz et al. collected the data at HUSM. However, the exact primary cohort paper is Mumtaz et al. (BSPC 2017) and Figshare (2016).",
        "action": "Supplement [18] with primary dataset citations: Mumtaz et al. BSPC 2017 and Figshare DOI"
    },
    {
        "ref_id": "[19]",
        "draft_attribution": "T-F representation showing amplitude and frequency varying over time.",
        "reference_entry": "K. Gupta, 'A robust deep learning system for screening of obstructive sleep apnea using tf spectrum of ecg signals,' CMBBE, vol. 28, no. 14, pp. 2137–2149, 2025.",
        "status": "Questionable / Self-cite",
        "audit_finding": "Paper is about ECG time-frequency for sleep apnea. The general definition of T-F representation is standard textbook knowledge (Cohen, 1995; Boashash, 2015).",
        "action": "Replace with standard foundational time-frequency reference (e.g. Cohen or Boashash)"
    },
    {
        "ref_id": "[20]",
        "draft_attribution": "SPWVD reducing cross-term interference by smoothing in time and frequency domains.",
        "reference_entry": "K. Gupta, V. Bajaj, I. A. Ansari, and U. R. Acharya, 'Hyp-net: Automated detection of hypertension using deep convolutional neural network and gabor transform techniques with ballistocardiogram signals,' BBE, vol. 42, no. 3, pp. 784–796, 2022.",
        "status": "Mismatched / Self-cite",
        "audit_finding": "Hyp-Net paper uses Gabor transform on ballistocardiogram for hypertension, NOT SPWVD. SPWVD reduction of cross-terms is foundational (Wigner 1932, Ville 1948, Classen & Mecklenbrauker 1980).",
        "action": "Replace with classic SPWVD reference (e.g. Classen & Mecklenbrauker 1980; Auger & Flandrin 1995)"
    },
    {
        "ref_id": "[21]",
        "draft_attribution": "ImageNet containing 1.2 million images across 1,000 classes.",
        "reference_entry": "M. Rostami, H. He, M. Chen, and D. Roth, 'Transfer learning via representation learning,' in Federated and Transfer Learning. Springer, 2022, pp. 233–257.",
        "status": "Indirect / Misattributed",
        "audit_finding": "Rostami et al. is a book chapter on transfer learning. The primary citation for ImageNet classification is Deng et al. (CVPR 2009) or Russakovsky et al. (IJCV 2015).",
        "action": "Cite Deng et al. (2009) / Russakovsky et al. (2015) for ImageNet"
    }
]

with open("reports/citation_audit.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["ref_id", "draft_attribution", "reference_entry", "status", "audit_finding", "action"])
    writer.writeheader()
    writer.writerows(citations)

print("Generated reports/citation_audit.csv with 21 audited references.")
