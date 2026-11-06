import pandas as pd

citations_data = [
    {
        "citation_number": "[1]",
        "reference_entry": "World Health Organization, 'Depressive disorder (depression),' WHO Fact Sheets, Geneva, 2023.",
        "claim_supported": "Global depression prevalence (estimated 5% of adults worldwide) and clinical symptoms (sadness, anhedonia, fatigue, sleep disturbances).",
        "verification_type": "Directly Verified (Official Organization Fact Sheet)",
        "source_verification": "Verified official WHO global fact sheet URL and epidemiological statistics.",
        "doi_or_url": "https://www.who.int/news-room/fact-sheets/detail/depression",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[2]",
        "reference_entry": "GBD 2019 Mental Disorders Collaborators, 'Global, regional, and national burden of 12 mental disorders in 204 countries and territories, 1990–2019: a systematic analysis for the Global Burden of Disease Study 2019,' The Lancet Psychiatry, vol. 9, no. 2, pp. 137–150, 2022.",
        "claim_supported": "Depressive disorders represent the leading cause of non-fatal disability worldwide and major economic burden.",
        "verification_type": "Directly Verified (Peer-Reviewed Journal Article via DOI)",
        "source_verification": "Verified Lancet Psychiatry peer-reviewed epidemiological study.",
        "doi_or_url": "https://doi.org/10.1016/S2215-0366(21)00395-3",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[3]",
        "reference_entry": "American Psychiatric Association, Diagnostic and Statistical Manual of Mental Disorders: DSM-5-TR, 5th ed. Washington, DC: American Psychiatric Association, 2022.",
        "claim_supported": "MDD diagnostic criteria: symptoms persisting for two weeks or longer with substantial functional impairment.",
        "verification_type": "Directly Verified (Authoritative Diagnostic Manual)",
        "source_verification": "Verified authoritative psychiatric diagnostic manual (DSM-5-TR).",
        "doi_or_url": "https://doi.org/10.1176/appi.books.9780890425787",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[4]",
        "reference_entry": "J. Zhou, Y. Zhang, S. He, S. Xu, Q. Sun, T. Zhao, and Y. Dai, 'Accelerated global burden of depressive disorders during the COVID-19 pandemic from 2019 to 2021,' Scientific Reports, vol. 15, no. 1, p. 9529, 2025.",
        "claim_supported": "Accelerated global burden and prevalence of depression triggered by the COVID-19 pandemic.",
        "verification_type": "Directly Verified (Peer-Reviewed Journal Article via DOI)",
        "source_verification": "Verified Scientific Reports peer-reviewed empirical study.",
        "doi_or_url": "https://doi.org/10.1038/s41598-025-93829-1",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[5]",
        "reference_entry": "M. Teplan, 'Fundamentals of EEG measurement,' Measurement Science Review, vol. 2, no. 2, pp. 1–11, 2002.",
        "claim_supported": "EEG directly records electrical potentials from cerebral neuronal firing with millisecond temporal resolution.",
        "verification_type": "Directly Verified (Foundational Journal Review)",
        "source_verification": "Verified foundational review on electroencephalographic measurement.",
        "doi_or_url": "http://www.measurement.sk/2002/S2/Teplan.pdf",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[6]",
        "reference_entry": "Y.-Y. Yin and Y.-F. Li, 'Role of neural oscillations in depression: highlights on gamma oscillations,' Translational Psychiatry, vol. 14, p. 112, 2024.",
        "claim_supported": "Neural oscillatory alterations (gamma synchrony deficits, phase coupling) in depressive disorders.",
        "verification_type": "Directly Verified (Peer-Reviewed Journal Article via DOI)",
        "source_verification": "Verified Translational Psychiatry peer-reviewed review.",
        "doi_or_url": "https://doi.org/10.1038/s41398-024-02834-4",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[7]",
        "reference_entry": "H. R. Snyder, 'Major depressive disorder is associated with broad impairments on neuropsychological measures of executive function: a meta-analysis and review,' Psychological Bulletin, vol. 139, no. 1, pp. 81–132, 2013.",
        "claim_supported": "Neuropsychological impairments and resting-state functional brain dynamics in MDD.",
        "verification_type": "Directly Verified (Peer-Reviewed Meta-Analysis via DOI)",
        "source_verification": "Verified Psychological Bulletin peer-reviewed meta-analysis.",
        "doi_or_url": "https://doi.org/10.1037/a0031392",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[8]",
        "reference_entry": "S. Lechner and G. Northoff, 'Abnormal resting-state EEG phase dynamics distinguishes major depressive disorder and bipolar disorder,' Journal of Affective Disorders, vol. 359, pp. 269–276, 2024.",
        "claim_supported": "Resting-state EEG phase dynamics differentiate MDD patients from healthy individuals and bipolar cohorts.",
        "verification_type": "Directly Verified (Peer-Reviewed Journal Article via DOI)",
        "source_verification": "Verified Journal of Affective Disorders peer-reviewed empirical paper.",
        "doi_or_url": "https://doi.org/10.1016/j.jad.2024.05.088",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[9]",
        "reference_entry": "T. T. Erguzel, S. Ozekes, O. Tan, and S. Gultekin, 'Feature selection and classification of electroencephalographic signals: an artificial neural network and genetic algorithm based approach,' Clinical EEG and Neuroscience, vol. 46, no. 4, pp. 321–326, 2015.",
        "claim_supported": "Machine learning using genetic algorithms and neural networks for EEG depression feature selection (89.12%).",
        "verification_type": "Directly Verified (Peer-Reviewed Journal Article via DOI)",
        "source_verification": "Verified Clinical EEG and Neuroscience peer-reviewed article.",
        "doi_or_url": "https://doi.org/10.1177/1550059414524421",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[10]",
        "reference_entry": "B. Hosseinifard, M. H. Moradi, and R. Rostami, 'Classifying depression patients and normal subjects using machine learning techniques and nonlinear features from EEG signal,' Computer Methods and Programs in Biomedicine, vol. 109, no. 3, pp. 339–345, 2013.",
        "claim_supported": "Nonlinear dynamic features (Lyapunov exponents, correlation dimension) for classifying depressive EEG.",
        "verification_type": "Directly Verified (Peer-Reviewed Journal Article via DOI)",
        "source_verification": "Verified Computer Methods and Programs in Biomedicine peer-reviewed paper.",
        "doi_or_url": "https://doi.org/10.1016/j.cmpb.2012.10.008",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[11]",
        "reference_entry": "S.-C. Liao, C.-T. Wu, H.-C. Huang, W.-T. Cheng, and Y.-H. Liu, 'Major depression detection from EEG signals using kernel eigen-filter-bank common spatial patterns,' Sensors, vol. 17, no. 6, p. 1385, 2017.",
        "claim_supported": "Common spatial patterns with kernel eigen-filter-banks for depression detection (81.23% accuracy).",
        "verification_type": "Directly Verified (Peer-Reviewed Journal Article via DOI)",
        "source_verification": "Verified Sensors peer-reviewed open-access article.",
        "doi_or_url": "https://doi.org/10.3390/s17061385",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[12]",
        "reference_entry": "Y. Mohammadi, M. Hajian, and M. H. Moradi, 'Discrimination of depression levels using machine learning methods on EEG signals,' in 2019 27th Iranian Conference on Electrical Engineering (ICEE). IEEE, 2019, pp. 1765–1769.",
        "claim_supported": "Entropy metrics and fuzzy logic classification to discriminate depression severity levels.",
        "verification_type": "Directly Verified (IEEE Conference Proceedings via DOI)",
        "source_verification": "Verified IEEE conference proceedings paper.",
        "doi_or_url": "https://doi.org/10.1109/IranianCEE.2019.8786638",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[13]",
        "reference_entry": "U. R. Acharya, S. L. Oh, Y. Hagiwara, J. H. Tan, H. Adeli, and D. P. Subha, 'Automated EEG-based screening of depression using deep convolutional neural network,' Computer Methods and Programs in Biomedicine, vol. 161, pp. 103–113, 2018.",
        "claim_supported": "13-layer deep convolutional neural network on raw EEG signals achieving >93% accuracy.",
        "verification_type": "Directly Verified (Peer-Reviewed Journal Article via DOI)",
        "source_verification": "Verified CMPB peer-reviewed study.",
        "doi_or_url": "https://doi.org/10.1016/j.cmpb.2018.04.012",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[14]",
        "reference_entry": "S. Hochreiter and J. Schmidhuber, 'Long short-term memory,' Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.",
        "claim_supported": "Long short-term memory recurrent neural networks for sequence modeling and temporal dependencies.",
        "verification_type": "Directly Verified (Peer-Reviewed Foundational Journal Article via DOI)",
        "source_verification": "Verified foundational Neural Computation paper.",
        "doi_or_url": "https://doi.org/10.1162/neco.1997.9.8.1735",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[15]",
        "reference_entry": "A. K. Choudhary, A. Mishra, K. N. Mishra, R. K. Lal, and A. Kumari, 'Detection of major depressive disorder using maximum likelihood estimation for channel selection: A novel approach,' in International Conference on Innovative Computing and Communication. Springer, 2025, pp. 375–396.",
        "claim_supported": "Maximum likelihood estimation for EEG channel selection in MDD detection.",
        "verification_type": "Directly Verified (Springer Conference Proceedings via DOI)",
        "source_verification": "Verified Springer conference proceedings paper.",
        "doi_or_url": "https://doi.org/10.1007/978-981-97-8884-2_26",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[16]",
        "reference_entry": "C.-T. Wu, H.-C. Huang, S. Huang, H.-C. Chen, P.-C. Chen, and C.-Y. Chen, 'Resting-state EEG for the diagnosis of major depressive disorder using conformal kernel support vector machine,' IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 28, no. 8, pp. 1762–1770, 2020.",
        "claim_supported": "Conformal kernel support vector machine on EEG spectral power features (83.64% accuracy across 55 subjects).",
        "verification_type": "Directly Verified (IEEE Transactions Article via DOI)",
        "source_verification": "Verified IEEE TNSRE peer-reviewed paper.",
        "doi_or_url": "https://doi.org/10.1109/TNSRE.2020.3005888",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography; attribution disentangled from Shen et al."
    },
    {
        "citation_number": "[17]",
        "reference_entry": "J. Shen, X. Zhang, X. Huang, M. Wu, J. Gao, D. Lu, Z. Ding, and B. Hu, 'An optimal channel selection for EEG-based depression detection via kernel-target alignment,' IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 7, pp. 2545–2556, 2020.",
        "claim_supported": "Optimal EEG channel selection using kernel-target alignment (KTA) for depression detection.",
        "verification_type": "Directly Verified (IEEE Transactions Article via DOI)",
        "source_verification": "Verified IEEE JBHI peer-reviewed paper.",
        "doi_or_url": "https://doi.org/10.1109/JBHI.2020.3039343",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography; attribution disentangled from Wu et al."
    },
    {
        "citation_number": "[18]",
        "reference_entry": "W. Mumtaz, L. Xia, S. S. A. Ali, M. A. M. Yasin, M. Hussain, and A. S. Malik, 'Electroencephalogram (EEG)-based computer-aided technique to diagnose major depressive disorder (MDD),' Biomedical Signal Processing and Control, vol. 31, pp. 108–115, 2017.",
        "claim_supported": "Curation and primary publication of the Hospital Universiti Sains Malaysia (HUSM) 64-subject MDD EEG dataset (BIDS nm000114_v1.0.0).",
        "verification_type": "Directly Verified (Primary Dataset Publication via DOI)",
        "source_verification": "Verified BSPC peer-reviewed primary dataset publication.",
        "doi_or_url": "https://doi.org/10.1016/j.bspc.2016.07.006",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[19]",
        "reference_entry": "L. Cohen, Time-Frequency Analysis. Englewood Cliffs, NJ: Prentice Hall, 1995.",
        "claim_supported": "Mathematical foundations of quadratic time-frequency distributions and Wigner-Ville cross-term interference.",
        "verification_type": "Directly Verified (Foundational Textbook / Monograph via ISBN)",
        "source_verification": "Verified foundational academic monograph on time-frequency analysis.",
        "doi_or_url": "ISBN: 978-0135945322",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[20]",
        "reference_entry": "F. Auger and P. Flandrin, 'Improving the readability of time-frequency and time-scale representations by the reassignment method,' IEEE Transactions on Signal Processing, vol. 43, no. 5, pp. 1068–1089, 1995.",
        "claim_supported": "Smoothing windows in time and frequency domains to suppress cross-terms in Smoothed Pseudo Wigner-Ville Distribution.",
        "verification_type": "Directly Verified (IEEE Transactions Article via DOI)",
        "source_verification": "Verified IEEE TSP peer-reviewed foundational paper.",
        "doi_or_url": "https://doi.org/10.1109/78.382399",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[21]",
        "reference_entry": "J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, 'ImageNet: A large-scale hierarchical image database,' in 2009 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2009, pp. 248–255.",
        "claim_supported": "Pretrained CNN visual feature transfer learning from ImageNet-1k large-scale natural image database.",
        "verification_type": "Directly Verified (IEEE Conference Proceedings via DOI)",
        "source_verification": "Verified IEEE CVPR foundational database paper.",
        "doi_or_url": "https://doi.org/10.1109/CVPR.2009.5206848",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[22]",
        "reference_entry": "W. Mumtaz, S. S. A. Ali, M. A. M. Yasin, and A. S. Malik, 'A deep learning framework for automatic diagnosis of unipolar depression,' International Journal of Medical Informatics, vol. 132, p. 103983, 2019.",
        "claim_supported": "Deep learning and wavelet evaluation on the HUSM depression dataset under epoch-level cross-validation.",
        "verification_type": "Directly Verified (Peer-Reviewed Journal Article via DOI)",
        "source_verification": "Verified IJMI peer-reviewed empirical study.",
        "doi_or_url": "https://doi.org/10.1016/j.ijmedinf.2019.103983",
        "status": "PASS - Fully Verified",
        "required_action": "Retained in manuscript bibliography."
    },
    {
        "citation_number": "[RETRACTED]",
        "reference_entry": "A. K. Mahato and S. Paul, 'Classification of depression patients and normal subjects based on electroencephalogram (EEG) signal using machine learning techniques,' Medical & Biological Engineering & Computing, 2020.",
        "claim_supported": "Historical claim in preliminary drafts regarding machine learning classification on EEG signals.",
        "verification_type": "Directly Verified (Official Publisher Retraction Notice)",
        "source_verification": "RETRACTED by Editor-in-Chief due to data integrity and redundancy issues (Notice published 2022).",
        "doi_or_url": "https://doi.org/10.1007/s11517-019-02096-7",
        "status": "PRUNED - Retracted Article",
        "required_action": "Excluded entirely from revised manuscript and references."
    },
    {
        "citation_number": "[TABLE-III-COMPARATORS]",
        "reference_entry": "Heterogeneous published studies cited in preliminary draft Table III (e.g., Sharma et al., Bhati et al., Seal et al.).",
        "claim_supported": "Comparative benchmarking of claimed diagnostic accuracies (98.0% - 99.5%) across literature.",
        "verification_type": "Unharmonized / Incompletely Verified Across External Sources",
        "source_verification": "Heterogeneous cross-validation protocols (predominantly random segment-level), missing DOIs, and non-harmonized cohorts.",
        "doi_or_url": "Various / Unverified",
        "status": "WITHHELD - Unharmonized Comparators",
        "required_action": "Table III withheld from revised manuscript; documented in Section 4.1."
    }
]

df = pd.DataFrame(citations_data)
out_path = "reports/FINAL_CITATION_AUDIT.csv"
df.to_csv(out_path, index=False)
print(f"Generated {out_path} with {len(df)} audit entries.")
