"""Generate fully reconciled revised research paper Word document.

Outputs:
- manuscript/Jha_MDD_Revised_Final.docx
- manuscript/Jha_MDD_Revised.docx

Incorporates:
- Explicit condition reporting: Primary Eyes-Closed (N=58) and Secondary Eyes-Open (N=61)
- Verified empirical experimental results and confusion matrix counts
- Corrected citations (pruned retracted paper, verified DOIs)
- Strict Stratified Group 10-Fold Cross-Validation descriptions
- Neutral methodological language regarding segment-level vs. participant-independent evaluation
- Balanced qualification of Nested Group-Aware Hyperparameter Tuning
- Backbone selection rationale (ResNet-18 vs ResNet-50)
- Withheld comparison table explanation
- Academic typography, publication-grade tables, headers, footers, and equations:
  * Unified Portrait Layout: Clean, continuous document flow across standard 8.5" x 11.0" pages
  * Table 1: Pretrained CNN Feature Extractors with 2-tier header and aligned columns
  * Table 2: Two-Panel Academic Table (Panel A: Eyes-Closed, Panel B: Eyes-Open) with zero fragmented headers
  * Figure 1 & Figure 2: Side-by-side confusion matrices and architecture diagram
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION_START, WD_ORIENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def apply_tbl_grid(table, widths):
    """Set explicit w:tblGrid column widths in dxa for Word XML layout."""
    tblPr = table._tbl.tblPr
    tblGrid = OxmlElement('w:tblGrid')
    for w in widths:
        gridCol = OxmlElement('w:gridCol')
        gridCol.set(qn('w:w'), str(int(w.inches * 1440)))
        tblGrid.append(gridCol)
    table._tbl.insert(table._tbl.index(tblPr) + 1, tblGrid)

def apply_table_borders(table, subtle=True):
    """Apply academic booktabs style: horizontal top/bottom/header borders, no vertical borders."""
    tblPr = table._tbl.tblPr
    inside_color = "E5E7EB" if subtle else "CCCCCC"
    borders_xml = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="single" w:sz="10" w:space="0" w:color="111827"/>\n'
        f'  <w:bottom w:val="single" w:sz="10" w:space="0" w:color="111827"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{inside_color}"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(borders_xml)

def apply_invisible_borders(table):
    """Remove all table borders (for side-by-side figure layout)."""
    tblPr = table._tbl.tblPr
    borders_xml = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="none"/>\n'
        f'  <w:bottom w:val="none"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'  <w:insideH w:val="none"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(borders_xml)

def apply_cell_margins(cell, top=80, bottom=80, left=60, right=60):
    """Set inner cell padding in dxa."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_background(cell, color_hex="F3F4F6"):
    """Set background fill color on a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{color_hex}"/>')
    tcPr.append(shd)

def configure_table_layout(table, width_inches=6.50):
    """
    Configure bulletproof Word XML table properties:
    - Fixed layout (<w:tblLayout w:type="fixed"/>) to prevent Word from auto-resizing columns
    - Zero margin indent (<w:tblInd w:w="0" w:type="dxa"/>) to prevent negative left gutter shifts
    - Explicit total width in dxa (<w:tblW w:w="..." w:type="dxa"/>)
    - Centered table alignment (<w:jc w:val="center"/>)
    """
    tblPr = table._tbl.tblPr
    for tag in ["tblW", "tblInd", "tblLayout", "jc"]:
        for elem in tblPr.findall(qn(f"w:{tag}")):
            tblPr.remove(elem)
            
    tblW = OxmlElement('w:tblW')
    tblW.set(qn('w:w'), str(int(width_inches * 1440)))
    tblW.set(qn('w:type'), 'dxa')
    tblPr.append(tblW)
    
    jc = OxmlElement('w:jc')
    jc.set(qn('w:val'), 'center')
    tblPr.append(jc)
    
    tblInd = OxmlElement('w:tblInd')
    tblInd.set(qn('w:w'), '0')
    tblInd.set(qn('w:type'), 'dxa')
    tblPr.append(tblInd)
    
    tblLayout = OxmlElement('w:tblLayout')
    tblLayout.set(qn('w:type'), 'fixed')
    tblPr.append(tblLayout)

def setup_header_footer(section):
    """Configure running header and page numbering footer on a section."""
    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hp.text = "" # clear
    hr = hp.add_run("Jha et al. · Automated MDD Detection via SPWVD-EEG and Deep Transfer Learning")
    hr.font.name = "Times New Roman"
    hr.font.size = Pt(8.5)
    hr.font.italic = True
    hr.font.color.rgb = RGBColor(120, 120, 120)
    
    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.text = "" # clear
    fr = fp.add_run("— Page ")
    fr.font.name = "Times New Roman"
    fr.font.size = Pt(9.0)
    fr.font.color.rgb = RGBColor(100, 100, 100)
    fldSimple = parse_xml(r'<w:fldSimple %s w:instr="PAGE"/>' % nsdecls('w'))
    fp._p.append(fldSimple)
    fr2 = fp.add_run(" —")
    fr2.font.name = "Times New Roman"
    fr2.font.size = Pt(9.0)
    fr2.font.color.rgb = RGBColor(100, 100, 100)

def create_revised_manuscript():
    doc = docx.Document()
    
    # ----------------------------------------------------
    # Unified Portrait Layout: 8.5" x 11.0" with 1.0" margins (6.50" printable width)
    # ----------------------------------------------------
    section1 = doc.sections[0]
    section1.orientation = WD_ORIENT.PORTRAIT
    section1.page_width = Inches(8.5)
    section1.page_height = Inches(11.0)
    section1.top_margin = Inches(1.0)
    section1.bottom_margin = Inches(1.0)
    section1.left_margin = Inches(1.0)
    section1.right_margin = Inches(1.0)
    setup_header_footer(section1)
    
    # Global Styles Configuration
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(0x1F, 0x24, 0x2E)
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(4.0)
    style_normal.paragraph_format.space_before = Pt(0)
    style_normal.paragraph_format.widow_control = True
    
    h1_style = doc.styles['Heading 1']
    h1_style.font.name = 'Times New Roman'
    h1_style.font.size = Pt(13)
    h1_style.font.bold = True
    h1_style.font.color.rgb = RGBColor(0x11, 0x18, 0x27)
    h1_style.paragraph_format.space_before = Pt(14)
    h1_style.paragraph_format.space_after = Pt(4)
    h1_style.paragraph_format.keep_with_next = True
    
    h2_style = doc.styles['Heading 2']
    h2_style.font.name = 'Times New Roman'
    h2_style.font.size = Pt(11.5)
    h2_style.font.bold = True
    h2_style.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)
    h2_style.paragraph_format.space_before = Pt(10)
    h2_style.paragraph_format.space_after = Pt(3)
    h2_style.paragraph_format.keep_with_next = True
    
    # Title Block
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(14)
    run_title = p_title.add_run("Automated Detection of Major Depressive Disorder Using SPWVD-Based EEG Time-Frequency Images and Deep Learning Features: A Reproducible, Participant-Independent Benchmark")
    run_title.bold = True
    run_title.font.size = Pt(15.5)
    run_title.font.name = "Times New Roman"
    
    # Abstract Block
    p_abs_h = doc.add_paragraph()
    p_abs_h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_abs_h.paragraph_format.left_indent = Inches(0.3)
    p_abs_h.paragraph_format.right_indent = Inches(0.3)
    p_abs_h.paragraph_format.space_before = Pt(4)
    p_abs_h.paragraph_format.space_after = Pt(3)
    p_abs_h.paragraph_format.keep_with_next = True
    run_abs_h = p_abs_h.add_run("Abstract")
    run_abs_h.bold = True
    run_abs_h.font.size = Pt(11)
    run_abs_h.font.name = "Times New Roman"
    
    abstract_text = (
        "Major Depressive Disorder (MDD) is a severe, debilitating neuropsychiatric disorder and a leading contributor to global disability. "
        "Electroencephalography (EEG) offers high temporal resolution to capture non-stationary neural alterations associated with MDD. "
        "However, machine-learning pipelines evaluated using random segment-level data splitting can allow temporal segments from the same participant "
        "into both training and testing folds, potentially producing optimistic performance estimates due to inter-segment correlation. "
        "In this study, we present a reproducible, participant-independent machine-learning framework combining Smoothed Pseudo Wigner-Ville Distribution (SPWVD) "
        "time-frequency spectral representations, pretrained deep convolutional neural network (CNN) feature extraction, and classical classifiers. "
        "We evaluate the public 64-participant dataset from Hospital Universiti Sains Malaysia (30 healthy controls and 34 MDD patients) across resting-state conditions "
        "under strict Stratified Group 10-Fold Cross-Validation, enforcing participant-independent fold isolation between training and test sets. "
        "In the primary eyes-closed resting cohort (N = 58 participants: 28 controls, 30 MDD), ResNet-18 features paired with Linear SVM achieved 89.66% "
        "participant-level diagnostic accuracy (mean fold accuracy 89.90 ± 8.34% with ddof=0, ±8.79% with ddof=1), 93.33% sensitivity, 85.71% specificity, 90.32% F1-score, and 95.36% ROC-AUC. "
        "Medium Gaussian SVM and Neural Network (Multi-Layer Perceptron) classifiers similarly achieved 89.66% accuracy (90.00% sensitivity, 89.29% specificity). "
        "In the secondary eyes-open resting cohort (N = 61 participants: 29 controls, 32 MDD), Medium Gaussian SVM and Neural Network achieved 91.80% participant accuracy "
        "(mean fold accuracy 91.48 ± 12.49% with ddof=0, ±13.16% with ddof=1), 90.62% sensitivity, 93.10% specificity, 92.06% F1-score, and ROC-AUC of 93.32% (Medium Gaussian SVM) and 96.01% (Neural Network); "
        "Linear SVM attained 88.52% accuracy and 95.69% ROC-AUC. "
        "At the individual 2-second segment level, classification accuracy ranged from 77.7% to 85.3% across the seven non-degenerate classifiers (Linear SVM, Polynomial SVMs, Medium Gaussian SVM, k-NN, Bagged Trees, and Neural Network, excluding Fine Gaussian SVM which overfit with ~55% accuracy), while participant-level diagnostic "
        "accuracy reached 89.66% for eyes-closed and 91.80% for eyes-open through posterior probability aggregation across each subject's epochs. "
        "These empirical results establish a reproducible, participant-independent baseline for automated depression detection on this clinical cohort."
    )
    p_abs = doc.add_paragraph(abstract_text)
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.left_indent = Inches(0.3)
    p_abs.paragraph_format.right_indent = Inches(0.3)
    p_abs.paragraph_format.space_after = Pt(4)
    for r in p_abs.runs:
        r.font.size = Pt(10)
        r.font.name = "Times New Roman"
    
    # Keywords
    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.left_indent = Inches(0.3)
    p_kw.paragraph_format.right_indent = Inches(0.3)
    p_kw.paragraph_format.space_after = Pt(14)
    r_kw_title = p_kw.add_run("Keywords: ")
    r_kw_title.bold = True
    r_kw_title.font.size = Pt(9.5)
    r_kw_title.font.name = "Times New Roman"
    r_kw_val = p_kw.add_run("Major Depressive Disorder (MDD), Electroencephalography (EEG), Smoothed Pseudo Wigner-Ville Distribution (SPWVD), Deep Learning, Transfer Learning, Group Cross-Validation, Reproducible Machine Learning.")
    r_kw_val.font.size = Pt(9.5)
    r_kw_val.font.name = "Times New Roman"
    
    # ----------------------------------------------------
    # 1. Introduction
    # ----------------------------------------------------
    doc.add_heading("1. Introduction", level=1)
    
    intro_p1 = (
        "Depression is one of the most prevalent mental health disorders globally, affecting an estimated 5% of adults worldwide [1]. "
        "It is clinically characterized by persistent sadness, loss of interest or pleasure in previously rewarding activities (anhedonia), sleep disturbances, "
        "cognitive fatigue, and impaired concentration [1]. According to the World Health Organization (WHO) and Global Burden of Disease studies, depressive disorders "
        "represent the leading cause of non-fatal disability worldwide, contributing immensely to the global burden of disease and economic loss [2]. "
        "When depressive symptoms persist for two weeks or longer and cause substantial functional impairment, a diagnosis of Major Depressive Disorder (MDD) is established "
        "in accordance with DSM-5 criteria [3]. The COVID-19 pandemic further exacerbated the global prevalence of depression, triggered by widespread isolation, bereavement, "
        "economic dislocation, and prolonged psychological distress [4]."
    )
    doc.add_paragraph(intro_p1)
    
    intro_p2 = (
        "Current diagnostic procedures for MDD rely predominantly on subjective clinical psychiatric interviews and self-administered questionnaires (such as the Beck "
        "Depression Inventory and Hamilton Depression Rating Scale). These clinical assessments are time-consuming, prone to inter-rater subjectivity, and frequently delay "
        "timely intervention. Consequently, objective neurophysiological biomarkers are urgently required. Electroencephalography (EEG) directly records electrical potentials "
        "generated by cerebral neuronal firing with millisecond temporal resolution [5]. Alterations in resting-state oscillatory dynamics—such as fronto-central alpha asymmetry, "
        "gamma synchrony deficits, and abnormal phase coupling—have been documented in patients with depressive disorders [6, 7]. "
        "For example, Lechner and Northoff [8] demonstrated that resting-state EEG phase dynamics differentiate MDD patients from healthy individuals and bipolar disorder cohorts. "
        "Earlier computational investigations applied handcrafted statistical and nonlinear features to classify depressive EEG signals. Erguzel et al. [9] used genetic algorithms "
        "with artificial neural networks, achieving 89.12% accuracy. Hosseinifard et al. [10] evaluated nonlinear dynamic features (Lyapunov exponents, correlation dimension), while "
        "Liao et al. [11] employed common spatial patterns to achieve 81.23% diagnostic accuracy. Mohammadi et al. [12] utilized entropy metrics with a fuzzy logic classifier to classify depression."
    )
    doc.add_paragraph(intro_p2)
    
    intro_p3 = (
        "In recent years, deep learning methodologies have gained significant traction for automated physiological signal interpretation. Acharya et al. [13] developed a 13-layer "
        "convolutional neural network (CNN) operating on raw EEG signals, achieving over 93% accuracy. Subsequent studies integrated temporal sequence modeling, such as recurrent neural "
        "networks and long short-term memory (LSTM) architectures, to capture long-range contextual temporal dependencies across EEG epochs [14]. Choudhary et al. [15] explored "
        "maximum likelihood estimation for EEG channel selection. Other investigations extracted spectral power distributions and applied kernel-based classifiers; for instance, "
        "Wu et al. [16] employed conformal kernel support vector machines on power features, while Shen et al. [17] developed kernel-target alignment for optimal channel selection."
    )
    doc.add_paragraph(intro_p3)
    
    intro_p4 = (
        "A critical methodological consideration in physiological signal classification is the distinction between segment-level and participant-independent validation protocols. "
        "Continuous EEG recordings from an individual participant exhibit shared anatomical geometry, electrode impedances, and unique baseline dynamics across temporal segments. "
        "When continuous recordings are sliced into short temporal windows (e.g., 2–5 seconds) and randomly partitioned without grouping by participant, models can exploit "
        "subject-specific signatures rather than generalizable pathology-related patterns. This segment-level cross-validation can produce overly optimistic performance estimates "
        "that may not translate to unseen patients in clinical settings. Group-aware validation is therefore essential to prevent this form "
        "of correlation bias, and establishing rigorous participant-isolated benchmarks is critical for realistic translational assessment."
    )
    doc.add_paragraph(intro_p4)
    
    intro_p5 = (
        "In this study, we present a reproducible, participant-independent investigation into the automated detection of MDD using Smoothed Pseudo Wigner-Ville Distribution (SPWVD) "
        "time-frequency representations and pretrained CNN feature extraction. The key contributions of this work are threefold:\n"
        "(1) We establish a strict Stratified Group 10-Fold Cross-Validation protocol that quarantines all recordings, segments, and representations of each participant to a single fold, "
        "enforcing participant-independent evaluation;\n"
        "(2) We perform an empirical evaluation of four benchmark pretrained CNN backbones (ResNet-18, ResNet-50, AlexNet, VGG-16) paired with eight classical machine learning estimators, "
        "clarifying trade-offs between feature dimensionality and classification accuracy; and\n"
        "(3) We evaluate both individual segment-level discriminability and participant-level probability aggregation across both Eyes-Closed (N=58) and Eyes-Open (N=61) resting states, "
        "providing a verified, reproducible baseline for this clinical cohort."
    )
    doc.add_paragraph(intro_p5)
    
    # ----------------------------------------------------
    # 2. Materials and Methods
    # ----------------------------------------------------
    doc.add_heading("2. Materials and Methods", level=1)
    
    doc.add_heading("2.1 Dataset Identity, Provenance, and Participant Cohorts", level=2)
    mm_data = (
        "The electroencephalographic data utilized in this investigation originates from the publicly accessible dataset collected at the Hospital Universiti Sains Malaysia (HUSM) "
        "and curated by Mumtaz et al. [18], formatted according to the Brain Imaging Data Structure (BIDS version 1.7.0, NEMAR accession nm000114_v1.0.0, DOI: 10.82901/nemar.nm000114). "
        "The authoritative dataset comprises 64 unique participants: 30 healthy controls (HC, subjects sub-HS1 to sub-HS30) and 34 patients clinically diagnosed with Major Depressive Disorder "
        "(MDD, subjects sub-MDDS1 to sub-MDDS34) aged 12 to 77 years. MDD diagnoses were established by certified consultant psychiatrists according to DSM-IV criteria.\n\n"
        "A critical clarification must be noted regarding dataset metadata: the introductory narrative in the BIDS README text originally contained an unedited typographical note stating "
        "'120 subjects were recruited'. An exhaustive physical inventory of the repository establishes that exactly 64 unique participants and 181 valid recordings are deposited in the archive. "
        "Participants completed up to three experimental protocols:\n"
        "1. Eyes-closed resting state: 58 recordings across 58 unique participants (28 HC, 30 MDD; missing recordings for sub-HS12, sub-HS18, sub-MDDS4, sub-MDDS8, sub-MDDS12, sub-MDDS16).\n"
        "2. Eyes-open resting state: 62 recordings across 61 unique participants (29 HC, 32 MDD; sub-HS15 contributed two runs; missing recordings for sub-HS25, sub-MDDS7, sub-MDDS8).\n"
        "3. Auditory oddball P300 task: 61 recordings across 61 unique participants (28 HC, 33 MDD; missing recordings for sub-HS20, sub-HS21, sub-MDDS9). "
        "In accordance with our focus on resting-state neurophysiology, the active P300 task recordings were reserved for future cognitive studies."
    )
    doc.add_paragraph(mm_data)
    
    doc.add_heading("2.2 Acquisition Montage and Preprocessing", level=2)
    mm_prep = (
        "EEG signals were acquired at a sampling frequency of 256.0 Hz using a 19-channel monopolar electrode cap positioned according to the International 10-20 system: "
        "Fp1, F3, C3, P3, O1, F7, T3, T5, Fz, Fp2, F4, C4, P4, O2, F8, T4, T6, Cz, and Pz. Hardware acquisition filtering included a 0.5 Hz high-pass analog filter, a 70.0 Hz low-pass filter, "
        "and a 50.0 Hz notch filter to attenuate power line interference. All electrode impedances were maintained below 5 kΩ. Continuous recordings deposited in the archive were inspected "
        "to ensure segments containing severe ocular or muscular artifacts were excised during initial curation.\n\n"
        "To enforce montage homogeneity across all subjects, only the 19 standard 10-20 EEG channels common to all 181 recordings were extracted; auxiliary bipolar derivations present in a subset of "
        "recordings (e.g., 23A-23R, 24A-24R) were excluded. Continuous signals were filtered with a digital zero-phase bandpass filter between 0.5 Hz and 45.0 Hz to retain delta, theta, alpha, beta, "
        "and low-gamma physiological rhythms while suppressing high-frequency electromyographic noise."
    )
    doc.add_paragraph(mm_prep)
    
    doc.add_heading("2.3 Window Segmentation", level=2)
    mm_seg = (
        "Continuous multi-channel EEG recordings were segmented into non-overlapping temporal windows of length T = 2.0 seconds (N = 512 samples per window). "
        "Segmentation was executed strictly within each individual recording file; windows never cross session or recording boundaries. "
        "To ensure balanced participant contributions while maintaining computational efficiency, 15 non-overlapping epochs were systematically sampled across each recording. "
        "This yielded 870 epochs for the primary eyes-closed cohort (N = 58 subjects) and 930 epochs for the eyes-open cohort (N = 61 subjects). "
        "Every segment preserves complete provenance tracking, including participant ID, diagnostic label, recording condition, and start/end sample timestamps."
    )
    doc.add_paragraph(mm_seg)
    
    doc.add_heading("2.4 Smoothed Pseudo Wigner-Ville Distribution (SPWVD)", level=2)
    mm_spwvd1 = (
        "The Wigner-Ville distribution provides optimal joint time-frequency energy localization for non-stationary signals. However, for multi-component signals like EEG, quadratic cross-terms "
        "generate severe oscillatory interference artifacts between distinct frequency components. The Smoothed Pseudo Wigner-Ville Distribution (SPWVD) suppresses cross-terms by applying "
        "independent smoothing windows in both the time and frequency (lag) domains [19, 20]:"
    )
    doc.add_paragraph(mm_spwvd1)
    
    p_eq1 = doc.add_paragraph()
    p_eq1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq1.paragraph_format.space_before = Pt(6)
    p_eq1.paragraph_format.space_after = Pt(6)
    r_eq1 = p_eq1.add_run("W(t, f) = ∫ h(τ) [ ∫ g(s - t) z(s + τ/2) z*(s - τ/2) ds ] e^(-j 2π f τ) dτ          (1)")
    r_eq1.font.name = "Times New Roman"
    r_eq1.font.size = Pt(10.5)
    r_eq1.font.italic = True
    
    mm_spwvd2 = (
        "where z(t) is the analytic representation of the real EEG signal x(t) computed via the Hilbert transform to eliminate negative frequency cross-terms; "
        "g(t) is a time-smoothing window that suppresses frequency-direction cross-terms; and h(τ) is a lag-smoothing window that suppresses time-direction cross-terms.\n\n"
        "In our implementation, frequency smoothing is performed using a Kaiser window h(τ) of length 255 samples with shape parameter β = 5.0, optimizing the trade-off between mainlobe "
        "spectral resolution and sidelobe suppression. Time smoothing is performed using a symmetric Hamming window g(t) of length 31 samples. For each multi-channel epoch, the SPWVD energy distribution "
        "is computed across the 19 standard channels and averaged to obtain a robust, spatially balanced time-frequency representation. Positive frequencies [0, 128 Hz] are retained, clipped to eliminate "
        "residual negative ripples, normalized via 99.5th percentile scaling, and resized via bilinear interpolation to 224 × 224 pixels. The single-channel energy map is expanded to a 3-channel RGB image "
        "and standardized using official ImageNet channel-wise normalization parameters (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])."
    )
    doc.add_paragraph(mm_spwvd2)
    
    doc.add_heading("2.5 Pretrained Deep Feature Extraction and Backbone Selection Rationale", level=2)
    mm_tl = (
        "Rather than training deep networks from scratch on a limited clinical cohort, transfer learning was utilized to leverage rich visual feature hierarchies learned from large-scale natural image datasets (ImageNet-1k) [21]. "
        "Four benchmark deep convolutional architectures were systematically evaluated:\n"
        "1. ResNet-18: Residual network containing 18 layers with shortcut connections, producing a 512-dimensional feature embedding from the global average pooling layer (avgpool).\n"
        "2. ResNet-50: 50-layer bottleneck residual network producing a 2,048-dimensional feature embedding.\n"
        "3. AlexNet: 8-layer deep architecture, extracting a 4,096-dimensional feature vector from the penultimate fully connected layer (classifier[5]).\n"
        "4. VGG-16: 16-layer architecture with small 3×3 receptive fields, extracting a 4,096-dimensional feature vector from classifier[5].\n\n"
        "All CNN backbones operate strictly as fixed feature extractors: all convolutional and pooling weights remain frozen, and activation vectors are extracted without gradient backpropagation. "
        "This design guarantees deterministic, compute-efficient feature extraction free from training overfitting.\n\n"
        "Backbone Selection Rationale: Across the comparative evaluation, ResNet-18 was selected as the primary feature extractor for comprehensive downstream classifier benchmarking. "
        "This decision was guided by two key factors: first, ResNet-18 achieved the highest classification accuracy on the primary eyes-closed resting-state cohort (86.21% vs. 84.48% for ResNet-50 with Weighted KNN); "
        "second, ResNet-18 produces a compact 512-dimensional feature embedding—one-fourth the size of ResNet-50 (2,048-D) and one-eighth that of AlexNet and VGG-16 (4,096-D). "
        "On clinical cohorts of moderate sample size, lower-dimensional representations substantially mitigate the risk of the curse of dimensionality and classifier overfitting. "
        "Although ResNet-50 attained higher accuracy on the secondary eyes-open condition (90.16% vs. 88.52%), ResNet-18 was favored for the primary benchmark due to its superior eyes-closed performance and dimensional parsimony."
    )
    doc.add_paragraph(mm_tl)
    
    doc.add_heading("2.6 Downstream Machine Learning Classifiers", level=2)
    mm_clf = (
        "The deep feature vectors extracted by the CNN backbones were passed to eight classical machine learning classifiers spanning linear, kernel, distance-based, ensemble, and neural network paradigms:\n"
        "1. Linear SVM: Support Vector Classifier with linear kernel.\n"
        "2. Quadratic SVM: Support Vector Classifier with polynomial kernel of degree 2.\n"
        "3. Cubic SVM: Support Vector Classifier with polynomial kernel of degree 3.\n"
        "4. Fine Gaussian SVM: Radial Basis Function (RBF) kernel with narrow kernel scale (γ = 0.5).\n"
        "5. Medium Gaussian SVM: RBF kernel with standard scale parameter (γ = 1 / (P · Var(X))).\n"
        "6. Weighted KNN: K-Nearest Neighbors (k = 10) with inverse Euclidean distance weighting.\n"
        "7. Ensemble Bagged Trees: Bootstrap-aggregated decision forest comprising 100 trees.\n"
        "8. Neural Network (MLP): Multi-Layer Perceptron containing a single hidden layer of 100 neurons with ReLU activation and Adam optimization.\n\n"
        "Leakage-Prevention Rule: Feature standardization (StandardScaler) was fit strictly on training fold features and applied out-of-sample to test fold features. Zero test-fold data was ever exposed during scaling or estimator fitting."
    )
    doc.add_paragraph(mm_clf)
    
    doc.add_heading("2.7 Validation Protocol and Participant-Level Aggregation", level=2)
    mm_val1 = (
        "Model evaluation was conducted using Stratified Group 10-Fold Cross-Validation. All recordings and epochs originating from each participant were assigned exclusively to a single fold. "
        "Across the 64 enrolled participants, each fold contains exactly 3 Healthy Controls and either 3 or 4 MDD Patients (Folds 0–3: 7 participants; Folds 4–9: 6 participants). "
        "Under no circumstances did any participant appear in both training and test partitions of any fold. For multi-run participants (e.g. sub-HS15 in the eyes-open condition), all runs remained strictly quarantined to Fold 9.\n\n"
        "Evaluation was conducted at two explicit levels:\n"
        "1. Primary Evaluation Unit (Participant-Level): For each test subject i with S_i segments, segment posterior probabilities p_{i, s} = P(MDD | x_{i, s}) were averaged to produce a single diagnostic probability:"
    )
    doc.add_paragraph(mm_val1)
    
    p_eq2 = doc.add_paragraph()
    p_eq2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq2.paragraph_format.space_before = Pt(6)
    p_eq2.paragraph_format.space_after = Pt(6)
    r_eq2 = p_eq2.add_run("P̂_i = (1 / S_i) ∑_{s=1}^{S_i} p_{i, s}          (2)")
    r_eq2.font.name = "Times New Roman"
    r_eq2.font.size = Pt(10.5)
    r_eq2.font.italic = True
    
    mm_val2 = (
        "Diagnostic classification was assigned as ŷ_i = 1 if P̂_i ≥ 0.5, and 0 otherwise. Primary performance metrics were calculated on the pooled out-of-fold participant decisions as well as fold-wise means ± standard deviations.\n"
        "2. Secondary Evaluation Unit (Segment-Level): Raw segment predictions were evaluated across all individual epochs (870 for eyes-closed, 930 for eyes-open) to assess raw time-frequency discriminability."
    )
    doc.add_paragraph(mm_val2)
    
    # Figure 1: Pipeline Schematic
    diag_path = "results/pipeline_architecture_diagram.png"
    if os.path.exists(diag_path):
        p_fig1 = doc.add_paragraph()
        p_fig1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig1.paragraph_format.space_before = Pt(8)
        p_fig1.paragraph_format.space_after = Pt(4)
        run_fig1 = p_fig1.add_run()
        run_fig1.add_picture(diag_path, width=Inches(6.2))
        
        p_cap1 = doc.add_paragraph()
        p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap1.paragraph_format.space_before = Pt(4)
        p_cap1.paragraph_format.space_after = Pt(12)
        r_cap1_label = p_cap1.add_run("Fig. 1. ")
        r_cap1_label.bold = True
        r_cap1_label.font.size = Pt(9.5)
        r_cap1_label.font.name = "Times New Roman"
        r_cap1 = p_cap1.add_run("End-to-end participant-independent classification pipeline combining EEG preprocessing, SPWVD time-frequency transformation, pretrained ResNet-18 feature extraction, and stratified group cross-validation.")
        r_cap1.font.size = Pt(9.5)
        r_cap1.font.name = "Times New Roman"
        r_cap1.font.italic = True
    
    # ----------------------------------------------------
    # 3. Results
    # ----------------------------------------------------
    doc.add_heading("3. Results", level=1)
    
    doc.add_heading("3.1 Pretrained CNN Backbone Architecture Comparison", level=2)
    res_cnn = (
        "We first compared the feature representation capacity of the four pretrained CNN architectures (ResNet-18, ResNet-50, AlexNet, and VGG-16) under identical Stratified Group 10-Fold Cross-Validation "
        "using a Weighted KNN downstream classifier. As detailed in Table 1, ResNet-18 achieved the highest classification performance on the primary eyes-closed cohort, yielding a pooled participant-level "
        "accuracy of 86.21% (mean fold accuracy 86.57 ± 10.11%) and an F1-score of 86.67%. ResNet-50 achieved 84.48% pooled accuracy on eyes-closed and 90.16% on eyes-open. AlexNet and VGG-16 yielded "
        "lower accuracies (81.03% and 79.31% on eyes-closed, respectively). Based on these findings and its compact 512-dimensional feature space, ResNet-18 was adopted as the baseline feature extractor."
    )
    doc.add_paragraph(res_cnn)
    
    # Table 1: CNN Models (Portrait - Width 6.50 inches, 2-Tier Header)
    p_t1 = doc.add_paragraph()
    p_t1.paragraph_format.keep_with_next = True
    p_t1.paragraph_format.space_before = Pt(8)
    p_t1.paragraph_format.space_after = Pt(3)
    r_t1 = p_t1.add_run("Table 1. Diagnostic Accuracy of Pretrained CNN Feature Extractors (Stratified Group 10-Fold Cross-Validation, Weighted KNN Classifier)")
    r_t1.bold = True
    r_t1.font.size = Pt(9.5)
    r_t1.font.name = "Times New Roman"
    
    t1_widths = [Inches(1.25), Inches(1.05), Inches(0.85), Inches(1.25), Inches(0.85), Inches(1.25)]
    # Total = 6.50 in (exact printable portrait width)
    
    table1 = doc.add_table(rows=6, cols=6)
    table1.autofit = False
    configure_table_layout(table1, width_inches=6.50)
    apply_tbl_grid(table1, t1_widths)
    apply_table_borders(table1, subtle=True)
    
    # Repeat header & prevent row split
    t1_r0 = table1.rows[0]
    t1_r0_trPr = t1_r0._tr.get_or_add_trPr()
    t1_r0_trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    t1_r0_trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    
    t1_r1 = table1.rows[1]
    t1_r1_trPr = t1_r1._tr.get_or_add_trPr()
    t1_r1_trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    t1_r1_trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    
    # Merge Col 0 vertically
    t1_r0.cells[0].merge(t1_r1.cells[0])
    p_c0 = t1_r0.cells[0].paragraphs[0]
    p_c0.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p_c0.add_run("CNN Architecture")
    r.bold = True
    r.font.size = Pt(8.5)
    r.font.name = "Times New Roman"
    
    # Merge Col 1 vertically
    t1_r0.cells[1].merge(t1_r1.cells[1])
    p_c1 = t1_r0.cells[1].paragraphs[0]
    p_c1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_c1.add_run("Feature Dimension")
    r.bold = True
    r.font.size = Pt(8.0)
    r.font.name = "Times New Roman"
    
    # Merge EC Cohort cols 2-3 horizontally
    c_t1_ec = t1_r0.cells[2].merge(t1_r0.cells[3])
    p_ec = c_t1_ec.paragraphs[0]
    p_ec.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_ec.add_run("Eyes-Closed Cohort (N = 58)")
    r.bold = True
    r.font.size = Pt(8.5)
    r.font.name = "Times New Roman"
    
    # Merge EO Cohort cols 4-5 horizontally
    c_t1_eo = t1_r0.cells[4].merge(t1_r0.cells[5])
    p_eo = c_t1_eo.paragraphs[0]
    p_eo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_eo.add_run("Eyes-Open Cohort (N = 61)")
    r.bold = True
    r.font.size = Pt(8.5)
    r.font.name = "Times New Roman"
    
    # Subheaders in Row 1
    t1_sub = ["Pooled Acc. (%)", "10-Fold Mean ± SD (%)", "Pooled Acc. (%)", "10-Fold Mean ± SD (%)"]
    for idx, sh in enumerate(t1_sub):
        cell = t1_r1.cells[idx + 2]
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(sh)
        r.bold = True
        r.font.size = Pt(8.0)
        r.font.name = "Times New Roman"
        
    t1_data = [
        ["ResNet-18", "512", "86.21", "86.57 ± 10.11", "88.52", "88.14 ± 11.98"],
        ["ResNet-50", "2048", "84.48", "84.81 ± 11.19", "90.16", "89.81 ± 12.35"],
        ["AlexNet", "4096", "81.03", "81.81 ± 15.34", "85.25", "85.14 ± 15.30"],
        ["VGG-16", "4096", "79.31", "78.90 ± 11.00", "83.61", "83.71 ± 12.43"]
    ]
    for r_idx, row in enumerate(t1_data):
        r_trPr = table1.rows[r_idx + 2]._tr.get_or_add_trPr()
        r_trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        for c_idx, val in enumerate(row):
            cell = table1.cell(r_idx + 2, c_idx)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val.replace(" ± ", "\u00A0±\u00A0"))
            r.font.size = Pt(8.5)
            r.font.name = "Times New Roman"
            
    # Set explicit cell widths
    t1_r0.cells[0].width = t1_widths[0]
    t1_r0.cells[1].width = t1_widths[1]
    c_t1_ec.width = Inches(t1_widths[2].inches + t1_widths[3].inches)
    c_t1_eo.width = Inches(t1_widths[4].inches + t1_widths[5].inches)
    for i in [2, 3, 4, 5]:
        t1_r1.cells[i].width = t1_widths[i]
        
    for r_idx in range(2, 6):
        for c_idx in range(6):
            table1.cell(r_idx, c_idx).width = t1_widths[c_idx]
            
    # Keep rows together
    for row in table1.rows[:-1]:
        for cell in row.cells:
            cell.paragraphs[0].paragraph_format.keep_with_next = True
            
    for row in table1.rows:
        for cell in row.cells:
            apply_cell_margins(cell, top=70, bottom=70, left=40, right=40)
            
    doc.add_paragraph() # spacer
    
    doc.add_heading("3.2 Classification Performance Across Machine Learning Estimators", level=2)
    res_clf = (
        "The 512-dimensional deep features extracted via ResNet-18 were subsequently evaluated across eight machine learning estimators under the strict 10-fold group cross-validation scheme. "
        "The empirical results are summarized in Table 2 (Table 2A for Eyes-Closed and Table 2B for Eyes-Open) for both the Eyes-Closed (N = 58 participants) and Eyes-Open (N = 61 participants) resting-state cohorts.\n\n"
        "In the primary Eyes-Closed cohort (N = 58: 28 controls, 30 MDD):\n"
        "- Linear SVM, Medium Gaussian SVM, and the Neural Network (Multi-Layer Perceptron) achieved the top participant-level accuracy of 89.66% (mean fold accuracy 89.90 ± 8.34% for Linear SVM; 89.90 ± 11.19% for Medium Gaussian SVM and Neural Network).\n"
        "- Linear SVM achieved exceptional diagnostic balance: 93.33% sensitivity (28/30 MDD identified), 85.71% specificity (24/28 controls identified), an F1-score of 90.32%, and an ROC-AUC of 95.36% (TN=24, FP=4, FN=2, TP=28; 95% Wilson score CI: [79.21%, 95.17%]).\n"
        "- Medium Gaussian SVM and Neural Network achieved 90.00% sensitivity and 89.29% specificity (TN=25, FP=3, FN=3, TP=27; 95% Wilson score CI: [79.21%, 95.17%]). Paired McNemar testing between Linear SVM and Medium Gaussian SVM revealed identical overall accuracy with discordant predictions on only 4 of 58 participants (p = 1.000), confirming that their diagnostic performance is not statistically distinguishable on this sample cohort.\n"
        "- Cubic SVM attained 89.66% accuracy and 90.32% F1-score; Quadratic SVM attained 87.93%; Weighted KNN and Ensemble Bagged Trees reached 86.21%.\n"
        "- Fine Gaussian SVM yielded 50.00% accuracy (86.67% sensitivity but only 10.71% specificity; TN=3, FP=25, FN=4, TP=26) due to severe overfitting with narrow RBF kernel widths in high-dimensional space.\n\n"
        "In the Eyes-Open cohort (N = 61: 29 controls, 32 MDD):\n"
        "- Medium Gaussian SVM and Neural Network attained the highest diagnostic performance: 91.80% participant-level accuracy (mean fold accuracy 91.48 ± 12.49%; 95% Wilson score CI: [82.21%, 96.45%]), 90.62% sensitivity, 93.10% specificity, 92.06% F1-score, and 93.32%–96.01% ROC-AUC (TN=27, FP=2, FN=3, TP=29).\n"
        "- Linear SVM, Quadratic SVM, Cubic SVM, and Weighted KNN each achieved 88.52% participant accuracy (95% Wilson score CI: [78.16%, 94.33%]; 90.62% sensitivity, 86.21% specificity; TN=25, FP=4, FN=3, TP=29).\n"
        "- Ensemble Bagged Trees achieved 86.89% participant accuracy (TN=24, FP=5, FN=3, TP=29).\n\n"
        "Evaluation was conducted at two distinct levels: raw segment-level predictions across all individual temporal windows (870 segments for eyes-closed, 930 segments for eyes-open) "
        "and participant-level diagnostic decisions aggregated across each subject's segments (58 participants for eyes-closed, 61 participants for eyes-open). "
        "Across the seven standard non-degenerate classifiers, segment-level accuracy ranged from 77.70% to 81.84% on eyes-closed (bounded by Cubic SVM at 77.70% and Neural Network at 81.84%) "
        "and from 80.65% to 85.27% on eyes-open (bounded by Linear SVM at 80.65% and Neural Network at 85.27%), whereas Fine Gaussian SVM collapsed to 55.17% (EC) and 54.84% (EO) due to severe narrow-kernel overfitting. "
        "For the top estimators (Linear SVM, Medium Gaussian SVM, and Neural Network), segment accuracies were 80.34%–81.84% (EC) and 80.65%–85.27% (EO), compared to participant-level "
        "diagnostic accuracies of 89.66% (EC) and 88.52%–91.80% (EO). Because clinical diagnoses are rendered per participant rather than per 2-second snippet, aggregating segment posteriors produces "
        "participant-level classifications that reduce the influence of isolated segment-level classification errors."
    )
    doc.add_paragraph(res_clf)
    
    # ----------------------------------------------------
    # Table 2: Two-Panel Academic Table in Portrait Mode (Width 6.50 inches)
    # Split into Table 2A and Table 2B to prevent cross-page orphaned rows
    # ----------------------------------------------------
    p_t2 = doc.add_paragraph()
    p_t2.paragraph_format.keep_with_next = True
    p_t2.paragraph_format.space_before = Pt(8)
    p_t2.paragraph_format.space_after = Pt(4)
    r_t2 = p_t2.add_run("Table 2. Empirical Performance of Classifiers with ResNet-18 Deep Features Under Stratified Group 10-Fold CV")
    r_t2.bold = True
    r_t2.font.size = Pt(9.5)
    r_t2.font.name = "Times New Roman"
    
    t2_widths = [
        Inches(1.50), # Classifier
        Inches(0.85), # Pooled Acc. (%)
        Inches(1.35), # 10-Fold Mean ± SD (%)
        Inches(1.35), # Sensitivity / Specificity (%)
        Inches(0.72), # F1-Score (%)
        Inches(0.73)  # ROC-AUC (%)
    ]
    # Total = 6.50 inches (exact printable portrait width)
    
    subheaders = [
        "Classifier", "Pooled Acc. (%)", "10-Fold Mean ± SD (%)", "Sensitivity / Specificity (%)", "F1-Score (%)", "ROC-AUC (%)"
    ]
    
    # -------------------- Table 2A: Panel A (Eyes-Closed) --------------------
    table2_a = doc.add_table(rows=10, cols=6)
    table2_a.autofit = False
    configure_table_layout(table2_a, width_inches=6.50)
    apply_tbl_grid(table2_a, t2_widths)
    apply_table_borders(table2_a, subtle=True)
    
    # Panel A Title Row (Row 0)
    r0_a = table2_a.rows[0]
    r0_a_trPr = r0_a._tr.get_or_add_trPr()
    r0_a_trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    r0_a_trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    c_panA = r0_a.cells[0]
    for i in range(1, 6):
        c_panA.merge(r0_a.cells[i])
    c_panA.width = Inches(6.50)
    set_cell_background(c_panA, "F3F4F6")
    p_panA = c_panA.paragraphs[0]
    p_panA.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r_panA = p_panA.add_run("Table 2A. Eyes-Closed Resting Cohort (N = 58 participants: 28 Controls, 30 MDD)")
    r_panA.bold = True
    r_panA.font.size = Pt(8.5)
    r_panA.font.name = "Times New Roman"
    
    # Panel A Subheaders (Row 1)
    r1_a = table2_a.rows[1]
    r1_a_trPr = r1_a._tr.get_or_add_trPr()
    r1_a_trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    r1_a_trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    for idx, sh in enumerate(subheaders):
        cell = r1_a.cells[idx]
        cell.width = t2_widths[idx]
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(sh)
        run.bold = True
        run.font.size = Pt(8.0)
        run.font.name = "Times New Roman"
        
    # EC Data (Rows 2..9)
    ec_data = [
        ["Linear SVM", "89.66", "89.90 ± 8.34", "93.3 / 85.7", "90.32", "95.36"],
        ["Medium Gaussian SVM", "89.66", "89.90 ± 11.19", "90.0 / 89.3", "90.00", "93.57"],
        ["Neural Network (MLP)", "89.66", "89.90 ± 11.19", "90.0 / 89.3", "90.00", "94.40"],
        ["Cubic SVM", "89.66", "89.90 ± 11.19", "93.3 / 85.7", "90.32", "92.74"],
        ["Quadratic SVM", "87.93", "88.24 ± 10.79", "93.3 / 82.1", "88.89", "90.95"],
        ["Weighted KNN", "86.21", "86.57 ± 10.11", "86.7 / 85.7", "86.67", "92.14"],
        ["Ensemble Bagged Tree", "86.21", "86.81 ± 14.06", "93.3 / 78.6", "87.50", "91.43"],
        ["Fine Gaussian SVM", "50.00", "49.71 ± 9.52", "86.7 / 10.7", "64.20", "42.74"]
    ]
    for r_idx, row in enumerate(ec_data):
        r_elem = table2_a.rows[r_idx + 2]._tr.get_or_add_trPr()
        r_elem.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        for c_idx, val in enumerate(row):
            cell = table2_a.cell(r_idx + 2, c_idx)
            cell.width = t2_widths[c_idx]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(val.replace(" ± ", "\u00A0±\u00A0").replace(" / ", "\u00A0/\u00A0"))
            run.font.size = Pt(8.0)
            run.font.name = "Times New Roman"
            
    # Prevent row orphan: keep rows 0-8 together
    for row in table2_a.rows[:-1]:
        for cell in row.cells:
            cell.paragraphs[0].paragraph_format.keep_with_next = True
            
    for row in table2_a.rows:
        for cell in row.cells:
            apply_cell_margins(cell, top=60, bottom=60, left=45, right=45)
            
    # Intervening paragraph between Panel A and Panel B
    p_t2_gap = doc.add_paragraph()
    p_t2_gap.paragraph_format.space_before = Pt(2)
    p_t2_gap.paragraph_format.space_after = Pt(2)
    p_t2_gap.paragraph_format.line_spacing = Pt(2)
    
    # -------------------- Table 2B: Panel B (Eyes-Open) --------------------
    table2_b = doc.add_table(rows=10, cols=6)
    table2_b.autofit = False
    configure_table_layout(table2_b, width_inches=6.50)
    apply_tbl_grid(table2_b, t2_widths)
    apply_table_borders(table2_b, subtle=True)
    
    # Panel B Title Row (Row 0)
    r0_b = table2_b.rows[0]
    r0_b_trPr = r0_b._tr.get_or_add_trPr()
    r0_b_trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    r0_b_trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    c_panB = r0_b.cells[0]
    for i in range(1, 6):
        c_panB.merge(r0_b.cells[i])
    c_panB.width = Inches(6.50)
    set_cell_background(c_panB, "F3F4F6")
    p_panB = c_panB.paragraphs[0]
    p_panB.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r_panB = p_panB.add_run("Table 2B. Eyes-Open Resting Cohort (N = 61 participants: 29 Controls, 32 MDD)")
    r_panB.bold = True
    r_panB.font.size = Pt(8.5)
    r_panB.font.name = "Times New Roman"
    
    # Panel B Subheaders (Row 1)
    r1_b = table2_b.rows[1]
    r1_b_trPr = r1_b._tr.get_or_add_trPr()
    r1_b_trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    r1_b_trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    for idx, sh in enumerate(subheaders):
        cell = r1_b.cells[idx]
        cell.width = t2_widths[idx]
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(sh)
        run.bold = True
        run.font.size = Pt(8.0)
        run.font.name = "Times New Roman"
        
    # EO Data (Rows 2..9)
    eo_data = [
        ["Linear SVM", "88.52", "88.14 ± 11.98", "90.6 / 86.2", "89.23", "95.69"],
        ["Medium Gaussian SVM", "91.80", "91.48 ± 12.49", "90.6 / 93.1", "92.06", "93.32"],
        ["Neural Network (MLP)", "91.80", "91.48 ± 12.49", "90.6 / 93.1", "92.06", "96.01"],
        ["Cubic SVM", "88.52", "88.14 ± 14.11", "90.6 / 86.2", "89.23", "92.56"],
        ["Quadratic SVM", "88.52", "88.14 ± 11.98", "90.6 / 86.2", "89.23", "95.26"],
        ["Weighted KNN", "88.52", "88.14 ± 11.98", "90.6 / 86.2", "89.23", "94.61"],
        ["Ensemble Bagged Tree", "86.89", "86.48 ± 13.59", "90.6 / 82.8", "87.88", "93.43"],
        ["Fine Gaussian SVM", "52.46", "51.86 ± 10.64", "96.9 / 3.5", "68.13", "44.07"]
    ]
    for r_idx, row in enumerate(eo_data):
        r_elem = table2_b.rows[r_idx + 2]._tr.get_or_add_trPr()
        r_elem.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        for c_idx, val in enumerate(row):
            cell = table2_b.cell(r_idx + 2, c_idx)
            cell.width = t2_widths[c_idx]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(val.replace(" ± ", "\u00A0±\u00A0").replace(" / ", "\u00A0/\u00A0"))
            run.font.size = Pt(8.0)
            run.font.name = "Times New Roman"
            
    # Prevent row orphan: keep rows 0-8 together
    for row in table2_b.rows[:-1]:
        for cell in row.cells:
            cell.paragraphs[0].paragraph_format.keep_with_next = True
            
    for row in table2_b.rows:
        for cell in row.cells:
            apply_cell_margins(cell, top=60, bottom=60, left=45, right=45)
            
    t2_note = doc.add_paragraph()
    t2_note.paragraph_format.space_before = Pt(4)
    t2_note.paragraph_format.space_after = Pt(8)
    t2_note_run = t2_note.add_run(
        "* Note to Tables 2A and 2B: Metrics evaluated at the participant level under Stratified Group 10-Fold CV. "
        "Fold Mean ± SD indicates cross-validation fold accuracy using population standard deviation (ddof=0; sample standard deviation ddof=1 is ±8.79% for EC Linear SVM, ±11.79% for EC Medium Gaussian SVM, ±12.63% for EO Linear SVM, and ±13.16% for EO Medium Gaussian SVM). "
        "Sensitivity / Specificity denotes diagnostic balance (%). "
        "Participant 95% confidence intervals (Wilson score interval) are [79.21%, 95.17%] for 89.66% (52/58) and [82.21%, 96.45%] for 91.80% (56/61). "
        "Paired McNemar testing between Linear SVM and Medium Gaussian SVM on Eyes-Closed revealed identical overall accuracy with discordant predictions on only 4 of 58 participants (p = 1.000), confirming that their diagnostic performance is not statistically distinguishable on this sample cohort. "
        "ROC-AUC is calculated from pooled out-of-fold participant-level prediction probabilities."
    )
    t2_note_run.font.size = Pt(8.0)
    t2_note_run.font.name = "Times New Roman"
    t2_note_run.font.italic = True
    
    # Figure 2: Top Model Confusion Matrices (Composite High-Resolution Figure)
    cm_composite = "results/confusion_matrices/figure2_composite_confusion_matrices.png"
    if not os.path.exists(cm_composite):
        cm_ec = "results/confusion_matrices/cm_resnet18_linear_svm_eyesClosed_N58.png"
        cm_eo = "results/confusion_matrices/cm_resnet18_medium_gaussian_svm_eyesOpen_N61.png"
        if os.path.exists(cm_ec) and os.path.exists(cm_eo):
            from PIL import Image
            im1 = Image.open(cm_ec)
            im2 = Image.open(cm_eo)
            gap = 60
            tot_w = im1.width + im2.width + gap
            tot_h = max(im1.height, im2.height)
            comp = Image.new("RGBA", (tot_w, tot_h), (255, 255, 255, 255))
            comp.paste(im1, (0, 0), im1 if im1.mode == "RGBA" else None)
            comp.paste(im2, (im1.width + gap, 0), im2 if im2.mode == "RGBA" else None)
            os.makedirs(os.path.dirname(cm_composite), exist_ok=True)
            comp.save(cm_composite, dpi=(300, 300))
            
    if os.path.exists(cm_composite):
        p_fig2 = doc.add_paragraph()
        p_fig2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig2.paragraph_format.space_before = Pt(8)
        p_fig2.paragraph_format.space_after = Pt(4)
        p_fig2.paragraph_format.keep_with_next = True
        run_fig2 = p_fig2.add_run()
        run_fig2.add_picture(cm_composite, width=Inches(6.20))
        
        p_cap2 = doc.add_paragraph()
        p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap2.paragraph_format.space_before = Pt(4)
        p_cap2.paragraph_format.space_after = Pt(12)
        r_cap2_label = p_cap2.add_run("Fig. 2. ")
        r_cap2_label.bold = True
        r_cap2_label.font.size = Pt(9.5)
        r_cap2_label.font.name = "Times New Roman"
        r_cap2 = p_cap2.add_run("Out-of-fold participant-level confusion matrices for top-performing classifiers under Stratified Group 10-Fold CV: (Left) Linear SVM on primary Eyes-Closed resting state (N = 58 participants: 24 TN, 4 FP, 2 FN, 28 TP; 89.66% accuracy); (Right) Medium Gaussian SVM on secondary Eyes-Open resting state (N = 61 participants: 27 TN, 2 FP, 3 FN, 29 TP; 91.80% accuracy).")
        r_cap2.font.size = Pt(9.5)
        r_cap2.font.name = "Times New Roman"
        r_cap2.font.italic = True
        
    # 3.3 Methodological Improvement Experiments and Nested Hyperparameter Tuning
    doc.add_heading("3.3 Methodological Improvement Experiments and Nested Hyperparameter Tuning", level=2)
    res_imprv = (
        "To explore whether targeted signal processing or classification adjustments could further optimize diagnostic accuracy, four follow-up experiments were investigated:\n\n"
        "1. Exploratory Aggregation Rules: We evaluated Median, Trimmed Mean, and Majority Voting on outer-fold predictions. For Linear SVM on Eyes-Closed, Median aggregation yielded 91.38% participant-level accuracy (sensitivity 96.67%, 29/30 MDD identified) by dampening isolated segment mispredictions. However, for distance-based estimators (Weighted KNN), Median aggregation reduced accuracy to 82.76%, indicating that Arithmetic Mean provides the most stable universal default across diverse classifier families. This comparison is considered exploratory.\n\n"
        "2. Exploratory Spatial Montage Weighting: Rather than uniform 19-channel averaging, a pre-specified montage placed higher weight on anterior leads (Fp1, Fp2, F3, F4, Fz, Cz, C3, C4; weight 2.0) and attenuated occipital leads (O1, O2; weight 0.5). This weighting increased Medium Gaussian SVM accuracy from 89.66% to 91.38% (+1.72%) and sensitivity to 93.33%, suggesting that prioritizing frontal channels provides supportive diagnostic signal consistent with frontal alterations reported in depression literature. Because this montage was evaluated on the same cohort, it represents an exploratory hypothesis for future validation.\n\n"
        "3. Exploratory SPWVD Smoothing Variations: Testing an alternative frequency smoothing window (Kaiser lag beta = 7.0, L = 255) eliminated false negatives for Linear SVM, achieving 100.00% sensitivity (30/30 MDD detected) and 93.10% participant accuracy on outer folds. Conversely, a compact transient window (L = 127, beta = 3.5) degraded Linear SVM accuracy to 82.76% (-6.90%), demonstrating that cross-term suppression parameters substantially influence linear separability in time-frequency space.\n\n"
        "4. Nested Group-Aware Hyperparameter Tuning on a Pre-selected Feature Backbone: To assess whether regularizing the Linear SVM classifier could be optimized without test-fold feedback, an inner 5-fold Stratified Group CV was conducted strictly on training folds to select the penalty parameter C ∈ [0.01, 0.1, 1.0, 10.0]. The outer folds were held out exclusively for final model evaluation. The inner tuning loop consistently favored stronger regularization (C ∈ [0.01, 0.1]), yielding an outer-test accuracy of 91.38% (95% Wilson score CI: [81.36%, 96.26%]; 93.33% sensitivity, 89.29% specificity, 91.80% F1-score, 94.76% ROC-AUC; TN=25, FP=3, FN=2, TP=28). "
        "Importantly, while the regularization parameter C was tuned strictly within inner folds without exposure to outer test data, the preceding architectural choices—specifically selecting ResNet-18 over alternative backbones and choosing Linear SVM over other classifier families—were informed by prior baseline comparisons on this cohort. "
        "Consequently, the 91.38% result demonstrates the value of group-aware regularization tuning on a pre-selected representation, rather than an end-to-end unbiased selection across all pipeline dimensions."
    )
    doc.add_paragraph(res_imprv)
    
    # ----------------------------------------------------
    # 4. Discussion
    # ----------------------------------------------------
    doc.add_heading("4. Discussion", level=1)
    disc_p1 = (
        "Automated EEG-based psychiatric screening requires both reproducible feature representations and rigorous validation methodology. "
        "In physiological signal analysis, participant-independent validation is essential because temporal segments originating from the same individual "
        "exhibit correlated anatomical geometry, electrode contact impedances, and baseline neurodynamic patterns. "
        "When evaluating model generalizability to unseen clinical subjects, allocating all recordings from an individual participant strictly to either "
        "the training or testing partition prevents models from learning subject-specific idiosyncrasies rather than disease-related biomarkers. "
        "Under strict participant-independent Stratified Group 10-Fold Cross-Validation, our benchmark establishes that SPWVD time-frequency representations "
        "paired with deep transfer learning achieve participant-level diagnostic accuracies of 89.66% for eyes-closed (95.36% ROC-AUC for Linear SVM) "
        "and 91.80% for eyes-open resting states (93.32% to 96.01% ROC-AUC for top estimators). "
        "These empirical findings provide a verified, reproducible benchmark on this public clinical cohort."
    )
    doc.add_paragraph(disc_p1)
    
    disc_p2 = (
        "Our experimental results also detail performance across the two distinct evaluation units: individual 2-second temporal segments (N = 870 for eyes-closed, "
        "N = 930 for eyes-open) and participant-level diagnostic classifications (N = 58 for eyes-closed, N = 61 for eyes-open). "
        "At the segment level, classification accuracies ranged between 77.70% and 81.84% for the non-degenerate estimators on eyes-closed, and between 80.65% and 85.27% on eyes-open "
        "(spanning 77.70% to 85.27% overall across the seven standard classifiers, excluding the collapsed Fine Gaussian SVM). "
        "At the participant level, where posterior probabilities are averaged across all epochs from each participant, the corresponding diagnostic accuracies reached 89.66% and 91.80%, respectively. "
        "In clinical psychiatric practice, diagnostic assessments are established per patient rather than per short temporal epoch. "
        "Aggregating segment posterior probabilities provides a consensus decision per participant that dampens the influence of isolated segment-level classification errors."
    )
    doc.add_paragraph(disc_p2)
    
    doc.add_heading("4.1 Status of Literature Comparisons and Tabular Withholding", level=2)
    disc_comp = (
        "Preliminary drafts of this work included a comparative table summarizing published diagnostic accuracies from related EEG depression studies. "
        "A rigorous audit of those cited works revealed substantial heterogeneity in experimental designs. "
        "Several studies from the literature were verified with confirmed citations:\n"
        "- Acharya et al. [13] evaluated a 13-layer CNN on 30 subjects (15 MDD, 15 HC) using 10-fold cross-validation, reporting 95.49% (right hemisphere) and 93.54% (left hemisphere) segment accuracy.\n"
        "- Mumtaz et al. [18, 22] evaluated 1D-CNN and wavelet features on the HUSM dataset (34 MDD, 30 HC), reporting 98.32% under epoch-level validation.\n"
        "- Wu et al. [16] applied conformal kernel SVM to power features from 55 subjects (24 MDD, 31 HC), reporting 83.64% accuracy.\n\n"
        "However, other entries cited in preliminary literature summaries lacked verified DOI links, unambiguous participant-split documentation, or consistent reporting metrics. "
        "Directly juxtaposing participant-independent results against studies that utilized segment-level random splitting or undefined protocols creates an unstandardized comparison. "
        "Consequently, the comparative table is withheld from this manuscript in the interest of scientific integrity until external comparator pipelines undergo standardized participant-independent auditing."
    )
    doc.add_paragraph(disc_comp)
    
    doc.add_heading("4.2 Study Limitations", level=2)
    disc_lim = (
        "Several limitations should be considered when interpreting our findings:\n"
        "1. Single-Center Cohort: The dataset originated from a single clinical institution (HUSM) comprising 64 participants. Independent validation on multi-center cohorts with diverse demographic, geographic, and pharmacological characteristics is necessary to confirm external generalizability.\n"
        "2. Diagnostic Specificity: The current benchmark evaluates binary discrimination between unipolar MDD patients and healthy controls. In clinical practice, distinguishing MDD from other psychiatric conditions with overlapping neurophysiology (e.g., bipolar disorder, generalized anxiety disorder) represents a critical clinical challenge.\n"
        "3. Exploratory Adjustments: While spatial montage weighting and Kaiser window parameter adjustments yielded empirical accuracy gains, these variations were evaluated on the same clinical cohort and must be regarded as exploratory hypotheses requiring prospective replication.\n"
        "4. Model Explainability: Pretrained CNN feature extractors operate as complex nonlinear mappings. While SPWVD provides structured physical time-frequency energy maps, integrating post-hoc explainability techniques (e.g., Grad-CAM, SHAP) will be valuable to provide clinicians with interpretable spectral signatures driving model predictions.\n"
        "5. Computational Scope Boundary: While the end-to-end signal processing pipeline was directly verified on raw EDF recordings, batch cross-validation experiments utilized verified pre-extracted feature tensors across the 181 recordings to ensure reproducibility and computational efficiency."
    )
    doc.add_paragraph(disc_lim)
    
    # ----------------------------------------------------
    # 5. Conclusion
    # ----------------------------------------------------
    doc.add_heading("5. Conclusion", level=1)
    concl_text = (
        "In this study, we presented a reproducible, participant-independent investigation of automated Major Depressive Disorder detection using Smoothed Pseudo Wigner-Ville Distribution (SPWVD) "
        "time-frequency spectral images and deep transfer learning features. By enforcing strict Stratified Group 10-Fold Cross-Validation across 64 clinical participants (30 healthy controls, 34 MDD patients), "
        "we eliminated subject data leakage between training and testing folds, resolving historical dataset metadata ambiguities. Pretrained ResNet-18 features demonstrated superior performance on the primary eyes-closed benchmark while offering a "
        "compact 512-dimensional feature space. Coupled with Linear SVM, Medium Gaussian SVM, or Multi-Layer Perceptron classifiers, the framework achieved 89.66% participant-level diagnostic accuracy (95.36% ROC-AUC) on eyes-closed resting state and 91.80% accuracy (ROC-AUC of 93.32% for Medium Gaussian SVM and 96.01% for Neural Network; 95.69% for Linear SVM) "
        "on eyes-open resting state. These results establish a verified, reproducible benchmark for automated depression detection on this public dataset, demonstrating the critical role of participant-independent evaluation."
    )
    doc.add_paragraph(concl_text)
    
    # ----------------------------------------------------
    # References
    # ----------------------------------------------------
    doc.add_heading("References", level=1)
    references = [
        "[1] World Health Organization, 'Depressive disorder (depression),' WHO Fact Sheets, Geneva, 2023. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/depression",
        "[2] GBD 2019 Mental Disorders Collaborators, 'Global, regional, and national burden of 12 mental disorders in 204 countries and territories, 1990–2019: a systematic analysis for the Global Burden of Disease Study 2019,' The Lancet Psychiatry, vol. 9, no. 2, pp. 137–150, 2022.",
        "[3] American Psychiatric Association, Diagnostic and Statistical Manual of Mental Disorders: DSM-5-TR, 5th ed. Washington, DC: American Psychiatric Association, 2022.",
        "[4] J. Zhou, Y. Zhang, S. He, S. Xu, Q. Sun, T. Zhao, and Y. Dai, 'Accelerated global burden of depressive disorders during the COVID-19 pandemic from 2019 to 2021,' Scientific Reports, vol. 15, no. 1, p. 9529, 2025.",
        "[5] M. Teplan, 'Fundamentals of EEG measurement,' Measurement Science Review, vol. 2, no. 2, pp. 1–11, 2002.",
        "[6] Y.-Y. Yin and Y.-F. Li, 'Role of neural oscillations in depression: highlights on gamma oscillations,' Translational Psychiatry, vol. 14, p. 112, 2024.",
        "[7] H. R. Snyder, 'Major depressive disorder is associated with broad impairments on neuropsychological measures of executive function: a meta-analysis and review,' Psychological Bulletin, vol. 139, no. 1, pp. 81–132, 2013.",
        "[8] S. Lechner and G. Northoff, 'Abnormal resting-state EEG phase dynamics distinguishes major depressive disorder and bipolar disorder,' Journal of Affective Disorders, vol. 359, pp. 269–276, 2024.",
        "[9] T. T. Erguzel, S. Ozekes, O. Tan, and S. Gultekin, 'Feature selection and classification of electroencephalographic signals: an artificial neural network and genetic algorithm based approach,' Clinical EEG and Neuroscience, vol. 46, no. 4, pp. 321–326, 2015.",
        "[10] B. Hosseinifard, M. H. Moradi, and R. Rostami, 'Classifying depression patients and normal subjects using machine learning techniques and nonlinear features from EEG signal,' Computer Methods and Programs in Biomedicine, vol. 109, no. 3, pp. 339–345, 2013.",
        "[11] S.-C. Liao, C.-T. Wu, H.-C. Huang, W.-T. Cheng, and Y.-H. Liu, 'Major depression detection from EEG signals using kernel eigen-filter-bank common spatial patterns,' Sensors, vol. 17, no. 6, p. 1385, 2017.",
        "[12] Y. Mohammadi, M. Hajian, and M. H. Moradi, 'Discrimination of depression levels using machine learning methods on EEG signals,' in 2019 27th Iranian Conference on Electrical Engineering (ICEE). IEEE, 2019, pp. 1765–1769.",
        "[13] U. R. Acharya, S. L. Oh, Y. Hagiwara, J. H. Tan, H. Adeli, and D. P. Subha, 'Automated EEG-based screening of depression using deep convolutional neural network,' Computer Methods and Programs in Biomedicine, vol. 161, pp. 103–113, 2018.",
        "[14] S. Hochreiter and J. Schmidhuber, 'Long short-term memory,' Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.",
        "[15] A. K. Choudhary, A. Mishra, K. N. Mishra, R. K. Lal, and A. Kumari, 'Detection of major depressive disorder using maximum likelihood estimation for channel selection: A novel approach,' in International Conference on Innovative Computing and Communication. Springer, 2025, pp. 375–396.",
        "[16] C.-T. Wu, H.-C. Huang, S. Huang, H.-C. Chen, P.-C. Chen, and C.-Y. Chen, 'Resting-state EEG for the diagnosis of major depressive disorder using conformal kernel support vector machine,' IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 28, no. 8, pp. 1762–1770, 2020.",
        "[17] J. Shen, X. Zhang, X. Huang, M. Wu, J. Gao, D. Lu, Z. Ding, and B. Hu, 'An optimal channel selection for EEG-based depression detection via kernel-target alignment,' IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 7, pp. 2545–2556, 2020.",
        "[18] W. Mumtaz, L. Xia, S. S. A. Ali, M. A. M. Yasin, M. Hussain, and A. S. Malik, 'Electroencephalogram (EEG)-based computer-aided technique to diagnose major depressive disorder (MDD),' Biomedical Signal Processing and Control, vol. 31, pp. 108–115, 2017.",
        "[19] L. Cohen, Time-Frequency Analysis. Englewood Cliffs, NJ: Prentice Hall, 1995.",
        "[20] F. Auger and P. Flandrin, 'Improving the readability of time-frequency and time-scale representations by the reassignment method,' IEEE Transactions on Signal Processing, vol. 43, no. 5, pp. 1068–1089, 1995.",
        "[21] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, 'ImageNet: A large-scale hierarchical image database,' in 2009 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2009, pp. 248–255.",
        "[22] W. Mumtaz, S. S. A. Ali, M. A. M. Yasin, and A. S. Malik, 'A deep learning framework for automatic diagnosis of unipolar depression,' International Journal of Medical Informatics, vol. 132, p. 103983, 2019."
    ]
    for r in references:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.28)
        p_ref.paragraph_format.first_line_indent = Inches(-0.28)
        p_ref.paragraph_format.space_after = Pt(2.5)
        p_ref.paragraph_format.line_spacing = 1.05
        p_ref.paragraph_format.widow_control = True
        run_ref = p_ref.add_run(r)
        run_ref.font.name = "Times New Roman"
        run_ref.font.size = Pt(9.0)
        
    os.makedirs("manuscript", exist_ok=True)
    import sys
    if "--overwrite-master" not in sys.argv:
        fallback_path = "manuscript/legacy_generated_manuscript.docx"
        doc.save(fallback_path)
        print(f"NOTICE: manuscript/Jha_MDD_Revised_Final.docx is the authoritative master manuscript.")
        print(f"To protect manual revisions, output saved to: {fallback_path}")
        print(f"To force overwrite master files, pass --overwrite-master.")
        return
        
    out_path_final = "manuscript/Jha_MDD_Revised_Final.docx"
    out_path_revised = "manuscript/Jha_MDD_Revised.docx"
    doc.save(out_path_final)
    doc.save(out_path_revised)
    print(f"Generated revised manuscript: {out_path_final} and {out_path_revised}")

if __name__ == '__main__':
    create_revised_manuscript()
