with open("src/generate_revised_manuscript.py", "r") as f:
    code = f.read()

# Add Figure 1 after Section 2.7
target_s27 = 'doc.add_paragraph(mm_val)'
fig1_code = '''doc.add_paragraph(mm_val)
    
    # Figure 1: Pipeline Schematic
    if os.path.exists("results/pipeline_architecture_diagram.png"):
        p_fig1 = doc.add_paragraph()
        p_fig1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_fig1 = p_fig1.add_run()
        run_fig1.add_picture("results/pipeline_architecture_diagram.png", width=Inches(6.0))
        p_cap1 = doc.add_paragraph()
        p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap1 = p_cap1.add_run("Fig. 1. Schematic of the proposed participant-independent MDD detection pipeline: 19-channel EEG preprocessing, SPWVD joint time-frequency distribution, ImageNet-pretrained ResNet-18 feature extraction (512-D), Stratified Group 10-Fold Cross-Validation, and participant-level posterior probability aggregation.")
        r_cap1.font.size = Pt(9.5)
        r_cap1.font.italic = True
        doc.add_paragraph()'''

assert target_s27 in code, "target_s27 not found"
code = code.replace(target_s27, fig1_code, 1)

# Add Figure 2 after Table 2 Note
target_t2_note = 'doc.add_paragraph() # spacer\n    \n    # 3.3 Methodological Improvement Experiments and Nested Hyperparameter Tuning'
fig2_code = '''# Figure 2: Top Model Confusion Matrices
    cm_ec = "results/confusion_matrices/cm_resnet18_linear_svm_eyesClosed_N58.png"
    cm_eo = "results/confusion_matrices/cm_resnet18_medium_gaussian_svm_eyesOpen_N61.png"
    if os.path.exists(cm_ec) and os.path.exists(cm_eo):
        table_cm = doc.add_table(rows=1, cols=2)
        table_cm.alignment = WD_TABLE_ALIGNMENT.CENTER
        c1 = table_cm.cell(0, 0).paragraphs[0]
        c1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c1.add_run().add_picture(cm_ec, width=Inches(3.0))
        c2 = table_cm.cell(0, 1).paragraphs[0]
        c2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c2.add_run().add_picture(cm_eo, width=Inches(3.0))
        
        p_cap2 = doc.add_paragraph()
        p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap2 = p_cap2.add_run("Fig. 2. Out-of-fold participant-level confusion matrices for top-performing classifiers under Stratified Group 10-Fold CV: (Left) Linear SVM on primary Eyes-Closed resting state (N = 58 participants: 24 TN, 4 FP, 2 FN, 28 TP; 89.66% accuracy); (Right) Medium Gaussian SVM on secondary Eyes-Open resting state (N = 61 participants: 27 TN, 2 FP, 3 FN, 29 TP; 91.80% accuracy).")
        r_cap2.font.size = Pt(9.5)
        r_cap2.font.italic = True
        doc.add_paragraph()
        
    # 3.3 Methodological Improvement Experiments and Nested Hyperparameter Tuning'''

assert target_t2_note in code, "target_t2_note not found"
code = code.replace(target_t2_note, fig2_code, 1)

with open("src/generate_revised_manuscript.py", "w") as f:
    f.write(code)

print("Successfully integrated Figures 1 and 2 into src/generate_revised_manuscript.py!")
