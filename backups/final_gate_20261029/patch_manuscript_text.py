import re

with open("src/generate_revised_manuscript.py", "r") as f:
    code = f.read()

# 1. Update Abstract
old_abs_target = (
    '        "At the individual 2-second segment level, accuracy ranged from 78% to 85%, demonstrating that participant-level probability aggregation across epochs "\n'
    '        "dampens isolated segment-level classification errors. "'
)
new_abs_replacement = (
    '        "At the individual 2-second segment level, classification accuracy ranged from 77.7% to 85.3% across top estimators, while participant-level diagnostic "\n'
    '        "accuracy reached 89.66% for eyes-closed and 91.80% for eyes-open through posterior probability aggregation across each subject\'s epochs. "'
)

# 2. Update Section 3.2
old_s32_target = (
    '        "Crucially, across both recording conditions, participant-level aggregation improved diagnostic accuracy by 7.0% to 9.5% compared to raw segment-level predictions. "\n'
    '        "Averaging posterior probabilities across multiple epochs effectively cancels out transient segment-level physiological noise (such as minor eye blinks or momentary drift)."'
)
new_s32_replacement = (
    '        "Evaluation was conducted at two distinct levels: raw segment-level predictions across all individual temporal windows (870 segments for eyes-closed, 930 segments for eyes-open) "\n'
    '        "and participant-level diagnostic decisions aggregated across each subject\'s segments (58 participants for eyes-closed, 61 participants for eyes-open). "\n'
    '        "On the primary eyes-closed cohort, segment-level classification accuracy was 80.34% for Linear SVM, 81.72% for Medium Gaussian SVM, and 81.84% for Neural Network (MLP), "\n'
    '        "compared to participant-level diagnostic accuracy of 89.66% for each of these classifiers. On the secondary eyes-open cohort, segment-level accuracy was 80.65% for Linear SVM, "\n'
    '        "84.19% for Medium Gaussian SVM, and 85.27% for Neural Network, compared to participant-level accuracy of 88.52% (Linear SVM) and 91.80% (Medium Gaussian SVM and Neural Network). "\n'
    '        "Because clinical diagnoses are rendered per participant rather than per 2-second snippet, aggregating segment posteriors produces participant-level classifications that "\n'
    '        "reduce the influence of isolated segment-level classification errors."'
)

# 3. Update Discussion Paragraph 1
old_disc1_target = (
    '    disc_p1 = (\n'
    '        "Automated EEG-based psychiatric screening requires both reproducible feature representations and rigorous validation methodology. "\n'
    '        "In preliminary investigations and across portions of the EEG literature, classification accuracies exceeding 98% are frequently reported. "\n'
    '        "Methodological analyses indicate that such high scores often stem from segment-level cross-validation protocols, where temporal epochs from the same participant "\n'
    '        "are allocated across both training and test partitions. Under such protocols, classifiers can exploit subject-specific physiological idiosyncratic traits rather "\n'
    '        "than generalizable disease biomarkers. By enforcing strict participant-independent Stratified Group 10-Fold Cross-Validation, our benchmark establishes that "\n'
    '        "SPWVD time-frequency representations paired with deep transfer learning achieve realistic participant-level diagnostic accuracies of 89.66% for eyes-closed "\n'
    '        "and 91.80% for eyes-open resting states (with ROC-AUC values of 93.32% to 96.01% among the top estimators). "\n'
    '        "These empirical findings provide a verified, reproducible benchmark on this public clinical cohort."\n'
    '    )'
)
new_disc1_replacement = (
    '    disc_p1 = (\n'
    '        "Automated EEG-based psychiatric screening requires both reproducible feature representations and rigorous validation methodology. "\n'
    '        "In physiological signal analysis, participant-independent validation is essential because temporal segments originating from the same individual "\n'
    '        "exhibit correlated anatomical geometry, electrode contact impedances, and baseline neurodynamic patterns. "\n'
    '        "When evaluating model generalizability to unseen clinical subjects, allocating all recordings from an individual participant strictly to either "\n'
    '        "the training or testing partition prevents models from learning subject-specific idiosyncrasies rather than disease-related biomarkers. "\n'
    '        "Under strict participant-independent Stratified Group 10-Fold Cross-Validation, our benchmark establishes that SPWVD time-frequency representations "\n'
    '        "paired with deep transfer learning achieve participant-level diagnostic accuracies of 89.66% for eyes-closed (95.36% ROC-AUC for Linear SVM) "\n'
    '        "and 91.80% for eyes-open resting states (93.32% to 96.01% ROC-AUC for top estimators). "\n'
    '        "These empirical findings provide a verified, reproducible benchmark on this public clinical cohort."\n'
    '    )'
)

# 4. Update Discussion Paragraph 2
old_disc2_target = (
    '    disc_p2 = (\n'
    '        "A key observation from our experiments is the substantial performance gain provided by participant-level probability aggregation. Individual 2-second EEG segments "\n'
    '        "achieved raw classification accuracies of 78% to 85%. Individual short epochs are vulnerable to transient physiological fluctuations, such as ocular micromovements or temporary attentional drift. "\n'
    '        "However, because these transient artifacts are largely uncorrelated across successive windows within a recording, averaging posterior probability estimates across multiple epochs per participant "\n'
    '        "substantially reduced misclassifications, elevating participant-level accuracy by approximately 7% to 9.5%. "\n'
    '        "Because clinical psychiatric diagnoses are rendered per patient rather than per temporal epoch, participant-level evaluation aligns directly with translational clinical requirements."\n'
    '    )'
)
new_disc2_replacement = (
    '    disc_p2 = (\n'
    '        "Our experimental results also detail performance across the two distinct evaluation units: individual 2-second temporal segments (N = 870 for eyes-closed, "\n'
    '        "N = 930 for eyes-open) and participant-level diagnostic classifications (N = 58 for eyes-closed, N = 61 for eyes-open). "\n'
    '        "At the segment level, classification accuracies ranged between 77.70% and 81.84% for the top estimators on eyes-closed, and between 80.65% and 85.27% on eyes-open. "\n'
    '        "At the participant level, where posterior probabilities are averaged across all epochs from each participant, the corresponding diagnostic accuracies reached 89.66% and 91.80%, respectively. "\n'
    '        "In clinical psychiatric practice, diagnostic assessments are established per patient rather than per short temporal epoch. "\n'
    '        "Aggregating segment posterior probabilities provides a consensus decision per participant that dampens the influence of isolated segment-level classification errors."\n'
    '    )'
)

assert old_abs_target in code, "old_abs_target not found"
assert old_s32_target in code, "old_s32_target not found"
assert old_disc1_target in code, "old_disc1_target not found"
assert old_disc2_target in code, "old_disc2_target not found"

code = code.replace(old_abs_target, new_abs_replacement)
code = code.replace(old_s32_target, new_s32_replacement)
code = code.replace(old_disc1_target, new_disc1_replacement)
code = code.replace(old_disc2_target, new_disc2_replacement)

with open("src/generate_revised_manuscript.py", "w") as f:
    f.write(code)

print("Successfully patched src/generate_revised_manuscript.py!")
