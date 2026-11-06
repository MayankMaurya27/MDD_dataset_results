with open("src/generate_revised_manuscript.py", "r") as f:
    code = f.read()

# 1. Fix Fold 3 -> Fold 9 for sub-HS15
old_fold = "For multi-run participants (e.g. sub-HS15 in the eyes-open condition), all runs remained strictly quarantined to Fold 3."
new_fold = "For multi-run participants (e.g. sub-HS15 in the eyes-open condition), all runs remained strictly quarantined to Fold 9."
assert old_fold in code, "old_fold not found"
code = code.replace(old_fold, new_fold)

# 2. Add Wilson CIs and McNemar's test to Section 3.2
old_s32_ec = (
    '        "- Linear SVM achieved exceptional diagnostic balance: 93.33% sensitivity (28/30 MDD identified), 85.71% specificity (24/28 controls identified), an F1-score of 90.32%, and an ROC-AUC of 95.36% (TN=24, FP=4, FN=2, TP=28).\\n"\n'
    '        "- Medium Gaussian SVM and Neural Network achieved 90.00% sensitivity and 89.29% specificity (TN=25, FP=3, FN=3, TP=27).\\n"'
)
new_s32_ec = (
    '        "- Linear SVM achieved exceptional diagnostic balance: 93.33% sensitivity (28/30 MDD identified), 85.71% specificity (24/28 controls identified), an F1-score of 90.32%, and an ROC-AUC of 95.36% (TN=24, FP=4, FN=2, TP=28; 95% Wilson score CI: [79.21%, 95.17%]).\\n"\n'
    '        "- Medium Gaussian SVM and Neural Network achieved 90.00% sensitivity and 89.29% specificity (TN=25, FP=3, FN=3, TP=27; 95% Wilson score CI: [79.21%, 95.17%]). Paired McNemar testing between Linear SVM and Medium Gaussian SVM revealed identical overall accuracy with discordant predictions on only 4 of 58 participants (p = 1.000), confirming that their diagnostic performance is not statistically distinguishable on this sample cohort.\\n"'
)
assert old_s32_ec in code, "old_s32_ec not found"
code = code.replace(old_s32_ec, new_s32_ec)

# 3. Add Wilson CIs to EO text in Section 3.2
old_s32_eo = (
    '        "- Medium Gaussian SVM and Neural Network attained the highest diagnostic performance: 91.80% participant-level accuracy (mean fold accuracy 91.48 ± 12.49%), 90.62% sensitivity, 93.10% specificity, 92.06% F1-score, and 93.32%–96.01% ROC-AUC (TN=27, FP=2, FN=3, TP=29).\\n"\n'
    '        "- Linear SVM, Quadratic SVM, Cubic SVM, and Weighted KNN each achieved 88.52% participant accuracy (90.62% sensitivity, 86.21% specificity; TN=25, FP=4, FN=3, TP=29).\\n"'
)
new_s32_eo = (
    '        "- Medium Gaussian SVM and Neural Network attained the highest diagnostic performance: 91.80% participant-level accuracy (mean fold accuracy 91.48 ± 12.49%; 95% Wilson score CI: [82.21%, 96.45%]), 90.62% sensitivity, 93.10% specificity, 92.06% F1-score, and 93.32%–96.01% ROC-AUC (TN=27, FP=2, FN=3, TP=29).\\n"\n'
    '        "- Linear SVM, Quadratic SVM, Cubic SVM, and Weighted KNN each achieved 88.52% participant accuracy (95% Wilson score CI: [78.16%, 94.33%]; 90.62% sensitivity, 86.21% specificity; TN=25, FP=4, FN=3, TP=29).\\n"'
)
assert old_s32_eo in code, "old_s32_eo not found"
code = code.replace(old_s32_eo, new_s32_eo)

# 4. Add Wilson CI to Nested Linear SVM in Section 3.3
old_nested = 'yielding an outer-test accuracy of 91.38% (93.33% sensitivity, 89.29% specificity, 91.80% F1-score, 94.76% ROC-AUC; TN=25, FP=3, FN=2, TP=28).'
new_nested = 'yielding an outer-test accuracy of 91.38% (95% Wilson score CI: [81.36%, 96.26%]; 93.33% sensitivity, 89.29% specificity, 91.80% F1-score, 94.76% ROC-AUC; TN=25, FP=3, FN=2, TP=28).'
assert old_nested in code, "old_nested not found"
code = code.replace(old_nested, new_nested)

# 5. Update Table 2 note with Wilson CIs
old_t2_note = (
    '        "* Note: Metrics evaluated at the participant level under Stratified Group 10-Fold CV. "\n'
    '        "Fold Mean ± SD indicates cross-validation fold accuracy using population standard deviation (ddof=0; sample standard deviation ddof=1 is ±8.79% for EC Linear SVM, ±11.79% for EC Medium Gaussian SVM, ±12.63% for EO Linear SVM, and ±13.16% for EO Medium Gaussian SVM). "\n'
    '        "ROC-AUC is calculated from pooled out-of-fold participant-level prediction probabilities."'
)
new_t2_note = (
    '        "* Note: Metrics evaluated at the participant level under Stratified Group 10-Fold CV. "\n'
    '        "Fold Mean ± SD indicates cross-validation fold accuracy using population standard deviation (ddof=0; sample standard deviation ddof=1 is ±8.79% for EC Linear SVM, ±11.79% for EC Medium Gaussian SVM, ±12.63% for EO Linear SVM, and ±13.16% for EO Medium Gaussian SVM). "\n'
    '        "Participant 95% confidence intervals (Wilson score interval) are [79.21%, 95.17%] for 89.66% (52/58) and [82.21%, 96.45%] for 91.80% (56/61). "\n'
    '        "ROC-AUC is calculated from pooled out-of-fold participant-level prediction probabilities."'
)
assert old_t2_note in code, "old_t2_note not found"
code = code.replace(old_t2_note, new_t2_note)

with open("src/generate_revised_manuscript.py", "w") as f:
    f.write(code)

print("Applied Phase 4 patches successfully!")
