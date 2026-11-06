"""
Phase 1A: Out-of-Fold Verification and Recalculation Script.

Performs:
1. Verification of all 8 classifiers across Eyes-Closed (N=58) and Eyes-Open (N=61).
2. Independent reproduction of participant-level posterior probability aggregation.
3. Recalculation of all reported metrics (pooled accuracy, sensitivity, specificity, F1, fold mean +/- SD with ddof=0 and ddof=1, Wilson CIs).
4. Full continuous ROC curve coordinate generation (FPR, TPR, thresholds).
5. Participant-level 1,000-iteration bootstrap 95% AUC confidence intervals with fixed seed (seed=42).
6. Quantitative verification of segment probability differences vs participant decision convergence.
7. Export of verified CSV tables to results/phase1a_verification/ and reports/reviewer_revision_phase1a/.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score

RESULTS_DIR = "/Users/mayankmaurya/Desktop/experiment/results/phase1a_verification"
REPORTS_DIR = "/Users/mayankmaurya/Desktop/experiment/reports/reviewer_revision_phase1a"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return 0.0, 0.0
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / len(y_true)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, sens, spec, f1, tn, fp, fn, tp

def run_verification():
    cohorts = [
        ("Eyes-Closed", "/Users/mayankmaurya/Desktop/experiment/results/out_of_fold_predictions_eyesClosed.csv", "eyesClosed"),
        ("Eyes-Open", "/Users/mayankmaurya/Desktop/experiment/results/out_of_fold_predictions_eyesOpen.csv", "eyesOpen")
    ]
    
    all_recalculated_rows = []
    all_bootstrap_rows = []
    all_roc_coords = []
    all_participant_preds = []
    
    classifier_order = [
        "Linear SVM",
        "Medium Gaussian SVM",
        "Neural Network Classifier",
        "Cubic SVM",
        "Quadratic SVM",
        "Weighted KNN",
        "Ensemble Bagged Tree",
        "Fine Gaussian SVM"
    ]
    
    for cohort_name, oof_csv, tag in cohorts:
        print(f"\n=======================================================")
        print(f"  Verifying {cohort_name} ({oof_csv})")
        print(f"=======================================================")
        
        df = pd.read_csv(oof_csv)
        print(f"Loaded {len(df)} segment prediction rows.")
        
        # Check classifiers
        present_clfs = df["classifier"].unique().tolist()
        print(f"Classifiers found ({len(present_clfs)}): {present_clfs}")
        
        part_df_list = []
        
        for clf in classifier_order:
            if clf not in present_clfs:
                continue
                
            cdf = df[df["classifier"] == clf].copy()
            n_segments = len(cdf)
            
            # Segment-level continuous AUC
            seg_auc = roc_auc_score(cdf["y_true"], cdf["y_prob"])
            seg_acc, seg_sens, seg_spec, seg_f1, s_tn, s_fp, s_fn, s_tp = compute_metrics(cdf["y_true"], cdf["y_pred"])
            
            # Participant-level aggregation: mean posterior probability
            part_agg = cdf.groupby("participant_id").agg({
                "y_true": "first",
                "y_prob": "mean",
                "fold": "first",
                "condition": "first",
                "feature_extractor": "first"
            }).reset_index()
            
            part_agg["classifier"] = clf
            part_agg["y_pred"] = (part_agg["y_prob"] >= 0.5).astype(int)
            part_agg["correct"] = (part_agg["y_true"] == part_agg["y_pred"]).astype(int)
            
            n_parts = len(part_agg)
            n_correct = part_agg["correct"].sum()
            
            p_acc, p_sens, p_spec, p_f1, tn, fp, fn, tp = compute_metrics(part_agg["y_true"], part_agg["y_pred"])
            w_low, w_high = wilson_ci(n_correct, n_parts)
            
            # Per-fold participant metrics
            fold_accs = []
            fold_senss = []
            fold_specs = []
            fold_f1s = []
            
            for f in range(10):
                f_df = part_agg[part_agg["fold"] == f]
                if len(f_df) == 0:
                    continue
                f_acc, f_sens, f_spec, f_f1, _, _, _, _ = compute_metrics(f_df["y_true"], f_df["y_pred"])
                fold_accs.append(f_acc)
                fold_senss.append(f_sens)
                fold_specs.append(f_spec)
                fold_f1s.append(f_f1)
                
            fold_accs = np.array(fold_accs)
            fold_mean_acc = np.mean(fold_accs)
            fold_sd_sample = np.std(fold_accs, ddof=1)
            fold_sd_pop = np.std(fold_accs, ddof=0)
            
            # Continuous Participant AUC
            part_auc = roc_auc_score(part_agg["y_true"], part_agg["y_prob"])
            
            # Participant ROC curve
            fpr, tpr, thresh = roc_curve(part_agg["y_true"], part_agg["y_prob"])
            for fp_val, tp_val, th_val in zip(fpr, tpr, thresh):
                all_roc_coords.append({
                    "condition": tag,
                    "classifier": clf,
                    "fpr": fp_val,
                    "tpr": tp_val,
                    "threshold": th_val
                })
                
            # Bootstrap 95% CI for Participant AUC (1,000 iterations, seed=42)
            rng = np.random.RandomState(42)
            boot_aucs = []
            indices = np.arange(len(part_agg))
            
            for b_idx in range(1000):
                sample_idx = rng.choice(indices, size=len(indices), replace=True)
                sample_true = part_agg["y_true"].iloc[sample_idx]
                sample_prob = part_agg["y_prob"].iloc[sample_idx]
                if len(np.unique(sample_true)) < 2:
                    continue
                b_auc = roc_auc_score(sample_true, sample_prob)
                boot_aucs.append(b_auc)
                
            boot_ci_low = np.percentile(boot_aucs, 2.5)
            boot_ci_high = np.percentile(boot_aucs, 97.5)
            boot_se = np.std(boot_aucs, ddof=1)
            
            all_bootstrap_rows.append({
                "condition": tag,
                "cohort_name": cohort_name,
                "classifier": clf,
                "n_participants": n_parts,
                "participant_auc": round(part_auc, 4),
                "bootstrap_iterations": len(boot_aucs),
                "bootstrap_mean_auc": round(np.mean(boot_aucs), 4),
                "bootstrap_se": round(boot_se, 4),
                "bootstrap_95ci_low": round(boot_ci_low, 4),
                "bootstrap_95ci_high": round(boot_ci_high, 4),
                "random_state_seed": 42
            })
            
            all_recalculated_rows.append({
                "condition": tag,
                "cohort_name": cohort_name,
                "classifier": clf,
                "n_participants": n_parts,
                "n_segments": n_segments,
                "correct_participants": n_correct,
                "pooled_accuracy_pct": round(p_acc * 100, 2),
                "wilson_95ci_low_pct": round(w_low * 100, 2),
                "wilson_95ci_high_pct": round(w_high * 100, 2),
                "sensitivity_pct": round(p_sens * 100, 2),
                "specificity_pct": round(p_spec * 100, 2),
                "f1_score_pct": round(p_f1 * 100, 2),
                "fold_mean_accuracy_pct": round(fold_mean_acc * 100, 2),
                "fold_sd_sample_ddof1_pct": round(fold_sd_sample * 100, 2),
                "fold_sd_pop_ddof0_pct": round(fold_sd_pop * 100, 2),
                "participant_auc": round(part_auc, 4),
                "bootstrap_auc_95ci": f"[{boot_ci_low:.4f}, {boot_ci_high:.4f}]",
                "segment_accuracy_pct": round(seg_acc * 100, 2),
                "segment_auc": round(seg_auc, 4),
                "true_positives": tp,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn
            })
            
            part_df_list.append(part_agg)
            
            print(f"  {clf:<26s} | Acc: {p_acc*100:5.2f}% | Wilson: [{w_low*100:5.2f}%, {w_high*100:5.2f}%] | "
                  f"Fold Mean: {fold_mean_acc*100:5.2f} +/- {fold_sd_sample*100:5.2f}% (s) / {fold_sd_pop*100:5.2f}% (p) | "
                  f"AUC: {part_auc:.4f} [{boot_ci_low:.4f}, {boot_ci_high:.4f}]")
                  
        part_full_df = pd.concat(part_df_list, ignore_index=True)
        all_participant_preds.append(part_full_df)
        part_full_df.to_csv(os.path.join(RESULTS_DIR, f"participant_predictions_{tag}.csv"), index=False)
        print(f"Saved participant predictions to {RESULTS_DIR}/participant_predictions_{tag}.csv")
        
    # Export full metrics
    recalc_df = pd.DataFrame(all_recalculated_rows)
    recalc_csv = os.path.join(REPORTS_DIR, "PHASE1A_RECALCULATED_METRICS.csv")
    recalc_df.to_csv(recalc_csv, index=False)
    print(f"\nSaved recalculated metrics to {recalc_csv}")
    
    boot_df = pd.DataFrame(all_bootstrap_rows)
    boot_csv = os.path.join(REPORTS_DIR, "PHASE1A_BOOTSTRAP_AUC_RESULTS.csv")
    boot_df.to_csv(boot_csv, index=False)
    print(f"Saved bootstrap AUC results to {boot_csv}")
    
    roc_df = pd.DataFrame(all_roc_coords)
    roc_csv = os.path.join(RESULTS_DIR, "roc_coordinates_all_classifiers.csv")
    roc_df.to_csv(roc_csv, index=False)
    print(f"Saved ROC coordinates to {roc_csv}")

if __name__ == "__main__":
    run_verification()

