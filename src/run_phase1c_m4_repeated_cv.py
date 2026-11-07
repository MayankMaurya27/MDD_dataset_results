"""Phase 1C — Module M4: Repeated Participant-Grouped Cross-Validation.

Implements 10 complete repetitions of 10-Fold Stratified Group Cross-Validation
across predefined seeds: [42, 101, 202, 303, 404, 505, 606, 707, 808, 909].

Statistical Standards (Reviewer Point 4):
- The complete 10-fold partition (one repetition) is the fundamental unit of observation.
- Each repetition yields ONE complete participant-level metric across all participants.
- Variability is reported as the mean +/- sample standard deviation (ddof=1) across
  the N = 10 complete repetitions.
- Strict Prohibition: 100 fold scores must NEVER be treated as 100 independent observations,
  which would severely violate statistical independence and artificially inflate degrees of freedom.
- Full fold-level metrics are recorded in an audit log for complete transparency.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath("."))
from src.participant_calibrated_svm import ParticipantCalibratedSVM

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = "results/phase1c/M4_repeated_cv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CACHE_DIR = "results/phase1c/M2_configuration_matrix/features_cache"

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]

BENCHMARK_PIPELINES = [
    {
        "name": "SPWVD_ResNet18_LinearSVM",
        "rep": "SPWVD",
        "bb": "ResNet-18",
        "clf": "Linear SVM"
    },
    {
        "name": "SPWVD_ResNet18_MLP",
        "rep": "SPWVD",
        "bb": "ResNet-18",
        "clf": "MLP"
    },
    {
        "name": "Welch_PSD_RBFSVM",
        "rep": "Welch_PSD",
        "bb": "Direct_Features",
        "clf": "RBF SVM"
    }
]

def get_calibrated_model(clf_name, seed):
    if clf_name == "Linear SVM":
        return ParticipantCalibratedSVM(kernel="linear", C=1.0, n_calib_splits=3, random_state=seed)
    elif clf_name == "RBF SVM":
        return ParticipantCalibratedSVM(kernel="rbf", C=1.0, gamma="scale", n_calib_splits=3, random_state=seed)
    elif clf_name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier {clf_name}")

def run_repeated_cv_cohort(task, cohort_name):
    print(f"\n================================================================================")
    print(f"   RUNNING 10-REPEAT x 10-FOLD GROUP CV ON {cohort_name.upper()} (Task: {task})")
    print(f"================================================================================")
    t0 = time.time()
    
    meta_path = f"results/features_cache/metadata_{task}.csv"
    df_meta = pd.read_csv(meta_path)
    y = df_meta["numeric_label"].values
    groups = df_meta["participant_id"].values
    
    # Load feature sets
    feat_dict = {
        ("SPWVD", "ResNet-18"): np.load(f"results/features_cache/features_resnet18_{task}.npy"),
        ("Welch_PSD", "Direct_Features"): np.load(os.path.join(CACHE_DIR, f"features_welch_{task}.npy"))
    }
    
    repeat_level_results = []
    fold_level_audit_logs = []
    
    for pipe in BENCHMARK_PIPELINES:
        pipe_name = pipe["name"]
        rep = pipe["rep"]
        bb = pipe["bb"]
        clf_name = pipe["clf"]
        X_full = feat_dict[(rep, bb)]
        
        print(f"\nEvaluating Pipeline: {pipe_name} across 10 Complete Repetitions...")
        
        rep_accs = []
        rep_aucs = []
        rep_senss = []
        rep_specs = []
        rep_f1s = []
        
        for rep_idx, seed in enumerate(SEEDS):
            t_rep0 = time.time()
            sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
            splits = list(sgkf.split(X_full, y, groups=groups))
            
            oof_predictions = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                X_tr, y_tr, g_tr = X_full[train_idx], y[train_idx], groups[train_idx]
                X_te, y_te, g_te = X_full[test_idx], y[test_idx], groups[test_idx]
                
                # Assert zero group leakage
                assert set(g_tr).isdisjoint(set(g_te)), "Leakage error: participants cross folds!"
                
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_te_scaled = scaler.transform(X_te)
                
                model = get_calibrated_model(clf_name, seed=seed + fold_idx)
                if isinstance(model, ParticipantCalibratedSVM):
                    model.fit(X_tr_scaled, y_tr, groups=g_tr)
                else:
                    model.fit(X_tr_scaled, y_tr)
                    
                probs = model.predict_proba(X_te_scaled)[:, 1]
                
                # Record fold-level metrics for auditability
                fold_df = pd.DataFrame({"participant_id": g_te, "y_true": y_te, "y_prob": probs})
                part_fold = fold_df.groupby("participant_id").agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
                f_true = part_fold["y_true"].values
                f_pred = (part_fold["y_prob"].values >= 0.5).astype(int)
                f_acc = accuracy_score(f_true, f_pred) * 100.0
                
                fold_level_audit_logs.append({
                    "cohort": cohort_name,
                    "pipeline": pipe_name,
                    "repetition_idx": rep_idx + 1,
                    "seed": seed,
                    "fold_idx": fold_idx,
                    "n_test_participants": len(part_fold),
                    "fold_accuracy_pct": round(f_acc, 2)
                })
                
                for i_te, orig_idx in enumerate(test_idx):
                    oof_predictions.append({
                        "participant_id": g_te[i_te],
                        "y_true": int(y_te[i_te]),
                        "y_prob": float(probs[i_te])
                    })
                    
            # Complete Repetition Participant-Level Evaluation
            rep_df = pd.DataFrame(oof_predictions).groupby("participant_id").agg(
                y_true=("y_true", "first"),
                y_prob=("y_prob", "mean")
            ).reset_index()
            
            y_true_rep = rep_df["y_true"].values
            y_prob_rep = rep_df["y_prob"].values
            y_pred_rep = (y_prob_rep >= 0.5).astype(int)
            
            acc = accuracy_score(y_true_rep, y_pred_rep) * 100.0
            auc = roc_auc_score(y_true_rep, y_prob_rep)
            f1 = f1_score(y_true_rep, y_pred_rep) * 100.0
            tn, fp, fn, tp = confusion_matrix(y_true_rep, y_pred_rep).ravel()
            sens = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
            
            rep_accs.append(acc)
            rep_aucs.append(auc)
            rep_senss.append(sens)
            rep_specs.append(spec)
            rep_f1s.append(f1)
            
            repeat_level_results.append({
                "cohort": cohort_name,
                "pipeline": pipe_name,
                "repetition_idx": rep_idx + 1,
                "seed": seed,
                "accuracy_pct": round(acc, 2),
                "sensitivity_pct": round(sens, 2),
                "specificity_pct": round(spec, 2),
                "f1_score_pct": round(f1, 2),
                "roc_auc": round(auc, 4),
                "repetition_runtime_sec": round(time.time() - t_rep0, 2)
            })
            
            print(f"  Rep {rep_idx+1:2d}/10 (seed={seed:3d}) -> Acc: {acc:5.2f}% | AUC: {auc:.4f} | Sens: {sens:5.2f}% | Spec: {spec:5.2f}% ({time.time()-t_rep0:.1f}s)")
            
        print(f"  >>> {pipe_name} Summary across 10 Complete Repetitions:")
        print(f"      Accuracy: {np.mean(rep_accs):.2f}% +/- {np.std(rep_accs, ddof=1):.2f}% [Min: {np.min(rep_accs):.2f}%, Max: {np.max(rep_accs):.2f}%]")
        print(f"      ROC-AUC:  {np.mean(rep_aucs):.4f} +/- {np.std(rep_aucs, ddof=1):.4f} [Min: {np.min(rep_aucs):.4f}, Max: {np.max(rep_aucs):.4f}]")
        
    df_reps = pd.DataFrame(repeat_level_results)
    df_folds = pd.DataFrame(fold_level_audit_logs)
    
    # Save CSV deliverables
    rep_csv = os.path.join(OUTPUT_DIR, f"repeated_cv_10repeats_{task}.csv")
    fold_csv = os.path.join(OUTPUT_DIR, f"repeated_cv_100folds_audit_{task}.csv")
    df_reps.to_csv(rep_csv, index=False)
    df_folds.to_csv(fold_csv, index=False)
    
    # Aggregate summary
    summary_rows = []
    for pipe_name, group in df_reps.groupby("pipeline"):
        summary_rows.append({
            "cohort": cohort_name,
            "pipeline": pipe_name,
            "n_repeats": len(group),
            "mean_accuracy_pct": round(float(group["accuracy_pct"].mean()), 2),
            "sd_accuracy_ddof1": round(float(group["accuracy_pct"].std(ddof=1)), 2),
            "min_accuracy_pct": round(float(group["accuracy_pct"].min()), 2),
            "max_accuracy_pct": round(float(group["accuracy_pct"].max()), 2),
            "mean_roc_auc": round(float(group["roc_auc"].mean()), 4),
            "sd_roc_auc_ddof1": round(float(group["roc_auc"].std(ddof=1)), 4),
            "min_roc_auc": round(float(group["roc_auc"].min()), 4),
            "max_roc_auc": round(float(group["roc_auc"].max()), 4),
            "mean_sensitivity_pct": round(float(group["sensitivity_pct"].mean()), 2),
            "mean_specificity_pct": round(float(group["specificity_pct"].mean()), 2),
            "mean_f1_score_pct": round(float(group["f1_score_pct"].mean()), 2)
        })
    df_summary = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(OUTPUT_DIR, f"repeated_cv_summary_{task}.csv")
    df_summary.to_csv(sum_csv, index=False)
    
    return df_summary

def run_m4_repeated_cv():
    t_start = time.time()
    for cohort_name, task in [("EyesClosed", "eyesClosed"), ("EyesOpen", "eyesOpen")]:
        run_repeated_cv_cohort(task, cohort_name)
        
    print(f"\n================================================================================")
    print(f"   MODULE M4 REPEATED CROSS-VALIDATION COMPLETED IN {time.time()-t_start:.2f} SECONDS")
    print(f"   Deliverables saved to: {OUTPUT_DIR}/")
    print(f"================================================================================")

if __name__ == "__main__":
    run_m4_repeated_cv()

