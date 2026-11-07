"""Phase 1C — Module M3: Autonomous Double-Loop Nested Cross-Validation.

Implements true double-loop nested cross-validation:
- Outer Loop: 10-Fold Stratified Group CV (results/fold_assignments.csv).
  Outer test participants are held out and tested exactly ONCE.
- Inner Loop: 5-Fold Stratified Group CV on outer training participants only.
  Performs autonomous model and hyperparameter selection across candidates.
- Leakage Prevention:
  * StandardScaler fit strictly on inner training data in inner loop,
    and outer training data in outer refit.
  * ParticipantCalibratedSVM performs participant-grouped Platt scaling strictly within training sets.
  * Zero outer-test or inner-val leakage.
- Model Selection Objective:
  Inner mean participant-level ROC-AUC: P_i = (1/S_i) * sum(p_{i, s}).
  Deterministic tie-breaking: Inner AUC -> Inner Balanced Acc -> Model simplicity.
- Output Deliverables:
  * nested_cv_participant_predictions.csv (unbiased out-of-fold predictions)
  * nested_cv_selected_models.csv (winning model selected in each outer fold)
  * nested_cv_unbiased_metrics.json (Accuracy, Sensitivity, Specificity, F1, AUC, 95% CIs)
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, balanced_accuracy_score

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath("."))
from src.participant_calibrated_svm import ParticipantCalibratedSVM

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = "results/phase1c/M3_nested_cv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CACHE_DIR = "results/phase1c/M2_configuration_matrix/features_cache"

# Define Candidate Pipelines for Autonomous Nested Selection
CANDIDATE_SEARCH_SPACE = [
    # SPWVD + ResNet-18
    {"rep": "SPWVD", "bb": "ResNet-18", "clf": "Linear SVM", "params": {"C": 1.0}},
    {"rep": "SPWVD", "bb": "ResNet-18", "clf": "Linear SVM", "params": {"C": 0.1}},
    {"rep": "SPWVD", "bb": "ResNet-18", "clf": "RBF SVM", "params": {"C": 1.0, "gamma": "scale"}},
    {"rep": "SPWVD", "bb": "ResNet-18", "clf": "MLP", "params": {"alpha": 0.0001}},
    {"rep": "SPWVD", "bb": "ResNet-18", "clf": "Logistic Regression", "params": {"C": 1.0}},
    {"rep": "SPWVD", "bb": "ResNet-18", "clf": "Random Forest", "params": {"n_estimators": 100}},
    
    # STFT + ResNet-18
    {"rep": "STFT", "bb": "ResNet-18", "clf": "Linear SVM", "params": {"C": 1.0}},
    {"rep": "STFT", "bb": "ResNet-18", "clf": "MLP", "params": {"alpha": 0.0001}},
    {"rep": "STFT", "bb": "ResNet-18", "clf": "Logistic Regression", "params": {"C": 1.0}},
    
    # CWT Morlet + ResNet-18
    {"rep": "CWT_Morlet", "bb": "ResNet-18", "clf": "Linear SVM", "params": {"C": 1.0}},
    {"rep": "CWT_Morlet", "bb": "ResNet-18", "clf": "MLP", "params": {"alpha": 0.0001}},
    {"rep": "CWT_Morlet", "bb": "ResNet-18", "clf": "Logistic Regression", "params": {"C": 1.0}},
    
    # Welch PSD (Direct Features, D = 95)
    {"rep": "Welch_PSD", "bb": "Direct_Features", "clf": "Linear SVM", "params": {"C": 1.0}},
    {"rep": "Welch_PSD", "bb": "Direct_Features", "clf": "RBF SVM", "params": {"C": 1.0, "gamma": "scale"}},
    {"rep": "Welch_PSD", "bb": "Direct_Features", "clf": "MLP", "params": {"alpha": 0.0001}},
    {"rep": "Welch_PSD", "bb": "Direct_Features", "clf": "Logistic Regression", "params": {"C": 1.0}},
    {"rep": "Welch_PSD", "bb": "Direct_Features", "clf": "Random Forest", "params": {"n_estimators": 100}}
]

def instantiate_candidate(candidate, random_state=42):
    clf_name = candidate["clf"]
    params = candidate["params"]
    
    if clf_name == "Linear SVM":
        return ParticipantCalibratedSVM(kernel="linear", C=params.get("C", 1.0), n_calib_splits=3, random_state=random_state)
    elif clf_name == "RBF SVM":
        return ParticipantCalibratedSVM(kernel="rbf", C=params.get("C", 1.0), gamma=params.get("gamma", "scale"), n_calib_splits=3, random_state=random_state)
    elif clf_name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(100,), alpha=params.get("alpha", 0.0001), max_iter=500, random_state=random_state)
    elif clf_name == "Logistic Regression":
        return LogisticRegression(C=params.get("C", 1.0), max_iter=500, random_state=random_state)
    elif clf_name == "Random Forest":
        return RandomForestClassifier(n_estimators=params.get("n_estimators", 100), max_depth=10, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier {clf_name}")

def run_nested_cv_cohort(task, cohort_name):
    print(f"\n================================================================================")
    print(f"   RUNNING NESTED CV ON {cohort_name.upper()} (Task: {task})")
    print(f"================================================================================")
    t0 = time.time()
    
    meta_path = f"results/features_cache/metadata_{task}.csv"
    df_meta = pd.read_csv(meta_path)
    y = df_meta["numeric_label"].values
    groups = df_meta["participant_id"].values
    folds = df_meta["fold"].values
    unique_folds = sorted(np.unique(folds))
    
    # Load feature sets
    feat_dict = {
        ("SPWVD", "ResNet-18"): np.load(f"results/features_cache/features_resnet18_{task}.npy"),
        ("STFT", "ResNet-18"): np.load(os.path.join(CACHE_DIR, f"features_stft_resnet18_{task}.npy")),
        ("CWT_Morlet", "ResNet-18"): np.load(os.path.join(CACHE_DIR, f"features_cwt_resnet18_{task}.npy")),
        ("Welch_PSD", "Direct_Features"): np.load(os.path.join(CACHE_DIR, f"features_welch_{task}.npy"))
    }
    
    selected_models_ledger = []
    outer_oof_predictions = []
    
    for outer_fold in unique_folds:
        t_fold0 = time.time()
        test_mask = (folds == outer_fold)
        train_mask = ~test_mask
        
        y_outer_train = y[train_mask]
        g_outer_train = groups[train_mask]
        
        # Inner CV: 5-Fold Stratified Group CV on Outer Train only
        inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + outer_fold)
        inner_splits = list(inner_cv.split(np.zeros(len(y_outer_train)), y_outer_train, groups=g_outer_train))
        
        candidate_scores = []
        
        for cand_idx, cand in enumerate(CANDIDATE_SEARCH_SPACE):
            rep = cand["rep"]
            bb = cand["bb"]
            X_full = feat_dict[(rep, bb)]
            X_outer_train = X_full[train_mask]
            
            inner_part_aucs = []
            inner_part_baccs = []
            
            for inner_train_idx, inner_val_idx in inner_splits:
                X_it = X_outer_train[inner_train_idx]
                y_it = y_outer_train[inner_train_idx]
                g_it = g_outer_train[inner_train_idx]
                
                X_iv = X_outer_train[inner_val_idx]
                y_iv = y_outer_train[inner_val_idx]
                g_iv = g_outer_train[inner_val_idx]
                
                # Fit scaler strictly on inner train
                sc = StandardScaler()
                X_it_scaled = sc.fit_transform(X_it)
                X_iv_scaled = sc.transform(X_iv)
                
                model = instantiate_candidate(cand, random_state=42 + outer_fold)
                if isinstance(model, ParticipantCalibratedSVM):
                    model.fit(X_it_scaled, y_it, groups=g_it)
                else:
                    model.fit(X_it_scaled, y_it)
                    
                val_probs = model.predict_proba(X_iv_scaled)[:, 1]
                
                # Pool to participant level
                val_df = pd.DataFrame({"participant_id": g_iv, "y_true": y_iv, "y_prob": val_probs})
                part_val = val_df.groupby("participant_id").agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
                
                y_pt = part_val["y_true"].values
                y_pp = part_val["y_prob"].values
                
                if len(np.unique(y_pt)) == 2:
                    inner_part_aucs.append(roc_auc_score(y_pt, y_pp))
                    inner_part_baccs.append(balanced_accuracy_score(y_pt, (y_pp >= 0.5).astype(int)))
                    
            mean_auc = float(np.mean(inner_part_aucs)) if inner_part_aucs else 0.5
            mean_bacc = float(np.mean(inner_part_baccs)) if inner_part_baccs else 0.5
            
            candidate_scores.append({
                "cand_idx": cand_idx,
                "candidate": cand,
                "mean_auc": mean_auc,
                "mean_bacc": mean_bacc
            })
            
        # Select winner with deterministic tie-breaking
        # 1. Higher mean AUC -> 2. Higher mean Balanced Acc -> 3. Simpler complexity
        complexity_rank = {"Linear SVM": 1, "Logistic Regression": 2, "RBF SVM": 3, "MLP": 4, "Random Forest": 5}
        candidate_scores.sort(
            key=lambda c: (
                round(c["mean_auc"], 5),
                round(c["mean_bacc"], 5),
                -complexity_rank.get(c["candidate"]["clf"], 99)
            ),
            reverse=True
        )
        
        winner = candidate_scores[0]
        winner_cand = winner["candidate"]
        
        # Refit winner on 100% of outer training data
        rep = winner_cand["rep"]
        bb = winner_cand["bb"]
        X_full = feat_dict[(rep, bb)]
        X_outer_train = X_full[train_mask]
        X_outer_test = X_full[test_mask]
        y_outer_test = y[test_mask]
        g_outer_test = groups[test_mask]
        
        outer_scaler = StandardScaler()
        X_ot_scaled = outer_scaler.fit_transform(X_outer_train)
        X_otest_scaled = outer_scaler.transform(X_outer_test)
        
        final_model = instantiate_candidate(winner_cand, random_state=42 + outer_fold)
        if isinstance(final_model, ParticipantCalibratedSVM):
            final_model.fit(X_ot_scaled, y_outer_train, groups=g_outer_train)
        else:
            final_model.fit(X_ot_scaled, y_outer_train)
            
        test_probs = final_model.predict_proba(X_otest_scaled)[:, 1]
        
        test_indices = np.where(test_mask)[0]
        for idx_in_test, global_idx in enumerate(test_indices):
            outer_oof_predictions.append({
                "segment_global_idx": df_meta["segment_global_idx"].iloc[global_idx],
                "participant_id": df_meta["participant_id"].iloc[global_idx],
                "fold": outer_fold,
                "y_true": int(y_outer_test[idx_in_test]),
                "y_prob": float(test_probs[idx_in_test]),
                "selected_rep": winner_cand["rep"],
                "selected_backbone": winner_cand["bb"],
                "selected_classifier": winner_cand["clf"]
            })
            
        ledger_entry = {
            "outer_fold": outer_fold,
            "cohort": cohort_name,
            "selected_rep": winner_cand["rep"],
            "selected_backbone": winner_cand["bb"],
            "selected_classifier": winner_cand["clf"],
            "selected_params": json.dumps(winner_cand["params"]),
            "inner_cv_mean_auc": round(winner["mean_auc"], 4),
            "inner_cv_mean_bacc": round(winner["mean_bacc"], 4),
            "fold_runtime_sec": round(time.time() - t_fold0, 2)
        }
        selected_models_ledger.append(ledger_entry)
        print(f"  Outer Fold {outer_fold:2d} -> Selected: {winner_cand['rep']} + {winner_cand['clf']} (Inner AUC: {winner['mean_auc']:.4f}) in {time.time()-t_fold0:.1f}s")

    df_oof = pd.DataFrame(outer_oof_predictions)
    df_ledger = pd.DataFrame(selected_models_ledger)
    
    # Participant-level evaluation
    part_df = df_oof.groupby("participant_id").agg(
        y_true=("y_true", "first"),
        y_prob=("y_prob", "mean")
    ).reset_index()
    
    y_true = part_df["y_true"].values
    y_prob = part_df["y_prob"].values
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred) * 100.0
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred) * 100.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
    
    # Bootstrap 95% AUC CI
    rng = np.random.default_rng(42)
    boot_aucs = []
    n_part = len(y_true)
    for _ in range(1000):
        b_idx = rng.choice(n_part, size=n_part, replace=True)
        if len(np.unique(y_true[b_idx])) == 2:
            boot_aucs.append(roc_auc_score(y_true[b_idx], y_prob[b_idx]))
    ci_low = np.percentile(boot_aucs, 2.5) if boot_aucs else auc
    ci_high = np.percentile(boot_aucs, 97.5) if boot_aucs else auc
    
    unbiased_metrics = {
        "cohort": cohort_name,
        "task": task,
        "n_participants": n_part,
        "n_segments": len(df_oof),
        "unbiased_accuracy_pct": round(acc, 2),
        "unbiased_sensitivity_pct": round(sens, 2),
        "unbiased_specificity_pct": round(spec, 2),
        "unbiased_f1_score_pct": round(f1, 2),
        "unbiased_roc_auc": round(auc, 4),
        "auc_bootstrap_95ci": [round(ci_low, 4), round(ci_high, 4)],
        "auc_ci_str": f"[{ci_low:.4f}, {ci_high:.4f}]",
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "runtime_seconds": round(time.time() - t0, 2)
    }
    
    print(f"\n>>> UNBIASED NESTED CV RESULTS ({cohort_name}):")
    print(f"  Accuracy:    {acc:.2f}%")
    print(f"  Sensitivity: {sens:.2f}%")
    print(f"  Specificity: {spec:.2f}%")
    print(f"  F1 Score:    {f1:.2f}%")
    print(f"  ROC-AUC:     {auc:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Confusion:   TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Runtime:     {unbiased_metrics['runtime_seconds']:.1f}s")
    
    return unbiased_metrics, df_ledger, df_oof

def run_m3_nested_cv():
    t_start = time.time()
    all_metrics = {}
    
    for cohort_name, task in [("EyesClosed", "eyesClosed"), ("EyesOpen", "eyesOpen")]:
        metrics, df_ledger, df_oof = run_nested_cv_cohort(task, cohort_name)
        all_metrics[cohort_name] = metrics
        
        ledger_path = os.path.join(OUTPUT_DIR, f"nested_cv_selected_models_{task}.csv")
        oof_path = os.path.join(OUTPUT_DIR, f"nested_cv_participant_predictions_{task}.csv")
        df_ledger.to_csv(ledger_path, index=False)
        df_oof.to_csv(oof_path, index=False)
        
    metrics_json_path = os.path.join(OUTPUT_DIR, "nested_cv_unbiased_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    print(f"\n================================================================================")
    print(f"   MODULE M3 NESTED CROSS-VALIDATION COMPLETED IN {time.time()-t_start:.2f} SECONDS")
    print(f"   Deliverables saved to: {OUTPUT_DIR}/")
    print(f"================================================================================")

if __name__ == "__main__":
    run_m3_nested_cv()

