"""Stage 2 & 3: Leakage-Safe Stratified Group 10-Fold Experiments.

Evaluates:
1. ResNet-18 Deep Features with 8 Classifiers:
   - Linear SVM
   - Quadratic SVM
   - Cubic SVM
   - Fine Gaussian SVM
   - Medium Gaussian SVM
   - Weighted KNN
   - Ensemble Bagged Trees
   - Neural Network Classifier
2. CNN Architecture Comparison (ResNet-18, ResNet-50, AlexNet, VGG-16)

Outputs condition-specific and combined deliverables with explicit titles,
participant counts, and condition tracking.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.classifiers import ALL_CLASSIFIER_NAMES, get_classifier
from src.evaluation import compute_classification_metrics, aggregate_participant_predictions

CACHE_DIR = os.path.abspath("results/features_cache")
RESULTS_DIR = os.path.abspath("results")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")

def plot_confusion_matrix(tp, fp, tn, fn, title, subtitle, save_path):
    cm = np.array([[tn, fp], [fn, tp]])
    total = tn + fp + fn + tp
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)
    
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / total * 100.0
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if val > total*0.4 else "black")
            
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["HC (0)", "MDD (1)"], fontweight="bold")
    ax.set_yticklabels(["HC (0)", "MDD (1)"], fontweight="bold")
    ax.set_xlabel("Predicted Diagnosis", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Clinical Label", fontweight="bold", labelpad=8)
    
    full_title = f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=10, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_experiment_for_task(task="eyesClosed"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CM_DIR, exist_ok=True)
    
    meta_path = os.path.join(CACHE_DIR, f"metadata_{task}.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata cache not found: {meta_path}. Run extract_features.py first.")
    df_meta = pd.read_csv(meta_path)
    
    n_participants = df_meta['participant_id'].nunique()
    n_segments = len(df_meta)
    folds = sorted(df_meta['fold'].unique())
    print(f"\n=======================================================")
    print(f"RUNNING EXPERIMENTS: Task={task} | Participants={n_participants} | Segments={n_segments}")
    print(f"=======================================================")
    
    y_all = df_meta['numeric_label'].values
    pids = df_meta['participant_id'].values
    fold_col = df_meta['fold'].values
    
    # -------------------------------------------------------------
    # 1. ResNet-18 Features across 8 Classifiers
    # -------------------------------------------------------------
    feat_path_r18 = os.path.join(CACHE_DIR, f"features_resnet18_{task}.npy")
    X_r18 = np.load(feat_path_r18)
    
    summary_records = []
    fold_metric_records = []
    oof_prediction_records = []
    
    for clf_name in ALL_CLASSIFIER_NAMES:
        print(f"  -> Evaluating {clf_name}...")
        fold_results_participant = []
        fold_results_segment = []
        oof_seg_df_list = []
        
        for f in folds:
            train_mask = (fold_col != f)
            test_mask = (fold_col == f)
            
            X_train, y_train = X_r18[train_mask], y_all[train_mask]
            X_test, y_test = X_r18[test_mask], y_all[test_mask]
            test_pids = pids[test_mask]
            
            # FIT SCALER STRICTLY ON TRAINING FOLD
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf = get_classifier(clf_name)
            clf.fit(X_train_scaled, y_train)
            
            preds = clf.predict(X_test_scaled)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_test_scaled)[:, 1]
            elif hasattr(clf, "decision_function"):
                df_val = clf.decision_function(X_test_scaled)
                probs = 1.0 / (1.0 + np.exp(-df_val))
            else:
                probs = preds.astype(float)
                
            m_seg = compute_classification_metrics(y_test, preds, probs)
            fold_results_segment.append(m_seg)
            
            df_fold_seg = pd.DataFrame({
                "condition": task,
                "participant_id": test_pids,
                "y_true": y_test,
                "y_pred": preds,
                "y_prob": probs,
                "fold": f,
                "classifier": clf_name,
                "feature_extractor": "ResNet-18"
            })
            oof_seg_df_list.append(df_fold_seg)
            
            df_fold_part = aggregate_participant_predictions(df_fold_seg)
            m_part = compute_classification_metrics(
                df_fold_part['y_true'],
                df_fold_part['y_pred'],
                df_fold_part['mean_prob']
            )
            fold_results_participant.append(m_part)
            
            fold_metric_records.append({
                "condition": task,
                "classifier": clf_name,
                "feature_extractor": "ResNet-18",
                "fold": f,
                "part_accuracy": m_part["accuracy"],
                "part_bal_acc": m_part["balanced_accuracy"],
                "part_precision": m_part["precision"],
                "part_sensitivity": m_part["sensitivity"],
                "part_specificity": m_part["specificity"],
                "part_f1": m_part["f1_score"],
                "part_roc_auc": m_part["roc_auc"],
                "seg_accuracy": m_seg["accuracy"],
                "seg_f1": m_seg["f1_score"]
            })
            
        full_oof_seg = pd.concat(oof_seg_df_list, ignore_index=True)
        oof_prediction_records.append(full_oof_seg)
        
        full_oof_part = aggregate_participant_predictions(full_oof_seg)
        
        pooled_part_metrics = compute_classification_metrics(
            full_oof_part['y_true'],
            full_oof_part['y_pred'],
            full_oof_part['mean_prob']
        )
        pooled_seg_metrics = compute_classification_metrics(
            full_oof_seg['y_true'],
            full_oof_seg['y_pred'],
            full_oof_seg['y_prob']
        )
        
        acc_vals = [r["accuracy"] for r in fold_results_participant]
        f1_vals = [r["f1_score"] for r in fold_results_participant]
        mean_acc, std_acc = np.mean(acc_vals), np.std(acc_vals)
        mean_f1, std_f1 = np.mean(f1_vals), np.std(f1_vals)
        
        clean_clf_name = clf_name.replace(" ", "_").lower()
        cm_filename = f"cm_resnet18_{clean_clf_name}_{task}_N{n_participants}.png"
        cm_path = os.path.join(CM_DIR, cm_filename)
        
        plot_confusion_matrix(
            tp=pooled_part_metrics["tp"],
            fp=pooled_part_metrics["fp"],
            tn=pooled_part_metrics["tn"],
            fn=pooled_part_metrics["fn"],
            title=f"{clf_name} (ResNet-18)",
            subtitle=f"Condition: {task} | Participant-Level (N={n_participants})",
            save_path=cm_path
        )
        
        summary_records.append({
            "condition": task,
            "experiment": "ResNet-18 Classifiers",
            "feature_extractor": "ResNet-18",
            "classifier": clf_name,
            "num_participants": n_participants,
            "num_segments": n_segments,
            "part_acc_mean": round(mean_acc, 2),
            "part_acc_std": round(std_acc, 2),
            "part_acc_pooled": pooled_part_metrics["accuracy"],
            "part_balanced_acc": pooled_part_metrics["balanced_accuracy"],
            "part_precision": pooled_part_metrics["precision"],
            "part_sensitivity": pooled_part_metrics["sensitivity"],
            "part_specificity": pooled_part_metrics["specificity"],
            "part_f1": pooled_part_metrics["f1_score"],
            "part_roc_auc": pooled_part_metrics["roc_auc"],
            "seg_acc_pooled": pooled_seg_metrics["accuracy"],
            "seg_f1_pooled": pooled_seg_metrics["f1_score"],
            "tp": pooled_part_metrics["tp"],
            "fp": pooled_part_metrics["fp"],
            "tn": pooled_part_metrics["tn"],
            "fn": pooled_part_metrics["fn"]
        })
        
    # -------------------------------------------------------------
    # 2. CNN Architecture Comparison (ResNet-18, ResNet-50, AlexNet, VGG-16)
    # -------------------------------------------------------------
    models = ["resnet18", "resnet50", "alexnet", "vgg16"]
    for m_name in models:
        feat_path = os.path.join(CACHE_DIR, f"features_{m_name}_{task}.npy")
        if not os.path.exists(feat_path):
            continue
        X_m = np.load(feat_path)
        
        fold_accs = []
        oof_part_list = []
        for f in folds:
            train_mask = (fold_col != f)
            test_mask = (fold_col == f)
            
            X_train, y_train = X_m[train_mask], y_all[train_mask]
            X_test, y_test = X_m[test_mask], y_all[test_mask]
            test_pids = pids[test_mask]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf = get_classifier("Weighted KNN")
            clf.fit(X_train_scaled, y_train)
            preds = clf.predict(X_test_scaled)
            probs = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else preds.astype(float)
            
            df_fold_seg = pd.DataFrame({
                "condition": task,
                "participant_id": test_pids,
                "y_true": y_test,
                "y_prob": probs
            })
            df_fold_part = aggregate_participant_predictions(df_fold_seg)
            m_part = compute_classification_metrics(df_fold_part['y_true'], df_fold_part['y_pred'], df_fold_part['mean_prob'])
            fold_accs.append(m_part["accuracy"])
            oof_part_list.append(df_fold_part)
            
        full_oof_part = pd.concat(oof_part_list, ignore_index=True)
        pooled = compute_classification_metrics(full_oof_part['y_true'], full_oof_part['y_pred'], full_oof_part['mean_prob'])
        mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
        
        summary_records.append({
            "condition": task,
            "experiment": "CNN Architecture Comparison",
            "feature_extractor": m_name.upper(),
            "classifier": "Weighted KNN",
            "num_participants": n_participants,
            "num_segments": n_segments,
            "part_acc_mean": round(mean_acc, 2),
            "part_acc_std": round(std_acc, 2),
            "part_acc_pooled": pooled["accuracy"],
            "part_balanced_acc": pooled["balanced_accuracy"],
            "part_precision": pooled["precision"],
            "part_sensitivity": pooled["sensitivity"],
            "part_specificity": pooled["specificity"],
            "part_f1": pooled["f1_score"],
            "part_roc_auc": pooled["roc_auc"],
            "seg_acc_pooled": None,
            "seg_f1_pooled": None,
            "tp": pooled["tp"],
            "fp": pooled["fp"],
            "tn": pooled["tn"],
            "fn": pooled["fn"]
        })
        
    df_task_summary = pd.DataFrame(summary_records)
    df_task_fold_metrics = pd.DataFrame(fold_metric_records)
    df_task_oof = pd.concat(oof_prediction_records, ignore_index=True)
    
    # Save condition-specific files
    df_task_summary.to_csv(os.path.join(RESULTS_DIR, f"metrics_summary_{task}.csv"), index=False)
    df_task_fold_metrics.to_csv(os.path.join(RESULTS_DIR, f"fold_metrics_{task}.csv"), index=False)
    df_task_oof.to_csv(os.path.join(RESULTS_DIR, f"out_of_fold_predictions_{task}.csv"), index=False)
    print(f"Condition-specific deliverables saved for {task}.")
    
    return df_task_summary, df_task_fold_metrics, df_task_oof

def run_all():
    tasks = ["eyesClosed", "eyesOpen"]
    all_summaries = []
    all_fold_metrics = []
    all_oofs = []
    
    for t in tasks:
        s, f, o = run_experiment_for_task(task=t)
        all_summaries.append(s)
        all_fold_metrics.append(f)
        all_oofs.append(o)
        
    # Combine into unified master files
    master_summary = pd.concat(all_summaries, ignore_index=True)
    master_fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    master_oof = pd.concat(all_oofs, ignore_index=True)
    
    master_summary.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)
    master_fold_metrics.to_csv(os.path.join(RESULTS_DIR, "fold_metrics.csv"), index=False)
    master_oof.to_csv(os.path.join(RESULTS_DIR, "out_of_fold_predictions.csv"), index=False)
    print("\nMaster unified deliverables (metrics_summary.csv, fold_metrics.csv, out_of_fold_predictions.csv) updated successfully.")

if __name__ == '__main__':
    run_all()
