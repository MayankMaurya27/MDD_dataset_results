"""Systematic Methodological Improvement Experiments Suite.

Evaluates four pre-specified, scientifically justified experimental axes:
1. Epoch probability aggregation strategies (Mean vs. Median vs. Trimmed Mean vs. Majority Vote).
2. Pre-specified neurobiological channel weighting (Frontal-Central dominance) vs. uniform averaging.
3. SPWVD smoothing parameter variations (narrow vs. compact window geometries).
4. Training-only group-aware nested hyperparameter tuning vs. fixed baseline defaults.

Guarantees:
- Strict zero-leakage nested and outer Stratified Group Cross-Validation.
- Identical eligible participants and outer fold assignments (SEED=42).
- Zero tuning on outer test folds.
- Comprehensive recording of rationale, execution time, parameters, and all outcomes (including null/unsuccessful).
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import trim_mean
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from src.preprocessing import load_eeg_recording, segment_recording, STANDARD_1020_CHANNELS
from src.spwvd import compute_spwvd, spwvd_to_rgb_image
from src.feature_extractor import DeepFeatureExtractor
from src.evaluation import compute_classification_metrics

EXP_DIR = os.path.abspath("results/experiments")
os.makedirs(EXP_DIR, exist_ok=True)

# -------------------------------------------------------------------------------------------------
# Utility: Compute classification metrics
# -------------------------------------------------------------------------------------------------
def extract_tensor_features(extractor, image_tensors, batch_size=64):
    feature_batches = []
    for b_start in range(0, len(image_tensors), batch_size):
        b_end = min(b_start + batch_size, len(image_tensors))
        batch = torch.stack(image_tensors[b_start:b_end], dim=0)
        feats = extractor.extract_features(batch)
        feature_batches.append(feats)
    return np.concatenate(feature_batches, axis=0)

def eval_part_predictions(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    sens = recall_score(y_true, y_pred, zero_division=0) * 100.0
    spec = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    auc = roc_auc_score(y_true, y_prob) * 100.0 if len(np.unique(y_true)) > 1 else np.nan
    return {
        "accuracy": round(acc, 2),
        "balanced_accuracy": round(bal_acc, 2),
        "sensitivity": round(sens, 2),
        "specificity": round(spec, 2),
        "precision": round(prec, 2),
        "f1_score": round(f1, 2),
        "roc_auc": round(auc, 2),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

# -------------------------------------------------------------------------------------------------
# Experiment 1: Epoch Probability Aggregation Strategies
# -------------------------------------------------------------------------------------------------
def run_experiment_1_aggregation():
    print("\n================================================================================")
    print("RUNNING EXPERIMENT 1: Epoch Probability Aggregation Strategies")
    print("================================================================================")
    t0 = time.time()
    out_dir = os.path.join(EXP_DIR, "exp1_aggregation")
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    
    for task in ["eyesClosed", "eyesOpen"]:
        oof_path = f"results/baseline/out_of_fold_predictions_{task}.csv"
        df = pd.read_csv(oof_path)
        
        classifiers = df["classifier"].unique()
        for clf in classifiers:
            c_df = df[df["classifier"] == clf]
            
            # Strategies to evaluate:
            # 1. Arithmetic Mean (Baseline)
            # 2. Median Probability
            # 3. Trimmed Mean (10% trim on both tails)
            # 4. Hard Majority Voting
            
            strategies = {
                "Mean (Baseline)": lambda s: np.mean(s),
                "Median": lambda s: np.median(s),
                "Trimmed Mean (10%)": lambda s: trim_mean(s, proportiontocut=0.10),
                "Majority Vote": lambda s: (np.mean(np.array(s) >= 0.5) >= 0.5).astype(float)
            }
            
            for strat_name, strat_fn in strategies.items():
                part_rows = []
                for p_id, p_group in c_df.groupby("participant_id"):
                    y_true = p_group["y_true"].iloc[0]
                    fold = p_group["fold"].iloc[0]
                    probs = p_group["y_prob"].values
                    agg_prob = float(strat_fn(probs))
                    pred_label = 1 if agg_prob >= 0.5 else 0
                    part_rows.append({
                        "participant_id": p_id,
                        "y_true": y_true,
                        "y_prob": agg_prob,
                        "y_pred": pred_label,
                        "fold": fold
                    })
                part_df = pd.DataFrame(part_rows)
                metrics = eval_part_predictions(part_df["y_true"], part_df["y_pred"], part_df["y_prob"])
                
                # Compute fold mean and std
                fold_accs = [
                    accuracy_score(f_df["y_true"], f_df["y_pred"]) * 100.0
                    for f in range(10)
                    for f_df in [part_df[part_df["fold"] == f]]
                    if len(f_df) > 0
                ]
                
                results.append({
                    "condition": task,
                    "classifier": clf,
                    "aggregation_strategy": strat_name,
                    "num_participants": len(part_df),
                    "accuracy": metrics["accuracy"],
                    "fold_mean_acc": round(np.mean(fold_accs), 2),
                    "fold_std_acc": round(np.std(fold_accs, ddof=1), 2),
                    "sensitivity": metrics["sensitivity"],
                    "specificity": metrics["specificity"],
                    "precision": metrics["precision"],
                    "f1_score": metrics["f1_score"],
                    "roc_auc": metrics["roc_auc"],
                    "tp": metrics["tp"], "fp": metrics["fp"], "tn": metrics["tn"], "fn": metrics["fn"]
                })
                
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(out_dir, "aggregation_comparison.csv"), index=False)
    print(f"Experiment 1 complete in {time.time()-t0:.2f}s. Saved to {out_dir}/aggregation_comparison.csv")
    return res_df

# -------------------------------------------------------------------------------------------------
# Experiment 2: Pre-specified Channel Weighting vs Uniform Averaging
# -------------------------------------------------------------------------------------------------
def run_experiment_2_channel_weighting():
    print("\n================================================================================")
    print("RUNNING EXPERIMENT 2: Pre-specified Channel Weighting (Frontal-Dominant Montage)")
    print("================================================================================")
    t0 = time.time()
    out_dir = os.path.join(EXP_DIR, "exp2_channel_weighting")
    os.makedirs(out_dir, exist_ok=True)
    
    bids_dir = os.path.abspath("nm000114_v1.0.0")
    inv = pd.read_csv("reports/dataset_inventory.csv")
    folds_df = pd.read_csv("results/fold_assignments.csv")
    inv = inv.merge(folds_df[['participant_id', 'fold']], on='participant_id', how='left')
    records = inv[inv['task'] == "eyesClosed"].copy()
    
    # Pre-specified Channel Weights (STANDARD_1020_CHANNELS order):
    # Frontal/Central = 2.0; Occipital = 0.5; Temporal/Parietal = 1.0
    channel_weight_map = {
        'Fp1': 2.0, 'F3': 2.0, 'C3': 2.0, 'Fz': 2.0, 'Fp2': 2.0, 'F4': 2.0, 'C4': 2.0, 'Cz': 2.0,
        'O1': 0.5, 'O2': 0.5,
        'P3': 1.0, 'F7': 1.0, 'T3': 1.0, 'T5': 1.0, 'P4': 1.0, 'F8': 1.0, 'T4': 1.0, 'T6': 1.0, 'Pz': 1.0
    }
    raw_weights = np.array([channel_weight_map[ch] for ch in STANDARD_1020_CHANNELS])
    norm_weights = raw_weights / np.sum(raw_weights)
    
    print("Pre-specified Channel Weights:")
    for ch, w in zip(STANDARD_1020_CHANNELS, norm_weights):
        print(f"  {ch:4s}: {w:.4f}")
        
    image_tensors = []
    metadata_rows = []
    
    for idx, (_, row) in enumerate(records.iterrows()):
        p_id = row['participant_id']
        full_path = os.path.join(bids_dir, row['filename'])
        group = row['group_label']
        label = row['numeric_label']
        fold = row['fold']
        
        data, fs, chs = load_eeg_recording(full_path, l_freq=0.5, h_freq=45.0)
        segs, indices = segment_recording(data, fs, window_sec=2.0)
        if len(segs) > 15:
            step = len(segs) // 15
            selected_idx = [i * step for i in range(15)]
            segs = segs[selected_idx]
            indices = [indices[i] for i in selected_idx]
            
        for seg_idx, (s, (start_idx, end_idx)) in enumerate(zip(segs, indices)):
            spwvd_stack = []
            for ch_idx in range(19):
                wvd, freqs = compute_spwvd(s[ch_idx], fs=fs, nfft=256)
                spwvd_stack.append(wvd)
            weighted_wvd = np.average(spwvd_stack, axis=0, weights=norm_weights)
            rgb_tensor = spwvd_to_rgb_image(weighted_wvd, target_size=(224, 224))
            image_tensors.append(rgb_tensor)
            
            metadata_rows.append({
                "segment_global_idx": len(metadata_rows),
                "participant_id": p_id,
                "group_label": group,
                "numeric_label": label,
                "fold": fold,
                "segment_local_idx": seg_idx
            })
            
    df_meta = pd.DataFrame(metadata_rows)
    print(f"Generated {len(image_tensors)} channel-weighted SPWVD images.")
    
    # Feature extraction with ResNet-18
    extractor = DeepFeatureExtractor(model_name="resnet18")
    features = extract_tensor_features(extractor, image_tensors, batch_size=64)
    np.save(os.path.join(out_dir, "resnet18_features_channel_weighted.npy"), features)
    df_meta.to_csv(os.path.join(out_dir, "metadata_channel_weighted.csv"), index=False)
    
    # Evaluate across 8 classifiers using identical 10 folds
    X = features
    y = df_meta["numeric_label"].values
    groups = df_meta["participant_id"].values
    folds = df_meta["fold"].values
    
    classifiers = {
        "Linear SVM": SVC(kernel='linear', C=1.0, probability=True, random_state=42),
        "Quadratic SVM": SVC(kernel='poly', degree=2, C=1.0, probability=True, random_state=42),
        "Cubic SVM": SVC(kernel='poly', degree=3, C=1.0, probability=True, random_state=42),
        "Fine Gaussian SVM": SVC(kernel='rbf', gamma=0.1, C=1.0, probability=True, random_state=42),
        "Medium Gaussian SVM": SVC(kernel='rbf', gamma='scale', C=1.0, probability=True, random_state=42),
        "Weighted KNN": KNeighborsClassifier(n_neighbors=10, weights='distance'),
        "Ensemble Bagged Tree": RandomForestClassifier(n_estimators=100, max_features=None, random_state=42),
        "Neural Network Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }
    
    eval_results = []
    for clf_name, clf in classifiers.items():
        oof_preds = []
        for test_fold in range(10):
            train_idx = np.where(folds != test_fold)[0]
            test_idx = np.where(folds == test_fold)[0]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            
            clf.fit(X_train, y[train_idx])
            probs = clf.predict_proba(X_test)[:, 1]
            preds = clf.predict(X_test)
            
            for i, idx_val in enumerate(test_idx):
                oof_preds.append({
                    "participant_id": df_meta.iloc[idx_val]["participant_id"],
                    "y_true": y[idx_val],
                    "y_prob": probs[i],
                    "y_pred": preds[i],
                    "fold": test_fold
                })
        oof_df = pd.DataFrame(oof_preds)
        
        # Participant-level aggregation
        part_agg = oof_df.groupby("participant_id").agg(
            y_true=("y_true", "first"),
            y_prob=("y_prob", "mean"),
            fold=("fold", "first")
        ).reset_index()
        part_agg["y_pred"] = (part_agg["y_prob"] >= 0.5).astype(int)
        
        m = eval_part_predictions(part_agg["y_true"], part_agg["y_pred"], part_agg["y_prob"])
        fold_accs = [
            accuracy_score(f_df["y_true"], f_df["y_pred"]) * 100.0
            for f in range(10)
            for f_df in [part_agg[part_agg["fold"] == f]]
            if len(f_df) > 0
        ]
        
        eval_results.append({
            "experiment": "Channel-Weighted SPWVD (Frontal Dominant)",
            "classifier": clf_name,
            "accuracy": m["accuracy"],
            "fold_mean_acc": round(np.mean(fold_accs), 2),
            "fold_std_acc": round(np.std(fold_accs, ddof=1), 2),
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "f1_score": m["f1_score"],
            "roc_auc": m["roc_auc"],
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"]
        })
        
    cw_df = pd.DataFrame(eval_results)
    cw_df.to_csv(os.path.join(out_dir, "metrics_summary_channel_weighted.csv"), index=False)
    
    # Comparison against baseline
    baseline_df = pd.read_csv("results/baseline/metrics_summary_eyesClosed.csv")
    baseline_df = baseline_df[baseline_df["experiment"] == "ResNet-18 Classifiers"]
    comp = cw_df[["classifier", "accuracy", "sensitivity", "specificity", "f1_score", "roc_auc"]].merge(
        baseline_df[["classifier", "part_acc_pooled", "part_sensitivity", "part_specificity", "part_f1", "part_roc_auc"]],
        on="classifier", suffixes=("_channel_weighted", "_baseline")
    )
    comp["acc_delta"] = comp["accuracy"] - comp["part_acc_pooled"]
    comp.to_csv(os.path.join(out_dir, "comparison_to_baseline.csv"), index=False)
    print(f"Experiment 2 complete in {time.time()-t0:.2f}s. Saved to {out_dir}")
    return cw_df

# -------------------------------------------------------------------------------------------------
# Experiment 3: SPWVD Smoothing Parameter Variations
# -------------------------------------------------------------------------------------------------
def run_experiment_3_spwvd_smoothing():
    print("\n================================================================================")
    print("RUNNING EXPERIMENT 3: SPWVD Smoothing Parameter Variations")
    print("================================================================================")
    t0 = time.time()
    out_dir = os.path.join(EXP_DIR, "exp3_spwvd_smoothing")
    os.makedirs(out_dir, exist_ok=True)
    
    bids_dir = os.path.abspath("nm000114_v1.0.0")
    inv = pd.read_csv("reports/dataset_inventory.csv")
    folds_df = pd.read_csv("results/fold_assignments.csv")
    inv = inv.merge(folds_df[['participant_id', 'fold']], on='participant_id', how='left')
    records = inv[inv['task'] == "eyesClosed"].copy()
    
    smoothing_configs = {
        "VarA_High_Suppression": {
            "g": signal.windows.hamming(31),
            "h": signal.windows.kaiser(255, beta=7.0),
            "desc": "Kaiser lag beta=7.0 (stronger cross-term suppression)"
        },
        "VarB_Compact_Transient": {
            "g": signal.windows.hamming(15),
            "h": signal.windows.kaiser(127, beta=3.5),
            "desc": "Hamming Lg=15, Kaiser Lh=127 beta=3.5 (compact transient)"
        }
    }
    
    all_var_results = []
    
    for var_name, var_cfg in smoothing_configs.items():
        print(f"\nEvaluating SPWVD Smoothing Configuration: {var_name} ({var_cfg['desc']})...")
        t_var = time.time()
        g_win = var_cfg["g"]
        h_win = var_cfg["h"]
        
        image_tensors = []
        metadata_rows = []
        
        for idx, (_, row) in enumerate(records.iterrows()):
            p_id = row['participant_id']
            full_path = os.path.join(bids_dir, row['filename'])
            group = row['group_label']
            label = row['numeric_label']
            fold = row['fold']
            
            data, fs, chs = load_eeg_recording(full_path, l_freq=0.5, h_freq=45.0)
            segs, indices = segment_recording(data, fs, window_sec=2.0)
            if len(segs) > 15:
                step = len(segs) // 15
                selected_idx = [i * step for i in range(15)]
                segs = segs[selected_idx]
                indices = [indices[i] for i in selected_idx]
                
            for seg_idx, (s, (start_idx, end_idx)) in enumerate(zip(segs, indices)):
                spwvd_stack = []
                for ch_idx in range(19):
                    wvd, freqs = compute_spwvd(s[ch_idx], fs=fs, g=g_win, h=h_win, nfft=256)
                    spwvd_stack.append(wvd)
                mean_wvd = np.mean(spwvd_stack, axis=0)
                rgb_tensor = spwvd_to_rgb_image(mean_wvd, target_size=(224, 224))
                image_tensors.append(rgb_tensor)
                
                metadata_rows.append({
                    "segment_global_idx": len(metadata_rows),
                    "participant_id": p_id,
                    "group_label": group,
                    "numeric_label": label,
                    "fold": fold,
                    "segment_local_idx": seg_idx
                })
                
        df_meta = pd.DataFrame(metadata_rows)
        extractor = DeepFeatureExtractor(model_name="resnet18")
        features = extract_tensor_features(extractor, image_tensors, batch_size=64)
        
        X = features
        y = df_meta["numeric_label"].values
        folds = df_meta["fold"].values
        
        # Test on top estimators: Linear SVM, Medium Gaussian SVM, Weighted KNN, Neural Network
        target_clfs = {
            "Linear SVM": SVC(kernel='linear', C=1.0, probability=True, random_state=42),
            "Medium Gaussian SVM": SVC(kernel='rbf', gamma='scale', C=1.0, probability=True, random_state=42),
            "Weighted KNN": KNeighborsClassifier(n_neighbors=10, weights='distance'),
            "Neural Network Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        }
        
        for clf_name, clf in target_clfs.items():
            oof_preds = []
            for test_fold in range(10):
                train_idx = np.where(folds != test_fold)[0]
                test_idx = np.where(folds == test_fold)[0]
                
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                
                clf.fit(X_train, y[train_idx])
                probs = clf.predict_proba(X_test)[:, 1]
                preds = clf.predict(X_test)
                
                for i, idx_val in enumerate(test_idx):
                    oof_preds.append({
                        "participant_id": df_meta.iloc[idx_val]["participant_id"],
                        "y_true": y[idx_val],
                        "y_prob": probs[i],
                        "y_pred": preds[i],
                        "fold": test_fold
                    })
            oof_df = pd.DataFrame(oof_preds)
            part_agg = oof_df.groupby("participant_id").agg(
                y_true=("y_true", "first"),
                y_prob=("y_prob", "mean"),
                fold=("fold", "first")
            ).reset_index()
            part_agg["y_pred"] = (part_agg["y_prob"] >= 0.5).astype(int)
            
            m = eval_part_predictions(part_agg["y_true"], part_agg["y_pred"], part_agg["y_prob"])
            all_var_results.append({
                "smoothing_variant": var_name,
                "description": var_cfg["desc"],
                "classifier": clf_name,
                "accuracy": m["accuracy"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "f1_score": m["f1_score"],
                "roc_auc": m["roc_auc"],
                "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
                "runtime_sec": round(time.time() - t_var, 2)
            })
            
    sm_df = pd.DataFrame(all_var_results)
    sm_df.to_csv(os.path.join(out_dir, "smoothing_comparison.csv"), index=False)
    print(f"Experiment 3 complete in {time.time()-t0:.2f}s. Saved to {out_dir}")
    return sm_df

# -------------------------------------------------------------------------------------------------
# Experiment 4: Training-Only Group-Aware Nested Hyperparameter Tuning
# -------------------------------------------------------------------------------------------------
def run_experiment_4_nested_tuning():
    print("\n================================================================================")
    print("RUNNING EXPERIMENT 4: Training-Only Group-Aware Nested Hyperparameter Tuning")
    print("================================================================================")
    t0 = time.time()
    out_dir = os.path.join(EXP_DIR, "exp4_nested_tuning")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load cached baseline ResNet-18 features for Eyes-Closed
    X = np.load("results/features_cache/features_resnet18_eyesClosed.npy")
    df_meta = pd.read_csv("results/features_cache/metadata_eyesClosed.csv")
    y = df_meta["numeric_label"].values
    groups = df_meta["participant_id"].values
    folds = df_meta["fold"].values
    
    # Parameter search grids
    param_grids = {
        "Linear SVM": [
            {"C": 0.01}, {"C": 0.1}, {"C": 1.0}, {"C": 10.0}
        ],
        "Medium Gaussian SVM": [
            {"C": 0.1, "gamma": 'scale'}, {"C": 1.0, "gamma": 'scale'}, {"C": 10.0, "gamma": 'scale'},
            {"C": 1.0, "gamma": 0.001}, {"C": 1.0, "gamma": 0.01}
        ],
        "Weighted KNN": [
            {"n_neighbors": 3}, {"n_neighbors": 5}, {"n_neighbors": 7},
            {"n_neighbors": 10}, {"n_neighbors": 15}
        ],
        "Neural Network Classifier": [
            {"alpha": 0.0001, "hidden_layer_sizes": (100,)},
            {"alpha": 0.001, "hidden_layer_sizes": (100,)},
            {"alpha": 0.01, "hidden_layer_sizes": (100,)},
            {"alpha": 0.0001, "hidden_layer_sizes": (64,)}
        ]
    }
    
    def instantiate_model(clf_name, params):
        if clf_name == "Linear SVM":
            return SVC(kernel='linear', C=params["C"], probability=True, random_state=42)
        elif clf_name == "Medium Gaussian SVM":
            return SVC(kernel='rbf', C=params["C"], gamma=params["gamma"], probability=True, random_state=42)
        elif clf_name == "Weighted KNN":
            return KNeighborsClassifier(n_neighbors=params["n_neighbors"], weights='distance')
        elif clf_name == "Neural Network Classifier":
            return MLPClassifier(hidden_layer_sizes=params["hidden_layer_sizes"], alpha=params["alpha"], max_iter=500, random_state=42)
        raise ValueError(f"Unknown classifier: {clf_name}")
        
    nested_results = []
    tuning_log = []
    
    for clf_name, grid in param_grids.items():
        print(f"\nEvaluating nested group-aware tuning for {clf_name} across {len(grid)} configurations...")
        outer_oof_preds = []
        
        for outer_fold in range(10):
            train_idx = np.where(folds != outer_fold)[0]
            test_idx = np.where(folds == outer_fold)[0]
            
            X_train_raw = X[train_idx]
            y_train_raw = y[train_idx]
            groups_train = groups[train_idx]
            
            X_test_raw = X[test_idx]
            y_test_raw = y[test_idx]
            
            # INNER 5-FOLD STRATIFIED GROUP CV on training fold only
            # Extract unique participants in train fold to perform balanced group split
            inner_sgkf = StratifiedGroupKFold(n_splits=5)
            
            best_score = -1.0
            best_param = None
            
            for param in grid:
                inner_part_scores = []
                for inner_train_rel, inner_val_rel in inner_sgkf.split(X_train_raw, y_train_raw, groups_train):
                    scaler_inner = StandardScaler()
                    X_in_train = scaler_inner.fit_transform(X_train_raw[inner_train_rel])
                    X_in_val = scaler_inner.transform(X_train_raw[inner_val_rel])
                    
                    y_in_train = y_train_raw[inner_train_rel]
                    y_in_val = y_train_raw[inner_val_rel]
                    groups_in_val = groups_train[inner_val_rel]
                    
                    m_inner = instantiate_model(clf_name, param)
                    m_inner.fit(X_in_train, y_in_train)
                    val_probs = m_inner.predict_proba(X_in_val)[:, 1]
                    
                    # Compute inner participant-level accuracy
                    val_df = pd.DataFrame({
                        "participant_id": groups_in_val,
                        "y_true": y_in_val,
                        "y_prob": val_probs
                    })
                    agg_val = val_df.groupby("participant_id").agg(
                        y_true=("y_true", "first"),
                        y_prob=("y_prob", "mean")
                    ).reset_index()
                    agg_val["y_pred"] = (agg_val["y_prob"] >= 0.5).astype(int)
                    inner_acc = accuracy_score(agg_val["y_true"], agg_val["y_pred"]) * 100.0
                    inner_part_scores.append(inner_acc)
                    
                mean_inner_score = np.mean(inner_part_scores)
                if mean_inner_score > best_score:
                    best_score = mean_inner_score
                    best_param = param
                    
            tuning_log.append({
                "classifier": clf_name,
                "outer_fold": outer_fold,
                "selected_param": str(best_param),
                "inner_cv_mean_acc": round(best_score, 2)
            })
            
            # Refit winning parameter on FULL outer training fold
            scaler_outer = StandardScaler()
            X_train_outer = scaler_outer.fit_transform(X_train_raw)
            X_test_outer = scaler_outer.transform(X_test_raw)
            
            best_model = instantiate_model(clf_name, best_param)
            best_model.fit(X_train_outer, y_train_raw)
            test_probs = best_model.predict_proba(X_test_outer)[:, 1]
            test_preds = best_model.predict(X_test_outer)
            
            for i, idx_val in enumerate(test_idx):
                outer_oof_preds.append({
                    "participant_id": df_meta.iloc[idx_val]["participant_id"],
                    "y_true": y_test_raw[i],
                    "y_prob": test_probs[i],
                    "y_pred": test_preds[i],
                    "outer_fold": outer_fold
                })
                
        outer_oof_df = pd.DataFrame(outer_oof_preds)
        part_agg = outer_oof_df.groupby("participant_id").agg(
            y_true=("y_true", "first"),
            y_prob=("y_prob", "mean"),
            outer_fold=("outer_fold", "first")
        ).reset_index()
        part_agg["y_pred"] = (part_agg["y_prob"] >= 0.5).astype(int)
        
        m = eval_part_predictions(part_agg["y_true"], part_agg["y_pred"], part_agg["y_prob"])
        fold_accs = [
            accuracy_score(f_df["y_true"], f_df["y_pred"]) * 100.0
            for f in range(10)
            for f_df in [part_agg[part_agg["outer_fold"] == f]]
            if len(f_df) > 0
        ]
        
        nested_results.append({
            "classifier": clf_name,
            "validation_scheme": "Nested Group CV (Inner 5-Fold, Outer 10-Fold)",
            "accuracy": m["accuracy"],
            "fold_mean_acc": round(np.mean(fold_accs), 2),
            "fold_std_acc": round(np.std(fold_accs, ddof=1), 2),
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "precision": m["precision"],
            "f1_score": m["f1_score"],
            "roc_auc": m["roc_auc"],
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"]
        })
        
    nested_df = pd.DataFrame(nested_results)
    nested_df.to_csv(os.path.join(out_dir, "nested_cv_results.csv"), index=False)
    pd.DataFrame(tuning_log).to_csv(os.path.join(out_dir, "best_hyperparams_per_fold.csv"), index=False)
    
    # Comparison against fixed baseline
    baseline_df = pd.read_csv("results/baseline/metrics_summary_eyesClosed.csv")
    comp = nested_df[["classifier", "accuracy", "sensitivity", "specificity", "f1_score", "roc_auc"]].merge(
        baseline_df[["classifier", "part_acc_pooled", "part_sensitivity", "part_specificity", "part_f1", "part_roc_auc"]],
        on="classifier"
    )
    comp["accuracy_delta"] = comp["accuracy"] - comp["part_acc_pooled"]
    comp.to_csv(os.path.join(out_dir, "comparison_to_fixed_baseline.csv"), index=False)
    print(f"Experiment 4 complete in {time.time()-t0:.2f}s. Saved to {out_dir}")
    return nested_df

# -------------------------------------------------------------------------------------------------
# Main Orchestrator
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    t_suite = time.time()
    print("================================================================================")
    print("STARTING SCIENTIFIC IMPROVEMENT EXPERIMENT SUITE")
    print("================================================================================")
    
    exp1_file = os.path.join(EXP_DIR, "exp1_aggregation", "aggregation_comparison.csv")
    if os.path.exists(exp1_file):
        print("Experiment 1 already completed. Loading results.")
        exp1_df = pd.read_csv(exp1_file)
    else:
        exp1_df = run_experiment_1_aggregation()
        
    exp2_file = os.path.join(EXP_DIR, "exp2_channel_weighting", "metrics_summary_channel_weighted.csv")
    if os.path.exists(exp2_file):
        print("Experiment 2 already completed. Loading results.")
        exp2_df = pd.read_csv(exp2_file)
    else:
        exp2_df = run_experiment_2_channel_weighting()
        
    exp3_file = os.path.join(EXP_DIR, "exp3_spwvd_smoothing", "smoothing_comparison.csv")
    if os.path.exists(exp3_file):
        print("Experiment 3 already completed. Loading results.")
        exp3_df = pd.read_csv(exp3_file)
    else:
        exp3_df = run_experiment_3_spwvd_smoothing()
        
    exp4_df = run_experiment_4_nested_tuning()
    
    print("\n================================================================================")
    print(f"ALL EXPERIMENTS COMPLETED SUCCESSFULLY IN {time.time()-t_suite:.1f}s")
    print("================================================================================")
