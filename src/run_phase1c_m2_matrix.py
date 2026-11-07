"""Phase 1C — Module M2: Experimental Configuration Matrix.

Evaluates exactly 35 valid model configurations across Eyes-Closed and Eyes-Open cohorts:

1. Time-Frequency Representations with CNN Backbones (30 configurations):
   - Representations (3): SPWVD, STFT, CWT Morlet
   - Backbones (2): ResNet-18 (512-D), ResNet-50 (2048-D)
   - Classifiers (5): Linear SVM, RBF SVM, MLP, Logistic Regression, Random Forest
   - Total: 3 x 2 x 5 = 30 valid pipelines.

2. Classical Spectral Representation with Direct Features (5 configurations):
   - Representation (1): Welch PSD
   - Backbone: None / Direct Features (D = 95, 5 canonical bands x 19 channels, log10-transformed)
   - Classifiers (5): Linear SVM, RBF SVM, MLP, Logistic Regression, Random Forest
   - Total: 1 x 1 x 5 = 5 valid pipelines.
   - Strict Constraint: Welch PSD features are NEVER passed into 2D CNN image backbones.

Methodological Guarantees:
- Stratified Group 10-Fold CV identical to historical partitions (results/fold_assignments.csv).
- Zero participant leakage across folds.
- StandardScaler fit strictly on training participants within each fold.
- ParticipantCalibratedSVM used for margin classifiers (Linear SVM, RBF SVM) with
  participant-grouped Platt scaling calibration (zero intra-participant calibration leakage).
- Segment-to-participant posterior mean aggregation: P_i = (1/S_i) * sum(p_{i, s}).
- Continuous ROC curves and 1,000-sample bootstrap 95% AUC confidence intervals.
"""

import os
import sys
import glob
import time
import json
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
import torch
import torchvision.models as models
import mne

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath("."))
from src.participant_calibrated_svm import ParticipantCalibratedSVM

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = "results/phase1c/M2_configuration_matrix"
CACHE_DIR = os.path.join(OUTPUT_DIR, "features_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

STANDARD_1020_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'F7', 'T3', 'T5', 'Fz',
    'Fp2', 'F4', 'C4', 'P4', 'O2',
    'F8', 'T4', 'T6', 'Cz', 'Pz'
]

BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0)
}

# -------------------------------------------------------------------------
# Feature Extraction Helpers
# -------------------------------------------------------------------------

def load_clean_recording(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    name_mapping = {}
    for ch in raw.ch_names:
        clean = ch.replace('EEG ', '').replace('-LE', '').replace('-A1', '').replace('-A2', '').strip()
        if clean in STANDARD_1020_CHANNELS:
            name_mapping[ch] = clean
    raw.rename_channels(name_mapping)
    raw.pick(STANDARD_1020_CHANNELS)
    raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
    data = raw.get_data() * 1e6 # microvolts
    fs = float(raw.info['sfreq'])
    return data, fs

def extract_welch_features(segments, fs=256):
    """Computes log10 Welch band powers: D = 95."""
    n_seg = len(segments)
    f, psd = signal.welch(segments, fs=fs, nperseg=256, noverlap=128, axis=-1)
    feats = []
    for b_name, (low, high) in BANDS.items():
        idx_band = np.where((f >= low) & (f <= high))[0]
        band_pwr = np.trapezoid(psd[:, :, idx_band], f[idx_band], axis=-1)
        feats.append(band_pwr)
    feat_mat = np.concatenate(feats, axis=-1)
    log_feats = np.log10(np.maximum(feat_mat, 1e-12))
    return log_feats

def tfr_to_cnn_embedding(tfr_maps, cnn_model, device, batch_size=64):
    """Interpolates 2D TFR (n_seg, H, W) to (3, 224, 224) and extracts CNN features."""
    n_seg = len(tfr_maps)
    embeddings = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    with torch.no_grad():
        for b in range(0, n_seg, batch_size):
            batch_maps = tfr_maps[b:b+batch_size]
            t_batch = torch.tensor(batch_maps, dtype=torch.float32, device=device).unsqueeze(1)
            t_batch = t_batch.repeat(1, 3, 1, 1)
            # Min-max normalize per segment image
            mins = t_batch.amin(dim=(-2, -1), keepdim=True)
            maxs = t_batch.amax(dim=(-2, -1), keepdim=True)
            diffs = torch.clamp(maxs - mins, min=1e-8)
            t_batch = (t_batch - mins) / diffs
            t_resized = torch.nn.functional.interpolate(t_batch, size=(224, 224), mode='bilinear', align_corners=False)
            t_norm = (t_resized - mean) / std
            emb = cnn_model(t_norm)
            embeddings.append(emb.cpu().numpy())
            
    return np.concatenate(embeddings, axis=0)

def compute_stft_maps(segments, fs=256):
    n_seg = len(segments)
    tfr_list = []
    for s in segments:
        f, t_pts, Zxx = signal.stft(s, fs=fs, window='hann', nperseg=64, noverlap=32)
        idx_f = np.where((f >= 0.5) & (f <= 45.0))[0]
        pwr = np.abs(Zxx[:, idx_f, :]) ** 2
        pwr_avg = np.mean(pwr, axis=0)
        tfr_list.append(np.log10(np.maximum(pwr_avg, 1e-12)))
    return np.array(tfr_list)

def compute_cwt_maps(segments, fs=256):
    freqs = np.logspace(np.log10(1.0), np.log10(45.0), 32)
    tfr = mne.time_frequency.tfr_array_morlet(
        segments, sfreq=fs, freqs=freqs, n_cycles=freqs / 2.0, output='power', verbose=False
    )
    pwr_avg = np.mean(tfr, axis=1) # Average across 19 channels
    return np.log10(np.maximum(pwr_avg, 1e-12))

# -------------------------------------------------------------------------
# Classifier Factory
# -------------------------------------------------------------------------

def get_classifier(clf_name, random_state=42):
    name = clf_name.lower().replace("-", " ").strip()
    if name == "linear svm":
        return ParticipantCalibratedSVM(kernel="linear", C=1.0, n_calib_splits=3, random_state=random_state)
    elif name in ["rbf svm", "medium gaussian svm"]:
        return ParticipantCalibratedSVM(kernel="rbf", C=1.0, gamma="scale", n_calib_splits=3, random_state=random_state)
    elif name in ["mlp", "neural network"]:
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state)
    elif name in ["logistic regression", "logreg"]:
        return LogisticRegression(C=1.0, max_iter=500, random_state=random_state)
    elif name in ["random forest", "ensemble bagged tree"]:
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")

# -------------------------------------------------------------------------
# Evaluation Harness
# -------------------------------------------------------------------------

def evaluate_pipeline(X, df_meta, clf_name, pipeline_desc, n_bootstrap=1000, seed=42):
    y = df_meta["numeric_label"].values
    groups = df_meta["participant_id"].values
    folds = df_meta["fold"].values
    unique_folds = sorted(np.unique(folds))
    
    oof_rows = []
    
    for fold in unique_folds:
        test_mask = (folds == fold)
        train_mask = ~test_mask
        
        X_train, y_train, g_train = X[train_mask], y[train_mask], groups[train_mask]
        X_test, y_test, g_test = X[test_mask], y[test_mask], groups[test_mask]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = get_classifier(clf_name, random_state=seed)
        
        if isinstance(clf, ParticipantCalibratedSVM):
            clf.fit(X_train_scaled, y_train, groups=g_train)
        else:
            clf.fit(X_train_scaled, y_train)
            
        probs = clf.predict_proba(X_test_scaled)[:, 1]
        
        test_indices = np.where(test_mask)[0]
        for idx_in_test, global_idx in enumerate(test_indices):
            oof_rows.append({
                "segment_global_idx": df_meta["segment_global_idx"].iloc[global_idx],
                "participant_id": df_meta["participant_id"].iloc[global_idx],
                "fold": fold,
                "y_true": int(y_test[idx_in_test]),
                "y_prob": float(probs[idx_in_test]),
                "pipeline": pipeline_desc,
                "classifier": clf_name
            })
            
    df_oof = pd.DataFrame(oof_rows)
    
    # Participant-level aggregation
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
    rng = np.random.default_rng(seed)
    boot_aucs = []
    n_part = len(y_true)
    for _ in range(n_bootstrap):
        b_idx = rng.choice(n_part, size=n_part, replace=True)
        if len(np.unique(y_true[b_idx])) == 2:
            boot_aucs.append(roc_auc_score(y_true[b_idx], y_prob[b_idx]))
    auc_ci_low = np.percentile(boot_aucs, 2.5) if boot_aucs else auc
    auc_ci_high = np.percentile(boot_aucs, 97.5) if boot_aucs else auc
    
    res = {
        "pipeline": pipeline_desc,
        "classifier": clf_name,
        "n_participants": n_part,
        "n_segments": len(df_oof),
        "accuracy_pct": round(acc, 2),
        "sensitivity_pct": round(sens, 2),
        "specificity_pct": round(spec, 2),
        "f1_score_pct": round(f1, 2),
        "roc_auc": round(auc, 4),
        "auc_95ci_low": round(auc_ci_low, 4),
        "auc_95ci_high": round(auc_ci_high, 4),
        "auc_ci_str": f"[{auc_ci_low:.4f}, {auc_ci_high:.4f}]"
    }
    return res, df_oof

# -------------------------------------------------------------------------
# Main Execution Orchestrator
# -------------------------------------------------------------------------

def run_m2_matrix():
    print("================================================================================")
    print("       PHASE 1C — MODULE M2: EXPERIMENTAL CONFIGURATION MATRIX (35 PIPELINES)   ")
    print("================================================================================")
    t_start = time.time()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Hardware compute acceleration: {device}")
    
    # Pretrained backbones
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    resnet18.fc = torch.nn.Identity()
    resnet18.eval()
    
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    resnet50.fc = torch.nn.Identity()
    resnet50.eval()
    
    classifiers = ["Linear SVM", "RBF SVM", "MLP", "Logistic Regression", "Random Forest"]
    
    for cohort_name, task in [("EyesClosed", "eyesClosed"), ("EyesOpen", "eyesOpen")]:
        print(f"\n>>> PROCESSING COHORT: {cohort_name} (Task: {task}) <<<")
        meta_path = f"results/features_cache/metadata_{task}.csv"
        df_meta = pd.read_csv(meta_path)
        print(f"Loaded metadata: {len(df_meta)} segments across {df_meta['participant_id'].nunique()} participants.")
        
        # Load raw segments
        raw_segments_cache = os.path.join(CACHE_DIR, f"raw_segments_{task}.npy")
        if os.path.exists(raw_segments_cache):
            raw_segments = np.load(raw_segments_cache)
            print(f"Loaded cached raw segments: {raw_segments.shape}")
        else:
            print("Extracting raw segments from EDF files in exact metadata order...")
            raw_segments = []
            rec_cache = {}
            for _, row in df_meta.iterrows():
                pid = row["participant_id"]
                s_start = int(row["sample_start"])
                s_end = int(row["sample_end"])
                if pid not in rec_cache:
                    p_files = sorted(glob.glob(f"nm000114_v1.0.0/{pid}/eeg/*{task}*.edf"))
                    if not p_files:
                        raise FileNotFoundError(f"No EDF found for {pid} {task}")
                    data, fs = load_clean_recording(p_files[0])
                    rec_cache[pid] = data
                data = rec_cache[pid]
                seg = data[:, s_start:s_end]
                raw_segments.append(seg)
            raw_segments = np.array(raw_segments)
            np.save(raw_segments_cache, raw_segments)
            print(f"Saved raw segments: {raw_segments.shape}")
            
        # 1. Welch PSD features (D = 95)
        welch_cache = os.path.join(CACHE_DIR, f"features_welch_{task}.npy")
        if os.path.exists(welch_cache):
            X_welch = np.load(welch_cache)
        else:
            print("Computing Welch PSD band-power features (D=95)...")
            X_welch = extract_welch_features(raw_segments, fs=256)
            np.save(welch_cache, X_welch)
        print(f"Welch PSD features ready: {X_welch.shape}")
        
        # 2. SPWVD features
        X_spwvd_r18 = np.load(f"results/features_cache/features_resnet18_{task}.npy")
        X_spwvd_r50 = np.load(f"results/features_cache/features_resnet50_{task}.npy")
        print(f"SPWVD features ready: ResNet-18 {X_spwvd_r18.shape}, ResNet-50 {X_spwvd_r50.shape}")
        
        # 3. STFT features
        stft_r18_cache = os.path.join(CACHE_DIR, f"features_stft_resnet18_{task}.npy")
        stft_r50_cache = os.path.join(CACHE_DIR, f"features_stft_resnet50_{task}.npy")
        if os.path.exists(stft_r18_cache) and os.path.exists(stft_r50_cache):
            X_stft_r18 = np.load(stft_r18_cache)
            X_stft_r50 = np.load(stft_r50_cache)
        else:
            print("Computing STFT time-frequency maps...")
            stft_maps = compute_stft_maps(raw_segments, fs=256)
            print("Extracting STFT embeddings (ResNet-18 & ResNet-50)...")
            X_stft_r18 = tfr_to_cnn_embedding(stft_maps, resnet18, device)
            X_stft_r50 = tfr_to_cnn_embedding(stft_maps, resnet50, device)
            np.save(stft_r18_cache, X_stft_r18)
            np.save(stft_r50_cache, X_stft_r50)
        print(f"STFT features ready: ResNet-18 {X_stft_r18.shape}, ResNet-50 {X_stft_r50.shape}")
        
        # 4. CWT Morlet features
        cwt_r18_cache = os.path.join(CACHE_DIR, f"features_cwt_resnet18_{task}.npy")
        cwt_r50_cache = os.path.join(CACHE_DIR, f"features_cwt_resnet50_{task}.npy")
        if os.path.exists(cwt_r18_cache) and os.path.exists(cwt_r50_cache):
            X_cwt_r18 = np.load(cwt_r18_cache)
            X_cwt_r50 = np.load(cwt_r50_cache)
        else:
            print("Computing CWT Morlet time-frequency maps...")
            cwt_maps = compute_cwt_maps(raw_segments, fs=256)
            print("Extracting CWT embeddings (ResNet-18 & ResNet-50)...")
            X_cwt_r18 = tfr_to_cnn_embedding(cwt_maps, resnet18, device)
            X_cwt_r50 = tfr_to_cnn_embedding(cwt_maps, resnet50, device)
            np.save(cwt_r18_cache, X_cwt_r18)
            np.save(cwt_r50_cache, X_cwt_r50)
        print(f"CWT features ready: ResNet-18 {X_cwt_r18.shape}, ResNet-50 {X_cwt_r50.shape}")
        
        # Assemble feature dictionary
        feature_dict = {
            # 2D TFR + ResNet-18 (512-D)
            ("SPWVD", "ResNet-18"): X_spwvd_r18,
            ("STFT", "ResNet-18"): X_stft_r18,
            ("CWT_Morlet", "ResNet-18"): X_cwt_r18,
            
            # 2D TFR + ResNet-50 (2048-D)
            ("SPWVD", "ResNet-50"): X_spwvd_r50,
            ("STFT", "ResNet-50"): X_stft_r50,
            ("CWT_Morlet", "ResNet-50"): X_cwt_r50,
            
            # 1D Classical Welch PSD (95-D Direct Features, NO CNN)
            ("Welch_PSD", "Direct_Features"): X_welch
        }
        
        print(f"\n--- Running 35 Valid Pipeline Evaluations ({cohort_name}) ---")
        cohort_results = []
        cohort_oof_dfs = []
        
        pipe_idx = 1
        for (rep, backbone), feat_matrix in feature_dict.items():
            for clf_name in classifiers:
                pipe_desc = f"{rep} + {backbone}" if backbone != "Direct_Features" else f"{rep} (Direct Features)"
                t_p0 = time.time()
                res, df_oof = evaluate_pipeline(feat_matrix, df_meta, clf_name, pipe_desc, n_bootstrap=1000, seed=42)
                res["representation"] = rep
                res["backbone"] = backbone
                res["cohort"] = cohort_name
                cohort_results.append(res)
                cohort_oof_dfs.append(df_oof)
                print(f"  [{pipe_idx:2d}/35] {pipe_desc:<35} | {clf_name:<18} -> Acc: {res['accuracy_pct']:5.2f}% | AUC: {res['roc_auc']:.4f} {res['auc_ci_str']} ({time.time()-t_p0:.1f}s)")
                pipe_idx += 1
                
        df_cohort_res = pd.DataFrame(cohort_results)
        df_cohort_oof = pd.concat(cohort_oof_dfs, ignore_index=True)
        
        res_csv = os.path.join(OUTPUT_DIR, f"m2_all_35_configurations_{task}.csv")
        oof_csv = os.path.join(OUTPUT_DIR, f"oof_predictions_all_35_models_{task}.csv")
        df_cohort_res.to_csv(res_csv, index=False)
        df_cohort_oof.to_csv(oof_csv, index=False)
        print(f"Saved {cohort_name} results: {res_csv}")
        
    print(f"\n================================================================================")
    print(f"   MODULE M2 ALL 35 PIPELINES COMPLETED IN {time.time()-t_start:.2f} SECONDS    ")
    print(f"================================================================================")

if __name__ == "__main__":
    run_m2_matrix()
