"""
Phase 1B: Classical Baseline Evaluation Harness.

Implements and evaluates under strict Stratified Group 10-Fold CV:
1. Baseline 1A: Welch PSD (95-D) + Linear SVM
2. Baseline 1B: Welch PSD (95-D) + RBF SVM
3. Baseline 2:  STFT Spectrogram + Frozen ResNet-18 (512-D) + Linear SVM
4. Baseline 3:  CWT Morlet Scalogram + Frozen ResNet-18 (512-D) + Linear SVM

Evaluates on both Eyes-Closed (N=58) and Eyes-Open (N=61) cohorts.
Enforces zero data leakage (StandardScaler fitted strictly on train folds).
Outputs saved to results/revision_experiments/baselines/.
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.models as models
from scipy import signal
from scipy.stats import norm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
import mne

OUTPUT_DIR = "/Users/mayankmaurya/Desktop/experiment/results/revision_experiments/baselines"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def load_eeg_raw_filtered(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    name_mapping = {}
    for ch in raw.ch_names:
        clean = ch.replace('EEG ', '').replace('-LE', '').replace('-A1', '').replace('-A2', '').strip()
        if clean in STANDARD_1020_CHANNELS:
            name_mapping[ch] = clean
    raw.rename_channels(name_mapping)
    raw.pick(STANDARD_1020_CHANNELS)
    raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
    data = raw.get_data() * 1e6 # uV
    fs = float(raw.info['sfreq'])
    return data, fs

def extract_welch_features(segments, fs=256):
    """
    Computes log Welch PSD band-power features.
    Input: segments (n_segments, 19, 512)
    Output: features (n_segments, 95)
    """
    n_seg = len(segments)
    f, psd = signal.welch(segments, fs=fs, nperseg=256, noverlap=128, axis=-1) # (n_seg, 19, 129)
    
    feats = []
    for b_name, (low, high) in BANDS.items():
        idx_band = np.where((f >= low) & (f <= high))[0]
        # integrate power over band using trapezoid rule
        band_pwr = np.trapezoid(psd[:, :, idx_band], f[idx_band], axis=-1) # (n_seg, 19)
        feats.append(band_pwr)
        
    # Stack along feature axis: 5 bands x 19 channels = 95 features
    feat_mat = np.concatenate(feats, axis=-1) # (n_seg, 95)
    # Log10 transform with floor
    log_feats = np.log10(np.maximum(feat_mat, 1e-12))
    return log_feats

def tfr_to_resnet_embedding(tfr_maps, resnet_model, device):
    """
    Resizes 2D TFR maps (n_seg, H, W) to (3, 224, 224), normalizes with ImageNet stats,
    and extracts 512-D features from frozen ResNet-18 avgpool.
    """
    n_seg = len(tfr_maps)
    embeddings = []
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    batch_size = 64
    with torch.no_grad():
        for b in range(0, n_seg, batch_size):
            b_maps = tfr_maps[b:b+batch_size]
            # Normalize each map: clip at 99.5th percentile
            b_tensors = []
            for m in b_maps:
                m_clip = np.maximum(m, 0)
                p99 = np.percentile(m_clip, 99.5)
                m_norm = np.clip(m_clip / p99, 0.0, 1.0) if p99 > 1e-8 else np.zeros_like(m_clip)
                b_tensors.append(torch.from_numpy(m_norm).float())
                
            # Stack into (B, 1, H, W)
            t = torch.stack(b_tensors).unsqueeze(1).to(device)
            # Resize via bilinear interpolation to (B, 1, 224, 224)
            t_resized = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
            # Repeat to 3 channels (B, 3, 224, 224)
            t_rgb = t_resized.repeat(1, 3, 1, 1)
            # Standardize
            t_std = (t_rgb - mean) / std
            
            # Forward pass through ResNet-18 avgpool
            feats = resnet_model(t_std).squeeze(-1).squeeze(-1) # (B, 512)
            embeddings.append(feats.cpu().numpy())
            
    return np.concatenate(embeddings, axis=0) # (n_seg, 512)

def extract_stft_features(segments, resnet_model, device, fs=256):
    """
    Computes spatially averaged STFT spectrograms and extracts 512-D ResNet-18 features.
    """
    stft_maps = []
    for s in segments:
        # s: (19, 512)
        f, t_s, z = signal.stft(s, fs=fs, nperseg=64, noverlap=32, axis=-1)
        # z: (19, 33, 17) -> magnitude
        mag = np.abs(z)
        # Spatial average across 19 channels
        mean_mag = np.mean(mag, axis=0) # (33, 17)
        stft_maps.append(mean_mag)
        
    return tfr_to_resnet_embedding(stft_maps, resnet_model, device)

def extract_cwt_features(segments, resnet_model, device, fs=256):
    """
    Computes spatially averaged Morlet CWT scalograms and extracts 512-D ResNet-18 features.
    """
    freqs = np.linspace(1.0, 45.0, 32)
    # MNE expects (n_epochs, n_channels, n_times)
    power = mne.time_frequency.tfr_array_morlet(
        segments, sfreq=fs, freqs=freqs, n_cycles=freqs / 2.0, output='power', verbose=False
    ) # (n_seg, 19, 32, 512)
    # Spatial average across 19 channels
    mean_power = np.mean(power, axis=1) # (n_seg, 32, 512)
    return tfr_to_resnet_embedding(mean_power, resnet_model, device)

def evaluate_baseline_group_cv(X, meta_df, clf_name="Linear SVM", svm_kernel="linear", baseline_tag="Welch PSD"):
    """
    Runs Stratified Group 10-Fold CV with zero test leakage.
    Scales training features only, predicts out-of-fold, pools per participant.
    """
    folds = sorted(meta_df['fold'].unique())
    n_seg = len(meta_df)
    
    oof_y_true = np.zeros(n_seg, dtype=int)
    oof_y_pred = np.zeros(n_seg, dtype=int)
    oof_y_prob = np.zeros(n_seg, dtype=float)
    
    for f in folds:
        train_idx = np.where(meta_df['fold'] != f)[0]
        test_idx = np.where(meta_df['fold'] == f)[0]
        
        X_train, y_train = X[train_idx], meta_df['numeric_label'].iloc[train_idx].values
        X_test, y_test = X[test_idx], meta_df['numeric_label'].iloc[test_idx].values
        
        # Zero leakage: fit scaler on train only
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        if svm_kernel == "linear":
            clf = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
        elif svm_kernel == "rbf":
            clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown kernel: {svm_kernel}")
            
        clf.fit(X_train_s, y_train)
        
        probs = clf.predict_proba(X_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        oof_y_true[test_idx] = y_test
        oof_y_pred[test_idx] = preds
        oof_y_prob[test_idx] = probs
        
    # Segment-level metrics
    s_acc, s_sens, s_spec, s_f1, _, _, _, _ = compute_metrics(oof_y_true, oof_y_pred)
    s_auc = roc_auc_score(oof_y_true, oof_y_prob)
    
    # Build segment DF
    seg_df = meta_df.copy()
    seg_df['y_true'] = oof_y_true
    seg_df['y_pred'] = oof_y_pred
    seg_df['y_prob'] = oof_y_prob
    seg_df['baseline_name'] = baseline_tag
    seg_df['classifier'] = clf_name
    
    # Participant-level aggregation
    part_agg = seg_df.groupby("participant_id").agg({
        "y_true": "first",
        "y_prob": "mean",
        "fold": "first",
        "group_label": "first"
    }).reset_index()
    part_agg['y_pred'] = (part_agg['y_prob'] >= 0.5).astype(int)
    part_agg['correct'] = (part_agg['y_true'] == part_agg['y_pred']).astype(int)
    part_agg['baseline_name'] = baseline_tag
    part_agg['classifier'] = clf_name
    
    n_parts = len(part_agg)
    n_correct = part_agg['correct'].sum()
    p_acc, p_sens, p_spec, p_f1, tn, fp, fn, tp = compute_metrics(part_agg['y_true'], part_agg['y_pred'])
    w_low, w_high = wilson_ci(n_correct, n_parts)
    
    # Fold-level participant metrics
    fold_accs = []
    for f in folds:
        f_df = part_agg[part_agg['fold'] == f]
        if len(f_df) == 0:
            continue
        f_acc, _, _, _, _, _, _, _ = compute_metrics(f_df['y_true'], f_df['y_pred'])
        fold_accs.append(f_acc)
    fold_accs = np.array(fold_accs)
    fold_mean = np.mean(fold_accs)
    fold_sd_sample = np.std(fold_accs, ddof=1)
    fold_sd_pop = np.std(fold_accs, ddof=0)
    
    # Continuous Participant AUC
    part_auc = roc_auc_score(part_agg['y_true'], part_agg['y_prob'])
    
    # 1,000 bootstrap iterations for AUC (seed=42)
    rng = np.random.RandomState(42)
    boot_aucs = []
    indices = np.arange(len(part_agg))
    for _ in range(1000):
        idx = rng.choice(indices, size=len(indices), replace=True)
        y_b = part_agg['y_true'].iloc[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_b, part_agg['y_prob'].iloc[idx]))
    boot_ci_low = np.percentile(boot_aucs, 2.5)
    boot_ci_high = np.percentile(boot_aucs, 97.5)
    boot_se = np.std(boot_aucs, ddof=1)
    
    summary_dict = {
        'baseline_name': baseline_tag,
        'classifier': clf_name,
        'n_participants': n_parts,
        'n_segments': n_seg,
        'correct_participants': n_correct,
        'pooled_accuracy_pct': round(p_acc * 100, 2),
        'wilson_95ci': f"[{w_low*100:.2f}%, {w_high*100:.2f}%]",
        'sensitivity_pct': round(p_sens * 100, 2),
        'specificity_pct': round(p_spec * 100, 2),
        'f1_score_pct': round(p_f1 * 100, 2),
        'fold_mean_acc_pct': round(fold_mean * 100, 2),
        'fold_sd_sample_ddof1': round(fold_sd_sample * 100, 2),
        'fold_sd_pop_ddof0': round(fold_sd_pop * 100, 2),
        'participant_auc': round(part_auc, 4),
        'bootstrap_auc_95ci': f"[{boot_ci_low:.4f}, {boot_ci_high:.4f}]",
        'bootstrap_se': round(boot_se, 4),
        'segment_accuracy_pct': round(s_acc * 100, 2),
        'segment_auc': round(s_auc, 4)
    }
    
    return seg_df, part_agg, summary_dict

def run_classical_baselines():
    t_start = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== Initializing Classical Baseline Evaluation Harness (Device: {device}) ===")
    
    # Pretrained frozen ResNet-18
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove final classification head to tap avgpool 512-D
    resnet.fc = torch.nn.Identity()
    resnet.eval().to(device)
    for p in resnet.parameters():
        p.requires_grad = False
        
    bids_dir = "/Users/mayankmaurya/Desktop/experiment/nm000114_v1.0.0"
    
    cohorts = [
        ("eyesClosed", "results/features_cache/metadata_eyesClosed.csv", "Eyes-Closed"),
        ("eyesOpen", "results/features_cache/metadata_eyesOpen.csv", "Eyes-Open")
    ]
    
    all_summary_rows = []
    
    for task_name, meta_path, cohort_label in cohorts:
        print(f"\n=================================================================")
        print(f"  Processing {cohort_label} Baseline Features ({meta_path})")
        print(f"=================================================================")
        
        meta_df = pd.read_csv(meta_path)
        print(f"Cohort metadata loaded: {len(meta_df)} segments across {meta_df['participant_id'].nunique()} participants.")
        
        # Load raw segments
        # Group by participant and recording to read each EDF only once
        segments_list = []
        inv_df = pd.read_csv("reports/dataset_inventory.csv")
        inv_task = inv_df[inv_df['task'] == task_name].set_index('participant_id')
        
        print(f"Loading raw EEG segments from EDF files...")
        t_load0 = time.time()
        
        # Map participant_id -> edf_path
        # For eyesOpen, sub-HS15 has 2 runs
        participant_edfs = {}
        for sub_id, grp in meta_df.groupby('participant_id'):
            if sub_id == "sub-HS15" and task_name == "eyesOpen":
                # Handle run-1 and run-2
                p_run1 = os.path.join(bids_dir, f"sub-HS15/eeg/sub-HS15_task-eyesOpen_run-1_eeg.edf")
                p_run2 = os.path.join(bids_dir, f"sub-HS15/eeg/sub-HS15_task-eyesOpen_run-2_eeg.edf")
                participant_edfs[sub_id] = [p_run1, p_run2]
            else:
                row = inv_task.loc[sub_id]
                rel_f = row['filename'] if isinstance(row, pd.Series) else row.iloc[0]['filename']
                participant_edfs[sub_id] = [os.path.join(bids_dir, rel_f)]
                
        # Cache raw data in memory
        loaded_raw = {}
        for sub_id, p_list in participant_edfs.items():
            loaded_raw[sub_id] = []
            for p in p_list:
                data, fs = load_eeg_raw_filtered(p)
                loaded_raw[sub_id].append((data, fs))
                
        # Now extract segments in exact metadata order
        raw_segments = []
        for idx, row in meta_df.iterrows():
            sub_id = row['participant_id']
            s_start = int(row['sample_start'])
            s_end = int(row['sample_end'])
            
            # Select proper run
            if sub_id == "sub-HS15" and task_name == "eyesOpen":
                # Row 0-14 is run-1, row 15-29 is run-2
                sub_rows = meta_df[meta_df['participant_id'] == "sub-HS15"]
                sub_local_idx = list(sub_rows.index).index(idx)
                run_idx = 0 if sub_local_idx < 15 else 1
                data, fs = loaded_raw[sub_id][run_idx]
            else:
                data, fs = loaded_raw[sub_id][0]
                
            seg = data[:, s_start:s_end]
            if seg.shape[1] < 512:
                # pad if boundary
                seg = np.pad(seg, ((0, 0), (0, 512 - seg.shape[1])), mode='constant')
            raw_segments.append(seg)
            
        raw_segments = np.array(raw_segments) # (n_segments, 19, 512)
        print(f"Extracted {len(raw_segments)} raw segments in {time.time()-t_load0:.2f}s. Shape: {raw_segments.shape}")
        
        # -------------------------------------------------------------
        # Feature Extraction: Baselines
        # -------------------------------------------------------------
        print(f"\nComputing Baseline 1: Welch PSD Features (95-D)...")
        t0 = time.time()
        X_welch = extract_welch_features(raw_segments, fs=256)
        print(f"  Welch PSD extracted in {time.time()-t0:.2f}s. Shape: {X_welch.shape}")
        
        print(f"Computing Baseline 2: STFT Spectrogram Features + ResNet-18 (512-D)...")
        t0 = time.time()
        X_stft = extract_stft_features(raw_segments, resnet, device, fs=256)
        print(f"  STFT + ResNet-18 extracted in {time.time()-t0:.2f}s. Shape: {X_stft.shape}")
        
        print(f"Computing Baseline 3: CWT Morlet Scalogram Features + ResNet-18 (512-D)...")
        t0 = time.time()
        X_cwt = extract_cwt_features(raw_segments, resnet, device, fs=256)
        print(f"  CWT + ResNet-18 extracted in {time.time()-t0:.2f}s. Shape: {X_cwt.shape}")
        
        # -------------------------------------------------------------
        # 10-Fold CV Evaluation
        # -------------------------------------------------------------
        baselines = [
            ("Welch PSD + Linear SVM", X_welch, "Linear SVM", "linear"),
            ("Welch PSD + RBF SVM", X_welch, "Medium Gaussian SVM", "rbf"),
            ("STFT + ResNet-18 (Linear SVM)", X_stft, "Linear SVM", "linear"),
            ("CWT Morlet + ResNet-18 (Linear SVM)", X_cwt, "Linear SVM", "linear"),
        ]
        
        cohort_seg_preds = []
        cohort_part_preds = []
        
        for b_name, X_feat, clf_title, kern in baselines:
            print(f"  Evaluating {b_name} under Stratified Group 10-Fold CV...")
            seg_pred, part_pred, summ = evaluate_baseline_group_cv(
                X_feat, meta_df, clf_name=clf_title, svm_kernel=kern, baseline_tag=b_name
            )
            summ['condition'] = task_name
            summ['cohort_name'] = cohort_label
            all_summary_rows.append(summ)
            cohort_seg_preds.append(seg_pred)
            cohort_part_preds.append(part_pred)
            
            print(f"    -> Acc: {summ['pooled_accuracy_pct']}% | Wilson: {summ['wilson_95ci']} | "
                  f"Fold Mean: {summ['fold_mean_acc_pct']} +/- {summ['fold_sd_sample_ddof1']}% | "
                  f"AUC: {summ['participant_auc']} (95% CI: {summ['bootstrap_auc_95ci']})")
                  
        # Save cohort predictions
        seg_all = pd.concat(cohort_seg_preds, ignore_index=True)
        seg_all.to_csv(os.path.join(OUTPUT_DIR, f"oof_predictions_baselines_{task_name}.csv"), index=False)
        
        part_all = pd.concat(cohort_part_preds, ignore_index=True)
        part_all.to_csv(os.path.join(OUTPUT_DIR, f"participant_predictions_baselines_{task_name}.csv"), index=False)
        
    # Summary table
    sum_df = pd.DataFrame(all_summary_rows)
    sum_csv = os.path.join(OUTPUT_DIR, "baseline_metrics_summary.csv")
    sum_df.to_csv(sum_csv, index=False)
    print(f"\nSaved complete baseline summary to {sum_csv}")
    
    # Save reproducibility configuration YAML
    config_dict = {
        'hardware': {
            'device': str(device),
            'platform': 'Apple Silicon (macOS)',
            'torch_version': torch.__version__,
            'mne_version': mne.__version__,
            'scikit_learn_version': '1.6.1'
        },
        'parameters': {
            'sampling_rate': 256.0,
            'epoch_duration_sec': 2.0,
            'epoch_samples': 512,
            'n_study_epochs_per_recording': 15,
            'bandpass_filter_hz': [0.5, 45.0],
            'standard_channels': STANDARD_1020_CHANNELS,
            'bands': BANDS,
            'cwt_freqs_hz': [1.0, 45.0],
            'cwt_n_scales': 32,
            'stft_window': 'hann',
            'stft_nperseg': 64,
            'stft_noverlap': 32,
            'cv_folds': 10,
            'cv_type': 'StratifiedGroupKFold',
            'random_seed': 42,
            'bootstrap_iterations': 1000
        },
        'execution_time_seconds': round(time.time() - t_start, 2)
    }
    config_path = os.path.join(OUTPUT_DIR, "baseline_execution_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print(f"Saved configuration YAML to {config_path}")
    print(f"Baseline harness completed in {time.time() - t_start:.2f} seconds.")

if __name__ == "__main__":
    run_classical_baselines()

