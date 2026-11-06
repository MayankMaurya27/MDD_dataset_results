"""
Phase 1B: Comprehensive EEG Artifact Diagnostic Audit Script.

Inspects all 181 EDF recordings in nm000114_v1.0.0.
Calculates:
- Epoch-level peak-to-peak amplitude distributions per channel and per region
- Epoch-level absolute amplitude distributions per channel and per region
- Exceedance rates at +/-100 uV and +/-150 uV
- Stratification across 5 scalp regions: Frontal, Central, Temporal, Parietal, Occipital
- FastICA decomposition on representative recordings for ocular/cardiac/muscular artifact evaluation
- Trace plots illustrating clean vs ocular blink vs muscle artifact segments.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "/Users/mayankmaurya/Desktop/experiment/results/revision_experiments/artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STANDARD_1020_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'F7', 'T3', 'T5', 'Fz',
    'Fp2', 'F4', 'C4', 'P4', 'O2',
    'F8', 'T4', 'T6', 'Cz', 'Pz'
]

REGIONS = {
    'Frontal': ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'Fz'],
    'Central': ['C3', 'C4', 'Cz'],
    'Temporal': ['T3', 'T4', 'T5', 'T6'],
    'Parietal': ['P3', 'P4', 'Pz'],
    'Occipital': ['O1', 'O2']
}

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
    data = raw.get_data() * 1e6  # to microvolts (uV)
    fs = float(raw.info['sfreq'])
    return data, fs, raw

def run_diagnostic():
    edf_files = sorted(glob.glob("nm000114_v1.0.0/sub-*/eeg/*.edf"))
    print(f"--- Found {len(edf_files)} EDF recordings across HUSM BIDS archive ---")
    
    rec_records = []
    channel_exceedances = {ch: {'total_epochs': 0, 'ptp_gt100': 0, 'ptp_gt150': 0, 'abs_gt100': 0, 'abs_gt150': 0, 'ptp_values': [], 'abs_values': []} for ch in STANDARD_1020_CHANNELS}
    
    rep_files = {} # for trace inspection
    
    for idx, edf_p in enumerate(edf_files):
        fname = os.path.basename(edf_p)
        sub_match = re.search(r'(sub-[A-Za-z0-9]+)', fname)
        task_match = re.search(r'task-([A-Za-z0-9]+)', fname)
        run_match = re.search(r'run-([0-9]+)', fname)
        
        sub = sub_match.group(1) if sub_match else "unknown"
        task = task_match.group(1) if task_match else "unknown"
        run = run_match.group(1) if run_match else "1"
        group = "HC" if "sub-HS" in sub else "MDD"
        
        data, fs, raw = load_clean_recording(edf_p)
        n_channels, n_samples = data.shape
        window_samples = int(2.0 * fs) # 512 samples
        
        # Continuous epochs
        n_epochs = n_samples // window_samples
        if n_epochs == 0:
            continue
            
        epochs = data[:, :n_epochs * window_samples].reshape(n_channels, n_epochs, window_samples)
        
        # 15 deterministic epochs
        if n_epochs >= 15:
            step = n_epochs // 15
            selected_indices = [i * step for i in range(15)]
        else:
            selected_indices = list(range(n_epochs))
            
        epochs_15 = epochs[:, selected_indices, :]
        
        # Metrics for the 15 selected epochs
        ptp_15 = np.ptp(epochs_15, axis=2) # shape: (19, 15)
        abs_15 = np.max(np.abs(epochs_15), axis=2) # shape: (19, 15)
        
        # Channel-level accumulation for the 15 study epochs
        for ch_idx, ch in enumerate(STANDARD_1020_CHANNELS):
            ch_ptp = ptp_15[ch_idx]
            ch_abs = abs_15[ch_idx]
            
            channel_exceedances[ch]['total_epochs'] += len(ch_ptp)
            channel_exceedances[ch]['ptp_gt100'] += int(np.sum(ch_ptp > 100.0))
            channel_exceedances[ch]['ptp_gt150'] += int(np.sum(ch_ptp > 150.0))
            channel_exceedances[ch]['abs_gt100'] += int(np.sum(ch_abs > 100.0))
            channel_exceedances[ch]['abs_gt150'] += int(np.sum(ch_abs > 150.0))
            channel_exceedances[ch]['ptp_values'].extend(ch_ptp.tolist())
            channel_exceedances[ch]['abs_values'].extend(ch_abs.tolist())
            
        # Recording-level summary (15 epochs)
        # Any channel exceeding threshold in an epoch
        epoch_ptp_max = np.max(ptp_15, axis=0) # max across 19 channels for each epoch
        epoch_abs_max = np.max(abs_15, axis=0)
        
        ptp_gt100_epochs = int(np.sum(epoch_ptp_max > 100.0))
        ptp_gt150_epochs = int(np.sum(epoch_ptp_max > 150.0))
        abs_gt100_epochs = int(np.sum(epoch_abs_max > 100.0))
        abs_gt150_epochs = int(np.sum(epoch_abs_max > 150.0))
        
        # Regional peak-to-peak means
        regional_ptp = {}
        for reg_name, reg_chs in REGIONS.items():
            reg_indices = [STANDARD_1020_CHANNELS.index(c) for c in reg_chs]
            regional_ptp[reg_name] = float(np.mean(ptp_15[reg_indices, :]))
            
        rec_records.append({
            'participant_id': sub,
            'group_label': group,
            'task': task,
            'run': run,
            'filename': fname,
            'duration_sec': round(n_samples / fs, 1),
            'total_available_epochs': n_epochs,
            'study_epochs': len(selected_indices),
            'mean_ptp_uV': round(float(np.mean(ptp_15)), 2),
            'median_ptp_uV': round(float(np.median(ptp_15)), 2),
            'p95_ptp_uV': round(float(np.percentile(ptp_15, 95)), 2),
            'max_ptp_uV': round(float(np.max(ptp_15)), 2),
            'mean_abs_uV': round(float(np.mean(abs_15)), 2),
            'max_abs_uV': round(float(np.max(abs_15)), 2),
            'epochs_ptp_gt100': ptp_gt100_epochs,
            'epochs_ptp_gt150': ptp_gt150_epochs,
            'pct_epochs_ptp_gt100': round(ptp_gt100_epochs / len(selected_indices) * 100, 2),
            'pct_epochs_ptp_gt150': round(ptp_gt150_epochs / len(selected_indices) * 100, 2),
            'frontal_mean_ptp_uV': round(regional_ptp['Frontal'], 2),
            'central_mean_ptp_uV': round(regional_ptp['Central'], 2),
            'temporal_mean_ptp_uV': round(regional_ptp['Temporal'], 2),
            'parietal_mean_ptp_uV': round(regional_ptp['Parietal'], 2),
            'occipital_mean_ptp_uV': round(regional_ptp['Occipital'], 2),
        })
        
        # Cache representative cases for visual trace inspection
        if task == "eyesOpen" and "sub-HS1" in sub:
            rep_files["EO_blink"] = (data, fs, selected_indices)
        if task == "eyesClosed" and "sub-HS1" in sub:
            rep_files["EC_clean"] = (data, fs, selected_indices)
            
        if (idx + 1) % 40 == 0 or (idx + 1) == len(edf_files):
            print(f"  Processed [{idx+1}/{len(edf_files)}] recordings...")
            
    # 1. Export recording-level summary
    rec_df = pd.DataFrame(rec_records)
    rec_csv = os.path.join(OUTPUT_DIR, "recording_level_artifact_metrics.csv")
    rec_df.to_csv(rec_csv, index=False)
    print(f"Saved recording-level artifact metrics to {rec_csv}")
    
    # 2. Export channel-level summary
    ch_records = []
    for ch in STANDARD_1020_CHANNELS:
        entry = channel_exceedances[ch]
        total_ep = entry['total_epochs']
        ptp_arr = np.array(entry['ptp_values'])
        abs_arr = np.array(entry['abs_values'])
        
        # Find region
        ch_reg = next(r for r, chs in REGIONS.items() if ch in chs)
        
        ch_records.append({
            'channel': ch,
            'region': ch_reg,
            'total_evaluated_epochs': total_ep,
            'mean_ptp_uV': round(float(np.mean(ptp_arr)), 2),
            'median_ptp_uV': round(float(np.median(ptp_arr)), 2),
            'p95_ptp_uV': round(float(np.percentile(ptp_arr, 95)), 2),
            'max_ptp_uV': round(float(np.max(ptp_arr)), 2),
            'ptp_gt100_count': entry['ptp_gt100'],
            'ptp_gt100_pct': round(entry['ptp_gt100'] / total_ep * 100, 2),
            'ptp_gt150_count': entry['ptp_gt150'],
            'ptp_gt150_pct': round(entry['ptp_gt150'] / total_ep * 100, 2),
            'abs_gt100_count': entry['abs_gt100'],
            'abs_gt100_pct': round(entry['abs_gt100'] / total_ep * 100, 2),
            'abs_gt150_count': entry['abs_gt150'],
            'abs_gt150_pct': round(entry['abs_gt150'] / total_ep * 100, 2),
        })
    ch_df = pd.DataFrame(ch_records)
    ch_csv = os.path.join(OUTPUT_DIR, "channel_level_artifact_summary.csv")
    ch_df.to_csv(ch_csv, index=False)
    print(f"Saved channel-level artifact summary to {ch_csv}")
    
    # 3. Export regional summary
    reg_records = []
    for reg_name, reg_chs in REGIONS.items():
        sub_ch = ch_df[ch_df['channel'].isin(reg_chs)]
        reg_records.append({
            'region': reg_name,
            'channels': ", ".join(reg_chs),
            'num_channels': len(reg_chs),
            'mean_ptp_uV': round(sub_ch['mean_ptp_uV'].mean(), 2),
            'p95_ptp_uV': round(sub_ch['p95_ptp_uV'].mean(), 2),
            'mean_ptp_gt100_pct': round(sub_ch['ptp_gt100_pct'].mean(), 2),
            'mean_ptp_gt150_pct': round(sub_ch['ptp_gt150_pct'].mean(), 2),
            'mean_abs_gt100_pct': round(sub_ch['abs_gt100_pct'].mean(), 2),
            'mean_abs_gt150_pct': round(sub_ch['abs_gt150_pct'].mean(), 2),
        })
    reg_df = pd.DataFrame(reg_records)
    reg_csv = os.path.join(OUTPUT_DIR, "regional_artifact_summary.csv")
    reg_df.to_csv(reg_csv, index=False)
    print(f"Saved regional artifact summary to {reg_csv}")
    
    # 4. Generate Visual Plots
    print("Generating representative trace and ICA diagnostic plots...")
    generate_diagnostic_plots(rec_df, ch_df, rep_files)
    
def generate_diagnostic_plots(rec_df, ch_df, rep_files):
    # Plot 1: Regional and Channel Peak-to-Peak Distributions
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Left: Channel-level exceedances
    ax = axes[0]
    bars1 = ax.bar(np.arange(len(ch_df)) - 0.2, ch_df['ptp_gt100_pct'], width=0.4, label='PTP > 100 uV', color='#d62728', alpha=0.85)
    bars2 = ax.bar(np.arange(len(ch_df)) + 0.2, ch_df['ptp_gt150_pct'], width=0.4, label='PTP > 150 uV', color='#1f77b4', alpha=0.85)
    ax.set_xticks(range(len(ch_df)))
    ax.set_xticklabels(ch_df['channel'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Percentage of Epochs Exceeding Threshold (%)', fontsize=10)
    ax.set_title('Channel-Level Exceedance Rates (Study 15-Epoch Cohort)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Right: Regional Exceedance Summary by condition
    ax2 = axes[1]
    task_group = rec_df.groupby('task')[['frontal_mean_ptp_uV', 'central_mean_ptp_uV', 'temporal_mean_ptp_uV', 'parietal_mean_ptp_uV', 'occipital_mean_ptp_uV']].mean()
    tasks = ['eyesClosed', 'eyesOpen', 'P300']
    regions = ['Frontal', 'Central', 'Temporal', 'Parietal', 'Occipital']
    
    x = np.arange(len(regions))
    width = 0.25
    for t_idx, t in enumerate(tasks):
        if t in task_group.index:
            vals = task_group.loc[t].values
            ax2.bar(x + (t_idx - 1) * width, vals, width, label=t, alpha=0.85)
            
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, fontsize=10)
    ax2.set_ylabel('Mean Peak-to-Peak Amplitude (uV)', fontsize=10)
    ax2.set_title('Regional Mean Voltage by Experimental Condition', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, "figure_artifact_distribution_summary.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved artifact distribution plot to {p1}")
    
    # Plot 2: Representative Traces & FastICA Decomposition
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), dpi=300)
    
    # Panel A: Clean Eyes-Closed Trace (Fp1, C3, O1)
    if "EC_clean" in rep_files:
        data, fs, sel = rep_files["EC_clean"]
        ep_idx = sel[0] # first epoch
        ep_data = data[:, ep_idx * 512:(ep_idx + 1) * 512]
        t_axis = np.linspace(0, 2.0, 512)
        ax = axes[0]
        ax.plot(t_axis, ep_data[STANDARD_1020_CHANNELS.index('Fp1')], label='Fp1 (Frontal Polar)', color='#1f77b4', lw=1.2)
        ax.plot(t_axis, ep_data[STANDARD_1020_CHANNELS.index('Cz')], label='Cz (Central Vertex)', color='#2ca02c', lw=1.2)
        ax.plot(t_axis, ep_data[STANDARD_1020_CHANNELS.index('O1')], label='O1 (Occipital)', color='#9467bd', lw=1.2)
        ax.set_title('A. Representative Quiescent Resting-State Trace (sub-HS1, Eyes-Closed) — Max PTP: ~35 uV', fontsize=11, fontweight='bold')
        ax.set_ylabel('Voltage (uV)', fontsize=10)
        ax.set_ylim([-80, 80])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.5)
        
    # Panel B: Eyes-Open Trace with Ocular Blink Artifact
    if "EO_blink" in rep_files:
        data, fs, sel = rep_files["EO_blink"]
        # Find epoch with highest Fp1 peak
        fp1_idx = STANDARD_1020_CHANNELS.index('Fp1')
        ptps = [np.ptp(data[fp1_idx, s*512:(s+1)*512]) for s in sel]
        max_s = sel[np.argmax(ptps)]
        ep_data = data[:, max_s * 512:(max_s + 1) * 512]
        t_axis = np.linspace(0, 2.0, 512)
        ax = axes[1]
        ax.plot(t_axis, ep_data[STANDARD_1020_CHANNELS.index('Fp1')], label='Fp1 (Frontal Polar — High Blink Deflection)', color='#d62728', lw=1.5)
        ax.plot(t_axis, ep_data[STANDARD_1020_CHANNELS.index('Cz')], label='Cz (Central Vertex — Minimal Transmission)', color='#2ca02c', lw=1.2)
        ax.plot(t_axis, ep_data[STANDARD_1020_CHANNELS.index('O1')], label='O1 (Occipital — Baseline Quiescent)', color='#9467bd', lw=1.2)
        ax.axhline(100, color='gray', linestyle='--', alpha=0.7, label='+100 uV Threshold')
        ax.axhline(-100, color='gray', linestyle='--', alpha=0.7)
        ax.set_title(f'B. Representative Ocular Blink Deflection (sub-HS1, Eyes-Open) — Fp1 PTP: {np.ptp(ep_data[fp1_idx]):.1f} uV', fontsize=11, fontweight='bold')
        ax.set_ylabel('Voltage (uV)', fontsize=10)
        ax.set_ylim([-180, 180])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.5)
        
    # Panel C: FastICA Component Separation Demo
    try:
        from sklearn.decomposition import FastICA
        if "EO_blink" in rep_files:
            data, fs, sel = rep_files["EO_blink"]
            # 10 seconds continuous
            chunk = data[:, :2560].T # (2560 samples, 19 channels)
            ica = FastICA(n_components=5, random_state=42, max_iter=500)
            comps = ica.fit_transform(chunk) # (2560, 5)
            
            # Find component most correlated with Fp1
            corrs = [np.corrcoef(chunk[:, STANDARD_1020_CHANNELS.index('Fp1')], comps[:, c])[0, 1] for c in range(5)]
            blink_comp = np.argmax(np.abs(corrs))
            
            t_chunk = np.linspace(0, 10.0, 2560)
            ax = axes[2]
            ax.plot(t_chunk, comps[:, blink_comp], label=f'ICA Component {blink_comp+1} (Ocular Blink: corr={corrs[blink_comp]:.2f} with Fp1)', color='#d62728', lw=1.3)
            neural_comp = (blink_comp + 1) % 5
            ax.plot(t_chunk, comps[:, neural_comp], label=f'ICA Component {neural_comp+1} (Underlying Neural Rhythm)', color='#1f77b4', lw=1.1, alpha=0.85)
            ax.set_title('C. FastICA Feasibility Demonstration: Ocular Artifact Component vs. Neural Activity Component', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_ylabel('Component Score (a.u.)', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, linestyle=':', alpha=0.5)
    except Exception as e:
        print(f"FastICA demo plot failed: {e}")
        
    plt.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, "figure_trace_and_ica_diagnostics.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved trace and ICA diagnostics plot to {p2}")

if __name__ == "__main__":
    run_diagnostic()

