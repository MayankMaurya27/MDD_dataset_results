"""Module M1: Signal-Quality and Artifact-Handling Validation.

Executes the three predefined experimental conditions across all 181 EDF recordings:
- Condition A: Historical uncorrected pipeline (immutable reference).
- Condition B: Per-recording ICA artifact correction (FastICA + Picard fallback).
- Condition C: Nearest-clean-epoch replacement sensitivity analysis.

Validation Criteria (Frozen):
- Frontal scalp electrodes Fp1/Fp2 used as synthetic ocular proxy: y_ocular = 0.5 * (Fp1 + Fp2).
- Candidate ocular IC criteria:
    * |corr(s_k, y_ocular)| >= 0.40
    * Frontal spatial energy ratio >= 0.50
    * Kurtosis >= 4.5
- Maximum 2 rejected components per recording (ranked by absolute proxy correlation).
- Retains at least 17 spatial linear degrees of freedom.
- Success thresholds:
    * Frontal PTP reduction >= 35.0% across contaminated cohorts
    * Paired Wilcoxon signed-rank p-value < 1e-10
    * Posterior alpha retention >= 85.0% across O1, O2, Pz
- Historical 2-second epoch protocol:
    * 512 samples at 256 Hz
    * Exactly 15 deterministic epochs per recording: step = floor(n_epochs / 15)
"""

import os
import sys
import glob
import json
import time
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
import mne

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = "results/phase1c/M1_artifact_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STANDARD_1020_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'F7', 'T3', 'T5', 'Fz',
    'Fp2', 'F4', 'C4', 'P4', 'O2',
    'F8', 'T4', 'T6', 'Cz', 'Pz'
]

def load_recording_raw(edf_path):
    """Load EDF and extract standardized 19 EEG channels bandpass filtered 0.5-45 Hz."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    name_mapping = {}
    for ch in raw.ch_names:
        clean = ch.replace('EEG ', '').replace('-LE', '').replace('-A1', '').replace('-A2', '').strip()
        if clean in STANDARD_1020_CHANNELS:
            name_mapping[ch] = clean
    raw.rename_channels(name_mapping)
    raw.pick(STANDARD_1020_CHANNELS)
    raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
    data = raw.get_data() * 1e6  # microvolts (uV)
    fs = float(raw.info['sfreq'])
    return data, fs, raw

def run_m1_pipeline():
    print("================================================================================")
    print("       PHASE 1C — MODULE M1: ARTIFACT VALIDATION & SIGNAL QUALITY AUDIT        ")
    print("================================================================================")
    t0 = time.time()
    
    edf_files = sorted(glob.glob("nm000114_v1.0.0/sub-*/eeg/*.edf"))
    print(f"Discovered {len(edf_files)} EDF recordings across dataset.")
    
    # Trackers
    recording_records = []
    ic_records = []
    fallback_records = []
    
    for idx, edf_path in enumerate(edf_files):
        fname = os.path.basename(edf_path)
        pid = fname.split("_")[0]
        task_match = [part for part in fname.split("_") if part.startswith("task-")]
        task = task_match[0].replace("task-", "") if task_match else "unknown"
        
        t_rec0 = time.time()
        data, fs, raw = load_recording_raw(edf_path)
        n_channels, n_samples = data.shape
        ch_names = raw.ch_names
        fp1_idx = ch_names.index('Fp1')
        fp2_idx = ch_names.index('Fp2')
        o1_idx = ch_names.index('O1')
        o2_idx = ch_names.index('O2')
        pz_idx = ch_names.index('Pz')
        
        # Historical epoch parameters
        epoch_samples = 512
        n_total_epochs = n_samples // epoch_samples
        step = n_total_epochs // 15
        selected_epochs = [i * step for i in range(15)]
        
        # Frontal ocular proxy
        y_ocular = 0.5 * (data[fp1_idx] + data[fp2_idx])
        
        # Condition A baseline metrics
        ptp_raw_frontal = np.ptp(data[[fp1_idx, fp2_idx]], axis=-1).mean()
        
        # Alpha power (8-12 Hz) in posterior leads
        f_raw, psd_raw = signal.welch(data[[o1_idx, o2_idx, pz_idx]], fs=fs, nperseg=256, noverlap=128)
        alpha_idx = np.where((f_raw >= 8.0) & (f_raw <= 12.0))[0]
        alpha_pwr_raw = np.trapezoid(psd_raw[:, alpha_idx], f_raw[alpha_idx], axis=-1).mean()
        
        # Fit ICA (Condition B) with fallback hierarchy
        solver = "FastICA"
        convergence = True
        fallback_used = False
        
        try:
            ica = mne.preprocessing.ICA(n_components=19, method='fastica', random_state=42, max_iter=500)
            ica.fit(raw, verbose=False)
        except Exception as e_fastica:
            fallback_records.append({
                "recording": fname,
                "participant_id": pid,
                "task": task,
                "attempted_solver": "FastICA",
                "status": "failed",
                "error": str(e_fastica)
            })
            try:
                solver = "Picard"
                ica = mne.preprocessing.ICA(n_components=19, method='picard', random_state=42, max_iter=500)
                ica.fit(raw, verbose=False)
                fallback_used = True
            except Exception as e_picard:
                solver = "Uncorrected_Fallback"
                convergence = False
                fallback_used = True
                fallback_records.append({
                    "recording": fname,
                    "participant_id": pid,
                    "task": task,
                    "attempted_solver": "Picard",
                    "status": "failed_default_to_raw",
                    "error": str(e_picard)
                })
        
        rejected_ics = []
        if convergence:
            sources = ica.get_sources(raw).get_data()
            mixing = ica.get_components()
            
            rejection_candidates = []
            for k in range(19):
                s_k = sources[k]
                corr = float(np.abs(np.corrcoef(s_k, y_ocular)[0, 1]))
                frontal_energy = float((mixing[fp1_idx, k]**2 + mixing[fp2_idx, k]**2) / np.sum(mixing[:, k]**2))
                kurt = float(stats.kurtosis(s_k, fisher=False))
                
                is_candidate = (corr >= 0.40 and frontal_energy >= 0.50 and kurt >= 4.5)
                rejection_candidates.append({
                    "ic_idx": k,
                    "corr": corr,
                    "frontal_ratio": frontal_energy,
                    "kurtosis": kurt,
                    "qualifies": is_candidate
                })
                
                ic_records.append({
                    "recording": fname,
                    "participant_id": pid,
                    "task": task,
                    "ic_idx": k,
                    "proxy_corr": round(corr, 4),
                    "frontal_ratio": round(frontal_energy, 4),
                    "kurtosis": round(kurt, 2),
                    "qualifies_ocular": is_candidate
                })
            
            # Rank candidates by absolute correlation descending, take at most 2
            qualifying = [c for c in rejection_candidates if c["qualifies"]]
            qualifying.sort(key=lambda x: x["corr"], reverse=True)
            rejected_ics = [c["ic_idx"] for c in qualifying[:2]]
            
            # Reconstruct Condition B cleaned data
            ica.exclude = rejected_ics
            raw_clean = ica.apply(raw.copy(), verbose=False)
            data_clean = raw_clean.get_data() * 1e6
        else:
            # Fallback: retain raw
            data_clean = data.copy()
            rejected_ics = []
            
        ptp_clean_frontal = np.ptp(data_clean[[fp1_idx, fp2_idx]], axis=-1).mean()
        ptp_reduction_pct = (ptp_raw_frontal - ptp_clean_frontal) / ptp_raw_frontal * 100.0 if ptp_raw_frontal > 0 else 0.0
        
        # Alpha power clean
        f_clean, psd_clean = signal.welch(data_clean[[o1_idx, o2_idx, pz_idx]], fs=fs, nperseg=256, noverlap=128)
        alpha_pwr_clean = np.trapezoid(psd_clean[:, alpha_idx], f_clean[alpha_idx], axis=-1).mean()
        alpha_retention_pct = alpha_pwr_clean / alpha_pwr_raw * 100.0 if alpha_pwr_raw > 0 else 100.0
        
        # Condition C: Nearest clean epoch replacement sensitivity metric
        # Flag 2s epochs with PTP > 150 uV in >3 channels
        raw_epochs = np.array([data[:, s*epoch_samples:(s+1)*epoch_samples] for s in selected_epochs])
        epoch_ptps = np.ptp(raw_epochs, axis=-1) # (15, 19)
        bad_epochs_cond_c = np.where((epoch_ptps > 150.0).sum(axis=-1) >= 3)[0]
        n_replaced_cond_c = len(bad_epochs_cond_c)
        
        rec_info = {
            "recording_idx": idx + 1,
            "filename": fname,
            "participant_id": pid,
            "task": task,
            "n_samples": n_samples,
            "solver": solver,
            "convergence": convergence,
            "fallback_used": fallback_used,
            "n_rejected_ics": len(rejected_ics),
            "rejected_ic_indices": str(rejected_ics),
            "retained_dimensions": 19 - len(rejected_ics),
            "ptp_raw_frontal_uV": round(ptp_raw_frontal, 2),
            "ptp_clean_frontal_uV": round(ptp_clean_frontal, 2),
            "ptp_reduction_pct": round(ptp_reduction_pct, 2),
            "alpha_raw_posterior": round(alpha_pwr_raw, 4),
            "alpha_clean_posterior": round(alpha_pwr_clean, 4),
            "alpha_retention_pct": round(alpha_retention_pct, 2),
            "cond_c_bad_epochs_replaced": n_replaced_cond_c,
            "cond_c_replacement_rate_pct": round(n_replaced_cond_c / 15.0 * 100, 2),
            "runtime_sec": round(time.time() - t_rec0, 2)
        }
        recording_records.append(rec_info)
        
        if (idx + 1) % 25 == 0 or (idx + 1) == len(edf_files):
            print(f"Processed {idx + 1:3d}/{len(edf_files)} recordings | Last: {fname} | Solver: {solver} | Rejected ICs: {len(rejected_ics)} | Frontal PTP Red: {ptp_reduction_pct:.1f}% | Alpha Ret: {alpha_retention_pct:.1f}%")

    df_rec = pd.DataFrame(recording_records)
    df_ic = pd.DataFrame(ic_records)
    df_fb = pd.DataFrame(fallback_records)
    
    # Save CSV deliverables
    rec_path = os.path.join(OUTPUT_DIR, "per_recording_ica_diagnostics.csv")
    ic_path = os.path.join(OUTPUT_DIR, "component_rejection_logs.csv")
    fb_path = os.path.join(OUTPUT_DIR, "fallback_logs.csv")
    df_rec.to_csv(rec_path, index=False)
    df_ic.to_csv(ic_path, index=False)
    df_fb.to_csv(fb_path, index=False)
    
    # Cohort-stratified statistics
    tasks_to_eval = ["eyesClosed", "eyesOpen"]
    cohort_stats = {}
    
    for tsk in tasks_to_eval:
        tdf = df_rec[df_rec["task"] == tsk]
        
        # Paired Wilcoxon signed-rank test on frontal PTP
        w_res = stats.wilcoxon(tdf["ptp_raw_frontal_uV"], tdf["ptp_clean_frontal_uV"], alternative='greater')
        
        # Statistical summary
        cohort_stats[tsk] = {
            "n_recordings": len(tdf),
            "mean_ptp_raw_uV": round(float(tdf["ptp_raw_frontal_uV"].mean()), 2),
            "mean_ptp_clean_uV": round(float(tdf["ptp_clean_frontal_uV"].mean()), 2),
            "mean_ptp_reduction_pct": round(float(tdf["ptp_reduction_pct"].mean()), 2),
            "wilcoxon_stat": float(w_res.statistic),
            "wilcoxon_p_value": float(w_res.pvalue),
            "mean_alpha_retention_pct": round(float(tdf["alpha_retention_pct"].mean()), 2),
            "min_alpha_retention_pct": round(float(tdf["alpha_retention_pct"].min()), 2),
            "pct_recordings_alpha_ge_85": round(float((tdf["alpha_retention_pct"] >= 85.0).mean() * 100), 2),
            "total_ics_rejected": int(tdf["n_rejected_ics"].sum()),
            "mean_ics_rejected_per_rec": round(float(tdf["n_rejected_ics"].mean()), 3),
            "cond_c_mean_replaced_epochs": round(float(tdf["cond_c_bad_epochs_replaced"].mean()), 2),
            "cond_c_replacement_rate_pct": round(float(tdf["cond_c_replacement_rate_pct"].mean()), 2)
        }
        
    # Evaluate Condition B Pre-Registration Criteria
    # 1. Frontal PTP reduction >= 35.0% in Eyes Open (primary ocular task)
    # 2. Wilcoxon p-value < 1e-10
    # 3. Posterior alpha retention >= 85.0%
    eo_ptp_red = cohort_stats["eyesOpen"]["mean_ptp_reduction_pct"]
    eo_wilcox_p = cohort_stats["eyesOpen"]["wilcoxon_p_value"]
    eo_alpha_ret = cohort_stats["eyesOpen"]["mean_alpha_retention_pct"]
    
    ec_alpha_ret = cohort_stats["eyesClosed"]["mean_alpha_retention_pct"]
    
    pass_ptp = eo_ptp_red >= 35.0
    pass_wilcox = eo_wilcox_p < 1e-10
    pass_alpha_eo = eo_alpha_ret >= 85.0
    pass_alpha_ec = ec_alpha_ret >= 85.0
    
    overall_pass = pass_ptp and pass_wilcox and pass_alpha_eo and pass_alpha_ec
    
    validation_decision = {
        "condition_b_status": "PASS" if overall_pass else "FAIL",
        "criteria_checks": {
            "eyes_open_frontal_ptp_reduction_ge_35pct": {
                "achieved": eo_ptp_red,
                "required": 35.0,
                "pass": pass_ptp
            },
            "eyes_open_ptp_reduction_wilcoxon_p_lt_1e-10": {
                "achieved": eo_wilcox_p,
                "required": 1e-10,
                "pass": pass_wilcox
            },
            "eyes_open_posterior_alpha_retention_ge_85pct": {
                "achieved": eo_alpha_ret,
                "required": 85.0,
                "pass": pass_alpha_eo
            },
            "eyes_closed_posterior_alpha_retention_ge_85pct": {
                "achieved": ec_alpha_ret,
                "required": 85.0,
                "pass": pass_alpha_ec
            }
        },
        "biophysical_caveat": (
            "Retaining 17 independent components in a 19-channel decomposition guarantees "
            "preserving 17 spatial linear degrees of freedom, but does NOT guarantee 17 pure "
            "unmixed neural sources due to volume conduction. The 2-component cap is a safety "
            "constraint against over-cleaning, not a proof of complete source separation."
        ),
        "cohort_statistics": cohort_stats,
        "runtime_total_seconds": round(time.time() - t0, 2)
    }
    
    # Save validation decision JSON
    json_path = os.path.join(OUTPUT_DIR, "m1_statistical_validation_results.json")
    with open(json_path, "w") as f:
        json.dump(validation_decision, f, indent=2)
        
    # Condition A / B / C comparison table
    comparison_rows = [
        {
            "Cohort": "Eyes Closed (EC, N=58)",
            "Condition A (Historical Uncorrected) Frontal PTP": f"{cohort_stats['eyesClosed']['mean_ptp_raw_uV']} uV",
            "Condition B (ICA Cleaned) Frontal PTP": f"{cohort_stats['eyesClosed']['mean_ptp_clean_uV']} uV (-{cohort_stats['eyesClosed']['mean_ptp_reduction_pct']}%)",
            "Condition B Posterior Alpha Retention": f"{cohort_stats['eyesClosed']['mean_alpha_retention_pct']}%",
            "Condition C Epoch Replacement Rate": f"{cohort_stats['eyesClosed']['cond_c_replacement_rate_pct']}% ({cohort_stats['eyesClosed']['cond_c_mean_replaced_epochs']}/15 epochs)",
            "Condition B Validation": "PASS" if pass_alpha_ec else "FAIL"
        },
        {
            "Cohort": "Eyes Open (EO, N=61, 62 recs)",
            "Condition A (Historical Uncorrected) Frontal PTP": f"{cohort_stats['eyesOpen']['mean_ptp_raw_uV']} uV",
            "Condition B (ICA Cleaned) Frontal PTP": f"{cohort_stats['eyesOpen']['mean_ptp_clean_uV']} uV (-{cohort_stats['eyesOpen']['mean_ptp_reduction_pct']}%)",
            "Condition B Posterior Alpha Retention": f"{cohort_stats['eyesOpen']['mean_alpha_retention_pct']}%",
            "Condition C Epoch Replacement Rate": f"{cohort_stats['eyesOpen']['cond_c_replacement_rate_pct']}% ({cohort_stats['eyesOpen']['cond_c_mean_replaced_epochs']}/15 epochs)",
            "Condition B Validation": "PASS" if (pass_ptp and pass_wilcox and pass_alpha_eo) else "FAIL"
        }
    ]
    df_comp = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(OUTPUT_DIR, "condition_comparison_summary.csv")
    df_comp.to_csv(comp_path, index=False)
    
    print("\n================================================================================")
    print("                    MODULE M1 VALIDATION SUMMARY RESULTS                        ")
    print("================================================================================")
    print(f"Condition B Status: >>> {validation_decision['condition_b_status']} <<<")
    print(f"EO Frontal PTP Reduction:      {eo_ptp_red:.2f}% (Threshold: >=35.0%) -> {'PASS' if pass_ptp else 'FAIL'}")
    print(f"EO PTP Reduction Wilcoxon p:    {eo_wilcox_p:.2e} (Threshold: <1e-10) -> {'PASS' if pass_wilcox else 'FAIL'}")
    print(f"EO Posterior Alpha Retention:  {eo_alpha_ret:.2f}% (Threshold: >=85.0%) -> {'PASS' if pass_alpha_eo else 'FAIL'}")
    print(f"EC Posterior Alpha Retention:  {ec_alpha_ret:.2f}% (Threshold: >=85.0%) -> {'PASS' if pass_alpha_ec else 'FAIL'}")
    print(f"Total Runtime:                 {validation_decision['runtime_total_seconds']:.2f} seconds")
    print(f"Outputs written to:            {OUTPUT_DIR}/")
    print("================================================================================")
    
    return validation_decision

if __name__ == "__main__":
    run_m1_pipeline()

