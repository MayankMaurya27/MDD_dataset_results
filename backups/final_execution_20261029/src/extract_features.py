"""Feature extraction and caching pipeline.

Loads EEG recordings, segments into windows, computes SPWVD images,
and extracts deep feature representations using:
- ResNet-18 (512-D)
- ResNet-50 (2048-D)
- AlexNet (4096-D)
- VGG-16 (4096-D)

Saves cached tensors and metadata in results/features_cache/.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch

from src.preprocessing import load_eeg_recording, segment_recording
from src.spwvd import compute_spwvd, spwvd_to_rgb_image
from src.feature_extractor import DeepFeatureExtractor

CACHE_DIR = os.path.abspath("results/features_cache")

def process_and_cache_dataset(task_filter="eyesClosed", max_segments_per_file=15, models=["resnet18", "resnet50", "alexnet", "vgg16"]):
    os.makedirs(CACHE_DIR, exist_ok=True)
    bids_dir = os.path.abspath("nm000114_v1.0.0")
    inv = pd.read_csv("reports/dataset_inventory.csv")
    folds_df = pd.read_csv("results/fold_assignments.csv")
    
    # Merge fold assignments
    inv = inv.merge(folds_df[['participant_id', 'fold']], on='participant_id', how='left')
    
    if task_filter:
        records = inv[inv['task'] == task_filter].copy()
    else:
        records = inv.copy()
        
    print(f"--- Processing dataset: task={task_filter or 'ALL'}, records={len(records)}, max_segments={max_segments_per_file} ---")
    
    image_tensors = []
    metadata_rows = []
    
    t_start = time.time()
    for idx, (_, row) in enumerate(records.iterrows()):
        p_id = row['participant_id']
        rel_path = row['filename']
        full_path = os.path.join(bids_dir, rel_path)
        group = row['group_label']
        label = row['numeric_label']
        fold = row['fold']
        task = row['task']
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(records):
            print(f"  [{idx+1}/{len(records)}] Processing {p_id} ({task})... elapsed: {time.time()-t_start:.1f}s")
            
        data, fs, chs = load_eeg_recording(full_path, l_freq=0.5, h_freq=45.0)
        
        # Segment 2.0s windows
        segs, indices = segment_recording(data, fs, window_sec=2.0)
        if len(segs) > max_segments_per_file:
            # Take evenly spaced segments across the recording
            step = len(segs) // max_segments_per_file
            selected_idx = [i * step for i in range(max_segments_per_file)]
            segs = segs[selected_idx]
            indices = [indices[i] for i in selected_idx]
            
        for seg_idx, (s, (start_idx, end_idx)) in enumerate(zip(segs, indices)):
            # Average multi-channel SPWVD energy across the 19 standard channels
            spwvd_stack = []
            for ch_idx in range(19):
                wvd, freqs = compute_spwvd(s[ch_idx], fs=fs, nfft=256)
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
                "task": task,
                "segment_local_idx": seg_idx,
                "sample_start": start_idx,
                "sample_end": end_idx
            })
            
    df_meta = pd.DataFrame(metadata_rows)
    meta_path = os.path.join(CACHE_DIR, f"metadata_{task_filter or 'all'}.csv")
    df_meta.to_csv(meta_path, index=False)
    print(f"Generated {len(image_tensors)} SPWVD images across {df_meta['participant_id'].nunique()} participants in {time.time()-t_start:.1f}s.")
    
    # Extract features for each requested CNN architecture
    batch_size = 64
    for model_name in models:
        print(f"Extracting features using pretrained {model_name}...")
        t_model = time.time()
        extractor = DeepFeatureExtractor(model_name=model_name)
        
        feature_batches = []
        for b_start in range(0, len(image_tensors), batch_size):
            b_end = min(b_start + batch_size, len(image_tensors))
            batch = torch.stack(image_tensors[b_start:b_end], dim=0)
            feats = extractor.extract_features(batch)
            feature_batches.append(feats)
            
        all_features = np.concatenate(feature_batches, axis=0)
        feat_path = os.path.join(CACHE_DIR, f"features_{model_name}_{task_filter or 'all'}.npy")
        np.save(feat_path, all_features)
        print(f"  Saved {all_features.shape} features to {feat_path} in {time.time()-t_model:.1f}s.")
        
    print("Feature extraction and caching completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="eyesClosed", help="Task filter: eyesClosed, eyesOpen, P300, or '' for all")
    parser.add_argument("--segments", type=int, default=15, help="Max segments per recording file")
    args = parser.parse_args()
    
    process_and_cache_dataset(task_filter=args.task, max_segments_per_file=args.segments)
