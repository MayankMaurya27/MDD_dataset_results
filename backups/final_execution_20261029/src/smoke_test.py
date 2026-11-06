"""Stage 1: Smoke Test.

Loads a small subset of recordings (e.g. 2 participants: 1 HC, 1 MDD),
runs preprocessing, segmentation, SPWVD, ResNet-18 feature extraction,
and fits a classifier end-to-end.
"""

import os
import time
import numpy as np
import pandas as pd
import torch

from src.preprocessing import load_eeg_recording, segment_recording
from src.spwvd import compute_spwvd, spwvd_to_rgb_image
from src.feature_extractor import DeepFeatureExtractor
from src.classifiers import get_classifier
from src.evaluation import compute_classification_metrics

def run_smoke_test():
    print("=== STARTING STAGE 1 SMOKE TEST ===")
    t0 = time.time()
    
    bids_dir = os.path.abspath("nm000114_v1.0.0")
    inv = pd.read_csv("reports/dataset_inventory.csv")
    
    # Pick 2 HC and 2 MDD with eyesClosed recordings
    sample_records = inv[inv['task'] == 'eyesClosed'].groupby('group_label').head(2).reset_index(drop=True)
    print(f"Testing on 4 participants: {sample_records['participant_id'].tolist()}")
    
    segments_list = []
    labels_list = []
    part_ids = []
    
    for _, row in sample_records.iterrows():
        p_id = row['participant_id']
        rel_path = row['filename']
        full_path = os.path.join(bids_dir, rel_path)
        label = row['numeric_label']
        
        print(f"Loading {p_id} from {rel_path}...")
        data, fs, chs = load_eeg_recording(full_path, l_freq=0.5, h_freq=45.0)
        assert data.shape[0] == 19
        
        # Take first 10 seconds (5 segments of 2s each) for fast smoke test
        data_subset = data[:, :int(10.0 * fs)]
        segs, indices = segment_recording(data_subset, fs, window_sec=2.0)
        print(f"  Extracted {len(segs)} segments of shape {segs[0].shape}")
        
        for s in segs:
            spwvd_stack = []
            for ch_idx in range(19):
                wvd, freqs = compute_spwvd(s[ch_idx], fs=fs, nfft=256)
                spwvd_stack.append(wvd)
            mean_wvd = np.mean(spwvd_stack, axis=0)
            
            rgb_img = spwvd_to_rgb_image(mean_wvd, target_size=(224, 224))
            
            segments_list.append(rgb_img)
            labels_list.append(label)
            part_ids.append(p_id)
            
    print(f"Total processed smoke test images: {len(segments_list)}")
    
    # Feature extraction with ResNet-18
    extractor = DeepFeatureExtractor(model_name="resnet18")
    features = extractor.extract_features(segments_list)
    print(f"Extracted feature matrix shape: {features.shape}")
    assert features.shape == (len(segments_list), 512)
    
    # Train: 1 HC + 1 MDD, Test: 1 HC + 1 MDD
    train_subs = [sample_records['participant_id'].iloc[0], sample_records['participant_id'].iloc[2]]
    train_mask = np.isin(part_ids, train_subs)
    test_mask = ~train_mask
    
    X_train, y_train = features[train_mask], np.array(labels_list)[train_mask]
    X_test, y_test = features[test_mask], np.array(labels_list)[test_mask]
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    clf = get_classifier("Weighted KNN")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    
    metrics = compute_classification_metrics(y_test, preds, probs)
    print(f"Smoke test completed in {time.time() - t0:.2f}s!")
    print(f"Smoke test evaluation: {metrics}")
    
    os.makedirs("logs", exist_ok=True)
    with open("logs/smoke_test.log", "w") as f:
        f.write(f"Smoke test passed successfully in {time.time() - t0:.2f}s.\n")
        f.write(f"Output shapes: features={features.shape}, test_metrics={metrics}\n")
    return True

if __name__ == '__main__':
    run_smoke_test()
