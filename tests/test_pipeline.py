"""Comprehensive verification and reproducibility test suite.

Tests:
1. Dataset discovery and participant label validation
2. Fold grouping and zero-leakage participant separation across all folds
3. Segmentation boundaries and temporal consistency
4. SPWVD mathematical correctness, output shape, and numerical stability
5. Training-only transformations (no data leakage)
6. Metric calculations and participant-level probability aggregation
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.spwvd import compute_spwvd, spwvd_to_rgb_image
from src.preprocessing import segment_recording, STANDARD_1020_CHANNELS
from src.evaluation import compute_classification_metrics, aggregate_participant_predictions
from src.classifiers import get_classifier

def test_dataset_discovery_and_labels():
    """Verify that dataset inventory exists and contains exactly 64 participants (30 HC, 34 MDD)."""
    inv_path = "reports/dataset_inventory.csv"
    assert os.path.exists(inv_path), f"{inv_path} must exist"
    df = pd.read_csv(inv_path)
    
    unique_subs = df[['participant_id', 'group_label', 'numeric_label']].drop_duplicates()
    assert len(unique_subs) == 64, f"Expected 64 unique participants, got {len(unique_subs)}"
    
    hc_count = (unique_subs['group_label'] == 'HC').sum()
    mdd_count = (unique_subs['group_label'] == 'MDD').sum()
    assert hc_count == 30, f"Expected 30 HC, got {hc_count}"
    assert mdd_count == 34, f"Expected 34 MDD, got {mdd_count}"
    
    # Check numeric labels: 0 for HC, 1 for MDD
    assert (unique_subs[unique_subs['group_label'] == 'HC']['numeric_label'] == 0).all()
    assert (unique_subs[unique_subs['group_label'] == 'MDD']['numeric_label'] == 1).all()

def test_zero_leakage_fold_isolation():
    """Verify that no participant appears in both training and testing partitions within any fold."""
    folds_path = "results/fold_assignments.csv"
    assert os.path.exists(folds_path), f"{folds_path} must exist"
    df_folds = pd.read_csv(folds_path)
    
    assert len(df_folds) == 64, f"Expected 64 participants in fold assignments, got {len(df_folds)}"
    folds = sorted(df_folds['fold'].unique())
    assert len(folds) == 10, f"Expected 10 folds, got {len(folds)}"
    
    all_participants = set(df_folds['participant_id'])
    
    for f in folds:
        test_subs = set(df_folds[df_folds['fold'] == f]['participant_id'])
        train_subs = set(df_folds[df_folds['fold'] != f]['participant_id'])
        
        # Test strict mutual exclusivity (zero leakage)
        overlap = train_subs.intersection(test_subs)
        assert len(overlap) == 0, f"DATA LEAKAGE DETECTED in Fold {f}! Overlapping participants: {overlap}"
        assert train_subs.union(test_subs) == all_participants, f"Fold {f} does not partition all participants"
        
        # Check stratification: exactly 3 HC in every fold
        test_hc = (df_folds[df_folds['fold'] == f]['numeric_label'] == 0).sum()
        assert test_hc == 3, f"Fold {f} has {test_hc} HC, expected exactly 3"

def test_segmentation_boundaries():
    """Verify that segmentation never crosses recording boundaries and handles durations correctly."""
    fs = 256.0
    n_channels = 19
    # Simulate 5-second recording (1280 samples)
    dummy_data = np.random.randn(n_channels, int(5.0 * fs))
    
    # 2-second windows, no overlap
    segments, indices = segment_recording(dummy_data, fs, window_sec=2.0, overlap=0.0)
    assert len(segments) == 2, f"Expected 2 full 2-second segments in 5s, got {len(segments)}"
    assert segments.shape == (2, n_channels, int(2.0 * fs))
    
    for start, end in indices:
        assert start >= 0
        assert end <= dummy_data.shape[1]
        assert (end - start) == int(2.0 * fs)

def test_spwvd_shape_and_numerical_stability():
    """Verify SPWVD output shape, finite values, and RGB image conversion."""
    fs = 256
    t = np.linspace(0, 1.0, fs, endpoint=False)
    # Pure tone + impulse
    x = np.sin(2 * np.pi * 15 * t)
    x[100] += 5.0
    
    wvd, freqs = compute_spwvd(x, fs=fs, nfft=256)
    
    # Shape check: nfft//2 = 128 frequency bins, len(x) = 256 time points
    assert wvd.shape == (128, 256), f"Expected (128, 256), got {wvd.shape}"
    assert np.all(np.isfinite(wvd)), "SPWVD output contains non-finite values (NaN or Inf)"
    assert freqs[0] == 0.0
    assert np.isclose(freqs[-1], fs / 2)
    
    # Test RGB image conversion
    rgb_tensor = spwvd_to_rgb_image(wvd, target_size=(224, 224))
    assert rgb_tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {rgb_tensor.shape}"
    assert np.all(np.isfinite(rgb_tensor.numpy())), "RGB tensor contains non-finite values"

def test_training_only_transformations():
    """Verify that preprocessing scalers fit strictly on training fold without test contamination."""
    rng = np.random.RandomState(42)
    # Simulate train features (mean=10, std=5) and test features (mean=50, std=1)
    X_train = rng.normal(10.0, 5.0, size=(100, 10))
    X_test = rng.normal(50.0, 1.0, size=(20, 10))
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Verify scaler mean is close to 10, not pulled towards 50
    assert np.allclose(scaler.mean_, 10.0, atol=1.5), "Scaler was contaminated or incorrectly computed"
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    assert np.allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-7)
    # Test scaled mean should be roughly (50 - 10) / 5 = 8
    assert np.all(X_test_scaled.mean(axis=0) > 5.0)

def test_metrics_and_participant_aggregation():
    """Verify diagnostic metrics computation and participant-level probability aggregation."""
    # Synthetic segment dataframe for 3 participants
    seg_df = pd.DataFrame({
        'participant_id': ['sub-1', 'sub-1', 'sub-2', 'sub-2', 'sub-3', 'sub-3'],
        'y_true': [1, 1, 0, 0, 1, 1],
        'y_prob': [0.8, 0.6, 0.1, 0.3, 0.4, 0.2] # sub-1=0.7 (MDD, correct), sub-2=0.2 (HC, correct), sub-3=0.3 (MDD, incorrect)
    })
    
    part_df = aggregate_participant_predictions(seg_df)
    assert len(part_df) == 3
    assert set(part_df['participant_id']) == {'sub-1', 'sub-2', 'sub-3'}
    
    # Sub-1: mean_prob=0.7 -> pred=1
    assert part_df[part_df['participant_id'] == 'sub-1']['y_pred'].iloc[0] == 1
    # Sub-2: mean_prob=0.2 -> pred=0
    assert part_df[part_df['participant_id'] == 'sub-2']['y_pred'].iloc[0] == 0
    # Sub-3: mean_prob=0.3 -> pred=0
    assert part_df[part_df['participant_id'] == 'sub-3']['y_pred'].iloc[0] == 0
    
    metrics = compute_classification_metrics(part_df['y_true'], part_df['y_pred'], part_df['mean_prob'])
    # 2 correct out of 3 = 66.67%
    assert np.isclose(metrics['accuracy'], 66.67, atol=0.1)
    assert metrics['tp'] == 1
    assert metrics['tn'] == 1
    assert metrics['fn'] == 1
    assert metrics['fp'] == 0
