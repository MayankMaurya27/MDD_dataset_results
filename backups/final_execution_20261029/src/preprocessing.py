"""EEG Preprocessing and Segmentation module.

Handles:
- MNE / EDF raw loading
- Standard 10-20 channel selection (19 channels)
- Optional bandpass filtering (0.5 - 45 Hz) and common average reference
- Fixed-duration non-overlapping window segmentation
- Strict segment origin tracking (participant, task, run, sample start/end)
"""

import os
import re
import mne
import numpy as np

# Standard 19 International 10-20 EEG channels
STANDARD_1020_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'F7', 'T3', 'T5', 'Fz',
    'Fp2', 'F4', 'C4', 'P4', 'O2',
    'F8', 'T4', 'T6', 'Cz', 'Pz'
]

def load_eeg_recording(edf_path, l_freq=0.5, h_freq=45.0, apply_car=False):
    """
    Load an EDF file using MNE and extract standardized 19 EEG channels.
    
    Parameters
    ----------
    edf_path : str
        Path to EDF file.
    l_freq : float, optional
        High-pass filter cutoff in Hz. Defaults to 0.5 Hz.
    h_freq : float, optional
        Low-pass filter cutoff in Hz. Defaults to 45.0 Hz.
    apply_car : bool, optional
        Whether to apply Common Average Referencing (CAR).
        
    Returns
    -------
    data : 2D numpy array of shape (19, n_samples)
        Preprocessed EEG signals in microvolts (uV).
    fs : float
        Sampling frequency in Hz.
    ch_names : list of str
        The 19 standard channel names in order.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # Map raw channel names (e.g. 'EEG Fp1-LE') to standard names (e.g. 'Fp1')
    name_mapping = {}
    for ch in raw.ch_names:
        clean = ch.replace('EEG ', '').replace('-LE', '').replace('-A1', '').strip()
        if clean in STANDARD_1020_CHANNELS:
            name_mapping[ch] = clean
            
    # Check that all 19 channels are found
    found_channels = set(name_mapping.values())
    missing = set(STANDARD_1020_CHANNELS) - found_channels
    if missing:
        raise ValueError(f"Recording {edf_path} is missing standard channels: {missing}")
        
    raw.rename_channels(name_mapping)
    raw.pick(STANDARD_1020_CHANNELS)
    
    # Optional bandpass filter
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        
    if apply_car:
        raw.set_eeg_reference('average', projection=False, verbose=False)
        
    # Extract data in microvolts (MNE defaults to Volts)
    data = raw.get_data() * 1e6 # Shape (19, n_samples)
    fs = float(raw.info['sfreq'])
    
    return data, fs, STANDARD_1020_CHANNELS

def segment_recording(data, fs, window_sec=2.0, overlap=0.0):
    """
    Segment multi-channel continuous EEG into fixed-length epochs.
    
    Parameters
    ----------
    data : 2D numpy array of shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz.
    window_sec : float
        Window duration in seconds.
    overlap : float
        Overlap fraction between successive windows [0.0, 1.0).
        
    Returns
    -------
    segments : 3D numpy array of shape (n_segments, n_channels, window_samples)
    sample_indices : list of (start_idx, end_idx) tuples
    """
    n_channels, n_samples = data.shape
    window_samples = int(window_sec * fs)
    stride_samples = int(window_samples * (1.0 - overlap))
    
    if stride_samples <= 0:
        raise ValueError("Overlap must be strictly less than 1.0.")
        
    segments = []
    sample_indices = []
    
    start = 0
    while start + window_samples <= n_samples:
        end = start + window_samples
        segments.append(data[:, start:end])
        sample_indices.append((start, end))
        start += stride_samples
        
    if len(segments) == 0:
        return np.empty((0, n_channels, window_samples)), []
        
    return np.array(segments), sample_indices
