"""Smoothed Pseudo Wigner-Ville Distribution (SPWVD) implementation.

Computes the discrete SPWVD of real or analytic signals using time and frequency smoothing.
"""

import numpy as np
from scipy import signal, ndimage

def compute_spwvd(x, fs=256, g=None, h=None, nfft=256):
    """
    Compute discrete Smoothed Pseudo Wigner-Ville Distribution.
    
    Parameters
    ----------
    x : 1D numpy array
        Input signal (real-valued).
    fs : float
        Sampling frequency in Hz (default 256).
    g : 1D numpy array, optional
        Time-smoothing window. Defaults to Hamming window of length 31.
    h : 1D numpy array, optional
        Frequency-smoothing (lag) window. Defaults to Kaiser window (length 255, beta=5.0).
    nfft : int
        Number of frequency bins for FFT (default 256).
        
    Returns
    -------
    wvd_pos : 2D numpy array of shape (nfft//2, len(x))
        SPWVD energy matrix for positive frequencies [0, fs/2].
    freqs : 1D numpy array of length nfft//2
        Frequency axis in Hz.
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N == 0:
        raise ValueError("Input signal is empty.")
        
    # Analytic signal via Hilbert transform
    z = signal.hilbert(x)
    
    # Time smoothing window g
    if g is None:
        g = signal.windows.hamming(31)
    else:
        g = np.asarray(g, dtype=np.float64)
    if len(g) % 2 == 0:
        g = np.pad(g, (0, 1), mode='constant')
    g = g / np.sum(g) # Normalize energy
    Lg = len(g)
    Mg = (Lg - 1) // 2
    
    # Frequency smoothing / lag window h (Kaiser window size 255)
    if h is None:
        h = signal.windows.kaiser(255, beta=5.0)
    else:
        h = np.asarray(h, dtype=np.float64)
    if len(h) % 2 == 0:
        h = np.pad(h, (0, 1), mode='constant')
    Lh = len(h)
    Mh = (Lh - 1) // 2
    h = h / h[Mh] # Peak normalized to 1.0
    
    # Zero-pad analytic signal for boundary lag products
    pad_len = Mh
    z_pad = np.pad(z, pad_len, mode='constant')
    
    # Autocorrelation tensor: R[tau_idx, t_idx] = z(t + tau) * conj(z(t - tau))
    tau_vals = np.arange(-Mh, Mh + 1)
    t_idx = np.arange(N) + pad_len
    
    # Outer product indexing
    z_plus = z_pad[t_idx[None, :] + tau_vals[:, None]]
    z_minus = np.conj(z_pad[t_idx[None, :] - tau_vals[:, None]])
    R = z_plus * z_minus
    
    # Time smoothing along time axis (axis 1)
    R_real = ndimage.convolve1d(np.real(R), g, axis=1, mode='constant')
    R_imag = ndimage.convolve1d(np.imag(R), g, axis=1, mode='constant')
    R_smoothed = R_real + 1j * R_imag
    
    # Frequency windowing along lag axis (axis 0)
    R_windowed = R_smoothed * h[:, None]
    
    # FFT along tau axis (axis 0)
    wvd = np.fft.fft(np.fft.ifftshift(R_windowed, axes=0), n=nfft, axis=0)
    wvd = np.real(wvd) # Theoretical SPWVD is real
    
    # Positive frequencies up to Nyquist
    n_freq = nfft // 2
    wvd_pos = wvd[:n_freq, :]
    freqs = np.linspace(0, fs / 2, n_freq)
    
    return wvd_pos, freqs

def spwvd_to_rgb_image(wvd, target_size=(224, 224)):
    """
    Convert a 2D SPWVD matrix to a standardized (3, 224, 224) RGB tensor.
    
    Uses robust percentile min-max scaling and bilinear interpolation.
    """
    import torch
    import torch.nn.functional as F
    
    # Robust normalization: clip negative cross-term ripples and upper outliers
    wvd_clipped = np.maximum(wvd, 0)
    p99 = np.percentile(wvd_clipped, 99.5)
    if p99 > 1e-8:
        wvd_norm = np.clip(wvd_clipped / p99, 0.0, 1.0)
    else:
        wvd_norm = np.zeros_like(wvd_clipped)
        
    # Resize to target size via PyTorch bilinear interpolation
    t = torch.from_numpy(wvd_norm).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    t_resized = F.interpolate(t, size=target_size, mode='bilinear', align_corners=False) # (1, 1, 224, 224)
    
    # Expand to 3 color channels (RGB)
    t_rgb = t_resized.repeat(1, 3, 1, 1) # (1, 3, 224, 224)
    
    # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    t_normalized = (t_rgb - mean) / std
    
    return t_normalized.squeeze(0) # (3, 224, 224)
