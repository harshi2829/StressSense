"""
Signal Filtering Utilities for WESAD Preprocessing
===================================================
Butterworth filters for physiological signal processing.
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample
from typing import Tuple, Optional


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth bandpass filter.
    
    Args:
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Tuple of (b, a) filter coefficients
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ensure frequencies are valid
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_lowpass(cutoff: float, fs: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth lowpass filter.
    
    Args:
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Tuple of (b, a) filter coefficients
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    normalized_cutoff = max(0.001, min(normalized_cutoff, 0.999))
    
    b, a = butter(order, normalized_cutoff, btype='low')
    return b, a


def butter_highpass(cutoff: float, fs: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth highpass filter.
    
    Args:
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Tuple of (b, a) filter coefficients
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    normalized_cutoff = max(0.001, min(normalized_cutoff, 0.999))
    
    b, a = butter(order, normalized_cutoff, btype='high')
    return b, a


def apply_bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to a signal.
    
    Args:
        signal: Input signal array
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    if len(signal) < 3 * order:
        return signal  # Signal too short for filtering
        
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    
    try:
        filtered = filtfilt(b, a, signal, padlen=min(len(signal) - 1, 3 * order))
    except ValueError:
        # Fallback if filtfilt fails
        filtered = signal
        
    return filtered


def apply_lowpass_filter(
    signal: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply a Butterworth lowpass filter to a signal.
    
    Args:
        signal: Input signal array
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    if len(signal) < 3 * order:
        return signal
        
    b, a = butter_lowpass(cutoff, fs, order)
    
    try:
        filtered = filtfilt(b, a, signal, padlen=min(len(signal) - 1, 3 * order))
    except ValueError:
        filtered = signal
        
    return filtered


def apply_highpass_filter(
    signal: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply a Butterworth highpass filter to a signal.
    
    Args:
        signal: Input signal array
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    if len(signal) < 3 * order:
        return signal
        
    b, a = butter_highpass(cutoff, fs, order)
    
    try:
        filtered = filtfilt(b, a, signal, padlen=min(len(signal) - 1, 3 * order))
    except ValueError:
        filtered = signal
        
    return filtered


def moving_average(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply a moving average smoothing filter.
    
    Args:
        signal: Input signal array
        window_size: Size of the moving average window
        
    Returns:
        Smoothed signal
    """
    if len(signal) < window_size:
        return signal
        
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(signal, kernel, mode='same')
    
    return smoothed


def resample_signal(
    signal: np.ndarray,
    original_fs: float,
    target_fs: float
) -> np.ndarray:
    """
    Resample a signal to a target sampling frequency.
    
    Args:
        signal: Input signal array
        original_fs: Original sampling frequency in Hz
        target_fs: Target sampling frequency in Hz
        
    Returns:
        Resampled signal
    """
    if original_fs == target_fs:
        return signal
        
    num_samples = int(len(signal) * target_fs / original_fs)
    resampled = resample(signal, num_samples)
    
    return resampled


def compute_acc_magnitude(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> np.ndarray:
    """
    Compute accelerometer magnitude from 3-axis components.
    
    Args:
        acc_x: X-axis accelerometer data
        acc_y: Y-axis accelerometer data
        acc_z: Z-axis accelerometer data
        
    Returns:
        Accelerometer magnitude
    """
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    return magnitude


def filter_ecg(signal: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 40.0, order: int = 4) -> np.ndarray:
    """
    Apply ECG-specific bandpass filter.
    
    Args:
        signal: Raw ECG signal
        fs: Sampling frequency in Hz
        lowcut: Low cutoff frequency (default: 0.5 Hz)
        highcut: High cutoff frequency (default: 40 Hz)
        order: Filter order
        
    Returns:
        Filtered ECG signal
    """
    return apply_bandpass_filter(signal, lowcut, highcut, fs, order)


def filter_eda(signal: np.ndarray, fs: float, cutoff: float = 5.0, order: int = 4) -> np.ndarray:
    """
    Apply EDA-specific lowpass filter.
    
    Args:
        signal: Raw EDA signal
        fs: Sampling frequency in Hz
        cutoff: Cutoff frequency (default: 5 Hz)
        order: Filter order
        
    Returns:
        Filtered EDA signal
    """
    return apply_lowpass_filter(signal, cutoff, fs, order)


def filter_temp(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply temperature smoothing filter.
    
    Args:
        signal: Raw temperature signal
        window_size: Moving average window size
        
    Returns:
        Smoothed temperature signal
    """
    return moving_average(signal, window_size)


def filter_acc(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply accelerometer smoothing filter.
    
    Args:
        signal: Raw accelerometer signal (magnitude or single axis)
        window_size: Moving average window size
        
    Returns:
        Smoothed accelerometer signal
    """
    return moving_average(signal, window_size)


def normalize_zscore(signal: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Apply z-score normalization.
    
    Args:
        signal: Input signal
        epsilon: Small value to prevent division by zero
        
    Returns:
        Normalized signal with mean=0, std=1
    """
    mean = np.mean(signal)
    std = np.std(signal)
    
    if std < epsilon:
        return signal - mean
        
    return (signal - mean) / std


def normalize_minmax(signal: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Apply min-max normalization to [0, 1] range.
    
    Args:
        signal: Input signal
        epsilon: Small value to prevent division by zero
        
    Returns:
        Normalized signal in [0, 1] range
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    range_val = max_val - min_val
    
    if range_val < epsilon:
        return np.zeros_like(signal)
        
    return (signal - min_val) / range_val
