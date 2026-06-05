"""
WESAD Preprocessing Module
==========================

Functions for preprocessing physiological signals from the WESAD dataset.
Outputs X.npy and y.npy for training.

Usage:
    from utils.preprocess import preprocess_dataset, CLASS_NAMES
    preprocess_dataset(input_dir, output_dir, config)
"""

import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from utils.signal_filters import (
    apply_bandpass_filter as bandpass_filter,
    apply_lowpass_filter as lowpass_filter,
    moving_average,
    compute_acc_magnitude
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WESAD sampling rates (Hz)
WESAD_SAMPLING_RATES = {
    'ECG': 700,
    'EDA': 4,
    'TEMP': 4,
    'ACC': 32,
    'label': 700
}

# Label mapping: WESAD labels -> Binary (0=non-stress, 1=stress)
LABEL_MAPPING = {
    0: -1,  # Not defined -> exclude
    1: 0,   # Baseline -> Non-stress
    2: 1,   # Stress -> Stress
    3: 0,   # Amusement -> Non-stress
    4: 0,   # Meditation -> Non-stress
    5: -1,  # Excluded
    6: -1,  # Excluded
    7: -1,  # Excluded
}

CLASS_NAMES = {0: 'Non-Stress', 1: 'Stress'}

# --------------------------
# Load subject data
# --------------------------
def load_wesad_subject(filepath: Union[str, Path]) -> Dict:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    logger.info(f"Loading subject data from {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

# --------------------------
# Extract chest signals
# --------------------------
def extract_chest_signals(data: Dict) -> Dict[str, np.ndarray]:
    chest = data['signal']['chest']
    signals = {
        'ECG': chest['ECG'].flatten(),
        'EDA': chest['EDA'].flatten(),
        'TEMP': chest['Temp'].flatten(),
        'ACC': chest['ACC'],  # Shape: (N, 3)
        'label': data['label'].flatten()
    }
    logger.info(f"Extracted signals - ECG: {signals['ECG'].shape}, "
                f"EDA: {signals['EDA'].shape}, TEMP: {signals['TEMP'].shape}, "
                f"ACC: {signals['ACC'].shape}")
    return signals

# --------------------------
# Filter signals
def filter_signals(signals: Dict[str, np.ndarray], config: Dict) -> Dict[str, np.ndarray]:
    filtered = {}
    filter_cfg = config.get('filtering', {})

    # ECG: Bandpass 0.5-40 Hz
    ecg_cfg = filter_cfg.get('ecg', {})
    filtered['ECG'] = bandpass_filter(
        signals['ECG'],
        lowcut=ecg_cfg.get('lowcut', 0.5),
        highcut=ecg_cfg.get('highcut', 40.0),
        fs=WESAD_SAMPLING_RATES['ECG'],
        order=ecg_cfg.get('order', 4)
    )

    # EDA: Lowpass 5 Hz
    eda_cfg = filter_cfg.get('eda', {})
    filtered['EDA'] = lowpass_filter(
        signals['EDA'],
        cutoff=eda_cfg.get('cutoff', 5.0),
        fs=WESAD_SAMPLING_RATES['EDA'],
        order=eda_cfg.get('order', 4)
    )

    # TEMP: Moving average smoothing
    temp_cfg = filter_cfg.get('temp', {})
    filtered['TEMP'] = moving_average(
        signals['TEMP'],
        window_size=temp_cfg.get('window_size', 5)
    )

    # ACC: Compute magnitude and smooth
    acc_cfg = filter_cfg.get('acc', {})
    acc_magnitude = compute_acc_magnitude(
        signals['ACC'][:, 0],  # x-axis
        signals['ACC'][:, 1],  # y-axis
        signals['ACC'][:, 2]   # z-axis
    )
    filtered['ACC'] = moving_average(
        acc_magnitude,
        window_size=acc_cfg.get('window_size', 5)
    )

    # Keep labels unchanged
    filtered['label'] = signals['label']

    logger.info("Signal filtering complete")
    return filtered
# --------------------------
# Resampling
# --------------------------
def resample_signal(signal: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    if original_rate == target_rate:
        return signal
    duration = len(signal) / original_rate
    original_times = np.linspace(0, duration, len(signal))
    target_samples = int(duration * target_rate)
    target_times = np.linspace(0, duration, target_samples)
    return np.interp(target_times, original_times, signal)

def resample_all_signals(signals: Dict[str, np.ndarray], target_rate: int = 32) -> Dict[str, np.ndarray]:
    resampled = {}
    for name, signal in signals.items():
        if name == 'label':
            resampled['label'] = resample_signal(signal, WESAD_SAMPLING_RATES['label'], target_rate).astype(int)
        else:
            original_rate = WESAD_SAMPLING_RATES[name]
            resampled[name] = resample_signal(signal, original_rate, target_rate)
    logger.info(f"All signals resampled to {target_rate} Hz")
    return resampled

# --------------------------
# Map labels
# --------------------------
def map_labels(labels: np.ndarray) -> np.ndarray:
    return np.array([LABEL_MAPPING.get(int(l), -1) for l in labels])

# --------------------------
# Windowing
# --------------------------
def create_windows(
    signals: Dict[str, np.ndarray],
    window_size: float = 10.0,
    overlap: float = 0.0,
    sampling_rate: int = 32,
    min_label_consistency: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    window_samples = int(window_size * sampling_rate)
    step_samples = int(window_samples * (1 - overlap))
    signal_names = ['ECG', 'EDA', 'TEMP', 'ACC']
    min_length = min(len(signals[name]) for name in signal_names)
    min_length = min(min_length, len(signals['label']))
    mapped_labels = map_labels(signals['label'][:min_length])
    windows, labels = [], []
    start = 0
    while start + window_samples <= min_length:
        end = start + window_samples
        window_labels = mapped_labels[start:end]
        valid_mask = window_labels >= 0
        if np.sum(valid_mask) < min_label_consistency * window_samples:
            start += step_samples
            continue
        valid_labels = window_labels[valid_mask]
        unique, counts = np.unique(valid_labels, return_counts=True)
        window_label = unique[np.argmax(counts)]
        if np.max(counts)/len(valid_labels) < min_label_consistency:
            start += step_samples
            continue
        window_data = [signals[name][start:end] for name in signal_names]
        windows.append(np.array(window_data))
        labels.append(window_label)
        start += step_samples
    if len(windows) == 0:
        return np.array([]), np.array([])
    X = np.array(windows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    logger.info(f"Created {len(windows)} windows with shape {X.shape}")
    return X, y

# --------------------------
# Normalization
# --------------------------
def normalize_windows(X: np.ndarray, method: str = 'zscore') -> np.ndarray:
    if method == 'none' or len(X) == 0:
        return X
    X_norm = X.copy()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            channel = X_norm[i, j, :]
            if method == 'zscore':
                mean, std = np.mean(channel), np.std(channel)
                X_norm[i, j, :] = (channel - mean)/std if std > 1e-8 else channel - mean
            elif method == 'minmax':
                min_val, max_val = np.min(channel), np.max(channel)
                X_norm[i, j, :] = (channel - min_val)/(max_val - min_val) if max_val - min_val > 1e-8 else 0.0
    logger.info(f"Applied {method} normalization")
    return X_norm

# --------------------------
# Preprocess single subject
# --------------------------
def preprocess_subject(subject_path: Union[str, Path], config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    data = load_wesad_subject(subject_path)
    signals = extract_chest_signals(data)
    filtered = filter_signals(signals, config)
    target_rate = config.get('sampling_rate', 32)
    resampled = resample_all_signals(filtered, target_rate)
    X, y = create_windows(
        resampled,
        window_size=config.get('window_size', 10.0),
        overlap=config.get('overlap', 0.0),
        sampling_rate=target_rate,
        min_label_consistency=config.get('quality', {}).get('min_label_consistency', 0.8)
    )
    X = normalize_windows(X, method=config.get('normalize', 'zscore'))
    return X, y

# --------------------------
# Preprocess entire dataset
# --------------------------
def preprocess_dataset(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Dict,
    subjects: Optional[List[str]] = None
) -> Dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if subjects is None:
        subject_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith('S')])
    else:
        subject_dirs = [input_dir / s for s in subjects]
    all_X, all_y, subject_stats = [], [], {}
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        pkl_file = subject_dir / f"{subject_id}.pkl"
        if not pkl_file.exists():
            logger.warning(f"Skipping {subject_id}: pickle file not found")
            continue
        try:
            logger.info(f"Processing {subject_id}...")
            X, y = preprocess_subject(pkl_file, config)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                unique, counts = np.unique(y, return_counts=True)
                subject_stats[subject_id] = {'windows': len(y), 'class_distribution': dict(zip(unique.tolist(), counts.tolist()))}
                logger.info(f"{subject_id}: {len(y)} windows")
            else:
                logger.warning(f"{subject_id}: No valid windows created")
        except Exception as e:
            logger.error(f"Error processing {subject_id}: {e}")
            continue
    if len(all_X) == 0:
        raise ValueError("No data was processed successfully")
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    np.save(output_dir / 'X.npy', X)
    np.save(output_dir / 'y.npy', y)
    logger.info(f"Saved X.npy shape: {X.shape}")
    logger.info(f"Saved y.npy shape: {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique.tolist(), counts.tolist()))
    summary = {
        'total_windows': len(y),
        'window_shape': X.shape[1:],
        'n_channels': X.shape[1],
        'window_samples': X.shape[2],
        'class_distribution': class_distribution,
        'subjects_processed': list(subject_stats.keys()),
        'subject_stats': subject_stats
    }
    return summary
