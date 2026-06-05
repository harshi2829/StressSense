#!/usr/bin/env python3
"""
Prediction Script for WESAD Stress Detection
=============================================
Load a trained model and make predictions on new data.

Usage:
    python predict.py --model models/stress_cnn.pth --input sample_data.npy
    python predict.py --model models/stress_cnn.pth --data_dir data_processed/
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

from models.cnn_model import StressCNN, load_model
from utils.signal_filters import (
    filter_ecg, filter_eda, filter_temp, filter_acc,
    resample_signal, normalize_zscore, compute_acc_magnitude
)


class StressPredictor:
    """
    Stress prediction using trained StressCNN model.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto'
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device for inference ('auto', 'cpu', 'cuda', 'mps')
        """
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = load_model(model_path, str(self.device))
        self.model.eval()
        
        # Class labels
        self.class_names = ['Non-Stress', 'Stress']
    
    def preprocess_signals(
        self,
        ecg: np.ndarray,
        eda: np.ndarray,
        temp: np.ndarray,
        acc: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        fs_ecg: float = 700.0,
        fs_eda: float = 4.0,
        fs_temp: float = 4.0,
        fs_acc: float = 32.0,
        target_fs: float = 32.0
    ) -> np.ndarray:
        """
        Preprocess raw signals for prediction.
        
        Args:
            ecg: Raw ECG signal
            eda: Raw EDA signal
            temp: Raw temperature signal
            acc: Accelerometer data (magnitude or 3-axis tuple)
            fs_*: Sampling frequencies for each signal
            target_fs: Target sampling frequency
            
        Returns:
            Preprocessed signal array of shape (channels, time_steps)
        """
        # Filter signals
        ecg_filt = filter_ecg(ecg, fs_ecg)
        eda_filt = filter_eda(eda, fs_eda)
        temp_filt = filter_temp(temp)
        
        # Handle accelerometer
        if isinstance(acc, tuple) and len(acc) == 3:
            acc_mag = compute_acc_magnitude(acc[0], acc[1], acc[2])
        else:
            acc_mag = acc
        acc_filt = filter_acc(acc_mag)
        
        # Resample to target frequency
        ecg_res = resample_signal(ecg_filt, fs_ecg, target_fs)
        eda_res = resample_signal(eda_filt, fs_eda, target_fs)
        temp_res = resample_signal(temp_filt, fs_temp, target_fs)
        acc_res = resample_signal(acc_filt, fs_acc, target_fs)
        
        # Ensure same length
        min_len = min(len(ecg_res), len(eda_res), len(temp_res), len(acc_res))
        
        # Normalize
        ecg_norm = normalize_zscore(ecg_res[:min_len])
        eda_norm = normalize_zscore(eda_res[:min_len])
        temp_norm = normalize_zscore(temp_res[:min_len])
        acc_norm = normalize_zscore(acc_res[:min_len])
        
        # Stack channels
        signals = np.stack([ecg_norm, eda_norm, temp_norm, acc_norm], axis=0)
        
        return signals.astype(np.float32)
    
    def predict(self, x: np.ndarray) -> Dict:
        """
        Make prediction on preprocessed data.
        
        Args:
            x: Input array of shape (channels, time_steps) or (batch, channels, time_steps)
            
        Returns:
            Dictionary with prediction results
        """
        # Handle single sample
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        
        # Convert to tensor
        x_tensor = torch.from_numpy(x).float().to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
        
        # Convert to numpy
        predictions = preds.cpu().numpy()
        probabilities = probs.cpu().numpy()
        
        results = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'class_names': [self.class_names[p] for p in predictions],
            'confidence': [float(probabilities[i, predictions[i]]) for i in range(len(predictions))]
        }
        
        return results
    
    def predict_batch(self, data_dir: str) -> Dict:
        """
        Make predictions on preprocessed data directory.
        
        Args:
            data_dir: Directory containing X.npy
            
        Returns:
            Dictionary with batch prediction results
        """
        x_path = os.path.join(data_dir, 'X.npy')
        
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"Data file not found: {x_path}")
        
        X = np.load(x_path).astype(np.float32)
        print(f"Loaded {len(X)} samples from {x_path}")
        
        results = self.predict(X)
        
        # Load labels if available
        y_path = os.path.join(data_dir, 'y.npy')
        if os.path.exists(y_path):
            y = np.load(y_path)
            results['labels'] = y.tolist()
            
            # Compute accuracy
            accuracy = np.mean(np.array(results['predictions']) == y)
            results['accuracy'] = float(accuracy)
            print(f"Accuracy: {accuracy:.4f}")
        
        return results
    
    def predict_from_raw(
        self,
        ecg: np.ndarray,
        eda: np.ndarray,
        temp: np.ndarray,
        acc: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        window_size: float = 10.0,
        target_fs: float = 32.0
    ) -> Dict:
        """
        Make predictions from raw signals.
        
        Args:
            ecg, eda, temp, acc: Raw physiological signals
            window_size: Window size in seconds
            target_fs: Target sampling frequency
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        signals = self.preprocess_signals(ecg, eda, temp, acc, target_fs=target_fs)
        
        # Window the data
        window_samples = int(window_size * target_fs)
        n_windows = signals.shape[1] // window_samples
        
        if n_windows == 0:
            raise ValueError(f"Signal too short for {window_size}s windows")
        
        windows = []
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            windows.append(signals[:, start:end])
        
        windows = np.array(windows)
        
        # Predict
        results = self.predict(windows)
        results['n_windows'] = n_windows
        results['window_size'] = window_size
        
        # Aggregate prediction
        stress_ratio = np.mean(np.array(results['predictions']))
        results['stress_ratio'] = float(stress_ratio)
        results['overall_prediction'] = 'Stress' if stress_ratio > 0.5 else 'Non-Stress'
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Stress Detection Prediction")
    parser.add_argument('--model', type=str, default='models/stress_cnn.pth',
                        help='Path to trained model')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input numpy file (channels x time_steps)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing X.npy (and optionally y.npy)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for inference (auto, cpu, cuda, mps)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for predictions (JSON format)')
    args = parser.parse_args()
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    # Initialize predictor
    predictor = StressPredictor(args.model, args.device)
    
    # Make predictions
    if args.input:
        # Single file prediction
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return
        
        X = np.load(args.input).astype(np.float32)
        print(f"Loaded input with shape: {X.shape}")
        
        results = predictor.predict(X)
        
    elif args.data_dir:
        # Batch prediction
        results = predictor.predict_batch(args.data_dir)
        
    else:
        # Demo with random data
        print("\nNo input provided, running demo with random data...")
        X = np.random.randn(5, 4, 320).astype(np.float32)  # 5 samples, 4 channels, 10s @ 32Hz
        results = predictor.predict(X)
    
    # Print results
    print("\n" + "=" * 60)
    print("Prediction Results")
    print("=" * 60)
    
    for i, (pred, prob, conf) in enumerate(zip(
        results['predictions'],
        results['probabilities'],
        results['confidence']
    )):
        class_name = predictor.class_names[pred]
        print(f"Sample {i+1}: {class_name} (confidence: {conf:.4f})")
        print(f"  Probabilities: Non-Stress={prob[0]:.4f}, Stress={prob[1]:.4f}")
    
    # Save results if output path provided
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
