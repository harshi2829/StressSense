"""
PyTorch Data Loader for WESAD Stress Detection
===============================================
Dataset and DataLoader classes for training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, List
import os


class StressDataset(Dataset):
    """
    PyTorch Dataset for stress detection.
    
    Loads preprocessed windows (X.npy) and labels (y.npy).
    """
    
    def __init__(
        self,
        data_dir: str,
        x_file: str = "X.npy",
        y_file: str = "y.npy",
        transform: Optional[callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing processed data
            x_file: Filename for input windows
            y_file: Filename for labels
            transform: Optional transform to apply to samples
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load data
        x_path = os.path.join(data_dir, x_file)
        y_path = os.path.join(data_dir, y_file)
        
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"Data file not found: {x_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Label file not found: {y_path}")
        
        self.X = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.int64)
        
        # Validate shapes
        if len(self.X) != len(self.y):
            raise ValueError(f"X and y length mismatch: {len(self.X)} vs {len(self.y)}")
        
        print(f"Loaded dataset: {len(self.X)} samples")
        print(f"  X shape: {self.X.shape}")
        print(f"  y shape: {self.y.shape}")
        print(f"  Classes: {np.unique(self.y)}")
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return torch.from_numpy(x), torch.tensor(y)
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced data.
        
        Returns:
            Tensor of class weights
        """
        classes, counts = np.unique(self.y, return_counts=True)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(classes)
        return torch.FloatTensor(weights)
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get the distribution of classes.
        
        Returns:
            Dictionary mapping class labels to counts
        """
        classes, counts = np.unique(self.y, return_counts=True)
        return dict(zip(classes.tolist(), counts.tolist()))


class SubjectDataset(Dataset):
    """
    Dataset that loads data per subject for leave-one-subject-out validation.
    """
    
    def __init__(
        self,
        data_dir: str,
        subjects: Optional[List[str]] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize per-subject dataset.
        
        Args:
            data_dir: Directory containing per-subject processed data
            subjects: List of subject IDs to include (None = all)
            transform: Optional transform
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Find subject files
        if subjects is None:
            subjects = self._find_subjects()
        
        self.subjects = subjects
        self.X_list = []
        self.y_list = []
        self.subject_indices = []
        
        # Load each subject's data
        for subj in subjects:
            x_path = os.path.join(data_dir, f"{subj}_X.npy")
            y_path = os.path.join(data_dir, f"{subj}_y.npy")
            
            if os.path.exists(x_path) and os.path.exists(y_path):
                x = np.load(x_path).astype(np.float32)
                y = np.load(y_path).astype(np.int64)
                
                start_idx = len(self.X_list)
                self.X_list.append(x)
                self.y_list.append(y)
                self.subject_indices.append((subj, start_idx, start_idx + len(x)))
        
        # Concatenate all data
        if self.X_list:
            self.X = np.concatenate(self.X_list, axis=0)
            self.y = np.concatenate(self.y_list, axis=0)
        else:
            self.X = np.array([])
            self.y = np.array([])
        
        print(f"Loaded {len(subjects)} subjects, {len(self.X)} total samples")
    
    def _find_subjects(self) -> List[str]:
        """Find all subject files in data directory."""
        subjects = []
        for f in os.listdir(self.data_dir):
            if f.endswith("_X.npy"):
                subj = f.replace("_X.npy", "")
                subjects.append(subj)
        return sorted(subjects)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return torch.from_numpy(x), torch.tensor(y)
    
    def get_subject_indices(self, subject: str) -> Optional[Tuple[int, int]]:
        """Get start and end indices for a specific subject."""
        for subj, start, end in self.subject_indices:
            if subj == subject:
                return (start, end)
        return None


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Directory containing X.npy and y.npy
        batch_size: Batch size for training
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducibility
        stratify: Whether to stratify splits by class
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load full dataset
    dataset = StressDataset(data_dir)
    
    # Get indices
    indices = np.arange(len(dataset))
    labels = dataset.y if stratify else None
    
    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=random_seed,
        stratify=labels
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    temp_labels = dataset.y[temp_idx] if stratify else None
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        random_state=random_seed,
        stratify=temp_labels
    )
    
    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    print(f"\nData split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_class_weights(data_loader: DataLoader) -> torch.Tensor:
    """
    Compute class weights from a data loader.
    
    Args:
        data_loader: DataLoader to compute weights from
        
    Returns:
        Tensor of class weights
    """
    all_labels = []
    for _, labels in data_loader:
        all_labels.extend(labels.numpy())
    
    labels = np.array(all_labels)
    classes, counts = np.unique(labels, return_counts=True)
    
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)
    
    return torch.FloatTensor(weights)
