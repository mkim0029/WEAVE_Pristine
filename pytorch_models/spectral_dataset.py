#!/usr/bin/env python3
"""
PyTorch Dataset and DataLoader Classes for WEAVE Spectral Data
==============================================================

This module provides PyTorch-compatible Dataset and DataLoader classes for
loading and batching processed spectral data for CNN training. The classes
handle HDF5 data loading, spectral transformations, and efficient batching
for deep learning workflows.

Key Features:
- Memory-efficient loading of large spectral datasets
- Flexible data transformations and augmentations
- Support for train/validation/test splits
- Integration with PyTorch training loops

Usage:
    from spectral_dataset import SpectralDataset, create_spectral_dataloaders
    
    # Load dataset
    dataset = SpectralDataset('processed_spectra.h5')
    
    # Create DataLoaders
    train_loader, val_loader = create_spectral_dataloaders(
        dataset, batch_size=32, train_split=0.8
    )
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Union
import warnings


class SpectralDataset(Dataset):
    """
    PyTorch Dataset for loading processed spectral data from HDF5 files.
    
    This dataset class handles loading of preprocessed spectral data and provides
    efficient access for training CNNs. It supports various data formats and
    transformations commonly used in astronomical spectral analysis.
    
    Parameters
    ----------
    hdf5_filepath : str or Path
        Path to the HDF5 file containing processed spectral data
    flux_key : str, optional
        HDF5 dataset key for flux data (default: 'flux_normalized')
    wavelength_key : str, optional
        HDF5 dataset key for wavelength data (default: 'wavelength')
    transform : callable, optional
        Transform function to apply to each spectrum (default: None)
    target_transform : callable, optional
        Transform function to apply to targets (default: None)
    load_targets : bool, optional
        Whether to load target data (for supervised learning, default: False)
    target_key : str, optional
        HDF5 dataset key for target data (default: 'targets')
    device : str, optional
        Device to load tensors on ('cpu', 'cuda', default: 'cpu')
    dtype : torch.dtype, optional
        Data type for tensors (default: torch.float32)

    *transform and target_transform can be used to augment and/or preprocess the data

    """
    
    def __init__(
        self,
        hdf5_filepath: Union[str, Path],
        flux_key: str = 'flux_normalized',
        wavelength_key: str = 'wavelength',
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        load_targets: bool = False,
        target_key: str = 'targets',
        use_target_indices: Optional[list[int]] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        self.hdf5_filepath = Path(hdf5_filepath)
        self.flux_key = flux_key
        self.wavelength_key = wavelength_key
        self.transform = transform
        self.target_transform = target_transform
        self.load_targets = load_targets
        self.target_key = target_key
        self.use_target_indices = use_target_indices
        self.device = device
        self.dtype = dtype

        self.target_mean = None
        self.target_std = None
        
        # Validate file exists
        if not self.hdf5_filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_filepath}")
        
        # Load metadata and validate dataset structure
        self._load_metadata()
        self._validate_data()
        
        print(f"SpectralDataset initialized:")
        print(f"  File: {self.hdf5_filepath}")
        print(f"  Spectra: {self.n_spectra}")
        print(f"  Wavelength points: {self.n_wavelengths}")
        print(f"  Wavelength range: {self.wavelength_min:.2f} - {self.wavelength_max:.2f} Å")
        print(f"  Flux key: {self.flux_key}")
        print(f"  Load targets: {self.load_targets}")
        if self.load_targets:
            print(f"  Target key: {self.target_key}")
            if self.use_target_indices:
                print(f"  Using target indices: {self.use_target_indices}")
        print(f"  Device: {self.device}")
        
    def _load_metadata(self):
        """Load metadata from HDF5 file without loading full datasets."""
        with h5py.File(self.hdf5_filepath, 'r') as f:
            # Check available datasets
            self.available_keys = list(f.keys())
            
            # Validate required keys exist
            if self.flux_key not in f:
                raise KeyError(f"Flux key '{self.flux_key}' not found in HDF5 file.")
            if self.wavelength_key not in f:
                raise KeyError(f"Wavelength key '{self.wavelength_key}' not found in HDF5 file.")
            
            # Load metadata
            flux_shape = f[self.flux_key].shape
            self.wavelength = f[self.wavelength_key][:]
            
            self.n_spectra = flux_shape[0]
            self.n_wavelengths = flux_shape[1]
            self.wavelength_min = self.wavelength[0]
            self.wavelength_max = self.wavelength[-1]

            if self.load_targets:
                if self.target_key not in f:
                    raise KeyError(f"Target key '{self.target_key}' not found in HDF5 file.")
                
                # Determine the number of targets based on the user's selection
                if self.use_target_indices:
                    self.n_targets = len(self.use_target_indices)
                else:
                    # If no indices are specified, use all targets
                    self.n_targets = f[self.target_key].shape[1]
            
            # Load any additional metadata from attributes
            self.metadata = dict(f.attrs)
    
    def _validate_data(self):
        """Validate data consistency."""
        if len(self.wavelength) != self.n_wavelengths:
            raise ValueError(f"Wavelength length ({len(self.wavelength)}) doesn't match expected ({self.n_wavelengths})")
        
        if self.n_spectra == 0:
            raise ValueError("No spectra found in dataset")
        
        print(f"Data validation passed")

    def set_target_stats(self, mean: np.ndarray, std: np.ndarray):
        """
        Set the mean and std for target normalization.

        Parameters
        ----------
        mean : np.ndarray
            Mean of each target parameter.
        std : np.ndarray
            Standard deviation of each target parameter.
        """
        print("Setting target normalization statistics...")
        self.target_mean = torch.from_numpy(mean).to(dtype=self.dtype, device=self.device)
        self.target_std = torch.from_numpy(std).to(dtype=self.dtype, device=self.device)
        # Add a small epsilon to std to avoid division by zero
        self.target_std[self.target_std == 0] = 1e-6
        print(f"  Mean: {self.target_mean.cpu().numpy()}")
        print(f"  Std: {self.target_std.cpu().numpy()}")

    def __len__(self) -> int:
        """Return the number of spectra in the dataset."""
        return self.n_spectra
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a single spectrum (and target if applicable) by index.
        
        Parameters
        ----------
        idx : int
            Index of the spectrum to retrieve
            
        Returns
        -------
        spectrum : torch.Tensor or tuple
            If load_targets=False: returns torch.Tensor of shape (n_wavelengths,)
            If load_targets=True: returns tuple (spectrum, target)
        """
        if idx >= self.n_spectra or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset with {self.n_spectra} spectra")
        
        # Load spectrum from HDF5 file (as numpy), then convert to torch.Tensor
        with h5py.File(self.hdf5_filepath, 'r') as f:
            spectrum_np = f[self.flux_key][idx].astype(np.float32)

        # Convert spectrum to torch tensor and move to device
        spectrum = torch.from_numpy(spectrum_np).to(dtype=self.dtype, device=self.device)

        # Apply spectrum transform if provided (transforms should accept torch.Tensor)
        if self.transform is not None:
            spectrum = self.transform(spectrum)

        if self.load_targets:
            # Load targets (as numpy) and select subset if requested
            with h5py.File(self.hdf5_filepath, 'r') as f:
                target_np = f[self.target_key][idx]

            if self.use_target_indices:
                target_np = target_np[self.use_target_indices]

            # Convert target to tensor and apply normalization/transform
            target = torch.from_numpy(target_np).to(dtype=self.dtype, device=self.device)

            if self.target_mean is not None and self.target_std is not None:
                target = (target - self.target_mean) / self.target_std
            elif self.target_transform is not None:
                target = self.target_transform(target)

            return spectrum, target
        else:
            return spectrum

    def inverse_transform_target(self, target: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse normalization transform to a target tensor.

        Parameters
        ----------
        target : torch.Tensor
            A normalized target tensor.

        Returns
        -------
        torch.Tensor
            The target tensor in its original physical scale.
        """
        if self.target_mean is None or self.target_std is None:
            warnings.warn("Target stats not set. Returning original tensor.")
            return target
        
        return target * self.target_std + self.target_mean

    def get_all_targets(self) -> Optional[np.ndarray]:
        """
        Load all targets from the HDF5 file into memory.

        Returns
        -------
        np.ndarray or None
            A numpy array of all targets, or None if not loading targets.
        """
        if not self.load_targets:
            return None
        with h5py.File(self.hdf5_filepath, 'r') as f:
            all_targets = f[self.target_key][:]
            if self.use_target_indices:
                return all_targets[:, self.use_target_indices]
            return all_targets
    
    def get_wavelength_tensor(self) -> torch.Tensor:
        """Get wavelength array as PyTorch tensor."""
        return torch.from_numpy(self.wavelength).to(dtype=self.dtype, device=self.device)
    
    def get_spectrum_range(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Get a range of spectra efficiently.
        
        Parameters
        ----------
        start_idx : int
            Starting index (inclusive)
        end_idx : int
            Ending index (exclusive)
            
        Returns
        -------
        spectra : torch.Tensor
            Tensor of shape (end_idx - start_idx, n_wavelengths)
        """
        if start_idx < 0 or end_idx > self.n_spectra or start_idx >= end_idx:
            raise ValueError(f"Invalid range [{start_idx}:{end_idx}] for dataset with {self.n_spectra} spectra")
        
        with h5py.File(self.hdf5_filepath, 'r') as f:
            spectra = f[self.flux_key][start_idx:end_idx].astype(np.float32)
        
        return torch.from_numpy(spectra).to(dtype=self.dtype, device=self.device)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Compute dataset statistics (mean, std, min, max).
        
        Returns
        -------
        stats : dict
            Dictionary containing statistical information
        """
        print("Computing dataset statistics...")
        
        with h5py.File(self.hdf5_filepath, 'r') as f:
            flux_data = f[self.flux_key]
            
            # Compute statistics in chunks to handle large datasets
            chunk_size = min(1000, self.n_spectra)
            all_values = []
            
            for i in range(0, self.n_spectra, chunk_size):
                end_idx = min(i + chunk_size, self.n_spectra)
                chunk = flux_data[i:end_idx]
                all_values.append(chunk.flatten())
            
            all_values = np.concatenate(all_values)
        
        stats = {
            'mean': float(np.mean(all_values)),
            'std': float(np.std(all_values)),
            'min': float(np.min(all_values)),
            'max': float(np.max(all_values)),
            'median': float(np.median(all_values)),
            'count': len(all_values)
        }
        
        print(f"Dataset statistics computed over {stats['count']} values")
        return stats


class SpectralTransforms:
    """Collection of common transforms for spectral data."""
    
    @staticmethod
    def normalize_to_unit_range(spectrum: torch.Tensor) -> torch.Tensor:
        """Normalize spectrum to [0, 1] range."""
        min_val = torch.min(spectrum)
        max_val = torch.max(spectrum)
        if max_val > min_val:
            return (spectrum - min_val) / (max_val - min_val)
        else:
            return spectrum
    
    @staticmethod
    def standardize(spectrum: torch.Tensor) -> torch.Tensor:
        """Standardize spectrum to zero mean and unit variance."""
        mean = torch.mean(spectrum)
        std = torch.std(spectrum)
        if std > 0:
            return (spectrum - mean) / std
        else:
            return spectrum - mean
    
    @staticmethod
    def add_gaussian_noise(noise_std: float = 0.01):
        """Return transform that adds Gaussian noise."""
        def transform(spectrum: torch.Tensor) -> torch.Tensor:
            noise = torch.randn_like(spectrum) * noise_std
            return spectrum + noise
        return transform
    
    @staticmethod
    def spectral_smoothing(kernel_size: int = 3):
        """Return transform that applies spectral smoothing."""
        def transform(spectrum: torch.Tensor) -> torch.Tensor:
            # Simple moving average smoothing
            if kernel_size <= 1:
                return spectrum
            
            padding = kernel_size // 2
            padded = torch.nn.functional.pad(spectrum, (padding, padding), mode='reflect')
            kernel = torch.ones(kernel_size) / kernel_size
            smoothed = torch.nn.functional.conv1d(
                padded.unsqueeze(0).unsqueeze(0), 
                kernel.unsqueeze(0).unsqueeze(0)
            ).squeeze()
            return smoothed
        return transform


def create_spectral_dataloaders(
    dataset: SpectralDataset,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.2,
    test_split: float = 0.0,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    random_seed: Optional[int] = 42
) -> Tuple[DataLoader, ...]:
    """
    Create train/validation/test DataLoaders from a SpectralDataset.
    
    Parameters
    ----------
    dataset : SpectralDataset
        The dataset to split into DataLoaders
    batch_size : int, optional
        Batch size for DataLoaders (default: 32)
    train_split : float, optional
        Fraction of data for training (default: 0.8)
    val_split : float, optional
        Fraction of data for validation (default: 0.2)
    test_split : float, optional
        Fraction of data for testing (default: 0.0)
    shuffle : bool, optional
        Whether to shuffle the data (default: True)
    num_workers : int, optional
        Number of worker processes for data loading (default: 0)
    pin_memory : bool, optional
        Whether to pin memory for GPU transfer (default: False)
    random_seed : int, optional
        Random seed for reproducible splits (default: 42)
        
    Returns
    -------
    dataloaders : tuple
        Tuple of DataLoaders (train, val) or (train, val, test) depending on splits
    """
    # Validate splits
    total_split = train_split + val_split + test_split
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {total_split}")
    
    # Set random seed for reproducible splits
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    print(f"Creating DataLoaders:")
    print(f"  Total dataset size: {dataset_size}")
    print(f"  Train size: {train_size}")
    print(f"  Validation size: {val_size}")
    if test_size > 0:
        print(f"  Test size: {test_size}")
    print(f"  Batch size: {batch_size}")
    
    # Split the dataset
    if test_size > 0:
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        datasets = [train_dataset, val_dataset, test_dataset]
        split_names = ['train', 'val', 'test']
    else:
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        datasets = [train_dataset, val_dataset]
        split_names = ['train', 'val']
    
    # Create DataLoaders
    dataloaders = []
    for dataset_split, name in zip(datasets, split_names):
        shuffle_split = shuffle and (name == 'train')  # Only shuffle training data
        
        dataloader = DataLoader(
            dataset_split,
            batch_size=batch_size,
            shuffle=shuffle_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(name == 'train')  # Drop last incomplete batch for training
        )
        
        dataloaders.append(dataloader)
        print(f"  {name.capitalize()} DataLoader: {len(dataloader)} batches")
    
    return tuple(dataloaders)


def main():
    """Example usage and testing of the spectral dataset classes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SpectralDataset functionality")
    parser.add_argument('--input', '-i', required=True,
                       help='Input HDF5 file with processed spectral data')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for DataLoader testing')
    parser.add_argument('--device', default='cpu',
                       help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SPECTRAL DATASET TESTING")
    print("=" * 60)
    
    try:
        # Create dataset
        print("\n1. Creating SpectralDataset")
        print("-" * 30)
        dataset = SpectralDataset(
            args.input,
            device=args.device,
            transform=SpectralTransforms.standardize
        )
        
        # Test single item access
        print("\n2. Testing single item access")
        print("-" * 30)
        spectrum = dataset[0]
        print(f"First spectrum shape: {spectrum.shape}")
        print(f"First spectrum stats: min={spectrum.min():.3f}, max={spectrum.max():.3f}, mean={spectrum.mean():.3f}")
        
        # Get dataset statistics
        print("\n3. Computing dataset statistics")
        print("-" * 30)
        stats = dataset.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value:.6f}")
        
        # Create DataLoaders
        print("\n4. Creating DataLoaders")
        print("-" * 30)
        train_loader, val_loader = create_spectral_dataloaders(
            dataset,
            batch_size=args.batch_size,
            train_split=0.8,
            val_split=0.2
        )
        
        # Test DataLoader iteration
        print("\n5. Testing DataLoader iteration")
        print("-" * 30)
        for i, batch in enumerate(train_loader):
            print(f"Batch {i+1}: shape={batch.shape}, dtype={batch.dtype}")
            if i >= 2:  # Only test first few batches
                break
        
        print("\n" + "=" * 60)
        print("SPECTRAL DATASET TESTING COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    main()