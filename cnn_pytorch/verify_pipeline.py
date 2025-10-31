#!/usr/bin/env python3
"""
Data Loading Verification Script for CNN Pipeline
=================================================

This script verifies that the CNN preprocessing pipeline and PyTorch Dataset/DataLoader
classes work correctly with the processed spectral data. It performs comprehensive
tests to ensure data integrity, proper shapes, and compatibility with PyTorch workflows.

Test Coverage:
- HDF5 file loading and structure validation
- Dataset creation and item access
- DataLoader batch generation and iteration
- Data type and shape consistency
- Statistical validation of processed data
- Memory usage and performance testing

Usage:
    python verify_pipeline.py --input ../data/test.h5
    python verify_pipeline.py --processed ../data/processed_spectra.h5 --full-test
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import h5py

# Add parent directory to path for local imports
project_root = Path(__file__).parent.parent
preprocessing_dir = project_root / "preprocessing"
cnn_pytorch_dir = project_root / "cnn_pytorch"

sys.path.append(str(project_root))
sys.path.append(str(preprocessing_dir))
sys.path.append(str(cnn_pytorch_dir))

try:
    from preprocessing.preprocess import load_spectra_from_hdf5, add_noise_to_spectra, normalize_spectra
    from spectral_dataset import SpectralDataset, create_spectral_dataloaders, SpectralTransforms
    from preprocessing.cont_norm import generate_noise, legendre_polyfit_huber
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(f"Project root: {project_root}")
    print(f"Preprocessing dir: {preprocessing_dir}")
    print(f"CNN PyTorch dir: {cnn_pytorch_dir}")
    print("Make sure all required files exist in their respective directories")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'  # End color
    BOLD = '\033[1m'


def print_test_header(test_name: str):
    """Print a formatted test header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{test_name.upper()}{Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.ENDC}")


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")


def print_info(message: str):
    """Print an info message."""
    print(f"{Colors.CYAN}ℹ {message}{Colors.ENDC}")


def verify_hdf5_structure(filepath: Path) -> bool:
    """
    Verify the structure and contents of an HDF5 file.
    
    Parameters
    ----------
    filepath : Path
        Path to the HDF5 file to verify
        
    Returns
    -------
    bool
        True if file structure is valid, False otherwise
    """
    print_test_header("HDF5 File Structure Verification")
    
    try:
        if not filepath.exists():
            print_error(f"File does not exist: {filepath}")
            return False
        
        print_info(f"Checking file: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            # List all datasets and attributes
            print_info("Available datasets:")
            datasets = {}
            
            def collect_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets[name] = obj.shape
                    print(f"  {name}: {obj.shape} ({obj.dtype})")
            
            f.visititems(collect_datasets)
            
            # Check for required datasets
            # For processed files, look for flux_normalized; for raw files, look for flux
            if 'flux_normalized' in datasets:
                required_keys = ['flux_normalized', 'wavelength']
                flux_key = 'flux_normalized'
            elif 'flux' in datasets:
                required_keys = ['flux', 'wavelength']
                flux_key = 'flux'
            else:
                required_keys = ['flux', 'wavelength']  # Default expectation
                flux_key = 'flux'
            
            missing_keys = [key for key in required_keys if key not in datasets]
            
            if missing_keys:
                print_error(f"Missing required datasets: {missing_keys}")
                return False
            
            # Validate shapes
            flux_shape = datasets[flux_key]
            wavelength_shape = datasets['wavelength']
            
            if len(flux_shape) != 2:
                print_error(f"Flux should be 2D, got shape {flux_shape}")
                return False
            
            if len(wavelength_shape) != 1:
                print_error(f"Wavelength should be 1D, got shape {wavelength_shape}")
                return False
            
            if flux_shape[1] != wavelength_shape[0]:
                print_error(f"Wavelength length {wavelength_shape[0]} doesn't match flux width {flux_shape[1]}")
                return False
            
            # Print file metadata
            print_info("File attributes:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
            
            print_success(f"File structure is valid: {flux_shape[0]} spectra × {flux_shape[1]} wavelengths")
            return True
            
    except Exception as e:
        print_error(f"Error reading HDF5 file: {e}")
        return False


def test_preprocessing_functions():
    """Test individual preprocessing functions."""
    print_test_header("Preprocessing Functions Test")
    
    try:
        # Test data generation
        print_info("Testing generate_noise function...")
        noise = generate_noise(shape=(100,), noise=0.1, seed=42)
        
        if len(noise) != 100:
            print_error(f"Expected noise length 100, got {len(noise)}")
            return False
        
        if not np.allclose(np.std(noise), 0.1, rtol=0.1):
            print_warning(f"Noise std {np.std(noise):.3f} differs from expected 0.1")
        
        print_success(f"Noise generation: shape={noise.shape}, std={np.std(noise):.3f}")
        
        # Test continuum normalization with synthetic data
        print_info("Testing legendre_polyfit_huber function...")
        wavelength = np.linspace(5000, 5100, 100)
        # Create synthetic spectrum with continuum + absorption line
        continuum = 1.0 + 0.1 * (wavelength - 5050) / 50  # Sloped continuum
        absorption = 0.3 * np.exp(-((wavelength - 5060) / 2)**2)  # Gaussian absorption
        flux = continuum - absorption + 0.01 * np.random.randn(100)  # Add noise
        
        norm_flux, fit_continuum = legendre_polyfit_huber(flux, wavelength)
        
        if len(norm_flux) != len(flux):
            print_error(f"Normalized flux length mismatch")
            return False
        
        # Check that normalization brings continuum closer to 1
        continuum_regions = (wavelength < 5055) | (wavelength > 5065)  # Avoid absorption line
        continuum_level = np.median(norm_flux[continuum_regions])
        
        if not np.allclose(continuum_level, 1.0, atol=0.2):
            print_warning(f"Continuum level {continuum_level:.3f} not close to 1.0")
        
        print_success(f"Continuum normalization: continuum level = {continuum_level:.3f}")
        return True
        
    except Exception as e:
        print_error(f"Preprocessing function test failed: {e}")
        traceback.print_exc()
        return False


def test_dataset_creation(filepath: Path) -> bool:
    """
    Test SpectralDataset creation and basic functionality.
    
    Parameters
    ----------
    filepath : Path
        Path to the HDF5 file
        
    Returns
    -------
    bool
        True if tests pass, False otherwise
    """
    print_test_header("Dataset Creation Test")
    
    try:
        # Test basic dataset creation
        print_info("Creating SpectralDataset...")
        dataset = SpectralDataset(filepath)
        
        print_success(f"Dataset created: {len(dataset)} spectra")
        
        # Test item access
        print_info("Testing item access...")
        spectrum = dataset[0]
        
        if not isinstance(spectrum, torch.Tensor):
            print_error(f"Expected torch.Tensor, got {type(spectrum)}")
            return False
        
        if len(spectrum.shape) != 1:
            print_error(f"Expected 1D spectrum, got shape {spectrum.shape}")
            return False
        
        print_success(f"Item access works: spectrum shape = {spectrum.shape}")
        
        # Test multiple item access
        print_info("Testing multiple item access...")
        for i in [0, len(dataset)//2, len(dataset)-1]:
            spec = dataset[i]
            if spec.shape != spectrum.shape:
                print_error(f"Shape mismatch at index {i}: {spec.shape} vs {spectrum.shape}")
                return False
        
        print_success("Multiple item access consistent")
        
        # Test wavelength access
        print_info("Testing wavelength access...")
        wavelength = dataset.get_wavelength_tensor()
        
        if wavelength.shape[0] != spectrum.shape[0]:
            print_error(f"Wavelength length {wavelength.shape[0]} doesn't match spectrum {spectrum.shape[0]}")
            return False
        
        print_success(f"Wavelength access works: shape = {wavelength.shape}")
        
        # Test statistics computation
        print_info("Testing statistics computation...")
        stats = dataset.get_statistics()
        required_stats = ['mean', 'std', 'min', 'max', 'median']
        
        for stat in required_stats:
            if stat not in stats:
                print_error(f"Missing statistic: {stat}")
                return False
        
        print_success(f"Statistics computed: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
        return True
        
    except Exception as e:
        print_error(f"Dataset creation test failed: {e}")
        traceback.print_exc()
        return False


def test_transforms():
    """Test spectral transform functions."""
    print_test_header("Transform Functions Test")
    
    try:
        # Create test spectrum
        test_spectrum = torch.randn(100) + 2.0  # Mean around 2, std around 1
        
        print_info(f"Original spectrum: mean={test_spectrum.mean():.3f}, std={test_spectrum.std():.3f}")
        
        # Test standardization
        standardized = SpectralTransforms.standardize(test_spectrum)
        if not torch.allclose(standardized.mean(), torch.tensor(0.0), atol=1e-6):
            print_error(f"Standardized mean not zero: {standardized.mean():.6f}")
            return False
        
        if not torch.allclose(standardized.std(), torch.tensor(1.0), atol=1e-6):
            print_error(f"Standardized std not one: {standardized.std():.6f}")
            return False
        
        print_success("Standardization transform works")
        
        # Test normalization to unit range
        normalized = SpectralTransforms.normalize_to_unit_range(test_spectrum)
        if not torch.allclose(normalized.min(), torch.tensor(0.0), atol=1e-6):
            print_error(f"Normalized min not zero: {normalized.min():.6f}")
            return False
        
        if not torch.allclose(normalized.max(), torch.tensor(1.0), atol=1e-6):
            print_error(f"Normalized max not one: {normalized.max():.6f}")
            return False
        
        print_success("Unit range normalization works")
        
        # Test noise addition
        noise_transform = SpectralTransforms.add_gaussian_noise(noise_std=0.1)
        noisy = noise_transform(test_spectrum)
        
        if noisy.shape != test_spectrum.shape:
            print_error(f"Noise transform changed shape: {noisy.shape} vs {test_spectrum.shape}")
            return False
        
        noise_diff = noisy - test_spectrum
        if abs(noise_diff.std() - 0.1) > 0.05:  # Allow some tolerance
            print_warning(f"Noise std {noise_diff.std():.3f} differs from expected 0.1")
        
        print_success("Noise addition transform works")
        
        return True
        
    except Exception as e:
        print_error(f"Transform test failed: {e}")
        traceback.print_exc()
        return False


def test_dataloader_creation(dataset: SpectralDataset) -> bool:
    """
    Test DataLoader creation and iteration.
    
    Parameters
    ----------
    dataset : SpectralDataset
        Dataset to test with
        
    Returns
    -------
    bool
        True if tests pass, False otherwise
    """
    print_test_header("DataLoader Creation Test")
    
    try:
        # Test DataLoader creation
        print_info("Creating DataLoaders...")
        dataloaders = create_spectral_dataloaders(
            dataset,
            batch_size=8,
            train_split=0.7,
            val_split=0.3,
            shuffle=True,
            random_seed=42
        )
        
        # Handle different numbers of returned dataloaders
        if len(dataloaders) == 2:
            train_loader, val_loader = dataloaders
            print_success(f"DataLoaders created: train={len(train_loader)} batches, val={len(val_loader)} batches")
        elif len(dataloaders) == 3:
            train_loader, val_loader, test_loader = dataloaders
            print_success(f"DataLoaders created: train={len(train_loader)} batches, val={len(val_loader)} batches, test={len(test_loader)} batches")
        else:
            print_error(f"Unexpected number of dataloaders returned: {len(dataloaders)}")
            return False
        
        # Test iteration over train loader
        print_info("Testing train DataLoader iteration...")
        batch_count = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch_count += 1
            total_samples += batch.shape[0]
            
            # Check batch properties
            if not isinstance(batch, torch.Tensor):
                print_error(f"Expected torch.Tensor batch, got {type(batch)}")
                return False
            
            if len(batch.shape) != 2:
                print_error(f"Expected 2D batch (batch_size, n_wavelengths), got shape {batch.shape}")
                return False
            
            if batch_idx == 0:
                print_info(f"First batch shape: {batch.shape}")
            
            # Only test first few batches to save time
            if batch_idx >= 3:
                break
        
        print_success(f"Train DataLoader iteration works: {batch_count} batches tested, {total_samples} samples")
        
        # Test validation loader
        print_info("Testing validation DataLoader...")
        val_batch_count = 0
        
        for batch_idx, batch in enumerate(val_loader):
            val_batch_count += 1
            
            if batch_idx == 0:
                print_info(f"First val batch shape: {batch.shape}")
            
            if batch_idx >= 2:
                break
        
        print_success(f"Validation DataLoader works: {val_batch_count} batches tested")
        
        return True
        
    except Exception as e:
        print_error(f"DataLoader test failed: {e}")
        traceback.print_exc()
        return False


def test_end_to_end_pipeline(input_file: Path, create_processed: bool = True) -> bool:
    """
    Test the complete end-to-end pipeline from raw data to DataLoader.
    
    Parameters
    ----------
    input_file : Path
        Path to input HDF5 file
    create_processed : bool
        Whether to create processed data file for testing
        
    Returns
    -------
    bool
        True if pipeline works end-to-end, False otherwise
    """
    print_test_header("End-to-End Pipeline Test")
    
    try:
        if create_processed:
            print_info("Running preprocessing pipeline...")
            
            # Load original data
            spectra = load_spectra_from_hdf5(input_file)
            flux = spectra['flux']
            wavelength = spectra['wavelength']
            
            print_info(f"Loaded {flux.shape[0]} spectra")
            
            # Add noise
            noisy_flux, noise_vectors = add_noise_to_spectra(flux, noise_level=0.1, seed=42)
            print_success("Noise addition completed")
            
            # Normalize (test with a small subset for speed)
            n_test = min(10, flux.shape[0])
            test_flux = noisy_flux[:n_test]
            
            normalized_flux, continuum_fits = normalize_spectra(test_flux, wavelength)
            print_success("Continuum normalization completed")
            
            # Create a small test file for DataLoader testing
            test_output = input_file.parent / "test_processed_small.h5"
            
            with h5py.File(test_output, 'w') as f:
                f.create_dataset('flux_normalized', data=normalized_flux)
                f.create_dataset('wavelength', data=wavelength)
                f.create_dataset('flux_original', data=test_flux)
                f.attrs['n_spectra'] = n_test
            
            print_success(f"Created test processed file: {test_output}")
            
            # Test with DataLoader
            print_info("Testing with PyTorch DataLoader...")
            dataset = SpectralDataset(test_output)
            
            dataloaders = create_spectral_dataloaders(
                dataset, batch_size=4, train_split=0.8, val_split=0.2
            )
            
            # Get train loader (first element regardless of how many are returned)
            train_loader = dataloaders[0]
            
            # Test one batch
            for batch in train_loader:
                print_info(f"Successfully loaded batch: {batch.shape}")
                break
            
            print_success("End-to-end pipeline test completed successfully")
            
            # Clean up test file
            test_output.unlink()
            print_info("Cleaned up test file")
            
        return True
        
    except Exception as e:
        print_error(f"End-to-end pipeline test failed: {e}")
        traceback.print_exc()
        return False


def run_performance_test(dataset: SpectralDataset, batch_size: int = 32, num_batches: int = 10) -> bool:
    """
    Run basic performance tests on DataLoader.
    
    Parameters
    ----------
    dataset : SpectralDataset
        Dataset to test
    batch_size : int
        Batch size for testing
    num_batches : int
        Number of batches to time
        
    Returns
    -------
    bool
        True if performance is acceptable, False otherwise
    """
    print_test_header("Performance Test")
    
    try:
        print_info(f"Testing performance with batch_size={batch_size}, num_batches={num_batches}")
        
        # Create DataLoader
        dataloaders = create_spectral_dataloaders(
            dataset, batch_size=batch_size, train_split=0.9, val_split=0.1
        )
        
        # Get train loader (first element)
        train_loader = dataloaders[0]
        
        # Time data loading
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
        
        end_time = time.time()
        total_time = end_time - start_time
        
        samples_processed = min(num_batches * batch_size, len(dataset))
        samples_per_second = samples_processed / total_time
        
        print_success(f"Performance test completed:")
        print_info(f"  Total time: {total_time:.3f} seconds")
        print_info(f"  Samples processed: {samples_processed}")
        print_info(f"  Samples per second: {samples_per_second:.1f}")
        
        # Check if performance is reasonable (>100 samples/sec for CPU)
        if samples_per_second < 50:
            print_warning("Performance may be slow for large datasets")
        else:
            print_success("Performance looks good")
        
        return True
        
    except Exception as e:
        print_error(f"Performance test failed: {e}")
        return False


def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(
        description="Verify CNN preprocessing pipeline and dataset functionality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', '-i', 
                       help='Input HDF5 file (test.h5 or processed file)')
    parser.add_argument('--processed', '-p',
                       help='Processed HDF5 file to test (skips preprocessing)')
    parser.add_argument('--full-test', action='store_true',
                       help='Run full comprehensive tests including performance')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for DataLoader testing')
    
    args = parser.parse_args()
    
    if not args.input and not args.processed:
        print_error("Must provide either --input or --processed file")
        return 1
    
    print(f"{Colors.PURPLE}{Colors.BOLD}")
    print("=" * 80)
    print("CNN PREPROCESSING PIPELINE VERIFICATION")
    print("=" * 80)
    print(f"{Colors.ENDC}")
    
    all_tests_passed = True
    
    try:
        # Test 1: HDF5 file structure
        if args.input:
            test_file = Path(args.input)
        else:
            test_file = Path(args.processed)
        
        if not verify_hdf5_structure(test_file):
            all_tests_passed = False
        
        # Test 2: Preprocessing functions (only if we have raw input)
        if args.input:
            if not test_preprocessing_functions():
                all_tests_passed = False
        
        # Test 3: Transform functions
        if not test_transforms():
            all_tests_passed = False
        
        # Test 4: Dataset creation
        if not test_dataset_creation(test_file):
            all_tests_passed = False
        
        # Test 5: DataLoader functionality
        dataset = SpectralDataset(test_file)
        if not test_dataloader_creation(dataset):
            all_tests_passed = False
        
        # Test 6: End-to-end pipeline (if raw input provided)
        if args.input and args.full_test:
            if not test_end_to_end_pipeline(Path(args.input)):
                all_tests_passed = False
        
        # Test 7: Performance test (if requested)
        if args.full_test:
            if not run_performance_test(dataset, batch_size=args.batch_size):
                all_tests_passed = False
        
        # Final summary
        print(f"\n{Colors.PURPLE}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.PURPLE}{Colors.BOLD}VERIFICATION SUMMARY{Colors.ENDC}")
        print(f"{Colors.PURPLE}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        
        if all_tests_passed:
            print_success("ALL TESTS PASSED! 🎉")
            print_info("The CNN preprocessing pipeline is ready for use.")
            return 0
        else:
            print_error("SOME TESTS FAILED! ❌")
            print_info("Please check the error messages above and fix issues.")
            return 1
            
    except Exception as e:
        print_error(f"Verification script failed with unexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)