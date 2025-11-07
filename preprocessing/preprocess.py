#!/usr/bin/env python3
"""
CNN Preprocessing Pipeline for WEAVE Spectral Data
=================================================

This script implements the preprocessing pipeline for training a CNN on stellar spectra:
1. Load spectra from test.h5 HDF5 file
2. Add consistent Gaussian noise (noise=0.1, seed=2025) to all spectra
3. Apply continuum normalization using legendre_polyfit_huber method
4. Save processed data for CNN training

The pipeline ensures consistent preprocessing across all spectra while maintaining
data integrity for machine learning applications.

Usage:
    python preprocess.py --input ../data/test.h5 --output ../data/processed_spectra.h5
    python preprocess.py --help  # for full options
"""

import argparse
import h5py
import numpy as np
import sys
from pathlib import Path    

# Add preprocessing directory to path for imports
sys.path.append(str(Path(__file__).parent))
from cont_norm import generate_noise, legendre_polyfit_huber
from hdf5_spectrum_reader import HDF5SpectrumReader


def load_spectra_from_hdf5(filepath):
    """
    Load spectral data from HDF5 file using HDF5SpectrumReader.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file containing spectral data
        
    Returns
    -------
    spectra : dict
        Dictionary containing 'flux', 'wavelength', and metadata
    """
    print(f"Loading spectral data from: {filepath}")
    
    # Use the existing HDF5SpectrumReader to load data
    reader = HDF5SpectrumReader(str(filepath))
    
    # Print dataset summary
    reader.print_dataset_summary()
    
    # Load all spectra data
    print(f"\nLoading flux data for {reader.n_spectra} spectra...")
    
    # Get wavelength grid (same for all spectra)
    wavelength = reader.wavelength_grid
    
    # Load all flux data efficiently
    with h5py.File(filepath, 'r') as f:
        flux = f['spectra/flux'][:]
    
    # Load labels and metadata
    stellar_parameters, param_names = reader.get_labels()
    filenames = reader.filenames
    
    spectra = {
        'flux': flux,
        'wavelength': wavelength,
        'stellar_parameters': stellar_parameters,
        'parameter_names': param_names,
        'filenames': filenames,
        'n_spectra': reader.n_spectra,
        'n_wavelength_points': reader.n_wavelength_points
    }
    
    print(f"Loaded {flux.shape[0]} spectra with {flux.shape[1]} wavelength points each")
    print(f"Wavelength range: {wavelength.min():.2f} - {wavelength.max():.2f} Å")
    print(f"Available stellar parameters: {param_names}")
    
    return spectra


def add_noise_to_spectra(flux_array, noise_level=0.1, seed=2025):
    """
    Add consistent Gaussian noise to all spectra.
    
    Parameters
    ----------
    flux_array : np.ndarray
        2D array of shape (n_spectra, n_wavelengths) containing flux values
    noise_level : float, optional
        Standard deviation of Gaussian noise to add (default: 0.1)
    seed : int, optional
        Random seed for reproducible noise generation (default: 2025)
        
    Returns
    -------
    noisy_flux : np.ndarray
        Flux array with added Gaussian noise
    """
    print(f"Adding Gaussian noise (σ={noise_level}) with seed={seed}")
    
    n_spectra, n_wavelengths = flux_array.shape

    # Generate different noise realizations for each spectrum using the same seed
    # This creates a single RNG with the specified seed, then draws multiple samples
    noise_vectors = generate_noise(shape=(n_spectra, n_wavelengths), noise=noise_level, seed=seed)
    
    # Add different noise vectors to each spectrum
    noisy_flux = flux_array + noise_vectors
    
    print(f"Added noise to {n_spectra} spectra")
    print(f"Noise statistics: mean={np.mean(noise_vectors):.6f}, std={np.std(noise_vectors):.6f}")
    
    return noisy_flux


def normalize_spectra(flux_array, wavelength, **normalization_kwargs):
    """
    Apply continuum normalization to all spectra using legendre_polyfit_huber.
    
    Parameters
    ----------
    flux_array : np.ndarray
        2D array of shape (n_spectra, n_wavelengths) containing flux values
    wavelength : np.ndarray
        1D array of wavelength values
    **normalization_kwargs : dict
        Additional arguments to pass to legendre_polyfit_huber
        
    Returns
    -------
    normalized_flux : np.ndarray
        Continuum-normalized flux array
    continuum_fits : np.ndarray
        Array of fitted continuum models for each spectrum
    """
    print(f"Applying continuum normalization using legendre_polyfit_huber")
    print(f"Normalization parameters: {normalization_kwargs}")
    
    n_spectra, _ = flux_array.shape
    normalized_flux = np.empty_like(flux_array)
    continuum_fits = np.empty_like(flux_array)
    
    # Apply normalization to each spectrum individually
    for i in range(n_spectra):
        try:
            norm_flux, continuum = legendre_polyfit_huber(
                flux_array[i], wavelength, **normalization_kwargs
            )
            normalized_flux[i] = norm_flux
            continuum_fits[i] = continuum
            
            if i % 100 == 0:  # Progress indicator for large datasets
                print(f"  Normalized {i+1}/{n_spectra} spectra")
                
        except Exception as e:
            print(f"Warning: Failed to normalize spectrum {i}: {e}")
            # Fallback: use original flux if normalization fails
            normalized_flux[i] = flux_array[i]
            continuum_fits[i] = np.ones_like(wavelength)
    
    print(f"Successfully normalized {n_spectra} spectra")
    
    # Print some statistics
    norm_mean = np.mean(normalized_flux)
    norm_std = np.std(normalized_flux)
    print(f"Normalized flux statistics: mean={norm_mean:.3f}, std={norm_std:.3f}")
    
    return normalized_flux, continuum_fits


def save_processed_data(output_filepath, processed_data):
    """
    Save processed spectral data to HDF5 file.
    
    Parameters
    ----------
    output_filepath : str or Path
        Path where to save the processed data
    processed_data : dict
        Dictionary containing processed spectral data and metadata
    """
    print(f"Saving processed data to: {output_filepath}")
    
    # Ensure output directory exists
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_filepath, 'w') as f:
        # Save main datasets
        for key, value in processed_data.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(key, data=value, compression='gzip')
                print(f"  Saved {key}: {value.shape}")
            else:
                # Handle metadata (scalars, strings, etc.)
                try:
                    f.attrs[key] = value
                    print(f"  Saved attribute {key}: {value}")
                except Exception as e:
                    print(f"  Warning: Could not save {key}: {e}")
        
        # Add processing metadata
        f.attrs['processing_version'] = '1.0'
        f.attrs['pipeline'] = 'cnn_preprocessing'
        f.attrs['description'] = 'Spectra processed for CNN training: noise added + continuum normalized'
    
    print(f"Successfully saved processed data with {len(processed_data)} datasets")


def main():
    """Main preprocessing pipeline execution."""
    parser = argparse.ArgumentParser(
        description="CNN preprocessing pipeline for WEAVE spectral data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Input HDF5 file containing spectral data')
    parser.add_argument('--output', '-o', required=True,
                       help='Output HDF5 file for processed data')
    
    # Noise parameters
    parser.add_argument('--noise', type=float, default=0.1,
                       help='Gaussian noise standard deviation')
    parser.add_argument('--seed', type=int, default=2025,
                       help='Random seed for noise generation')
    
    # Normalization parameters
    parser.add_argument('--degree', type=int, default=2,
                       help='Degree of Legendre polynomial for continuum fitting')
    parser.add_argument('--sigma-lower', type=float, default=2.5,
                       help='Lower sigma threshold for sigma clipping')
    parser.add_argument('--sigma-upper', type=float, default=2.5,
                       help='Upper sigma threshold for sigma clipping')
    parser.add_argument('--huber-f', type=float, default=1.0,
                       help='Huber loss parameter')
    parser.add_argument('--max-iter', type=int, default=3,
                       help='Maximum iterations for sigma clipping')
    parser.add_argument('--enable-clipping', type=bool, default=True,
                       help='Enable sigma clipping during normalization')

    # Processing options
    parser.add_argument('--skip-noise', action='store_true',
                       help='Skip noise addition step')
    parser.add_argument('--skip-normalization', action='store_true',
                       help='Skip continuum normalization step')
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print("=" * 60)
    print("CNN PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Output file: {args.output}")
    print(f"Noise level: {args.noise} (seed: {args.seed})")
    print(f"Normalization: degree={args.degree}, sigma=({args.sigma_lower}, {args.sigma_upper})")
    print("=" * 60)
    
    try:
        # Step 1: Load spectral data
        print("\n1. LOADING SPECTRAL DATA")
        print("-" * 30)
        spectra = load_spectra_from_hdf5(input_path)
        
        # Get the flux and wavelength arrays
        flux = spectra['flux']
        wavelength = spectra['wavelength']
        
        # Step 2: Add noise (if not skipped)
        if not args.skip_noise:
            print("\n2. ADDING GAUSSIAN NOISE")
            print("-" * 30)
            noisy_flux = add_noise_to_spectra(
                flux, noise_level=args.noise, seed=args.seed
            )
        else:
            print("\n2. SKIPPING NOISE ADDITION")
            noisy_flux = flux.copy()
        
        # Step 3: Continuum normalization (if not skipped)
        if not args.skip_normalization:
            print("\n3. CONTINUUM NORMALIZATION")
            print("-" * 30)
            normalization_kwargs = {
                'degree': args.degree,
                'sigma_lower': args.sigma_lower,
                'sigma_upper': args.sigma_upper,
                'huber_f': args.huber_f,
                'max_iter': args.max_iter
            }
            
            normalized_flux, continuum_fits = normalize_spectra(
                noisy_flux, wavelength, **normalization_kwargs
            )
        else:
            print("\n3. SKIPPING CONTINUUM NORMALIZATION")
            normalized_flux = noisy_flux.copy()
            continuum_fits = np.ones_like(noisy_flux)
        
        # Step 4: Prepare output data
        print("\n4. PREPARING OUTPUT DATA")
        print("-" * 30)
        
        processed_data = {
            'flux_normalized': normalized_flux,
            'wavelength': wavelength,
            'continuum_fits': continuum_fits,
            'noise_level': args.noise,
            'noise_seed': args.seed,
        }
        
        # Copy additional metadata from original file
        for key, value in spectra.items():
            if key not in ['flux', 'wavelength']:
                processed_data[f'original_{key}'] = value
        
        # Step 5: Save processed data
        print("\n5. SAVING PROCESSED DATA")
        print("-" * 30)
        save_processed_data(args.output, processed_data)
        
        print("\n" + "=" * 60)
        print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Input spectra: {flux.shape[0]} spectra × {flux.shape[1]} wavelengths")
        print(f"Output file: {args.output}")
        print(f"Processing steps: {'Noise + ' if not args.skip_noise else ''}{'Normalization' if not args.skip_normalization else 'No normalization'}")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with exception: {e}")
        raise


if __name__ == "__main__":
    main()