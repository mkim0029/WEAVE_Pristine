def parse_file_for_pool(args):
    cls, file_path = args
    try:
        return cls.parse_single_file(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
from importlib.metadata import files
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Tuple, List
from scipy import interpolate
import time
import concurrent.futures
import random


class SpectrumReaderHDF5:
    """
    Efficient SpectrumReader class using HDF5 storage and numpy arrays.
    
    This class processes multiple stellar spectrum files and stores them
    efficiently in HDF5 format.
    """
    
    def __init__(self):
        """Initialize the SpectrumReader."""
        # Storage for processed data (numpy arrays)
        self.spectra_wavelength = None    # 2D array: [n_spectra, n_wavelength_points] 
        self.stellar_parameters = None    # 2D array: [n_spectra, 6] (teff, log_g, met, micro, macro, vsini)
        self.filenames = None            # 1D array of strings
        
        # Common grid for interpolation
        self.common_wavelength_grid = None
        self.interpolated_flux = None    # 2D array: [n_spectra, n_grid_points]
        
        # Counters
        self.n_spectra = 0
        self.n_processed = 0
        
    def parse_single_file(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Parse a single spectrum file and return arrays.
        
        Args:
            file_path (Path): Path to the spectrum file
            
        Returns:
            tuple: (wavelength_array, flux_array, stellar_params_array, filename)
        """
        wavelengths = []
        fluxes = []
        stellar_params = np.full(6, np.nan)  # [teff, log_g, met, micro, macro, vsini]
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Parse headers
        found_teff_header = False
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
                
            if line == "# wave, flux":
                break
                
            # Look for the teff header line
            if line.startswith("# teff, log g, met, micro, macro, vsini"):
                found_teff_header = True
                continue
                
            # Parse stellar parameters
            if found_teff_header and line.startswith("#"):
                values = line.strip("# ").split()
                if len(values) >= 6:
                    try:
                        stellar_params[0] = float(values[0])  # teff
                        stellar_params[1] = float(values[1])  # log_g
                        stellar_params[2] = float(values[2])  # metallicity
                        stellar_params[3] = float(values[3])  # microturbulence
                        stellar_params[4] = float(values[4])  # macroturbulence
                        stellar_params[5] = float(values[5])  # vsini
                        found_teff_header = False
                    except ValueError:
                        pass
        
        # Parse data
        data_started = False
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
                
            if line == "# wave, flux":
                data_started = True
                continue
                
            if data_started and not line.startswith("#"):
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        wavelength = float(parts[0])
                        flux = float(parts[1])
                        wavelengths.append(wavelength)
                        fluxes.append(flux)
                except ValueError:
                    continue
        
        return (np.array(wavelengths), np.array(fluxes), stellar_params, file_path.name)
    
    def read_multiple_files(self, directory_path: str, pattern: str = "Star*_N", 
                          max_files: Optional[int] = None, chunk_size: int = 500, checkpoint_path: Optional[str] = None, n_workers: int = 4, grid_file: Optional[str] = None, randomize: bool = False, seed: Optional[int] = None) -> None:
        """
        Read multiple spectrum files from a directory efficiently.
        
        Args:
            directory_path (str): Path to directory containing spectrum files
            pattern (str): File pattern to match
            max_files (int, optional): Maximum number of files to read
            chunk_size (int): Process files in chunks to manage memory
            grid_file (str, optional): Path to the common wavelength grid file
            # Interpolation method is always 'linear'
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find matching files
        files = list(directory.glob(pattern))
        if not files:
            raise ValueError(f"No files found matching pattern '{pattern}' in {directory}")
        
        files.sort()
        # Optionally randomize order before truncating to `max_files` so
        # the output contains a random sample rather than the first N files
        # in sorted order.
        if randomize:
            if seed is not None:
                random.seed(seed)
            random.shuffle(files)

        if max_files:
            files = files[:max_files]
        
        print(f"Found {len(files)} files to process...")
        
        # Process files in chunks
        start_time = time.time()
        batch_size = chunk_size
        n_files = len(files)
        self.n_processed = 0
        if grid_file is None:
            raise ValueError("A grid_file must be provided to interpolate spectra before saving.")
        self.create_common_wavelength_grid(grid_file=grid_file)
        n_grid = len(self.common_wavelength_grid)
        total_written = 0
        h5file = None
        if checkpoint_path is not None:
            h5file = h5py.File(checkpoint_path, 'a')
            if 'flux' not in h5file:
                h5file.create_dataset('flux', shape=(0, n_grid), maxshape=(None, n_grid), dtype='f4', compression='gzip', chunks=True)
                h5file.create_dataset('stellar_parameters', shape=(0, 6), maxshape=(None, 6), dtype='f4', compression='gzip', chunks=True)
                h5file.create_dataset('filenames', shape=(0,), maxshape=(None,), dtype='S50', compression=None, chunks=True)
                h5file.create_dataset('wavelength_grid', data=self.common_wavelength_grid, dtype='f4')
        all_fluxes = []
        all_stellar_params = []
        all_filenames = []
        pool_args = [(self, file_path) for file_path in files]
        skipped_nan = []
        fixed_duplicate = 0
        skipped_nan = []
        fixed_duplicate = 0
        nan_threshold = 0.1  # Skip files with >10% NaN in flux

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            for i, result in enumerate(executor.map(parse_file_for_pool, pool_args), 1):
                if result is None:
                    continue
                wavelength, flux, stellar_params, filename = result
                if len(wavelength) > 0 and not np.all(np.isnan(stellar_params)):
                    # Skip if >10% of flux is NaN
                    frac_nan = np.mean(np.isnan(flux))
                    if frac_nan > nan_threshold:
                        skipped_nan.append(filename)
                        continue
                    # Debug: Check for Inf in wavelength or flux
                    if np.any(np.isinf(wavelength)) or np.any(np.isinf(flux)):
                        skipped_nan.append(filename)
                        continue
                    # Debug: Check for duplicate wavelengths
                    if np.any(np.diff(wavelength) == 0):
                        # Remove duplicates, keep first occurrence
                        _, unique_idx = np.unique(wavelength, return_index=True)
                        unique_idx.sort()
                        fixed_duplicate += 1
                        wavelength = wavelength[unique_idx]
                        flux = flux[unique_idx]
                    # Interpolate immediately (always linear)
                    try:
                        f_interp = interpolate.interp1d(
                            wavelength, flux, kind='linear', bounds_error=False, fill_value='extrapolate')
                        interp_flux = f_interp(self.common_wavelength_grid).astype('f4')
                        all_fluxes.append(interp_flux)
                        all_stellar_params.append(stellar_params.astype('f4'))
                        all_filenames.append(filename)
                        self.n_processed += 1
                    except Exception as e:
                        print(f"Interpolation failed for {filename}: {e}")
                        print(f"  wavelength: {wavelength}")
                        print(f"  flux: {flux}")
                        continue
                if i % batch_size == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    eta = (n_files - i) / rate
                    print(f"Processed {i}/{n_files} files... "
                        f"({self.n_processed} successful, {rate:.1f} files/s, ETA: {eta:.0f}s)")
                    if checkpoint_path is not None and len(all_fluxes) > 0:
                        n_batch = len(all_fluxes)
                        flux_arr = np.stack(all_fluxes)
                        param_arr = np.stack(all_stellar_params)
                        fname_arr = np.array(all_filenames, dtype='S50')
                        for key, arr in zip(['flux', 'stellar_parameters', 'filenames'], [flux_arr, param_arr, fname_arr]):
                            dset = h5file[key]
                            dset.resize((dset.shape[0] + n_batch,) + dset.shape[1:])
                            dset[-n_batch:, ...] = arr
                        total_written += n_batch
                        all_fluxes.clear()
                        all_stellar_params.clear()
                        all_filenames.clear()
                        h5file.flush()
        
        # Handle final batch if there are remaining files
        if checkpoint_path is not None and len(all_fluxes) > 0:
            n_batch = len(all_fluxes)
            flux_arr = np.stack(all_fluxes)
            param_arr = np.stack(all_stellar_params)
            fname_arr = np.array(all_filenames, dtype='S50')
            for key, arr in zip(['flux', 'stellar_parameters', 'filenames'], [flux_arr, param_arr, fname_arr]):
                dset = h5file[key]
                dset.resize((dset.shape[0] + n_batch,) + dset.shape[1:])
                dset[-n_batch:, ...] = arr
            total_written += n_batch
            h5file.flush()
        
        if checkpoint_path is not None:
            h5file.close()
            print(f"Checkpointed {total_written} spectra to {checkpoint_path}")
            # Print checkpoint structure
            self._print_hdf5_structure(checkpoint_path)
            with h5py.File(checkpoint_path, 'r') as h5f:
                self.interpolated_flux = h5f['flux'][:]
                self.stellar_parameters = h5f['stellar_parameters'][:]
                self.filenames = h5f['filenames'][:]
                self.common_wavelength_grid = h5f['wavelength_grid'][:]
            self.n_spectra = self.interpolated_flux.shape[0]
            print(f"Successfully loaded {self.n_spectra} spectra from checkpoint.")
            print(f"Processing took {time.time() - start_time:.1f} seconds")
        else:
            self.n_spectra = len(all_stellar_params)
            if self.n_spectra > 0:
                self.interpolated_flux = np.stack(all_fluxes)
                self.stellar_parameters = np.stack(all_stellar_params)
                self.filenames = np.array(all_filenames, dtype='S50')
                print(f"Successfully loaded {self.n_processed} spectra from {len(files)} files")
                print(f"Processing took {time.time() - start_time:.1f} seconds")
            else:
                print("No valid spectra found!")

        # Write skipped files to skipped.txt
        with open('../data/skipped.txt', 'w') as fskip:
            for fname in skipped_nan:
                fskip.write(f"{fname}\n")

        # Print summary
        print(f"\nSummary of skipped/fixed files:")
        print(f"  NaN flux threshold: {nan_threshold*100:.0f}%")
        print(f"  Skipped due to >{nan_threshold*100:.0f}% NaN in flux: {len(skipped_nan)} out of {len(files)}")
        print(f"  Files with duplicate wavelength values fixed: {fixed_duplicate} out of {len(files)}")
        print(f"  Skipped files written to ../data/skipped.txt")

    def check_if_calibration_needed(self) -> bool:
        """
        Check if spectra have different wavelength grids and need calibration.
        
        Returns:
            bool: True if calibration is needed, False if all spectra have identical grids
        """
        if self.n_spectra <= 1:
            return False
        
        # Check if all spectra have the same length
        # No need to check lengths after rescaling
        
        # Check if the first few spectra have identical wavelength grids
        reference_wavelength = self.spectra_wavelength[0]
        
        # Check first 5 spectra for identical wavelength grids
        n_check = min(5, self.n_spectra)
        for i in range(1, n_check):
            current_wavelength = self.spectra_wavelength[i]
            if not np.allclose(reference_wavelength, current_wavelength, rtol=1e-10):
                print(f"Calibration needed: Wavelength grids differ between spectra")
                return True
        
        print("All spectra have identical wavelength grids - calibration not needed")
        return False
    
    def create_common_wavelength_grid(self, grid_file: str = "../data/grid_wavelengths_windows.txt") -> np.ndarray:
        """
        Load a common wavelength grid from a file (e.g., grid_wavelengths_windows.txt).
        Args:
            grid_file (str): Path to the grid file
        Returns:
            np.ndarray: Common wavelength grid
        """
        if not Path(grid_file).exists():
            raise FileNotFoundError(f"Grid file not found: {grid_file}")
        grid = np.loadtxt(grid_file)
        self.common_wavelength_grid = grid
        print(f"Loaded common wavelength grid from {grid_file}: {grid[0]:.4f} - {grid[-1]:.4f} Å")
        print(f"Grid points: {len(grid)}")
        return self.common_wavelength_grid
    
    def interpolate_spectra_to_grid(self, force_interpolation: bool = False) -> None:
        """
        Interpolate all spectra to the common wavelength grid.
        
        Args:
            force_interpolation (bool): Force interpolation even if not needed
        """
        if self.common_wavelength_grid is None:
            raise ValueError("Common wavelength grid not created. Use create_common_wavelength_grid() first.")

        # Check if calibration is actually needed
        if not force_interpolation and not self.check_if_calibration_needed():
            print("Skipping interpolation - using original wavelength grids")
            # Use the original flux data as "interpolated" data
            self.interpolated_flux = self.interpolated_flux.copy()
            # Update the common grid to match the original grid
            self.common_wavelength_grid = self.spectra_wavelength[0].copy()
            # Always replace original arrays
            self.interpolated_flux = self.interpolated_flux
            self.spectra_wavelength = np.tile(self.common_wavelength_grid, (self.n_spectra, 1))
            return

        n_grid_points = len(self.common_wavelength_grid)
        interpolated_flux_temp = np.full((self.n_spectra, n_grid_points), np.nan)

        print(f"Interpolating {self.n_spectra} spectra to common grid...")

        successful_interpolations = 0

        for i in range(self.n_spectra):
            try:
                # Get valid data points for this spectrum
                wavelength = self.spectra_wavelength[i]
                flux = self.interpolated_flux[i]

                # Create interpolation function
                f_interp = interpolate.interp1d(
                    wavelength, flux,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )

                # Interpolate to common grid
                interpolated_flux_temp[i, :] = f_interp(self.common_wavelength_grid)
                successful_interpolations += 1

            except Exception as e:
                print(f"Error interpolating spectrum {i}: {e}")
                continue

        print(f"Successfully interpolated {successful_interpolations} spectra")

        # Replace with interpolated data
        self.interpolated_flux = interpolated_flux_temp
        self.spectra_wavelength = np.tile(self.common_wavelength_grid, (self.n_spectra, 1))
    
    def save_to_hdf5(self, output_path: str, compress: bool = True) -> None:
        """
        Save all data to an HDF5 file.
        
        Args:
            output_path (str): Path for output HDF5 file
            compress (bool): Whether to use compression
        """
        if self.stellar_parameters is None:
            raise ValueError("No data to save. Use read_multiple_files() first.")
        
        compression = 'gzip' if compress else None
        
        with h5py.File(output_path, 'w') as f:
            # Create groups for organization
            metadata_group = f.create_group('metadata')
            spectra_group = f.create_group('spectra')
            labels_group = f.create_group('labels')

            # Save metadata
            metadata_group.attrs['n_spectra'] = self.n_spectra
            metadata_group.attrs['stellar_param_names'] = ['teff', 'log_g', 'metallicity',
                                                          'microturbulence', 'macroturbulence', 'vsini']

            # Save flux data (interpolated)
            spectra_group.create_dataset('flux', data=self.interpolated_flux, compression=compression)
            spectra_group.create_dataset('wavelength_grid', data=self.common_wavelength_grid)

            # Save labels (stellar parameters)
            labels_group.create_dataset('stellar_parameters', data=self.stellar_parameters, compression=compression)
            labels_group.create_dataset('filenames', data=self.filenames)

        print(f"Data saved to {output_path}")
        print(f"File size: {Path(output_path).stat().st_size / (1024**2):.1f} MB")
        
        # Print HDF5 structure
        self._print_hdf5_structure(output_path)

    def _print_hdf5_structure(self, file_path: str, indent: str = "") -> None:
        """
        Print the hierarchical structure of an HDF5 file.
        
        Args:
            file_path (str): Path to the HDF5 file
            indent (str): Indentation for nested structure
        """
        def print_structure(name, obj):
            print(f"{indent}{name}")
            if hasattr(obj, 'shape'):
                print(f"{indent}  └─ Dataset: shape={obj.shape}, dtype={obj.dtype}")
            elif hasattr(obj, 'attrs'):
                if len(obj.attrs) > 0:
                    print(f"{indent}  └─ Attributes: {dict(obj.attrs)}")
        
        print(f"\nHDF5 File Structure: {file_path}")
        print("=" * 50)
        
        with h5py.File(file_path, 'r') as f:
            f.visititems(print_structure)
        
        print("=" * 50)

def main():
    """Example usage of the SpectrumReaderHDF5 class."""
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser(description="Spectrum grid reader with external wavelength grid support")
    parser.add_argument('--data-dir', type=str, default="/home/minjihk/projects/rrg-kyi/astro/data/weave/nlte-grids/train_nlte", help="Directory with spectrum files")
    parser.add_argument('--grid-file', type=str, default="../data/grid_wavelengths_windows.txt", help="Path to wavelength grid file")
    parser.add_argument('--output', type=str, default="../data/weave_nlte_grids.h5", help="Output HDF5 file")
    parser.add_argument('--max-files', type=int, default=None, help="Maximum number of files to read")
    parser.add_argument('--randomize', action='store_true', help='Randomly sample files before selecting max-files')
    parser.add_argument('--seed', type=int, default=None, help='Random seed when --randomize is used') #2025
    args = parser.parse_args()

    print("=" * 60)
    reader = SpectrumReaderHDF5()
    # Use checkpointing: write every 500 spectra to output file
    reader.read_multiple_files(args.data_dir, max_files=args.max_files, checkpoint_path=args.output, chunk_size=500, grid_file=args.grid_file, randomize=args.randomize, seed=args.seed)
    
    # Only save to HDF5 if data was loaded
    if reader.stellar_parameters is not None:
        reader.save_to_hdf5(args.output)
        print("\nProcessing complete!")
    else:
        print("\nNo data was loaded - skipping HDF5 save.")
        print("Check the summary above to see why files were skipped.")


if __name__ == "__main__":
    main()