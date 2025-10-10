import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Tuple, List, Union


class HDF5SpectrumReader:
    """
    A class to read and extract data from HDF5 spectrum files created by SpectrumReaderHDF5.
    
    This class provides convenient methods to:
    1. Extract wavelength & flux data for plotting
    2. Extract stellar parameter labels
    3. Access specific spectra by index or filename
    4. Get summary information about the dataset
    """
    
    def __init__(self, hdf5_file_path: str):
        """
        Initialize the HDF5 spectrum reader.
        
        Args:
            hdf5_file_path (str): Path to the HDF5 file
        """
        self.file_path = Path(hdf5_file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_file_path}")
        
        # Load metadata and basic info
        self._load_metadata()
        
    def _load_metadata(self) -> None:
        """Load metadata from the HDF5 file (updated for new structure)."""
        with h5py.File(self.file_path, 'r') as f:
            # Get metadata
            metadata = f['metadata']
            self.n_spectra = metadata.attrs['n_spectra']
            self.stellar_param_names = metadata.attrs['stellar_param_names']
            if isinstance(self.stellar_param_names[0], bytes):
                self.stellar_param_names = [name.decode('utf-8') for name in self.stellar_param_names]

            # Load filenames
            self.filenames = f['labels/filenames'][:]
            if isinstance(self.filenames[0], bytes):
                self.filenames = [name.decode('utf-8') for name in self.filenames]

            # New structure: single wavelength grid for all spectra
            self.wavelength_grid = f['spectra/wavelength_grid'][:]
            self.n_wavelength_points = len(self.wavelength_grid)
    
    def get_wavelength_flux(self, spectrum_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract wavelength and flux arrays for a specific spectrum (updated for new structure).
        
        Args:
            spectrum_index (int): Index of the spectrum (0-based)
        
        Returns:
            tuple: (wavelength_array, flux_array)
        """
        if spectrum_index >= self.n_spectra or spectrum_index < 0:
            raise IndexError(f"Spectrum index {spectrum_index} out of range [0, {self.n_spectra-1}]")

        with h5py.File(self.file_path, 'r') as f:
            wavelength = self.wavelength_grid
            flux = f['spectra/flux'][spectrum_index]
        return wavelength, flux
    
    
    def get_labels(self, parameter_names: Optional[List[str]] = None) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """
        Extract stellar parameter labels.
        
        Args:
            parameter_names (List[str], optional): Specific parameters to extract.
                                                  If None, returns all parameters.
        
        Returns:
            np.ndarray or tuple: If parameter_names is None, returns (labels_array, parameter_names)
                                Otherwise returns just the labels_array for requested parameters
        """
        with h5py.File(self.file_path, 'r') as f:
            all_labels = f['labels/stellar_parameters'][:]
        
        if parameter_names is None:
            return all_labels, self.stellar_param_names.copy()
        
        # Extract specific parameters
        param_indices = []
        for param in parameter_names:
            if param in self.stellar_param_names:
                param_indices.append(self.stellar_param_names.index(param))
            else:
                raise ValueError(f"Parameter '{param}' not found. Available: {self.stellar_param_names}")
        
        return all_labels[:, param_indices]
    
    def get_spectrum_by_filename(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get wavelength and flux for a spectrum by its filename (updated for new structure).
        
        Args:
            filename (str): Name of the spectrum file
        
        Returns:
            tuple: (wavelength_array, flux_array)
        """
        try:
            index = self.filenames.index(filename)
            return self.get_wavelength_flux(index)
        except ValueError:
            raise ValueError(f"Filename '{filename}' not found in dataset")
    
    def get_spectrum_info(self, spectrum_index: int) -> dict:
        """
        Get comprehensive information about a specific spectrum (updated for new structure).
        
        Args:
            spectrum_index (int): Index of the spectrum
        
        Returns:
            dict: Dictionary containing filename, stellar parameters, and data info
        """
        if spectrum_index >= self.n_spectra or spectrum_index < 0:
            raise IndexError(f"Spectrum index {spectrum_index} out of range")

        labels, param_names = self.get_labels()

        info = {
            'index': spectrum_index,
            'filename': self.filenames[spectrum_index],
            'stellar_parameters': {}
        }

        # Add stellar parameters
        for i, param_name in enumerate(param_names):
            value = labels[spectrum_index, i]
            info['stellar_parameters'][param_name] = value if not np.isnan(value) else None

        # Add wavelength info (single grid for all spectra)
        info['wavelength_points'] = len(self.wavelength_grid)
        info['wavelength_range'] = (float(self.wavelength_grid.min()), float(self.wavelength_grid.max()))

        return info
    
    
    def get_dataset_summary(self) -> dict:
        """
        Get a comprehensive summary of the entire dataset (updated for new structure).
        
        Returns:
            dict: Summary statistics and information about the dataset
        """
        labels, param_names = self.get_labels()

        summary = {
            'file_path': str(self.file_path),
            'file_size_mb': self.file_path.stat().st_size / (1024**2),
            'n_spectra': self.n_spectra,
            'stellar_parameters': param_names,
            'parameter_statistics': {}
        }

        # Calculate statistics for each parameter
        for i, param_name in enumerate(param_names):
            values = labels[:, i]
            valid_values = values[~np.isnan(values)]

            if len(valid_values) > 0:
                summary['parameter_statistics'][param_name] = {
                    'count': len(valid_values),
                    'min': float(valid_values.min()),
                    'max': float(valid_values.max()),
                    'mean': float(valid_values.mean()),
                    'std': float(valid_values.std())
                }
            else:
                summary['parameter_statistics'][param_name] = {
                    'count': 0,
                    'min': None,
                    'max': None,
                    'mean': None,
                    'std': None
                }

        # Add wavelength info (single grid for all spectra)
        summary['wavelength_points'] = len(self.wavelength_grid)
        summary['wavelength_range'] = (float(self.wavelength_grid.min()), float(self.wavelength_grid.max()))

        return summary
    
    def print_dataset_summary(self) -> None:
        """Print a formatted summary of the dataset (updated for new structure)."""
        summary = self.get_dataset_summary()

        print(f"HDF5 Spectrum Dataset Summary")
        print(f"=" * 40)
        print(f"File: {summary['file_path']}")
        print(f"Size: {summary['file_size_mb']:.1f} MB")
        print(f"Number of spectra: {summary['n_spectra']}")
        print(f"Wavelength points: {summary['wavelength_points']}")
        print(f"Wavelength range: {summary['wavelength_range'][0]:.1f} - {summary['wavelength_range'][1]:.1f} Å")

        print(f"\nStellar Parameters:")
        for param_name in summary['stellar_parameters']:
            stats = summary['parameter_statistics'][param_name]
            if stats['count'] > 0:
                print(f"  {param_name}: {stats['min']:.2f} - {stats['max']:.2f} "
                      f"(mean: {stats['mean']:.2f} ± {stats['std']:.2f}, n={stats['count']})")
            else:
                print(f"  {param_name}: No valid data")
    
    def list_filenames(self, max_display: int = 20) -> List[str]:
        """
        List the filenames in the dataset.
        
        Args:
            max_display (int): Maximum number of filenames to display
        
        Returns:
            List[str]: List of all filenames
        """
        print(f"Dataset contains {len(self.filenames)} spectra:")
        
        display_count = min(max_display, len(self.filenames))
        for i in range(display_count):
            print(f"  {i:3d}: {self.filenames[i]}")
        
        if len(self.filenames) > max_display:
            print(f"  ... and {len(self.filenames) - max_display} more")
        
        return self.filenames.copy()