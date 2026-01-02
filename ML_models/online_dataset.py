
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from scipy.special import eval_hermite
import sys
from pathlib import Path

# Import cont_norm
sys.path.append(str(Path(__file__).parents[1]))
try:
    from preprocessing import cont_norm
except ImportError:
    # Fallback
    import cont_norm

class WeaveOnlineDataset(Dataset):
    """
    PyTorch Dataset that performs on-the-fly augmentation and processing.
    
    Pipeline:
    1. Load raw spectrum from HDF5.
    2. Interpolate to log-linear grid.
    3. Apply LSF (randomized resolution).
    4. Apply Noise & Masking (randomized SNR).
    5. Continuum Normalization (robust Legendre fit).
    """
    
    def __init__(self, raw_hdf5_path, metadata_path, target_cols=None, wave_start=4040.0, wave_end=6850.0, wave_step_ref=0.05):
        self.raw_hdf5_path = raw_hdf5_path
        self.metadata_path = metadata_path
        
        # Load Metadata
        self.meta_csv = raw_hdf5_path.replace(".h5", "_metadata.csv")
        if not Path(self.meta_csv).exists():
            raise FileNotFoundError(f"Metadata CSV {self.meta_csv} not found.")
        self.meta_df = pd.read_csv(self.meta_csv)

        self.target_cols = target_cols
        if self.target_cols is None:
             # Auto-detect numeric columns
             # Exclude common non-target columns
             exclude = ['star_name', 'hdf5_key', 'original_chunk', 'spectrum_path', 'original_file', 'index', 'Unnamed: 0']
             # Select numeric columns
             numeric_cols = self.meta_df.select_dtypes(include=[np.number]).columns.tolist()
             self.target_cols = [c for c in numeric_cols if c not in exclude]
        
        # Grid Setup
        self.wave_start = wave_start
        self.wave_end = wave_end
        self.log_wave_start = np.log(wave_start)
        self.log_wave_end = np.log(wave_end)
        self.log_wave_step = (wave_step_ref / wave_start)
        
        n_points = int((self.log_wave_end - self.log_wave_start) / self.log_wave_step) + 1
        self.log_wave_grid = np.linspace(self.log_wave_start, self.log_wave_end, n_points)
        self.wave_grid = np.exp(self.log_wave_grid)
        
        # Arm Definitions
        self.blue_arm = (4040, 4650)
        self.green_arm = (4730, 5450)
        self.red_arm = (5950, 6850)
        
        # Load Golden Sample for sampling
        self.golden_sample_df = self._load_golden_sample(metadata_path)
        
        # Open HDF5 (will be opened in __getitem__ or worker_init_fn to be safe with workers)
        self.hf = None
        
        # Normalization stats
        self.target_mean = None
        self.target_std = None

    def set_target_stats(self, mean, std):
        self.target_mean = torch.tensor(mean, dtype=torch.float32)
        self.target_std = torch.tensor(std, dtype=torch.float32)

    def _load_golden_sample(self, fits_path):
        from astropy.io import fits
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            mask = data['MODE'] == 'HIGHRES'
            filtered = data[mask]
            # Note: FITS data is big-endian, but pandas requires little-endian.
            # We must cast to native byte order to avoid "ValueError: Big-endian buffer not supported"
            df = pd.DataFrame({
                'RESOL': filtered['RESOL'].astype(np.float32),
                'SNR_blue': filtered['SNR_blue_QAG'].astype(np.float32),
                'SNR_green': filtered['SNR_green_QAG'].astype(np.float32),
                'SNR_red': filtered['SNR_red_QAG'].astype(np.float32)
            })
            
            # Fill NaNs with 0 to handle missing arms (Blue/Green are mutually exclusive)
            df = df.fillna(0)
            
            # Filter: Red must be present, and at least one of Blue or Green
            df = df[((df['SNR_blue'] > 0) | (df['SNR_green'] > 0)) & (df['SNR_red'] > 0)]
            
        return df

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.raw_hdf5_path, 'r')
            
        row = self.meta_df.iloc[idx]
        key = row['hdf5_key']
        
        # Load Raw Data
        data = self.hf["raw_spectra"][key][:]
        raw_wave = data[:, 0]
        raw_flux = data[:, 1]
        
        # Process
        processed_flux = self._process_spectrum(raw_wave, raw_flux)
        
        # Targets
        targets = torch.tensor([row[c] for c in self.target_cols], dtype=torch.float32)
        
        if self.target_mean is not None and self.target_std is not None:
            targets = (targets - self.target_mean) / self.target_std
        
        # Convert to Tensor
        return torch.from_numpy(processed_flux).float(), targets

    def _process_spectrum(self, raw_wave, raw_flux):
        # 1. Interpolate
        f_interp = interp1d(raw_wave, raw_flux, kind='linear', bounds_error=False, fill_value=0.0)
        flux_interp = f_interp(self.wave_grid)
        
        # 2. LSF
        meta_row = self.golden_sample_df.sample(n=1).iloc[0]
        resol = meta_row['RESOL']
        lambda_center = (self.wave_start + self.wave_end) / 2.0
        r_eff = lambda_center / resol
        sigma_pix = 1.0 / (r_eff * self.log_wave_step * 2.355)
        
        h3 = np.random.normal(0, 0.005)
        h4 = np.random.uniform(0.02, 0.07)
        
        kernel = self._generate_lsf_kernel(sigma_pix, h3, h4)
        flux_conv = convolve1d(flux_interp, kernel, mode='constant', cval=0.0)
        
        # 3. Noise & Masking
        snr_blue = meta_row['SNR_blue']
        snr_green = meta_row['SNR_green']
        snr_red = meta_row['SNR_red']
        
        mask_blue = (self.wave_grid >= self.blue_arm[0]) & (self.wave_grid <= self.blue_arm[1])
        mask_green = (self.wave_grid >= self.green_arm[0]) & (self.wave_grid <= self.green_arm[1])
        mask_red = (self.wave_grid >= self.red_arm[0]) & (self.wave_grid <= self.red_arm[1])
        
        final_flux = flux_conv.copy()
        
        if np.isnan(snr_blue) or snr_blue <= 0: final_flux[mask_blue] = 0
        if np.isnan(snr_green) or snr_green <= 0: final_flux[mask_green] = 0
        if np.isnan(snr_red) or snr_red <= 0: final_flux[mask_red] = 0
        
        rng = np.random.default_rng()
        if not np.isnan(snr_blue) and snr_blue > 0:
            sigma = np.abs(final_flux[mask_blue]) / snr_blue
            final_flux[mask_blue] += rng.normal(0, 1, size=np.sum(mask_blue)) * sigma
        if not np.isnan(snr_green) and snr_green > 0:
            sigma = np.abs(final_flux[mask_green]) / snr_green
            final_flux[mask_green] += rng.normal(0, 1, size=np.sum(mask_green)) * sigma
        if not np.isnan(snr_red) and snr_red > 0:
            sigma = np.abs(final_flux[mask_red]) / snr_red
            final_flux[mask_red] += rng.normal(0, 1, size=np.sum(mask_red)) * sigma
            
        # 4. Normalize
        # Normalize Blue
        if np.any(mask_blue) and np.any(final_flux[mask_blue] > 0):
            flux_b = final_flux[mask_blue]
            wave_b = self.wave_grid[mask_blue]
            norm_flux_b, _ = cont_norm.legendre_polyfit_huber(
                flux_b, wave_b, degree=4, sigma_lower=2.0, sigma_upper=2.0
            )
            final_flux[mask_blue] = norm_flux_b

        # Normalize Green
        if np.any(mask_green) and np.any(final_flux[mask_green] > 0):
            flux_g = final_flux[mask_green]
            wave_g = self.wave_grid[mask_green]
            norm_flux_g, _ = cont_norm.legendre_polyfit_huber(
                flux_g, wave_g, degree=4, sigma_lower=2.0, sigma_upper=2.0
            )
            final_flux[mask_green] = norm_flux_g
            
        # Normalize Red
        if np.any(mask_red) and np.any(final_flux[mask_red] > 0):
            flux_r = final_flux[mask_red]
            wave_r = self.wave_grid[mask_red]
            norm_flux_r, _ = cont_norm.legendre_polyfit_huber(
                flux_r, wave_r, degree=5, sigma_lower=1.5, sigma_upper=1.5
            )
            final_flux[mask_red] = norm_flux_r
            
        # 5. Cleanup
        mask_gaps = ~(mask_bg | mask_r)
        final_flux[mask_gaps] = 0
        
        return final_flux

    def _generate_lsf_kernel(self, sigma_pix, h3, h4, size_sigma=5):
        half_size = int(np.ceil(size_sigma * sigma_pix))
        x = np.arange(-half_size, half_size + 1)
        y = x / sigma_pix
        gauss = np.exp(-0.5 * y**2) / (np.sqrt(2 * np.pi) * sigma_pix)
        H3 = eval_hermite(3, y)
        H4 = eval_hermite(4, y)
        kernel = gauss * (1 + h3 * H3 + h4 * H4)
        kernel /= np.sum(kernel)
        return kernel
