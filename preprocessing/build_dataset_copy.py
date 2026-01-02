
import os
import glob
import numpy as np
import pandas as pd
import h5py
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from scipy.special import eval_hermite
import multiprocessing
from functools import partial
from pathlib import Path
import sys

# Add current directory to path to import cont_norm
sys.path.append(str(Path(__file__).parent))
try:
    import cont_norm
except ImportError:
    # Fallback if running from root
    from preprocessing import cont_norm

# Configuration
RAW_DATA_DIR = os.path.expanduser("~/scratch/mpia_results_10k")
METADATA_FILE = "data/GA-QAG_OPR3bv3_MasterTableCodev4_GoldenSample_RELEASE.fits"
RAW_HDF5_FILE = "data/raw_spectra_1.h5"
PROCESSED_HDF5_FILE = "data/processed_spectra_1.h5"

# Grid Definition
WAVE_START = 4040.0
WAVE_END = 6850.0
WAVE_STEP_REF = 0.05  # at 4040 A
LOG_WAVE_START = np.log(WAVE_START)
LOG_WAVE_END = np.log(WAVE_END)
LOG_WAVE_STEP = WAVE_STEP_REF / WAVE_START

# Arm Definitions (Angstroms)
BLUE_ARM = (4040, 4650)
GREEN_ARM = (4730, 5450)
RED_ARM = (5950, 6850)

def get_wavelength_grid():
    """Create the global log-linear wavelength grid."""
    n_points = int((LOG_WAVE_END - LOG_WAVE_START) / LOG_WAVE_STEP) + 1
    log_wave = np.linspace(LOG_WAVE_START, LOG_WAVE_END, n_points)
    return np.exp(log_wave), log_wave

def create_raw_hdf5():
    """Step 1: Ingest raw text files into a master HDF5 file."""
    print(f"Scanning {RAW_DATA_DIR}...")
    
    # Find all chunks
    chunk_dirs = glob.glob(os.path.join(RAW_DATA_DIR, "chunk_*"))
    
    with h5py.File(RAW_HDF5_FILE, 'w') as hf:
        # Create extensible datasets or groups
        # Using a group per spectrum might be slow if 10k+
        # Better to store as one large array if lengths are same? 
        # Raw spectra might have different lengths/grids.
        # So we should store them as variable length or individual datasets.
        # Given Step 2 interpolates them, they are likely on different grids.
        # Storing as individual datasets in a group is safest.
        
        grp = hf.create_group("raw_spectra")
        
        # Metadata storage
        meta_list = []
        
        for chunk_dir in chunk_dirs:
            csv_files = glob.glob(os.path.join(chunk_dir, "*.csv"))
            if not csv_files: continue
            
            labels_df = pd.read_csv(csv_files[0])
            chunk_id = os.path.basename(chunk_dir) # e.g. chunk_0, chunk_1
            
            for idx, row in labels_df.iterrows():
                star_name = row['star_name']
                spec_path = os.path.join(chunk_dir, f"{star_name}_N")
                
                if os.path.exists(spec_path):
                    # Read file
                    try:
                        # Assuming simple format
                        data = np.loadtxt(spec_path) # shape (N, 2)
                        # Store in HDF5
                        # Make unique key by combining chunk and star name
                        dset_name = f"{chunk_id}_{star_name}"
                        if dset_name in grp:
                            del grp[dset_name]
                        grp.create_dataset(dset_name, data=data, compression="gzip")
                        
                        # Store label info
                        row_dict = row.to_dict()
                        row_dict['hdf5_key'] = dset_name
                        row_dict['original_chunk'] = chunk_id
                        meta_list.append(row_dict)
                        
                    except Exception as e:
                        print(f"Failed to read {spec_path}: {e}")

        # Save metadata table
        if meta_list:
            meta_df = pd.DataFrame(meta_list)
            # Save as dataset in HDF5 (structured array) or separate CSV
            # Saving as CSV is easier for pandas
            meta_df.to_csv(RAW_HDF5_FILE.replace(".h5", "_metadata.csv"), index=False)
            print(f"Created raw HDF5 with {len(meta_list)} spectra.")
        else:
            print("No spectra found.")

def load_golden_sample_metadata(fits_path):
    """Load and filter Golden Sample metadata."""
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"Metadata file not found: {fits_path}")
    
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        # Filter for HIGHRES
        mask = data['MODE'] == 'HIGHRES'
        filtered_data = data[mask]
        
        # Extract relevant columns
        # Note: FITS data is big-endian, but pandas requires little-endian.
        # We must cast to native byte order to avoid "ValueError: Big-endian buffer not supported"
        df = pd.DataFrame({
            'RESOL': filtered_data['RESOL'].astype(np.float32),
            'SNR_blue': filtered_data['SNR_blue_QAG'].astype(np.float32),
            'SNR_green': filtered_data['SNR_green_QAG'].astype(np.float32),
            'SNR_red': filtered_data['SNR_red_QAG'].astype(np.float32)
        })
        
        # Fill NaNs with 0 to handle missing arms (Blue/Green are mutually exclusive)
        df = df.fillna(0)
        
        # Filter: Red must be present, and at least one of Blue or Green
        df = df[((df['SNR_blue'] > 0) | (df['SNR_green'] > 0)) & (df['SNR_red'] > 0)]
        
    return df

def generate_lsf_kernel(sigma_pix, h3, h4, size_sigma=5):
    """
    Generate Gauss-Hermite LSF kernel.
    LSF(x) = Gauss(x) * [1 + h3*H3(x) + h4*H4(x)]
    """
    # Create grid in pixels
    half_size = int(np.ceil(size_sigma * sigma_pix))
    x = np.arange(-half_size, half_size + 1)
    
    # Normalized coordinate
    y = x / sigma_pix
    
    # Gaussian part
    gauss = np.exp(-0.5 * y**2) / (np.sqrt(2 * np.pi) * sigma_pix)
    
    # Hermite polynomials (Probabilist's or Physicist's? 
    # Usually in this expansion, H3 and H4 are the standard Hermite polynomials)
    # Using scipy.special.eval_hermite
    H3 = eval_hermite(3, y)
    H4 = eval_hermite(4, y)
    
    # Combine
    kernel = gauss * (1 + h3 * H3 + h4 * H4)
    
    # Normalize kernel to sum to 1
    kernel /= np.sum(kernel)
    
    return kernel

def process_spectrum_data(args):
    """
    Process a single spectrum from data.
    args: (raw_wave, raw_flux, label_row, wave_grid, log_wave_grid, golden_sample_df)
    """
    raw_wave, raw_flux, label_row, wave_grid, log_wave_grid, golden_sample_df = args
    
    try:
        if len(raw_wave) == 0:
            return None
            
        # 2. Interpolate to Log Grid
        # Use linear interpolation
        f_interp = interp1d(raw_wave, raw_flux, kind='linear', bounds_error=False, fill_value=0.0)
        flux_interp = f_interp(wave_grid)
        
        # 3. LSF Convolution
        # Randomly select a row from Golden Sample
        # Check if golden_sample_df is empty
        if golden_sample_df.empty:
            raise ValueError("Golden Sample DataFrame is empty after filtering.")
            
        meta_row = golden_sample_df.sample(n=1).iloc[0]
        resol = meta_row['RESOL']
        
        # Calculate sigma
        lambda_center = (WAVE_START + WAVE_END) / 2.0
        r_eff = lambda_center / resol
        
        sigma_pix = 1.0 / (r_eff * LOG_WAVE_STEP * 2.355)
        
        # Randomize h3, h4
        h3 = np.random.normal(0, 0.005)
        h4 = np.random.uniform(0.02, 0.07)
        
        kernel = generate_lsf_kernel(sigma_pix, h3, h4)
        flux_conv = convolve1d(flux_interp, kernel, mode='constant', cval=0.0)
        
        # 4. Noise & Masking
        snr_blue = meta_row['SNR_blue']
        snr_green = meta_row['SNR_green']
        snr_red = meta_row['SNR_red']
        
        # Create masks for arms
        mask_blue = (wave_grid >= BLUE_ARM[0]) & (wave_grid <= BLUE_ARM[1])
        mask_green = (wave_grid >= GREEN_ARM[0]) & (wave_grid <= GREEN_ARM[1])
        mask_red = (wave_grid >= RED_ARM[0]) & (wave_grid <= RED_ARM[1])
        
        final_flux = flux_conv.copy()
        
        # Apply masking if SNR is NaN or <= 0
        if np.isnan(snr_blue) or snr_blue <= 0:
            final_flux[mask_blue] = 0
        if np.isnan(snr_green) or snr_green <= 0:
            final_flux[mask_green] = 0
        if np.isnan(snr_red) or snr_red <= 0:
            final_flux[mask_red] = 0
            
        # Add Noise
        rng = np.random.default_rng()
        
        # Blue
        if not np.isnan(snr_blue) and snr_blue > 0:
            sigma_blue = np.abs(final_flux[mask_blue]) / snr_blue
            noise_blue = rng.normal(0, 1, size=np.sum(mask_blue)) * sigma_blue
            final_flux[mask_blue] += noise_blue
            
        # Green
        if not np.isnan(snr_green) and snr_green > 0:
            sigma_green = np.abs(final_flux[mask_green]) / snr_green
            noise_green = rng.normal(0, 1, size=np.sum(mask_green)) * sigma_green
            final_flux[mask_green] += noise_green
            
        # Red
        if not np.isnan(snr_red) and snr_red > 0:
            sigma_red = np.abs(final_flux[mask_red]) / snr_red
            noise_red = rng.normal(0, 1, size=np.sum(mask_red)) * sigma_red
            final_flux[mask_red] += noise_red
            
        # 5. Continuum Normalization
        # Normalize Blue
        if np.any(mask_blue) and np.any(final_flux[mask_blue] > 0):
            flux_b = final_flux[mask_blue]
            wave_b = wave_grid[mask_blue]
            # Use Legendre polynomials for better stability
            # Adjust parameters for Blue arm (shorter, maybe lower degree)
            norm_flux_b, cont_b = cont_norm.legendre_polyfit_huber(
                flux_b, wave_b, degree=4, sigma_lower=2.0, sigma_upper=2.0
            )
            # Check if normalization failed (fallback to ones)
            if np.all(cont_b == 1):
                return None
            final_flux[mask_blue] = norm_flux_b

        # Normalize Green
        if np.any(mask_green) and np.any(final_flux[mask_green] > 0):
            flux_g = final_flux[mask_green]
            wave_g = wave_grid[mask_green]
            norm_flux_g, cont_g = cont_norm.legendre_polyfit_huber(
                flux_g, wave_g, degree=4, sigma_lower=2.0, sigma_upper=2.0
            )
            if np.all(cont_g == 1):
                return None
            final_flux[mask_green] = norm_flux_g
            
        # Normalize Red
        if np.any(mask_red) and np.any(final_flux[mask_red] > 0):
            flux_r = final_flux[mask_red]
            wave_r = wave_grid[mask_red]
            # User suggested degree=5, sigma=1.5 for Red
            norm_flux_r, cont_r = cont_norm.legendre_polyfit_huber(
                flux_r, wave_r, degree=5, sigma_lower=1.5, sigma_upper=1.5
            )
            if np.all(cont_r == 1):
                return None
            final_flux[mask_red] = norm_flux_r
            
        # 6. Final Cleanup
        mask_gaps = ~(mask_blue | mask_green | mask_red)
        final_flux[mask_gaps] = 0
        
        return final_flux, label_row
        
    except Exception as e:
        print(f"Error processing spectrum: {e}")
        return None

def process_from_raw_hdf5():
    """Step 2-6: Process spectra from Raw HDF5."""
    if not os.path.exists(RAW_HDF5_FILE):
        print(f"Raw HDF5 file {RAW_HDF5_FILE} not found. Run create_raw_hdf5 first.")
        return

    print("Loading metadata...")
    meta_csv = RAW_HDF5_FILE.replace(".h5", "_metadata.csv")
    if not os.path.exists(meta_csv):
        print("Metadata CSV not found.")
        return
        
    meta_df = pd.read_csv(meta_csv)
    wave_grid, log_wave_grid = get_wavelength_grid()
    golden_sample_df = load_golden_sample_metadata(METADATA_FILE)
    
    print(f"Processing {len(meta_df)} spectra...")
    
    # We need to read data from HDF5 and pass to workers.
    # Reading HDF5 in main process and passing large arrays to workers via pickle is slow.
    # Better: Workers open HDF5 in read mode.
    # But HDF5 and multiprocessing can be tricky.
    # Safest: Read all data into memory if it fits (10k spectra is small enough).
    
    tasks = []
    with h5py.File(RAW_HDF5_FILE, 'r') as hf:
        grp = hf["raw_spectra"]
        for idx, row in meta_df.iterrows():
            key = row['hdf5_key']
            if key in grp:
                data = grp[key][:]
                # Assuming data is [N, 2] (wave, flux) or similar
                # If it was saved as np.loadtxt, it's likely [N, 2]
                if data.ndim == 2 and data.shape[1] >= 2:
                    raw_wave = data[:, 0]
                    raw_flux = data[:, 1]
                    tasks.append((raw_wave, raw_flux, row, wave_grid, log_wave_grid, golden_sample_df))
    
    # Process
    processed_data = []
    processed_labels = []
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_spectrum_data, tasks)
        
    for res in results:
        if res is not None:
            flux, label = res
            processed_data.append(flux)
            processed_labels.append(label)
            
    # Save
    print(f"Saving to {PROCESSED_HDF5_FILE}...")
    print(f"Total spectra processed: {len(processed_data)}")
    print(f"Excluded spectra: {len(tasks) - len(processed_data)}")

    if not processed_data:
        print("No data processed.")
        return

    processed_data = np.array(processed_data)
    labels_df_final = pd.DataFrame(processed_labels)
    
    with h5py.File(PROCESSED_HDF5_FILE, 'w') as hf:
        hf.create_dataset('spectra', data=processed_data)
        hf.create_dataset('wavelength', data=wave_grid)
        
        grp = hf.create_group('labels')
        for col in labels_df_final.columns:
            if labels_df_final[col].dtype == 'object':
                dt = h5py.special_dtype(vlen=str)
                data = labels_df_final[col].values.astype(object)
                grp.create_dataset(col, data=data, dtype=dt)
            else:
                grp.create_dataset(col, data=labels_df_final[col].values)
                
    print("Done!")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['ingest', 'process', 'all'], default='all', help='Pipeline step')
    args = parser.parse_args()
    
    if args.step in ['ingest', 'all']:
        create_raw_hdf5()
    
    if args.step in ['process', 'all']:
        process_from_raw_hdf5()

if __name__ == "__main__":
    main()
