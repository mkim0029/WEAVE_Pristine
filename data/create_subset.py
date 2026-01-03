import h5py
import pandas as pd
import os

# Paths
DATA_DIR = "data"
SOURCE_H5 = os.path.join(DATA_DIR, "raw_spectra_10k.h5")
SOURCE_META = os.path.join(DATA_DIR, "raw_spectra_10k_metadata.csv")
TARGET_H5 = os.path.join(DATA_DIR, "raw_spectra_4.h5")
TARGET_META = os.path.join(DATA_DIR, "raw_spectra_4_metadata.csv")

# Indices to select
INDICES = [1, 223, 45, 3465]

def create_subset():
    print(f"Reading metadata from {SOURCE_META}...")
    df = pd.read_csv(SOURCE_META)
    
    # Select rows
    subset_df = df.iloc[INDICES].copy()
    print(f"Selected {len(subset_df)} rows.")
    
    # Save subset metadata
    subset_df.to_csv(TARGET_META, index=False)
    print(f"Saved metadata to {TARGET_META}")
    
    # Create subset HDF5
    print(f"Creating subset HDF5 at {TARGET_H5}...")
    with h5py.File(SOURCE_H5, 'r') as source_hf, h5py.File(TARGET_H5, 'w') as target_hf:
        grp_out = target_hf.create_group("raw_spectra")
        grp_in = source_hf["raw_spectra"]
        
        for idx, row in subset_df.iterrows():
            key = row['hdf5_key']
            if key in grp_in:
                data = grp_in[key][:]
                grp_out.create_dataset(key, data=data, compression="gzip")
                print(f"Copied {key}")
            else:
                print(f"Warning: Key {key} not found in source HDF5")

if __name__ == "__main__":
    create_subset()
