#!/usr/bin/env python3
"""
Slice processed spectra by wavelength ranges
------------------------------------------------

Reads a processed HDF5 file (output from `preprocess.py`) and slices each
spectrum to the wavelength ranges defined in a simple text file
(`test_slice_grid.txt`). The script writes a new HDF5 file containing only
the sliced flux and the sliced wavelength grid. Other metadata / label
datasets (e.g., stellar parameters, filenames) are copied across.

Usage:
    python data/slice.py \
        --input ../data/processed_spectra_full.h5 \
        --grid ../data/test_slice_grid.txt \
        --output ../data/processed_spectra_full_sliced.h5

Grid file format (flexible): each non-empty line may contain either
 - two numbers separated by whitespace or a hyphen: `start end` or `start-end`
 - a single number (will include the closest wavelength point)
Comments (lines starting with `#`) and blank lines are ignored.
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
import sys


def parse_grid_file(path: Path):
    ranges = []
    vals = []
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            # Accept formats: "start end" or "start-end" or single value
            if '-' in line and not line.strip().count(' ') >= 1:
                parts = [p.strip() for p in line.split('-', 1)]
            else:
                parts = line.split()
            try:
                nums = [float(p) for p in parts if p != '']
            except ValueError:
                raise ValueError(f"Could not parse numbers from grid line: {line}")

            if len(nums) == 2:
                start, end = nums
                if end < start:
                    start, end = end, start
                ranges.append((start, end))
            elif len(nums) == 1:
                vals.append(nums[0])
            else:
                raise ValueError(f"Unexpected number of numeric values in line: {line}")

    return ranges, vals


def choose_keys(h5):
    # Heuristics to find flux and wavelength datasets
    candidates = list(h5.keys())
    flux_key = None
    wavelength_key = None

    # Common keys
    for key in ['flux_normalized', 'spectra/flux_normalized']:
        if key in h5:
            flux_key = key
            break

    for key in ['wavelength', 'wavelength_grid', 'spectra/wavelength', 'wavelengths']:
        if key in h5:
            wavelength_key = key
            break

    # Try nested groups (e.g., 'spectra' group)
    if flux_key is None and 'spectra' in h5:
        grp = h5['spectra']
        for key in ['flux', 'flux_normalized']:
            if key in grp:
                flux_key = 'spectra/' + key
                break

    if wavelength_key is None and 'spectra' in h5:
        grp = h5['spectra']
        for key in ['wavelength', 'wavelength_grid']:
            if key in grp:
                wavelength_key = 'spectra/' + key
                break

    return flux_key, wavelength_key


def main():
    parser = argparse.ArgumentParser(description='Slice processed spectra HDF5 by wavelength ranges')
    parser.add_argument('--input', '-i', required=True, help='Input processed HDF5 file')
    parser.add_argument('--grid', '-g', default='test_slice_grid.txt', help='Grid file listing wavelength ranges (one per line)')
    parser.add_argument('--output', '-o', default='processed_spectra_full_sliced.h5', help='Output HDF5 path (sliced)')
    parser.add_argument('--chunk', type=int, default=512, help='Chunk size when writing spectra')
    args = parser.parse_args()

    input_path = Path(args.input)
    grid_path = Path(args.grid)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)
    if not grid_path.exists():
        print(f"Grid file not found: {grid_path}")
        sys.exit(1)

    ranges, single_vals = parse_grid_file(grid_path)
    print(f"Parsed {len(ranges)} ranges and {len(single_vals)} single points from {grid_path}")

    with h5py.File(input_path, 'r') as inf:
        flux_key, wavelength_key = choose_keys(inf)
        if flux_key is None or wavelength_key is None:
            print("Could not find suitable flux/wavelength datasets in the input file. Available keys:", list(inf.keys()))
            sys.exit(1)

        print(f"Using flux key: {flux_key}")
        print(f"Using wavelength key: {wavelength_key}")

        wavelength = inf[wavelength_key][:]
        # Build boolean mask for selected wavelength indices
        mask = np.zeros_like(wavelength, dtype=bool)

        for (start, end) in ranges:
            mask |= (wavelength >= start) & (wavelength <= end)

        if len(single_vals) > 0:
            for val in single_vals:
                # pick closest wavelength point
                idx = np.argmin(np.abs(wavelength - val))
                mask[idx] = True

        selected_idx = np.where(mask)[0]
        if selected_idx.size == 0:
            print("No wavelength points matched the provided grid ranges/values. Exiting.")
            sys.exit(1)

        new_wavelength = wavelength[selected_idx]
        print(f"Selected {len(selected_idx)} wavelength points (range {new_wavelength[0]:.4f} - {new_wavelength[-1]:.4f})")

        # Determine input flux dataset shape and dtype
        flux_ds = inf[flux_key]
        n_spectra = flux_ds.shape[0]
        dtype = flux_ds.dtype

        # Open output file and create datasets
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, 'w') as outf:
            # Create wavelength dataset for sliced grid
            outf.create_dataset('wavelength', data=new_wavelength, dtype=new_wavelength.dtype)

            # Create flux dataset with shape (n_spectra, n_selected)
            n_selected = selected_idx.size
            dset = outf.create_dataset(
                'flux_normalized',
                shape=(0, n_selected),
                maxshape=(None, n_selected),
                dtype=dtype,
                compression='gzip',
                chunks=(min(args.chunk, n_spectra), n_selected)
            )

            # Reserve and write in chunks
            written = 0
            chunk = args.chunk
            for start in range(0, n_spectra, chunk):
                end = min(start + chunk, n_spectra)
                block = flux_ds[start:end, :]
                # slice columns
                sliced = block[:, selected_idx]
                # append to dset
                old_rows = dset.shape[0]
                new_rows = old_rows + sliced.shape[0]
                dset.resize((new_rows, n_selected))
                dset[old_rows:new_rows, :] = sliced
                written += sliced.shape[0]
                print(f"Wrote spectra {start}:{end} (total written: {written})")

            # Copy label / metadata datasets if present (common names)
            copy_keys = ['original_stellar_parameters', 'stellar_parameters', 'targets', 'filenames', 'parameters', 'parameter_names']
            for key in copy_keys:
                if key in inf:
                    try:
                        outf.create_dataset(key, data=inf[key][:], compression='gzip')
                        print(f"Copied dataset: {key}")
                    except Exception:
                        # try attribute copy if not dataset
                        pass

            # Also copy any top-level attrs
            for k, v in inf.attrs.items():
                try:
                    outf.attrs[k] = v
                except Exception:
                    # skip attributes that cannot be serialized
                    pass

            # Add provenance info
            outf.attrs['source_file'] = str(input_path)
            outf.attrs['source_flux_key'] = flux_key
            outf.attrs['sliced_by_grid'] = str(grid_path)

    print(f"Sliced HDF5 written to: {output_path}")


if __name__ == '__main__':
    main()
