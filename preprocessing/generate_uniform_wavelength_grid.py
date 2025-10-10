#!/usr/bin/env python3
"""
generate_uniform_wavelength_grid.py

Sample spectra and produce a single uniform wavelength grid saved as an ASCII file.

Why this script exists:
 - When you want a single wavelength scale (same min, max, step) for many spectra,
   it's best to derive a representative grid from the dataset rather than guessing.
 - This script samples a subset of spectra (default 1000) and computes robust
   statistics (central percentiles and median Δλ) to produce a stable grid.

Usage example:
    source module_loads.txt && python generate_uniform_wavelength_grid.py \
        /home/minjihk/projects/rrg-kyi/astro/data/weave/nlte-grids/train_nlte \
        --sample 1000 --out grid_wavelengths.txt

Notes:
 - The script only inspects the wavelength columns and does not alter any spectra.
 - It prints the computed min, max, step and the number of grid points.
 - The default behaviour uses the median per-file Δλ as the step, and the central
   percentile range (2.5%/97.5%) of per-file min/max values to avoid outliers.
"""

import argparse
from pathlib import Path
import random
import numpy as np
import sys
from typing import Optional, Tuple
import os


def parse_wavelengths_from_file(path: Path):
    """Parse and return the wavelength column from a single spectrum file.

    The files have a header with a "# wave, flux" marker; this function only
    collects the first column after that marker. It is defensive and skips
    unparsable lines.

    Args:
        path: Path to the spectrum file

    Returns:
        numpy array of wavelength floats (may be empty if parsing failed)
    """
    waves = []
    data_started = False
    try:
        with path.open('r') as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    # skip blank lines
                    continue
                if line == "# wave, flux":
                    # marker that data lines follow
                    data_started = True
                    continue
                if data_started and not line.startswith('#'):
                    # data lines: first column is wavelength
                    parts = line.split()
                    if len(parts) >= 1:
                        try:
                            waves.append(float(parts[0]))
                        except ValueError:
                            # skip lines that don't parse to float
                            continue
    except Exception as e:
        # Do not crash on single-file read errors; report and continue
        print(f"Warning: failed to read {path}: {e}", file=sys.stderr)
    return np.array(waves, dtype=float)


def _read_last_data_wavelength(path: Path) -> Optional[float]:
    """Return the last non-comment data line's first column as float without reading the whole file.

    This function reads the file from the end in chunks and searches backwards for the
    last line that does not start with '#'. It is robust to files with trailing newlines
    and large sizes.
    """
    block_size = 4096
    try:
        with path.open('rb') as fh:
            fh.seek(0, 2)
            file_size = fh.tell()
            if file_size == 0:
                return None
            data = bytearray()
            pos = file_size
            # Read backwards until we find a candidate data line
            while pos > 0:
                to_read = min(block_size, pos)
                pos -= to_read
                fh.seek(pos)
                chunk = fh.read(to_read)
                data = chunk + data
                # If there's at least one newline, try to parse trailing lines
                if b'\n' in data:
                    lines = data.splitlines()
                    # Walk lines from the end and find last non-comment line
                    for raw in reversed(lines):
                        try:
                            s = raw.decode('utf-8', errors='ignore').strip()
                        except Exception:
                            continue
                        if not s:
                            continue
                        if s.startswith('#'):
                            continue
                        parts = s.split()
                        if len(parts) >= 1:
                            try:
                                return float(parts[0])
                            except ValueError:
                                continue
                # If we've reached the start of file, try everything we've got
                if pos == 0:
                    lines = data.splitlines()
                    for raw in reversed(lines):
                        try:
                            s = raw.decode('utf-8', errors='ignore').strip()
                        except Exception:
                            continue
                        if not s or s.startswith('#'):
                            continue
                        parts = s.split()
                        if len(parts) >= 1:
                            try:
                                return float(parts[0])
                            except ValueError:
                                continue
    except Exception:
        return None
    return None


def parse_wavelengths_preview(path: Path, preview_lines_first: int = 1000) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Fast-ish parsing: return the first up-to-`preview_lines_first` wavelengths
    found after the "# wave, flux" marker, and the last data wavelength (if found)
    using a tail reader. This mirrors the earlier simpler behavior.

    Returns (first_waves_array_or_None, last_wave_or_None)
    """
    first_waves = []
    data_started = False
    try:
        with path.open('r') as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                if line == "# wave, flux":
                    data_started = True
                    continue
                if data_started and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 1:
                        try:
                            first_waves.append(float(parts[0]))
                        except ValueError:
                            continue
                    if len(first_waves) >= preview_lines_first:
                        break
    except Exception:
        pass

    last_wave = _read_last_data_wavelength(path)
    arr = np.array(first_waves, dtype=float) if first_waves else None
    return arr, last_wave


def main():
    """Main entry: sample files, compute robust grid statistics, write ASCII grid."""
    parser = argparse.ArgumentParser(description='Generate uniform wavelength grid from sampled spectra')
    parser.add_argument('directory', type=str, help='Directory containing spectrum files')
    parser.add_argument('--sample', type=int, default=500, help='Number of spectra to randomly sample (default: 1000)')
    parser.add_argument('--fast', action='store_true', help='Use fast preview mode (reads only first ~N lines + tail) to speed up large files')
    parser.add_argument('--out', type=str, default='../data/grid_wavelengths.txt', help='Output ASCII file for wavelength grid')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling')
    parser.add_argument('--percentile_low', type=float, default=2.5, help='Lower percentile for central range (default 2.5)')
    parser.add_argument('--percentile_high', type=float, default=97.5, help='Upper percentile for central range (default 97.5)')
    args = parser.parse_args()

    # Set reproducible sampling
    random.seed(args.seed)

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Directory not found: {directory}")
        sys.exit(1)

    # Collect candidate files using a streaming approach to avoid building a large list in memory.
    # We use reservoir sampling so we can pick `args.sample` random files without first listing all files.
    sample_n_desired = int(args.sample)
    reservoir = []
    total_files = 0
    try:
        with os.scandir(directory) as it:
            for entry in it:
                # Use DirEntry.is_file (fast) instead of Path.is_file which does an extra stat
                try:
                    if not entry.is_file():
                        continue
                except Exception:
                    # On strange filesystem errors, skip the entry
                    continue
                total_files += 1
                if len(reservoir) < sample_n_desired:
                    reservoir.append(Path(entry.path))
                else:
                    # Replace elements with decreasing probability
                    r = random.randint(0, total_files - 1)
                    if r < sample_n_desired:
                        reservoir[r] = Path(entry.path)
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}", file=sys.stderr)
        sys.exit(1)

    if total_files == 0:
        print(f"No files found in {directory}")
        sys.exit(1)

    sample_n = len(reservoir)
    sampled = reservoir
    print(f"Found {total_files} files; sampling {sample_n} of them (reservoir sampling)...")

    # Containers for per-file statistics
    min_list = []         # per-file minimum wavelength
    max_list = []         # per-file maximum wavelength
    median_steps = []     # per-file median Δλ
    lengths = []          # per-file number of wavelength points

    print(f"Sampling {sample_n} files from {total_files} available...")

    # Inspect each sampled file: parse wavelengths and compute basic stats.
    # If --fast is set, use a preview parser that reads only the first few data lines
    # and the last data line. This drastically reduces IO for very large files.
    preview_lines = 1000 if args.fast else None
    for i, f in enumerate(sampled, 1):
        if args.fast:
            first_waves, last_wave = parse_wavelengths_preview(f, preview_lines_first=preview_lines)
            # If we got a preview array, use its min as the file min; otherwise try last_wave
            if first_waves is None and last_wave is None:
                continue
            if first_waves is not None and first_waves.size > 0:
                file_min = float(first_waves.min())
                file_max = float(first_waves.max()) if last_wave is None else float(last_wave)
                lengths.append(first_waves.size)
                min_list.append(file_min)
                max_list.append(file_max)
                if first_waves.size > 1:
                    steps = np.diff(first_waves)
                    positive_steps = steps[steps > 0]
                    if positive_steps.size > 0:
                        median_steps.append(np.median(positive_steps))
            else:
                # no preview, but last_wave found -> treat as single-point file
                if last_wave is not None:
                    lengths.append(1)
                    min_list.append(last_wave)
                    max_list.append(last_wave)
        else:
            waves = parse_wavelengths_from_file(f)
            if waves.size == 0:
                # skip files that yielded no data
                continue
            lengths.append(waves.size)
            min_list.append(waves.min())
            max_list.append(waves.max())
            if waves.size > 1:
                steps = np.diff(waves)
                # Guard against any non-positive steps (shouldn't normally happen)
                positive_steps = steps[steps > 0]
                if positive_steps.size > 0:
                    median_steps.append(np.median(positive_steps))
        # Small progress indicator for large samples
        if i % 200 == 0:
            print(f"  inspected {i}/{sample_n} files...")

    if len(min_list) == 0:
        print("No wavelength data parsed from sampled files.")
        sys.exit(1)

    # Convert lists to arrays for percentile/statistics
    min_arr = np.array(min_list)
    max_arr = np.array(max_list)
    lengths = np.array(lengths)

    # Filter out non-finite entries which can appear if preview parsing failed for some files
    finite_mask = np.isfinite(min_arr) & np.isfinite(max_arr)
    if not np.any(finite_mask):
        print("No finite per-file min/max values could be computed from sampled files.")
        sys.exit(1)
    min_arr = min_arr[finite_mask]
    max_arr = max_arr[finite_mask]
    lengths = lengths[finite_mask]

    # Compute the central wavelength range using percentiles to exclude outliers
    lowp = args.percentile_low
    highp = args.percentile_high
    global_min = float(np.percentile(min_arr, lowp))
    global_max = float(np.percentile(max_arr, highp))

    median_steps = np.array(median_steps)
    median_steps = median_steps[np.isfinite(median_steps) & (median_steps > 0)]
    if median_steps.size == 0:
        print("Could not compute per-file steps; not enough valid step data points.")
        sys.exit(1)
    # Typical step is the median across per-file median steps (robust)
    median_step = float(np.median(median_steps))

    # Validate median_step and fall back to robust estimates if necessary
    if not np.isfinite(median_step) or median_step <= 0.0:
        print("Warning: median per-file step is invalid (<=0 or not finite). Attempting fallback estimates...")
        # Estimate per-file step from (max-min)/(n_points-1) where possible
        valid = (lengths > 1)
        alt_steps = None
        if np.any(valid):
            alt_steps = (max_arr[valid] - min_arr[valid]) / (lengths[valid] - 1)
            alt_steps = alt_steps[np.isfinite(alt_steps) & (alt_steps > 0)]
        if alt_steps is not None and alt_steps.size > 0:
            median_step = float(np.median(alt_steps))
            print(f"Using median step from per-file ranges: {median_step:.6e}")
        else:
            # Last resort: choose a default coarse step so we can build a grid
            approx_points = 50000
            median_step = max((global_max - global_min) / float(approx_points), 1e-12)
            print(f"Falling back to default step ~ (range/{approx_points}) = {median_step:.6e}")

    # Build a uniform grid spanning the computed min/max using the chosen step.
    # Use np.arange as in the original simple implementation.
    grid = np.arange(global_min, global_max + median_step * 0.5, median_step)

    # Print informative summary so the user can choose num_points or review results
    print("\nResults from sampled spectra:")
    print(f"  files sampled: {sample_n}")
    print(f"  per-file lengths: min={lengths.min()}, median={int(np.median(lengths))}, 95p={int(np.percentile(lengths,95))}, max={lengths.max()}")
    # central fraction reported (usually ~95%)
    central_fraction = highp - lowp
    print(f"  central-{central_fraction:.1f}% wavelength range -> min={global_min:.6f}, max={global_max:.6f}")
    print(f"  chosen step (median per-file Δλ) = {median_step:.6e}")
    print(f"  resulting grid points = {grid.size}")

    out_path = Path(args.out)
    # Write ASCII file: one wavelength per line with 6 decimal places
    np.savetxt(out_path, grid, fmt='%.6f')
    print(f"Wavelength grid written to: {out_path.resolve()}")


if __name__ == '__main__':
    main()
