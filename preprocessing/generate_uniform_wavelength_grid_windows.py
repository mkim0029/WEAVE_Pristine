#!/usr/bin/env python3
"""
generate_uniform_wavelength_grid_windows.py

Create a uniform wavelength grid by sampling random windows inside each spectrum file.

Behavior differences vs the original script:
 - This copy does NOT read the head (first N) lines. Instead, for each sampled file it
   samples `--preview-windows` random windows (byte-offset based) inside the data region
   and collects the first-column wavelength values found in those windows. It also reads
   the last data line (tail) to get the file-end wavelength.
 - Default behavior: 3 windows of ~300 lines each per file.

Usage example:
    python generate_uniform_wavelength_grid_windows.py /path/to/spectra --sample 1000 \
        --preview-windows 3 --preview-window-size 300 --out grid_windows.txt

Notes:
 - This is intended for large files where reading the full file or only the head may be
   non-representative; random windows sample across the file without scanning it all.
 - Window positions are chosen randomly (using --seed for reproducibility).
"""

import argparse
from pathlib import Path
import random
import numpy as np
import sys
import os
from typing import Optional, Tuple


def _read_last_data_wavelength(path: Path) -> Optional[float]:
    block_size = 4096
    try:
        with path.open('rb') as fh:
            fh.seek(0, 2)
            file_size = fh.tell()
            if file_size == 0:
                return None
            data = bytearray()
            pos = file_size
            while pos > 0:
                to_read = min(block_size, pos)
                pos -= to_read
                fh.seek(pos)
                chunk = fh.read(to_read)
                data = chunk + data
                if b'\n' in data:
                    lines = data.splitlines()
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


def parse_wavelengths_random_windows(path: Path, windows: int = 3, window_size: int = 300,
                                     rng: Optional[random.Random] = None) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Sample random windows inside the file and return collected wavelengths + last wave.

    Returns (collected_array_or_None, last_wave_or_None)
    """
    if rng is None:
        rng = random

    # Find the byte position where data likely starts by searching for the marker
    data_start_byte = None
    try:
        with path.open('rb') as fh:
            # Look in the first 64KB for the marker
            fh.seek(0)
            head = fh.read(65536)
            idx = head.find(b"# wave, flux")
            if idx != -1:
                end_idx = head.find(b"\n", idx)
                if end_idx != -1:
                    data_start_byte = end_idx + 1
            else:
                # streaming search until found or EOF
                fh.seek(0)
                while True:
                    chunk = fh.read(65536)
                    if not chunk:
                        break
                    idx = chunk.find(b"# wave, flux")
                    if idx != -1:
                        pos = fh.tell() - len(chunk) + idx
                        fh.seek(pos)
                        _ = fh.readline()
                        data_start_byte = fh.tell()
                        break
    except Exception:
        data_start_byte = None

    collected = []

    try:
        file_size = path.stat().st_size
    except Exception:
        file_size = None

    # If we couldn't find data_start_byte or file_size, fall back to a light full parse
    if data_start_byte is None or file_size is None or file_size <= data_start_byte:
        # Try a lighter full parse: read lines and collect first column up to a cap
        try:
            waves = []
            data_started = False
            with path.open('r', errors='ignore') as fh:
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
                                waves.append(float(parts[0]))
                            except ValueError:
                                continue
                        if len(waves) >= windows * window_size:
                            break
            arr = np.array(waves, dtype=float) if waves else None
            last_wave = _read_last_data_wavelength(path)
            return arr, last_wave
        except Exception:
            last_wave = _read_last_data_wavelength(path)
            return None, last_wave

    # Choose random start byte offsets for windows
    for _ in range(windows):
        # pick a random offset between data_start_byte and file_size-1
        start_byte = rng.randint(data_start_byte, file_size - 1)
        back = max(data_start_byte, start_byte - 2048)
        try:
            with path.open('rb') as fh:
                fh.seek(back)
                # Move to the next newline to start at a line boundary (unless already at data_start_byte)
                if back != data_start_byte:
                    fh.readline()
                start_of_window = fh.tell()
                # Read a chunk likely to contain window_size lines
                chunk = fh.read(window_size * 128)
                # Ensure the chunk ends at the end of a line: read until next newline or EOF
                extra = b''
                while True:
                    c = fh.read(1024)
                    if not c:
                        break
                    extra += c
                    if b'\n' in c:
                        idx = extra.find(b'\n')
                        chunk += extra[:idx+1]
                        break
                try:
                    text = chunk.decode('utf-8', errors='ignore')
                except Exception:
                    text = ''
                for line in text.splitlines():
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    parts = s.split()
                    if len(parts) >= 1:
                        try:
                            collected.append(float(parts[0]))
                        except ValueError:
                            continue
                    if len(collected) >= windows * window_size:
                        break
        except Exception:
            continue
        if len(collected) >= windows * window_size:
            break

    last_wave = _read_last_data_wavelength(path)
    arr = np.array(collected, dtype=float) if collected else None
    return arr, last_wave


def main():
    parser = argparse.ArgumentParser(description='Generate uniform wavelength grid from sampled spectra using random windows')
    parser.add_argument('directory', type=str, help='Directory containing spectrum files')
    parser.add_argument('--sample', type=int, default=1000, help='Number of spectra to randomly sample (default: 1000)')
    parser.add_argument('--preview-windows', type=int, default=3, help='Number of random windows per file (default: 3)')
    parser.add_argument('--preview-window-size', type=int, default=300, help='Approx number of data lines per window (default: 300)')
    parser.add_argument('--out', type=str, default='../data/grid_wavelengths_windows.txt', help='Output ASCII file for wavelength grid')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling')
    parser.add_argument('--percentile_low', type=float, default=5, help='Lower percentile for central range (default 5)')
    parser.add_argument('--percentile_high', type=float, default=95, help='Upper percentile for central range (default 95)')
    args = parser.parse_args()

    rng = random.Random(args.seed)

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Directory not found: {directory}")
        sys.exit(1)

    sample_n_desired = int(args.sample)
    reservoir = []
    total_files = 0
    try:
        with os.scandir(directory) as it:
            for entry in it:
                try:
                    if not entry.is_file():
                        continue
                except Exception:
                    continue
                total_files += 1
                if len(reservoir) < sample_n_desired:
                    reservoir.append(Path(entry.path))
                else:
                    r = rng.randint(0, total_files - 1)
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

    min_list = []
    max_list = []
    median_steps = []
    lengths = []

    print(f"Sampling {sample_n} files from {total_files} available...")

    for i, f in enumerate(sampled, 1):
        first_waves, last_wave = parse_wavelengths_random_windows(f, windows=args.preview_windows, window_size=args.preview_window_size, rng=rng)
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
            if last_wave is not None:
                lengths.append(1)
                min_list.append(last_wave)
                max_list.append(last_wave)
        if i % 200 == 0:
            print(f"  inspected {i}/{sample_n} files...")

    if len(min_list) == 0:
        print("No wavelength data parsed from sampled files.")
        sys.exit(1)

    min_arr = np.array(min_list)
    max_arr = np.array(max_list)
    lengths = np.array(lengths)

    finite_mask = np.isfinite(min_arr) & np.isfinite(max_arr)
    if not np.any(finite_mask):
        print("No finite per-file min/max values could be computed from sampled files.")
        sys.exit(1)
    min_arr = min_arr[finite_mask]
    max_arr = max_arr[finite_mask]
    lengths = lengths[finite_mask]

    lowp = args.percentile_low
    highp = args.percentile_high

    global_min = float(np.percentile(min_arr, lowp))
    global_max = float(np.percentile(max_arr, highp))

    # Outlier check: min and max should not differ by more than an order of magnitude
    if global_min > 0 and global_max / global_min > 10:
        print(f"ERROR: Computed min ({global_min:.6f}) and max ({global_max:.6f}) differ by more than an order of magnitude.")
        print("This likely means a random window hit a non-representative region or a file with a different unit/outlier.")
        print("Try increasing --percentile_low, filtering outliers, or inspecting files with very small min values.")
        # Print outlier files (min < 0.1 * global_max)
        outlier_threshold = 0.1 * global_max
        print(f"\nSampled files with per-file min < {outlier_threshold:.3f} (potential outliers):")
        for file_path, file_min in zip(sampled, min_list):
            if file_min < outlier_threshold:
                print(f"  {file_path}  min={file_min:.6f}")
        sys.exit(1)

    median_steps = np.array(median_steps)
    median_steps = median_steps[np.isfinite(median_steps) & (median_steps > 0)]
    if median_steps.size == 0:
        print("Could not compute per-file steps; not enough valid step data points.")
        sys.exit(1)
    median_step = float(np.median(median_steps))

    grid = np.arange(global_min, global_max + median_step * 0.5, median_step)

    print("\nResults from sampled spectra:")
    print(f"  files sampled: {sample_n}")
    print(f"  per-file lengths: min={lengths.min()}, median={int(np.median(lengths))}, 95p={int(np.percentile(lengths,95))}, max={lengths.max()}")
    central_fraction = highp - lowp
    print(f"  central-{central_fraction:.1f}% wavelength range -> min={global_min:.6f}, max={global_max:.6f}")
    print(f"  chosen step (median per-file Δλ) = {median_step:.6e}")
    print(f"  resulting grid points = {grid.size}")

    out_path = Path(args.out)
    np.savetxt(out_path, grid, fmt='%.6f')
    print(f"Wavelength grid written to: {out_path.resolve()}")


if __name__ == '__main__':
    main()
