# preprocessing/

Purpose
-------
Contains preprocessing scripts, helper modules, notebooks, and job scripts used to convert raw spectra into the canonical HDF5 dataset and to inspect the results.

Key files
---------
- `generate_uniform_wavelength_grid.py` creates a single, uniform wavelength grid by sampling a subset of input spectra (reservoir sampling), optionally inspecting only the first N data lines (--fast mode) plus the file tail per sampled file in fast/preview mode, computing robust per-file min/max and median Δλ (using central percentiles to exclude outliers), and building an np.arange grid spanning the central min–max with the median step; the grid is written to an ASCII file.
- `generate_uniform_wavelength_grid_windows.py` — main script to the uniform wavelength grids; this time, for each sampled file, it extracts wavelength samples from several random byte-offset windows (plus the file tail) instead of reading the head or whole file.
- `spectrum_grid_reader.py` — main conversion pipeline for raw spectra (in ascii) to a hdf5 disk: reads raw spectra, validates/fixes problems (NaNs, unsorted or duplicate wavelengths), interpolates flux to the common grid from `generate_uniform_wavelength_grid_windows.py`, batches results, checkpointing, and writes `data/weave_nlte_grids.h5`.
 - `spectrum_grid_reader.py` — main conversion pipeline for raw spectra (in ascii) to a hdf5 disk: reads raw spectra, validates/fixes problems (NaNs, unsorted or duplicate wavelengths), interpolates flux to the common grid from `generate_uniform_wavelength_grid_windows.py`, batches results, checkpointing, and writes `data/weave_nlte_grids.h5`.
 - `preprocess.py` — a lightweight preprocessing wrapper (added for CNN/data experiments). It can load an existing HDF5 file, add a consistent Gaussian noise vector, apply continuum normalization (e.g. `legendre_polyfit_huber`), and write out a compact `processed_spectra.h5` containing `wavelength`, `flux_normalized`, and optional `continuum_fits` and metadata.
- `hdf5_spectrum_reader.py` — a convenience reader class to open the generated HDF5 file and extract spectra, labels, and dataset summaries. 
 - `hdf5_spectrum_reader.py` — a convenience reader class to open the generated HDF5 file and extract spectra, labels, and dataset summaries. 
- `spectrum_plotting.ipynb` — example Jupyter notebook that demonstrates loading data with `HDF5SpectrumReader`, plotting full spectra and zoomed-in regions, and saving results.
 - `spectrum_plotting.ipynb` — example Jupyter notebook that demonstrates loading data with `HDF5SpectrumReader`, plotting full spectra and zoomed-in regions, and saving results.
 - `../testing/plot_spectrum.ipynb` — quick test notebook that plots wavelength vs flux for the first spectrum in `data/processed_spectra.h5` (handy for smoke tests).
- `grid_to_hdf5.sh` — SLURM job script to run the conversion on a cluster (submit using `sbatch`).

Dependencies
------------
Typical Python packages required:
- numpy
- scipy
- h5py
- matplotlib
- pandas

See /data/modules_loads.txt for running it on a Compute Canada cluster (I used Narval).

```

Usage notes
-----------
- To generate the full HDF5 dataset locally, run `spectrum_grid_reader.py`. Check the script's top-level help or docstring for CLI options.
- For quick preprocessing, testing, or to prepare a dataset for CNN training, use `preprocessing/preprocess.py` which produces a `processed_spectra.h5` with `flux_normalized` suitable for `pytorch_models` code.
- To run on a cluster, either copy the relevant scripts to ~/scratch or call the job script in /data from ~/scratch and run `sbatch grid_to_hdf5.sh` or similar.

```

Tips
----
- When re-running preprocessing, back up `data/weave_nlte_grids.h5` and `data/skipped.txt`.
- The conversion script supports checkpointing: if interrupted it can resume from the last checkpoint filel check the `spectrum_grid_reader.py` documentation for how to resume.
- If you see outlier wavelength min/max values when generating grids, try raising the percentile bounds (e.g., `--percentile_low`) or inspect the flagged files listed by the grid script.
