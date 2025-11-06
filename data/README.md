# data/

Purpose
-------
This directory holds inputs for running the preprocessing scripts (excluding the raw spectra files) and the outputs produced by the preprocessing pipeline. 

Contents
----------------
- `weave_nlte_grids.h5` — re-sampled HDF5 dataset containing all valid synthetic spectra from MPIA/NLTE dataset. Structure:
  - `labels/filenames` (n_spectra,)
  - `labels/stellar_parameters` (n_spectra, n_params)
  - `metadata` attributes (e.g., `n_spectra`, `stellar_param_names`)
  - `spectra/flux` (n_spectra, n_wavelengths) linearly interpolated flux
  - `spectra/wavelength_grid` (n_wavelengths,)

- `processed_spectra.h5` — example processed HDF5 file produced by `preprocessing/preprocess.py` (or other downstream steps). Typical structure used for CNN training:
  - `wavelength` (n_wavelengths,)  — 1D wavelength grid
  - `flux_normalized` (n_spectra, n_wavelengths) — continuum-normalized flux ready for ML
  - `continuum_fits` (n_spectra, n_wavelengths) — fitted continuum models (optional)
  - `original_*` datasets/attributes — any copied metadata from the original HDF5 (e.g., `original_stellar_parameters`)

- `skipped.txt` — list of filenames skipped during preprocessing (NaN-dominated or unrecoverable files).

- `grid_wavelengths_windows.txt` — common wavelength grid to be used for re-sampling raw spectra and calibrate them. Sample size of 500 gives very similar results to sample size of 200.

```
Results from sampled spectra:
  files sampled: 500
  per-file lengths: min=900, median=900, 95p=900, max=900
  central-90.0% wavelength range -> min=4851.924430, max=5400.000000
  chosen step (median per-file Δλ) = 5.100000e-03
  resulting grid points = 107467
```

- `grid_wavelengths_500.txt` — another wavelength grid made using a different (more naive) method, see `generate_uniform_wavelength_grid.py`.

```
Results from sampled spectra:
  files sampled: 500
  per-file lengths: min=1000, median=1000, 95p=1000, max=1000
  central-95.0% wavelength range -> min=4830.000000, max=5400.000000
  chosen step (median per-file Δλ) = 4.900000e-03
  resulting grid points = 116328
```

How files are produced
---------------------
Run the preprocessing conversion script (see `preprocessing/spectrum_grid_reader.py`) to read raw spectra, validate/clean them, interpolate onto a common wavelength grid, and write the consolidated HDF5 file. For smaller jobs or quick experiments use `preprocessing/preprocess.py` which loads an HDF5 file, optionally adds noise, normalizes spectra (e.g. via `legendre_polyfit_huber`) and writes a `processed_spectra.h5` file with keys like `flux_normalized`.

Quick check
-----------
To quickly inspect the HDF5 file from Python:

```python
from preprocessing.hdf5_spectrum_reader import HDF5SpectrumReader
r = HDF5SpectrumReader('data/weave_nlte_grids.h5')
r.print_dataset_summary()
```

Notes
-----
- This folder should contain only data files. Store scripts and notebooks in `preprocessing/`.
- Back up `weave_nlte_grids.h5` if you plan to re-run the conversion pipeline.
- Full list of labels or headers available in the raw MPIA/NLTE spectra (currently looking at 4 labels: teff, log g, met, micro):
```python
# teff, log g, met, micro, macro, vsini
# 5000.0   3.60   0.30   2.00   0.00   0.00
# wmin, wmax, min_sw, max_sw, sw_crit
# R=    0.0
#       4830.0000   5400.0000  5.0000  1.5000   0.10
# NLTE elements
#   01.01  08.01  12.01  14.01  20.01  20.02  22.01  22.02  26.01  25.01  24.01  27.01
#  elements with corrected abundance: corrections [X/Fe]
#     8   10   12   14   16   18   20   22:  0.25  0.25  0.25  0.25  0.25  0.25  0.25  0.25
# HEADER:
########################################################
# wave, flux
```
