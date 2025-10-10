WEAVE Pristine Pipeline — Preprocessing Workflow
==========================================

This README documents the current preprocessing workflow (how raw spectra become the consolidated HDF5 dataset) and points to the main scripts and outputs.

For how each script works, please refer to /preprocessing/README.md.

Workflow
-----------------------
1. Create a representative wavelength grid
   - File: 
     `preprocessing/generate_uniform_wavelength_grid_windows.py`
   - Using default arguments and --fast.
   - Output: ASCII grid file (default: `data/grid_wavelengths.txt` or `data/grid_wavelengths_windows.txt`).

2. Prepare input files
   - Location: `/home/minjihk/projects/rrg-kyi/astro/data/weave/nlte-grids/train_nlte`
   - Files are expected to contain a header marker `# wave, flux` followed by two-column data (wavelength, flux).

3. Convert and validate spectra in parallel
   - File: `preprocessing/spectrum_grid_reader.py`
   - What: The conversion pipeline scans the input directory, then reads spectra in parallel, validating each file:
     - Checks for NaN/Inf in wavelength or flux
     - Detects duplicate wavelengths (keeps first occurrence)
     - Skips files dominated by NaNs (logged to `data/skipped.txt`)
   - Techniques: `numpy` for array ops, `scipy.interpolate` for interpolation, `concurrent.futures` for parallelism.
   - Parallel processing may not have been necessary, but significantly sped up the job time, it ended up taking me ~ 10 min to create the file in step 6.

4. Interpolate flux to the uniform grid
   - File: `preprocessing/spectrum_grid_reader.py` (uses grid created in step 1)
   - What: Linearly interpolate flux arrays onto the single wavelength grid created earlier.
   - Need to be careful for spectra with sharp absorptions?

5. Batch accumulation and checkpointing
   - File: `preprocessing/spectrum_grid_reader.py`
   - What: Accumulate results into memory-efficient batches and periodically write checkpoints so long runs can resume if interrupted.

6. Write final HDF5 dataset using cluster / batch execution
    - File: `preprocessing/grid_to_hdf5.sh`
    - What: SLURM job script that sets resources and runs the conversion pipeline on a cluster. Use `sbatch` to submit.
   - Output file: `data/weave_nlte_grids.h5`
   - Structure: see `data/README.md`
   - Techniques: `h5py` with batch writing for performance.

7. Read and explore the HDF5 dataset
   - File: `preprocessing/hdf5_spectrum_reader.py`
   - What: Convenience class to open `weave_nlte_grids.h5`, read `get_wavelength_flux(index)` and `get_spectrum_info(index)`, and print dataset summaries.

8. Interactive plotting and inspection
   - File: `preprocessing/spectrum_plotting.ipynb`
   - What: Notebook examples to load the HDF5 with `HDF5SpectrumReader`, plot full spectra and zoomed regions, and save plots/CSV for individual spectra.

Quick usage examples
--------------------
- Generate a grid (fast preview mode):

```bash
python preprocessing/generate_uniform_wavelength_grid.py /path/to/raw_spectra --sample 500 --fast --out data/grid_wavelengths.txt
```

- Convert raw spectra to HDF5 (local run):

```bash
python preprocessing/spectrum_grid_reader.py --input-dir data/raw --grid data/grid_wavelengths.txt --out data/weave_nlte_grids.h5
```

- Quick inspect from Python:

```python
from preprocessing.hdf5_spectrum_reader import HDF5SpectrumReader
r = HDF5SpectrumReader('data/weave_nlte_grids.h5')
r.print_dataset_summary()
wavelength, flux = r.get_wavelength_flux(0)
```

Where things live
-----------------
- `preprocessing/`: scripts, helper modules, notebooks, SLURM job scripts.
- `data/`: generated HDF5 file (`weave_nlte_grids.h5`), grid files, and `skipped.txt`.

