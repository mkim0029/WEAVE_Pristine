WEAVE Pristine Pipeline
=======================

This repository contains the pipeline for processing WEAVE-Pristine synthetic spectra, training Machine Learning models (CNNs), and evaluating their performance.

For a detailed description of the methodology and model architecture, please see [METHOD.md](METHOD.md).

Workflow
--------

### 1. Data Preparation
**Goal**: Create a standardized, normalized HDF5 dataset from raw synthetic spectra.

Run the full pipeline:
```bash
python preprocessing/build_dataset.py --step all
```
Or

1.  **Generate Wavelength Grid, Master HDF5 file (spectra) and Master CSV file (labels)**:
    ```bash
    python preprocessing/build_dataset.py --step ingest
    ```
    Creates a log-linear wavelength grid (4040-6850 Å).

2.  **Build Dataset (Offline Processing)**:
    ```bash
    python preprocessing/build_dataset.py --step process
    ```
    *   Ingests raw spectra.
    *   Interpolates to the common grid.
    *   Performs continuum normalization (Legendre fit (Huber loss) + sigma clipping).
    *   Saves to `data/processed_spectra_10k.h5`.

For online processing, see **Online Training** below (replacing step 2 above), which relies on the outputs of step 1 above.

### 2. Model Training
**Goal**: Train a CNN to predict stellar parameters and abundances.

*   **Offline Training** (Uses pre-processed HDF5):
    ```bash
    sbatch jobs/train_cnn_offline.sh
    ```
    Or locally:
    ```bash
    python ML_models/cnn.py --input data/processed_spectra_10k.h5 --model-path ML_models/output/cnn_model_offline
    ```

*   **Online Training** (On-the-fly augmentation):
    ```bash
    sbatch jobs/train_cnn_online.sh
    ```

### 3. Evaluation & Testing
**Goal**: Assess model performance on a held-out test set.

1.  **Run Inference**:
    ```bash
    python results/evaluate_model.py \
        --input data/processed_spectra_10k.h5 \
        --model-path ML_models/output/cnn_model_offline.pth \
        --indices ML_models/output/cnn_model_offline_test_indices.npy \
        --output-dir results/test_output
    ```
    Generates `test_predictions.npz`.

2.  **Visualize Results**:
    Open `results/plots.ipynb` to generate:
    *   Violin plots of residuals.
    *   Global RMSE/Bias summary for all 28 targets.
    *   Parity plots (Truth vs Predicted).

Directory Structure
-------------------
*   `preprocessing/`: Scripts for grid generation, normalization, and HDF5 creation.
*   `ML_models/`: PyTorch model definitions (`cnn.py`) and dataset loaders.
*   `jobs/`: SLURM scripts for cluster execution.
*   `results/`: Evaluation scripts and visualization notebooks.
*   `data/`: Stores generated HDF5 files and grids.