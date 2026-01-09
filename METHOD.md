# WEAVE-Pristine Pipeline Methodology

This document details the data processing pipeline and machine learning model architecture used in the WEAVE-Pristine project.

## 1. Data Processing Pipeline

The data processing pipeline converts raw synthetic spectra into a standardized, normalized HDF5 dataset suitable for training machine learning models.

### 1.1 Wavelength Grid Generation
*   **Script**: `preprocessing/build_dataset.py` (Internal function `get_wavelength_grid`)
*   **Method**: Generates a log-linear wavelength grid spanning **4040 Å to 6850 Å**.
*   **Resolution**: The grid is defined by a reference step of 0.05 Å at 4040 Å, resulting in a constant logarithmic step size (`LOG_WAVE_STEP`). This ensures uniform velocity sampling across the entire spectral range, matching the requirements for high-resolution spectroscopy.

### 1.2 Dataset Construction (Offline)
*   **Script**: `preprocessing/build_dataset.py`
*   **Input**: Raw text/CSV files containing wavelength and flux.
*   **Process**:
    1.  **Ingestion & Filtering**: Scans raw data directories and matches spectra with metadata.
        *   **Performance**: Utilizes parallel workers and high-speed I/O (`pandas`) for efficient ingestion of 10k+ spectra.
        *   **Quality Control**: Automatically excludes spectra with unphysical flux values (negative or $>10^{17}$).
    2.  **Interpolation**: Linearly interpolates raw flux onto the common log-linear wavelength grid.
    3.  **LSF Convolution**: Convolves the interpolated spectrum with a Gauss-Hermite Line Spread Function (LSF) to simulate the instrument resolution.
        *   **Resolution**: Randomly sampled from the Golden Sample metadata (`RESOL`).
        *   **Kernel**: Gaussian core with randomized Hermite coefficients ($h_3, h_4$) to model asymmetric line profiles.
    4.  **Noise Injection**: Adds Gaussian noise based on Signal-to-Noise Ratios (SNR) for Blue, Green, and Red arms, sampled from the Golden Sample.
    5.  **Masking**: Zeros out flux in gap regions between the Blue (4040-4650 Å), Green (4730-5450 Å), and Red (5950-6850 Å) arms.
    6.  **Normalization**: Applies continuum normalization using `preprocessing/cont_norm.py` to each arm, independently.
        *   **Algorithm**: Robust Legendre polynomial fitting.
        *   **Loss Function**: Huber loss (robust to outliers/lines).
        *   **Clipping**: Asymmetric sigma clipping (sigma_lower=0.5) to reject absorption lines.
        *   **Clamping**: Continuum is clamped to be at least 1% of the median flux to prevent division by zero or extreme values.
    7.  **Storage**: Saves the processed data to an HDF5 file (`processed_spectra_10k.h5`).
        *   `spectra`: Normalized flux arrays.
        *   `continuua`: Fitted continuum arrays (for debugging/analysis).
        *   `labels`: Stellar parameters (Teff, logg, [Fe/H], vmic, etc.) and abundances.
        *   `metadata`: Additional info (star names, S/N).

### 1.3 Online Augmentation (Optional)
*   **Script**: `ML_models/online_dataset.py`
*   **Method**: Instead of using pre-normalized spectra, this module loads raw spectra and applies normalization and noise injection on-the-fly during training. This allows for infinite variations of noise realizations.

## 2. Machine Learning Model

The core model is a 1D Convolutional Neural Network (CNN) inspired by the StarNet architecture, adapted for the high-resolution WEAVE spectra.

### 2.1 Architecture (`ML_models/cnn.py`)
*   **Input**: 1D array of normalized flux values (~42,000 pixels).
*   **Output**: 28 continuous variables (Stellar parameters + Chemical abundances).

**Layer Configuration:**
1.  **Conv1D**: 1 input channel $\to$ 4 output channels, Kernel size = 15, ReLU activation.
2.  **Conv1D**: 4 input channels $\to$ 16 output channels, Kernel size = 15, ReLU activation.
3.  **MaxPool1D**: Pooling size = 20 (Reduces dimensionality significantly).
4.  **Flatten**: Converts 3D tensor to 1D vector.
5.  **Fully Connected (Dense)**: 256 neurons, ReLU activation.
6.  **Fully Connected (Dense)**: 128 neurons, ReLU activation.
7.  **Output Layer**: 28 neurons (Linear activation).

### 2.2 Training Strategy
*   **Loss Function**: Mean Squared Error (MSE).
*   **Optimizer**: Adam (Learning rate $\approx 10^{-4}$).
*   **Target Normalization**: Target labels (Teff, logg, etc.) are standardized (z-score normalization) using statistics from the training set. These statistics are saved in the model checkpoint for inverse transformation during inference.

## 3. Evaluation Pipeline

### 3.1 Inference
*   **Script**: `results/evaluate_model.py`
*   **Process**:
    1.  Loads the trained model checkpoint (`.pth`).
    2.  Loads the test set indices (`_test_indices.npy`).
    3.  Runs forward pass on test spectra.
    4.  Inverse-transforms predictions back to physical units.
    5.  Saves results to `results/test_output/test_predictions.npz`.

### 3.2 Visualization
*   **Notebook**: `results/plots.ipynb`
*   **Diagnostics**:
    *   **Violin Plots**: Distribution of residuals (Prediction - Truth) for each parameter.
    *   **Global Summary**: Bar charts of RMSE and Bias for all 28 parameters.
    *   **Parity Plots**: Scatter plots of Predicted vs. True values.
    *   **Gradient Analysis**: Monitoring gradient norms to detect vanishing/exploding gradients.
