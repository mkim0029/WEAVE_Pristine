#!/usr/bin/env python3
"""
Multilayer Perceptron (MLP) for Stellar Parameter Prediction
============================================================

Usage:
        python MLP.py \\
                --input ../data/processed_spectra.h5 \\
                --epochs 50 \\
                --batch-size 64 \\
                --lr 1e-4 \\
                --model-path ./mlp_model

Description:
        This script trains a simple feed-forward neural network (MLP) to predict
        stellar parameters from preprocessed spectral data stored in an HDF5 file.

        The script performs the following high-level steps:
        1. Parse command-line arguments for dataset path, training settings, and device.
        2. Load the custom `SpectralDataset` which provides spectra and target labels.
        3. Split the dataset into training and validation sets using helper utilities.
        4. Compute normalization statistics from the training split and apply them to
             the dataset so targets are standardized for training.
        5. Instantiate an MLP model, loss function, and optimizer.
        6. Train the model while tracking training and validation loss.
        7. Save the model state along with normalization statistics and a small
             training history file for later analysis / inference.

Notes:
        - This file only contains a relatively small example MLP. For production use
            you may want to add logging, checkpointing, better error handling, and
            configurable model architectures.
        - No changes to the runtime behavior or model logic are made by the
            comments and docstrings in this file — they are purely explanatory.
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path for local imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from pytorch_models.spectral_dataset import SpectralDataset, create_spectral_dataloaders, SpectralTransforms
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure that 'spectral_dataset.py' is in the 'pytorch_models' directory.")
    sys.exit(1)

class MLP(nn.Module):
    """
    Small fully-connected feed-forward neural network (MLP).

    This model uses a fixed architecture with two hidden layers (default sizes
    1024 and 512) and ReLU activations. It is intended for regression: the
    final layer outputs `output_size` continuous values (e.g., stellar params).

    Parameters
    ----------
    input_size : int
        Number of input features (the number of wavelength points per spectrum).
    output_size : int
        Number of output targets (e.g., Teff, logg, [Fe/H], [alpha/Fe]).
    hidden_layers : list
        Two-element list specifying hidden layer sizes. The list is also stored
        on the model instance to make it easy to save or inspect the config.
    """
    def __init__(self, input_size: int, output_size: int, hidden_layers: list = [1024, 512]):
        super().__init__()
        # preserve hidden layer sizes on the model for saving/inspection
        self.hidden_layers = hidden_layers

        # Build a simple sequential network: Linear -> ReLU -> Linear -> ReLU -> Linear
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, output_size).
        """
        return self.linear_relu_stack(x)

def train_model(
    model: nn.Module,
    dataset: SpectralDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: str
) -> dict:
    """
    Train the provided model using the given DataLoaders.

    The function runs a standard supervised training loop for `epochs` epochs.
    It records average training loss per epoch and computes validation loss
    after each epoch using `evaluate_model`.

    Parameters
    ----------
    model : nn.Module
        Model instance to train.
    dataset : SpectralDataset
        Dataset object (not directly iterated here but kept for context).
    train_loader : DataLoader
        DataLoader yielding training batches (spectra, targets).
    val_loader : DataLoader
        DataLoader for validation/evaluation.
    optimizer : optim.Optimizer
        Optimizer for parameter updates.
    criterion : nn.Module
        Loss function (e.g., nn.MSELoss for regression).
    epochs : int
        Number of training epochs.
    device : str
        Device string ('cuda' or 'cpu') to move tensors to.

    Returns
    -------
    dict
        Dictionary containing lists for 'train_loss' and 'val_loss'.
    """
    history = {'train_loss': [], 'val_loss': []}
    model.to(device)

    print("Starting training...")
    for epoch in range(epochs):
        # Set model to training mode (enables dropout, batchnorm behavior, etc.)
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch, (spectra, targets) in enumerate(train_loader):
            # Move inputs to device and compute predictions
            pred = model(spectra.to(device))
            loss = criterion(pred, targets.to(device))

            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
        
        # Average training loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_loss)

        # Compute validation loss using the helper function
        val_loss = evaluate_model(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        
        epoch_duration = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Duration: {epoch_duration:.2f}s"
        )

    print("Training finished.")
    return history

def evaluate_model(model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Evaluate the model over all batches provided by `loader` and return the
    average loss.

    This function runs in inference mode (no gradient computation) to get a
    deterministic estimate of performance on validation or test splits.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for spectra, targets in loader:
            pred = model(spectra.to(device))
            loss = criterion(pred, targets.to(device))
            running_loss += loss.item()

    return running_loss / len(loader)

def main():
    """
    Main entry point for training.

    The function parses CLI arguments, loads the dataset, computes training
    statistics for normalizing the targets, constructs the model, and runs the
    training loop. After training completes, it saves a small history file and
    a checkpoint containing both the model weights and the target normalization
    statistics (mean/std) required for correct inference later.
    """
    parser = argparse.ArgumentParser(
        description="Train an MLP for stellar parameter prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', '-i', required=True, help='Path to the processed HDF5 spectral data file.')
    parser.add_argument('--model-path', type=str, default='mlp_model.pth', help='Path to save the trained model.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cuda/cpu).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    args = parser.parse_args()

    # Set random seed for reproducibility across runs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("MLP Training for Stellar Parameter Prediction")
    print("=" * 60)
    print(f"Arguments: {vars(args)}")

    # --- 1. Load Dataset ---
    print("\n1. Loading dataset...")
    try:
        # This dataset object will be shared by the DataLoaders
        dataset = SpectralDataset(
            hdf5_filepath=args.input,
            load_targets=True,
            target_key='original_stellar_parameters',
            use_target_indices=[0, 1, 2, 3],  # Use Teff, logg, [Fe/H], [a/Fe]
            transform=SpectralTransforms.standardize,
            device=args.device
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # --- 2. Create DataLoaders ---
    print("\n2. Creating DataLoaders...")
    dataloaders = create_spectral_dataloaders(
        dataset,
        batch_size=args.batch_size,
        train_split=0.8,
        val_split=0.2,
        random_seed=args.seed
    )
    train_loader, val_loader = dataloaders[0], dataloaders[1]

    # --- 3. Calculate and Set Target Normalization Statistics ---
    print("\n3. Calculating target statistics from the TRAINING set...")
    # Get all targets from the dataset (already subsetted if use_target_indices was passed)
    all_targets_np = dataset.get_all_targets()
    
    # Get the indices corresponding to the training split (DataLoader keeps these)
    train_indices = train_loader.dataset.indices
    
    # Select the targets for the training set ONLY to avoid data leakage
    train_targets_np = all_targets_np[train_indices]
    
    # Calculate mean and std on the training split and store them for later use
    target_mean = np.mean(train_targets_np, axis=0)
    target_std = np.std(train_targets_np, axis=0)
    
    # Attach the statistics to the dataset so __getitem__ returns normalized targets
    # This ensures consistent normalization across all dataloaders and during inference
    dataset.set_target_stats(target_mean, target_std)

    # --- 4. Initialize Model, Loss, and Optimizer ---
    print("\n4. Initializing model...")
    input_size = dataset.n_wavelengths
    output_size = dataset.n_targets  # Number of targets selected in the dataset
    
    print(f"  Input features (wavelengths): {input_size}")
    print(f"  Output features (targets): {output_size}")

    # Instantiate the model and print a summary
    model = MLP(input_size=input_size, output_size=output_size)
    print(model)

    # Mean Squared Error is a common choice for regression problems
    criterion = nn.MSELoss()
    # Use Adam optimizer with a small weight decay for light regularization
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # --- 5. Train the Model ---
    history = train_model(
        model=model,
        dataset=dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=args.epochs,
        device=args.device
    )

    # Save training history as a numpy .npz file for efficient reading
    history_path = Path(args.model_path).with_suffix('.history.npz')
    np.savez(
        history_path,
        train_loss=np.array(history['train_loss']),
        val_loss=np.array(history['val_loss'])
    )
    print(f"Training history saved to {history_path}")

    # --- 6. Save the Model and Normalization Stats ---
    print(f"\n6. Saving trained model to {Path(args.model_path).with_suffix('.pth')}...")
    try:
        # Save a checkpoint dict containing weights and the normalization stats
        # so the model can be restored and used for inference with proper scaling.
        torch.save({
            'model_state_dict': model.state_dict(),
            'target_mean': target_mean,
            'target_std': target_std,
            'input_size': input_size,
            'output_size': output_size,
            'hidden_layers': getattr(model, 'hidden_layers', None)
            #'hidden_layers':model.network[0].out_features
        }, Path(args.model_path).with_suffix('.pth'))
        print("Model and normalization stats saved successfully.")
    except Exception as e:
        # Print a user-friendly error if the save fails (e.g., permission/IO issue)
        print(f"Error saving model: {e}")

    print("\n" + "=" * 60)
    print("MLP Training Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
