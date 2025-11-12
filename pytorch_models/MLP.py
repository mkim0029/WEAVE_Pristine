#!/usr/bin/env python3
"""
Multilayer Perceptron (MLP) for Stellar Parameter Prediction
============================================================

This script implements a robust MLP model using PyTorch to predict stellar
parameters from processed stellar spectra. It is rewritten for clarity and
correctness, ensuring that target normalization and subsetting are handled
properly.

The script is designed to work with the `SpectralDataset` class, loading
processed HDF5 files and performing a regression task on the stellar parameters.

Key Features:
- Flexible MLP architecture with configurable hidden layers.
- Correct handling of target normalization (standardization).
- Support for training on a subset of available stellar parameters.
- Training and validation loops with RMSE reporting in physical units.
- Command-line interface for easy configuration of training parameters.
- Model and normalization statistics saving for inference.

Usage:
    python MLP.py \\
        --input ../data/processed_spectra.h5 \\
        --epochs 50 \\
        --batch-size 64 \\
        --lr 1e-4 \\
        --model-path ./mlp_model
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
    A simple Multilayer Perceptron model for spectral data regression.

    Parameters
    ----------
    input_size : int
        The number of input features (e.g., wavelength points).
    output_size : int
        The number of output targets (e.g., stellar parameters).
    hidden_layers : list of int, optional
        A list specifying the number of neurons in each hidden layer.
        Defaults to [1024, 512, 256].
    """
    def __init__(self, input_size: int, output_size: int, hidden_layers: list = [1024, 512, 256]):
        super(MLP, self).__init__()

        # preserve hidden layer sizes on the model for saving/inspection
        self.hidden_layers = hidden_layers

        layers = []
        current_size = input_size

        # Create hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            current_size = hidden_size

        # Create output layer
        layers.append(nn.Linear(current_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_size).
        """
        return self.network(x)


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute the L2 norm of gradients across all model parameters.
    
    Returns
    -------
    float
        L2 norm of all gradients, or 0.0 if no gradients exist.
    """
    total_norm = 0.0
    param_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count == 0:
        return 0.0
    
    return total_norm ** 0.5


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
    Train the MLP model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train.
    dataset : SpectralDataset
        The full dataset, used for inverse transforming targets.
    train_loader : DataLoader
        DataLoader for the training set.
    val_loader : DataLoader
        DataLoader for the validation set.
    optimizer : optim.Optimizer
        The optimizer to use for training.
    criterion : nn.Module
        The loss function.
    epochs : int
        The number of epochs to train for.
    device : str
        The device to train on ('cpu' or 'cuda').

    Returns
    -------
    dict
        A dictionary containing training and validation loss history, plus gradient statistics.
    """
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'grad_norm_mean': [], 'grad_norm_max': [], 'grad_norm_min': []}
    model.to(device)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_grad_norms = []
        start_time = time.time()

        for i, (spectra, targets) in enumerate(train_loader):
            spectra, targets = spectra.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(spectra)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Compute gradient norm before optimizer step
            grad_norm = compute_gradient_norm(model)
            epoch_grad_norms.append(grad_norm)
            
            # Check for exploding gradients (> 10.0 is concerning)
            if grad_norm > 10.0:
                print(f"  WARNING: Large gradient norm detected: {grad_norm:.4f} (Epoch {epoch+1}, Batch {i+1})")
            elif grad_norm < 1e-6:
                print(f"  WARNING: Very small gradient norm detected: {grad_norm:.6f} (Epoch {epoch+1}, Batch {i+1})")
            
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_loss)
        
        # Store gradient statistics
        if epoch_grad_norms:
            history['grad_norm_mean'].append(np.mean(epoch_grad_norms))
            history['grad_norm_max'].append(np.max(epoch_grad_norms))
            history['grad_norm_min'].append(np.min(epoch_grad_norms))
        else:
            history['grad_norm_mean'].append(0.0)
            history['grad_norm_max'].append(0.0)
            history['grad_norm_min'].append(0.0)

        # Validation
        val_loss, val_rmse = evaluate_model(model, dataset, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        
        epoch_duration = time.time() - start_time
        grad_mean = history['grad_norm_mean'][-1]
        grad_max = history['grad_norm_max'][-1]
        
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Grad Norm (avg/max): {grad_mean:.4f}/{grad_max:.4f} | "
            f"Duration: {epoch_duration:.2f}s"
        )
        print(f"  Validation RMSE (physical units): {np.round(val_rmse, 2)}")

    print("Training finished.")
    return history


def evaluate_model(
    model: nn.Module,
    dataset: SpectralDataset,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple[float, np.ndarray]:
    """
    Evaluate the model on a given dataset.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    dataset : SpectralDataset
        The full dataset, used for inverse transforming targets.
    loader : DataLoader
        DataLoader for the dataset.
    criterion : nn.Module
        The loss function.
    device : str
        The device to use for evaluation.

    Returns
    -------
    tuple[float, np.ndarray]
        A tuple containing:
        - The average loss over the dataset.
        - The Root Mean Squared Error in physical units for each target parameter.
    """
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for spectra, targets in loader:
            spectra, targets = spectra.to(device), targets.to(device)
            
            outputs = model(spectra)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Inverse transform to get physical units
    physical_outputs = dataset.inverse_transform_target(all_outputs)
    physical_targets = dataset.inverse_transform_target(all_targets)

    # Calculate RMSE in physical units
    err = physical_outputs - physical_targets
    rmse_per_param = torch.sqrt(torch.mean(err**2, dim=0)).numpy()
    
    return total_loss / len(loader), rmse_per_param


def main():
    """Main function to run the MLP training and evaluation."""
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

    # Set random seed for reproducibility
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
    
    # Get the indices corresponding to the training split
    train_indices = train_loader.dataset.indices
    
    # Select the targets for the training set ONLY
    train_targets_np = all_targets_np[train_indices]
    
    # Calculate mean and std
    target_mean = np.mean(train_targets_np, axis=0)
    target_std = np.std(train_targets_np, axis=0)
    
    # Set these stats on the main dataset object. This will activate normalization
    # for all subsequent calls to __getitem__ for all dataloaders.
    dataset.set_target_stats(target_mean, target_std)

    # --- 4. Initialize Model, Loss, and Optimizer ---
    print("\n4. Initializing model...")
    input_size = dataset.n_wavelengths
    output_size = dataset.n_targets  # This is now correctly set in the dataset
    
    print(f"  Input features (wavelengths): {input_size}")
    print(f"  Output features (targets): {output_size}")

    model = MLP(input_size=input_size, output_size=output_size)
    print(model)

    criterion = nn.MSELoss()  # Mean Squared Error for regression
    # Add small weight decay for regularization
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
        val_loss=np.array(history['val_loss']),
        val_rmse=np.array(history['val_rmse']),
        grad_norm_mean=np.array(history['grad_norm_mean']),
        grad_norm_max=np.array(history['grad_norm_max']),
        grad_norm_min=np.array(history['grad_norm_min'])
    )
    print(f"Training history saved to {history_path}")

    # --- 6. Save the Model and Normalization Stats ---
    print(f"\n6. Saving trained model to {Path(args.model_path).with_suffix('.pth')}...")
    try:
        # It's crucial to save the normalization stats with the model
        # for correct inference later.
        torch.save({
            'model_state_dict': model.state_dict(),
            'target_mean': target_mean,
            'target_std': target_std,
            'input_size': input_size,
            'output_size': output_size,
            # Use the model's hidden_layers attribute if present; fall back to None
            'hidden_layers': getattr(model, 'hidden_layers', None)
        }, Path(args.model_path).with_suffix('.pth'))
        print("Model and normalization stats saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("\n" + "=" * 60)
    print("MLP Training Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
