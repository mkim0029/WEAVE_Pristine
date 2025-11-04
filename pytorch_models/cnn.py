#!/usr/bin/env python3
"""
Convolutional Neural Network (CNN) for Stellar Parameter Prediction
============================================================

Usage:
    python CNN.py \\
        --input ../data/processed_spectra.h5 \\
        --epochs 50 \\
        --batch-size 64 \\
        --lr 1e-4 \\
        --model-path ./cnn_model
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
    def __init__(self, input_size: int, output_size: int, hidden_layers: list = [1024, 512]):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    model.to(device)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch, (spectra, targets) in enumerate(train_loader):
            # Compute prediction and loss
            pred = model(spectra.to(device))
            loss = criterion(pred, targets.to(device))

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_loss)

        # Validation
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
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for spectra, targets in loader:
            pred = model(spectra.to(device))
            loss = criterion(pred, targets.to(device))
            running_loss += loss.item()

    return running_loss / len(loader)

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
        val_rmse=np.array(history['val_rmse'])
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
            'hidden_layers': model.network[0].out_features # A bit of a hack to get first hidden layer size
        }, Path(args.model_path).with_suffix('.pth'))
        print("Model and normalization stats saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("\n" + "=" * 60)
    print("MLP Training Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()