#!/usr/bin/env python3
"""
Convolutional Neural Network (CNN) for Stellar Parameter Prediction
============================================================

Usage:
    python cnn.py \\
        --input ../data/processed_spectra.h5 \\
        --epochs 50 \\
        --batch-size 64 \\
        --lr 1e-4 \\
        --model-path ./cnn_model

Description:
    This script trains a 1D convolutional neural network (CNN) following the
    StarNet2017 architecture to predict stellar parameters from preprocessed 
    spectral data stored in an HDF5 file.

    The CNN architecture consists of:
    1. Two 1D convolutional layers (1->4->16 filters) with ReLU activation
    2. Max pooling layer (pool_size=4)
    3. Flattening and two fully connected layers (256->128 neurons)
    4. Final linear layer for regression output

    The script performs the following high-level steps:
    1. Parse command-line arguments for dataset path, training settings, and device.
    2. Load the custom `SpectralDataset` which provides spectra and target labels.
    3. Split the dataset into training and validation sets using helper utilities.
    4. Compute normalization statistics from the training split and apply them to
         the dataset so targets are standardized for training.
    5. Instantiate a CNN model, loss function, and optimizer.
    6. Train the model while tracking training and validation loss.
    7. Save the model state along with normalization statistics and a small
         training history file for later analysis / inference.

Notes:
    - This CNN expects 1D spectral input with shape (batch, 1, length).
    - The model automatically handles input reshaping if needed.
    - Architecture parameters are saved with the checkpoint for easy inference.
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

class CNN(nn.Module):
    def __init__(self, kernel_size: int, input_length: int, output_size: int):
        super().__init__()
        
        # Store parameters for saving/inspection
        self.kernel_size = kernel_size
        self.input_length = input_length
        self.output_size = output_size
        
        # Convolutional and pooling layers
        self.conv1 = nn.Conv1d(1, 4, kernel_size)
        self.conv2 = nn.Conv1d(4, 16, kernel_size)
        self.pool = nn.MaxPool1d(4, 4)
        
        # Calculate output size after conv and pooling operations
        # After conv1: length = input_length - kernel_size + 1
        # After conv2: length = (input_length - kernel_size + 1) - kernel_size + 1 = input_length - 2*kernel_size + 2
        # After pool: length = floor((input_length - 2*kernel_size + 2) / 4)
        conv_output_length = input_length - 2 * kernel_size + 2
        pool_output_length = conv_output_length // 4
        
        if pool_output_length <= 0:
            raise ValueError(f"Input length {input_length} too small for kernel size {kernel_size}")
        
        flattened_size = 16 * pool_output_length
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has channel dimension
        if x.dim() == 2:  # (batch_size, input_length)
            x = x.unsqueeze(1)  # (batch_size, 1, input_length)
            
        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

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
    Train the provided model using the given DataLoaders.

    The function runs a standard supervised training loop for `epochs` epochs.
    It records average training loss per epoch and computes validation loss
    after each epoch using `evaluate_model`. Also monitors gradient norms
    to detect vanishing/exploding gradients.

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
        Dictionary containing lists for 'train_loss', 'val_loss', and gradient statistics.
    """
    history = {'train_loss': [], 'val_loss': [], 'grad_norm_mean': [], 'grad_norm_max': [], 'grad_norm_min': []}
    model.to(device)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_grad_norms = []
        start_time = time.time()

        for batch, (spectra, targets) in enumerate(train_loader):
            # Compute prediction and loss
            pred = model(spectra.to(device))
            loss = criterion(pred, targets.to(device))

            # Backpropagation
            loss.backward()
            
            # Compute gradient norm before optimizer step
            grad_norm = compute_gradient_norm(model)
            epoch_grad_norms.append(grad_norm)
            
            # Check for exploding gradients (> 10.0 is concerning)
            if grad_norm > 10.0:
                print(f"  WARNING: Large gradient norm detected: {grad_norm:.4f} (Epoch {epoch+1}, Batch {batch+1})")
            elif grad_norm < 1e-6:
                print(f"  WARNING: Very small gradient norm detected: {grad_norm:.6f} (Epoch {epoch+1}, Batch {batch+1})")
            
            optimizer.step()
            optimizer.zero_grad()

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
        val_loss = evaluate_model(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        
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
    """Main function to run the CNN training and evaluation."""
    parser = argparse.ArgumentParser(
        description="Train a CNN for stellar parameter prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', '-i', required=True, help='Path to the processed HDF5 spectral data file.')
    parser.add_argument('--model-path', type=str, default='cnn_model.pth', help='Path to save the trained model.')
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
    print("CNN Training for Stellar Parameter Prediction")
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
    input_length = dataset.n_wavelengths
    output_size = dataset.n_targets  # This is now correctly set in the dataset
    kernel_size = 8  # You can make this a command line argument if desired
    
    print(f"  Input length (wavelengths): {input_length}")
    print(f"  Output features (targets): {output_size}")
    print(f"  Kernel size: {kernel_size}")

    # Instantiate your CNN model
    model = CNN(kernel_size=kernel_size, input_length=input_length, output_size=output_size)
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
        grad_norm_mean=np.array(history['grad_norm_mean']),
        grad_norm_max=np.array(history['grad_norm_max']),
        grad_norm_min=np.array(history['grad_norm_min'])
    )
    print(f"Training history saved to {history_path}")

    # --- 6. Save the Model and Architecture Info ---
    print(f"\n6. Saving trained model to {Path(args.model_path).with_suffix('.pth')}...")
    try:
        # It's crucial to save the normalization stats with the model
        # for correct inference later.
        torch.save({
            'model_state_dict': model.state_dict(),
            'target_mean': target_mean,
            'target_std': target_std,
            'input_length': input_length,
            'output_size': output_size,
            'kernel_size': model.kernel_size
        }, Path(args.model_path).with_suffix('.pth'))
        print("Model and normalization stats saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("\n" + "=" * 60)
    print("CNN Training Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()