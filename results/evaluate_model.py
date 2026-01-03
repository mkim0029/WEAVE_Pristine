#!/usr/bin/env python3
"""
Evaluate CNN Model on Test Set
==============================

This script evaluates a trained CNN model on a test set defined by indices.
It generates predictions and saves them to an NPZ file for analysis.

Usage:
    python evaluate_model.py \
        --input ../data/processed_spectra_10k.h5 \
        --model-path ../ML_models/output/cnn_model_offline.pth \
        --indices ../ML_models/output/cnn_model_offline_test_indices.npy \
        --output-dir ../results/test_output \
        --device cuda

"""

import argparse
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Subset

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from ML_models.spectral_dataset import SpectralDataset
    from ML_models.cnn import CNN
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN on test set")
    parser.add_argument('--input', required=True, help='Path to processed HDF5 file')
    parser.add_argument('--model-path', required=True, help='Path to trained model .pth file')
    parser.add_argument('--indices', required=True, help='Path to test indices .npy file')
    parser.add_argument('--output-dir', required=True, help='Directory to save results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input)
    model_path = Path(args.model_path)
    indices_path = Path(args.indices)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating model: {model_path}")
    print(f"Input data: {input_path}")
    print(f"Test indices: {indices_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")

    # 1. Load Test Indices
    if not indices_path.exists():
        raise FileNotFoundError(f"Indices file not found: {indices_path}")
    test_indices = np.load(indices_path)
    print(f"Loaded {len(test_indices)} test indices")

    # 2. Load Checkpoint
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Set weights_only=False to allow loading numpy arrays and other objects saved in the checkpoint
    checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
    
    # Extract metadata from checkpoint
    target_mean = checkpoint.get('target_mean')
    target_std = checkpoint.get('target_std')
    input_length = checkpoint.get('input_length')
    output_size = checkpoint.get('output_size')
    kernel_size = checkpoint.get('kernel_size', 15) # Default to 15 if not found
    
    print(f"Model config: Input={input_length}, Output={output_size}, Kernel={kernel_size}")
    
    if target_mean is None or target_std is None:
        print("Warning: Normalization stats not found in checkpoint. Predictions will be in normalized space.")
    
    # 3. Load Dataset
    # Note: We use flux_key='spectra' as per the offline pipeline
    dataset = SpectralDataset(
        input_path, 
        load_targets=True, 
        flux_key='spectra',
        device=args.device
    )
    
    # Apply normalization stats to dataset if available
    if target_mean is not None and target_std is not None:
        dataset.set_target_stats(target_mean, target_std)
        print("Applied normalization stats from checkpoint to dataset.")

    # Create Subset
    test_subset = Subset(dataset, test_indices)
    
    # Create DataLoader
    test_loader = DataLoader(
        test_subset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # 4. Initialize Model
    model = CNN(kernel_size=kernel_size, input_length=input_length, output_size=output_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    # 5. Inference Loop
    print("Starting inference...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            
            outputs = model(inputs)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")
                
    # Concatenate results
    preds_normalized = np.concatenate(all_preds, axis=0)
    targets_normalized = np.concatenate(all_targets, axis=0)
    
    print(f"Predictions shape: {preds_normalized.shape}")
    
    # 6. Inverse Transform (to physical units)
    if target_mean is not None and target_std is not None:
        # Convert stats to numpy if they are tensors
        if isinstance(target_mean, torch.Tensor):
            target_mean = target_mean.cpu().numpy()
        if isinstance(target_std, torch.Tensor):
            target_std = target_std.cpu().numpy()
            
        preds_phys = preds_normalized * target_std + target_mean
        targets_phys = targets_normalized * target_std + target_mean
    else:
        preds_phys = preds_normalized
        targets_phys = targets_normalized
        
    # 7. Save Results
    output_file = output_dir / "test_predictions.npz"
    np.savez(
        output_file,
        indices=test_indices,
        preds_normalized=preds_normalized,
        targets_normalized=targets_normalized,
        preds_phys=preds_phys,
        targets_phys=targets_phys,
        target_cols=dataset.target_cols
    )
    
    print(f"Results saved to {output_file}")
    print("Done.")

if __name__ == "__main__":
    main()
