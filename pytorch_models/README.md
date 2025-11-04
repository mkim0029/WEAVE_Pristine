# pytorch_models

This folder contains simple PyTorch model training scripts, a dataset helper, and a small verification script used for the WEAVE_Pristine project.

Files
-----
- `MLP.py` - An earlier MLP training script (historical). See `new_MLP.py` for the current, documented trainer.
- `new_MLP.py` - Documented Multilayer Perceptron trainer. Loads a `SpectralDataset`, computes training split normalization
  statistics, trains an MLP, and saves a checkpoint and training history.
- `cnn.py` - Convolutional model(s) for experiments (architecture and training logic implemented here).
- `spectral_dataset.py` - Dataset and dataloader utilities used by the trainers. Provides `SpectralDataset`, `create_spectral_dataloaders`,
  and `SpectralTransforms` used to standardize inputs/targets.
- `verify_pipeline.py` - Small script that runs quick checks to verify the dataset -> dataloader -> model pipeline.
- `mlp_model.pth`, `mlp_model.history.npz` - Example/previously saved model checkpoint and its training history.
- `new_mlp_model.pth`, `new_mlp_model.history.npz` - Example/previously saved model checkpoint and history for `new_MLP.py`.

Quick start: train `new_MLP.py` from the terminal
-----------------------------------------------
1. Ensure you have the processed HDF5 dataset available in the repository `data/` folder (example: `../data/processed_spectra.h5`).

2. From this folder (`pytorch_models`) run the trainer. Example command that trains for 50 epochs:

```bash
python new_MLP.py \
  --input ../data/processed_spectra.h5 \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-4 \
  --model-path ./new_mlp_model
```

Notes on the command-line arguments:
- `--input` / `-i`: Path to the processed HDF5 file with spectra and targets.
- `--model-path`: Base path to save the model checkpoint (a `.pth` suffix will be added).
- `--epochs`: Number of training epochs.
- `--batch-size`: Batch size used by the DataLoaders.
- `--lr`: Learning rate for the optimizer.
- `--device`: Optional, defaults to `'cuda'` when available, otherwise `'cpu'`.
- `--seed`: Random seed for reproducibility (default 42).

What the training produces
-------------------------
After successful training, `new_MLP.py` will create two files (using the `--model-path` base):

- `<model-path>.pth` - PyTorch checkpoint containing:
  - `model_state_dict` (state_dict of the trained model)
  - `target_mean` and `target_std` (numpy arrays used to normalize targets during training)
  - `input_size`, `output_size`, and `hidden_layers` metadata

- `<model-path>.history.npz` - NumPy compressed archive containing training history arrays:
  - `train_loss`: array of per-epoch training loss values
  - `val_loss`: array of per-epoch validation loss values

Restoring and inference
-----------------------
To perform inference later you should:
1. Load the checkpoint with `torch.load(path)`.
2. Recreate the `MLP` model with the saved `input_size` and `output_size` and `hidden_layers` if needed.
3. Call `model.load_state_dict(checkpoint['model_state_dict'])` and put model in `eval()` mode.
4. Use `checkpoint['target_mean']` and `checkpoint['target_std']` to de-normalize predictions (or to normalize inputs/targets in the same way used during training).

Quick verification
------------------
You can run `verify_pipeline.py` to perform a small smoke test of the dataset and model pipeline. This script is intended to help detect
issues with data shapes, missing files, or obvious runtime errors before running a full training session.

Tips and next steps
-------------------
- For experiments change model hyperparameters in the trainer CLI (hidden layer sizes are defined in the model class defaults).
- Add checkpointing or early stopping if you plan to run longer experiments — `new_MLP.py` currently saves only the final weights and history.
- Consider adding a small unit test that loads `new_MLP.py` model, builds a dummy input and runs a forward pass to validate shapes.

Contact
-------
If you need the exact dataset format or more training examples, check the repository `data/` at the project root.
