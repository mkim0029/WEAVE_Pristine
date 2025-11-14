#!/bin/bash
# SLURM job script to run CNN training on a GPU

#SBATCH --job-name=cnn_train
#SBATCH --output=cnn_train.out
#SBATCH --error=cnn_train.err
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --account=def-sfabbro 

# Activate project virtualenv if available (adjust path if needed)
if [ -f /home/minjihk/projects/def-sfabbro/minjihk/WEAVE_Pristine/.venv/bin/activate ]; then
    echo "Activating virtualenv..."
    source /home/minjihk/projects/def-sfabbro/minjihk/WEAVE_Pristine/.venv/bin/activate
else
    echo "Warning: virtualenv not found at repository path. Using system Python."
fi

# Run the training command
# Note: cnn.py should be copied into /home/minjihk/scratch before running.
# Use absolute path for the input HDF5 (repo location) to avoid relative-path confusion.
python cnn.py \
    --input /home/minjihk/projects/def-sfabbro/minjihk/WEAVE_Pristine/data/processed_spectra.h5 \
    --epochs 20 \
    --batch-size 64 \
    --lr 1e-4 \
    --model-path ./cnn_model

EXIT_CODE=$?

echo "Training finished with exit code: ${EXIT_CODE}"

# Deactivate virtualenv if activated
if [[ "$(basename "$VIRTUAL_ENV" 2>/dev/null)" != "" ]]; then
    deactivate || true
fi

exit ${EXIT_CODE}
