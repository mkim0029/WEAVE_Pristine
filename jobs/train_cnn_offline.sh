#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --job-name=cnn_offline
#SBATCH --output=jobs/logs/cnn_offline_%j.out
#SBATCH --error=jobs/logs/cnn_offline_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1

# Activate environment
source /lustre06/project/6016730/minjihk/WEAVE_Pristine/.venv/bin/activate

# Define paths
PROJECT_ROOT="/home/minjihk/projects/def-sfabbro/minjihk/WEAVE_Pristine"
PROCESSED_DATA="$PROJECT_ROOT/data/processed_spectra_10k.h5"
OUTPUT_DIR="$PROJECT_ROOT/ML_models/output"
MODEL_PATH="$OUTPUT_DIR/cnn_model_offline"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
mkdir -p jobs/logs

# Run Training
echo "Starting Offline Training..."
python $PROJECT_ROOT/ML_models/cnn.py \
    --input $PROCESSED_DATA \
    --mode offline \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-4 \
    --model-path $MODEL_PATH \
    --device cuda

echo "Job finished."
