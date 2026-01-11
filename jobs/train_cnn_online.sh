#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --job-name=cnn_online
#SBATCH --output=jobs/logs/cnn_online_%j.out
#SBATCH --error=jobs/logs/cnn_online_%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1

# Activate environment
source /lustre06/project/6016730/minjihk/WEAVE_Pristine/.venv/bin/activate

# Define paths
PROJECT_ROOT="/home/minjihk/projects/def-sfabbro/minjihk/WEAVE_Pristine"
RAW_DATA="$PROJECT_ROOT/data/raw_spectra_10k.h5"
METADATA="$PROJECT_ROOT/data/GA-QAG_OPR3bv3_MasterTableCodev4_GoldenSample_RELEASE.fits"
OUTPUT_DIR="$PROJECT_ROOT/ML_models/output"
MODEL_PATH="$OUTPUT_DIR/cnn_model_online"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
mkdir -p jobs/logs

# Run Training
echo "Starting Online Training at $(date)"
START_TIME=$(date +%s)

python $PROJECT_ROOT/ML_models/cnn.py \
    --input $RAW_DATA \
    --metadata $METADATA \
    --mode online \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-4 \
    --model-path $MODEL_PATH \
    --device cuda

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Job finished at $(date)."
echo "Total execution time: $(($DURATION / 3600))h $(($DURATION % 3600 / 60))m $(($DURATION % 60))s"
