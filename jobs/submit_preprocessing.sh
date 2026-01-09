#!/bin/bash
#SBATCH --account=def-sfabbro
#SBATCH --job-name=weave_preprocess
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Load modules (matching your environment)
module purge
module load StdEnv/2020 scipy-stack hdf5/1.10.6

# Activate virtual environment
source /home/minjihk/projects/def-sfabbro/minjihk/WEAVE_Pristine/.venv/bin/activate

# Run the pipeline
echo "Starting preprocessing pipeline..."
python preprocessing/build_dataset.py --step all
# python preprocessing/build_dataset.py --step process
echo "Job finished."
