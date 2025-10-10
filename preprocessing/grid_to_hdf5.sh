#!/bin/bash
#SBATCH --job-name=grid_to_hdf5
#SBATCH --output=../data/grid_to_hdf5.out
#SBATCH --error=../data/grid_to_hdf5.err
#SBATCH --time=03:00:00 
#SBATCH --account=def-sfabbro  
#SBATCH --mem=50G #rough estimate 60G
#SBATCH --cpus-per-task=4

# Remove old output files
rm -f ../data/*.out ../data/*.err

# Load modules and activate environment
source ../data/module_loads.txt

# Run your script
python spectrum_grid_reader.py --grid-file ../data/grid_wavelengths_windows.txt --output ../data/weave_nlte_grids.h5
echo 'Job completed.'