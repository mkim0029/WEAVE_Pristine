#!/bin/bash
#SBATCH --job-name=test_width
#SBATCH --output=../data/test_width.out
#SBATCH --error=../data/test_width.err
#SBATCH --time=00:30:00 
#SBATCH --account=def-sfabbro  
#SBATCH --mem=10G 

# Load modules and activate environment
source home/minjihk/projects/def-sfabbro/minjihk/WEAVE_Pristine/data/module_loads.txt

# Run your script
python test.py
echo 'Job completed.'