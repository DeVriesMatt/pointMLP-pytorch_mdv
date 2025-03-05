#!/bin/bash
#SBATCH --job-name=pointmlp_no_down
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --output=/data/scratch/DBI/DUDBI/DYNCESYS/mvries/PointMIL/outputs/pointmlp_no_down_add_intra.txt
#SBATCH --error=/data/scratch/DBI/DUDBI/DYNCESYS/mvries/PointMIL/errors/pointmlp_no_down_add_intra.txt
#SBATCH --partition=gpuhm
module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh

conda activate cs


python classification_ModelNet40_new/main.py --batch_size 32 --model poinMLPElite --pooling 'additive'

