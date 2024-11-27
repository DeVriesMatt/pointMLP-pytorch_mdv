#!/bin/bash
#SBATCH --job-name=pointmlp_no_down
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --output=/data/scratch/DBI/DUDBI/DYNCESYS/mvries/PointMIL/outputs/pointmlp_no_down.txt
#SBATCH --error=/data/scratch/DBI/DUDBI/DYNCESYS/mvries/PointMIL/errors/pointmlp_no_down.txt
#SBATCH --partition=gpuhm
module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh

conda activate cs

export NCCL_SOCKET_IFNAME=^docker0,lo

srun python classification_ModelNet40_new/main.py --batch_size 8 --model pointMLP

~                 
