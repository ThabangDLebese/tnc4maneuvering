#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email@here.edu>
#SBATCH --time=7-00:00:00
#SBATCH --output=tnc4m_one_ds.out
#SBATCH --job-name=tnc4m_one_ds

module purge
module load gcc/8.1.0
module load cuda/11.2.152
cd ~/<your_virtual_environemt>/
source bin/activate
python3 -m tnc4maneuvering.tnc4maneuvering --data one_ds --train --cv 2  --w 0.01