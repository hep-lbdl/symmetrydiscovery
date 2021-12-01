#!/bin/bash
#SBATCH --time=240
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --account=m1759
#SBATCH --dependency=singleton
#SBATCH --mail-user=mail-user=krish.desai@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --output a-b.txt

module load cgpu
module load tensorflow/gpu-2.2.0-py37
conda activate tf-gpu

srun -n 1 -c 1 python3 /global/u1/k/kdesai/SymmetryDiscovery/SymmetryGAN/4O2Slices/a-b/a-b.py