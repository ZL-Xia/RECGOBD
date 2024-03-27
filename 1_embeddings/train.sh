#!/bin/bash
#SBATCH --job-name=MultiGPU
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p g-3090-2
#SBATCH --gres=gpu:2


source activate tensorflow 

srun --gres=gpu:2 python main.py -e train -c ./config/config_graph.json && srun --gres=gpu:2 python main.py -e test -c ./config/config_graph.json

