#!/bin/sh
#SBATCH --job-name=oracle_random_5pt
#SBATCH --array=0-99
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --mem=640G
#SBATCH -t 0-72:00
#SBATCH -c 8

## Run the experiment
./launchlite.sh 0,1,2,3,4,5,6,7 1 data/fineweb10B/5pct-retain-random.bin --save-checkpoint --run-description "oracle_random_5pct_retain_${SLURM_ARRAY_TASK_ID}"
