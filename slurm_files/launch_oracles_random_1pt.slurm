#!/bin/sh
#SBATCH --job-name=oracle_random_1pt
#SBATCH --array=0-99%12
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-4:00
#SBATCH -c 1
#SBATCH --output=/n/home04/rrinberg/catered_out/klom_bench/oracle_random_1pt-%A__%a.out
#SBATCH --mail-user=royrinberg@gmail.com  
#SBATCH --mail-type=ALL
#SBATCH --account kempner_sham_lab

module load python/3.10.12-fasrc01
module load intelpython/3.9.16-fasrc01
module load cuda cudnn

mamba activate ffcv

CODE_DIR=/n/home04/rrinberg/code/data-unlearning-bench
pushd $CODE_DIR

CHECKPOINT_FOLDER=/n/home04/rrinberg/data_dir/KLOM_bench/nano_gpt_models/oracle_models/

export CHECKPOINT_FOLDER=$CHECKPOINT_FOLDER
mkdir -p $CHECKPOINT_FOLDER

## Run the experiment
./launchlite.sh 0 2 data/fineweb10B/1pct-retain-random.bin --save-checkpoint --run-description "oracle_random_1pct_retain_${SLURM_ARRAY_TASK_ID}"
