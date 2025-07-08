#!/bin/sh
#SBATCH --job-name=full_model
#SBATCH --array=0-99%12
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-2:30
#SBATCH -c 1
#SBATCH --output=/n/home04/rrinberg/catered_out/klom_bench/full_model-%A__%a.out
#SBATCH --mail-user=royrinberg@gmail.com  
#SBATCH --mail-type=ALL
#SBATCH --account kempner_sham_lab

module load python/3.10.12-fasrc01
module load intelpython/3.9.16-fasrc01
module load cuda cudnn

mamba activate ffcv

CODE_DIR=/n/home04/rrinberg/code/data-unlearning-bench
pushd $CODE_DIR

CHECKPOINT_FOLDER=/n/home04/rrinberg/data_dir/KLOM_bench/nano_gpt_models/pretrain_models/

export CHECKPOINT_FOLDER=$CHECKPOINT_FOLDER
mkdir -p $CHECKPOINT_FOLDER

## Run the experiment
./launchlite.sh 0 2 data/fineweb10B/fineweb_train_subset.bin --save-checkpoint --run-description "pretrain_fineweb_subset_${SLURM_ARRAY_TASK_ID}"
