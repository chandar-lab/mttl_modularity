#!/bin/bash
#SBATCH --job-name=safety_model
#SBATCH --output=out/%A%a.out
#SBATCH --error=out/%A%a.err
#SBATCH --cpus-per-task=8 
#SBATCH --gres=gpu:a100l:1                           
# SBATCH --gres=gpu:1
#SBATCH --mem=32G 
#SBATCH --time=10:00:00
#SBATCH --partition=main
#SBATCH --constraint='ampere'



module load python/3.10
module load cuda/12.1.1
source ../../../envs/mttl_env_1/bin/activate 
# HYDRA_FULL_ERROR=1

# model_path=$1
# type=$2
# echo $model_path
# python my_safety_evaluator.py --model_path=$model_path --type_behaviour=$type

python main_pipline.py





 
