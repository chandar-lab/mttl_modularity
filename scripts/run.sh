#!/bin/bash
#SBATCH --job-name=safety_model
#SBATCH --output=out/%A%a.out
#SBATCH --error=out/%A%a.err
#SBATCH --cpus-per-task=24 
#SBATCH --gres=gpu:a100l:1                           
# SBATCH --gres=gpu:1
#SBATCH --mem=64G 
#SBATCH --time=3:00:00
# SBATCH --partition=short-unkillable
#SBATCH --constraint='ampere'



module load python/3.10
module load cuda/12.1.1
source ../../../envs/mttl_env_1/bin/activate 
# HYDRA_FULL_ERROR=1

model_path=$1


echo $model_path
python safety_2.py --model_path=$model_path

# python main_4.py peft.r=4 training.num_train_epochs=5
# output_dir='model_ckpts_action_sg_lora/'$model'_'$size'_'$sg_name
# cache_dir='cache_action_sg_'$model'_'$size'_lora/_'$sg_name'/'





 
