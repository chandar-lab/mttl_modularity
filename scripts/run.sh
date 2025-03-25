#!/bin/bash
#SBATCH --job-name=safety_model
#SBATCH --output=out/%A%a.out
#SBATCH --error=out/%A%a.err
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:a100l:4
# SBATCH --gres=gpu:l40s:4                          
# SBATCH --gres=gpu:1
#SBATCH --mem=64G 
#SBATCH --time=3:00:00
# SBATCH --partition=main
# SBATCH --constraint='ampere'
#SBATCH --partition=short-unkillable



module load python/3.10
module load cuda/12.1.1
source /home/mila/m/maryam.hashemzadeh/projects/envs/mttl_env_1/bin/activate 


HYDRA_FULL_ERROR=1

# model_path=$1
# type=$2
# echo $model_path
# python my_safety_evaluator.py --model_path=$model_path --type_behaviour=$type


TASK_NAME=$1  # Take task name from command line argument

echo "Running with task: $TASK_NAME"

# python main_pipline.py expert_training=True tasks="[$TASK_NAME]" path="local://trained_Llama-3-8B-Instruct_experts_lora_$TASK_NAME"

# python main_pipline.py

accelerate launch main_pipline.py train_config.model="meta-llama/Llama-3.1-8B-Instruct" \
    expert_training=True tasks="[$TASK_NAME]" \
    path="local://trained_Llama-3-8B-Instruct_experts_lora_$TASK_NAME" \
    train_config.micro_batch_size=1 train_config.train_batch_size=16 \
    # checkpoint_path=$CHECKPOINT_FILE \
    # resume_from_checkpoint=$RESUME




 
