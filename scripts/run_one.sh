#!/bin/bash
#SBATCH --job-name=safety_model
#SBATCH --output=out/%A%a.out
#SBATCH --error=out/%A%a.err
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:a100l:4
# SBATCH --gres=gpu:l40s:4                          
# SBATCH --gres=gpu:1
#SBATCH --mem=128G 
#SBATCH --time=03:00:00
# SBATCH --partition=main
# SBATCH --constraint='ampere'
#SBATCH --partition=short-unkillable
# SBATCH --signal=USR1@90  # Preemption warning 90s before timeout
#SBATCH --signal=B:TERM@90



# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

module load python/3.10
module load cuda/12.1.1
source /home/mila/m/maryam.hashemzadeh/projects/envs/mttl_env_1/bin/activate

TASK_NAME=$1
MODEL=$2

# CHECKPOINT_PATH="out_ckpt/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_ckpt/"
# CHECKPOINT_PATH="out_ckpt/${SLURM_JOB_ID}_${TASK_NAME}_ckpt/"
# CHECKPOINT_PATH="out_ckpt/${SLURM_JOB_NAME}_ckpt.pth"
CHECKPOINT_PATH="out_ckpt/${MODEL}_${TASK_NAME}_ckpt/"
LAST_CHECKPOINT="${CHECKPOINT_PATH}checkpoint-10/"
echo "............ LAST_CHECKPOINT ....... ${LAST_CHECKPOINT}"

# # Trap signal and requeue the job
# function _resubmit_on_timeout() {
#     echo "Caught signal, saving checkpoint and requeueing... ${SLURM_JOB_ID}"
#     trap "" SIGTERM
#     scancel -s SIGTERM --full $SLURM_JOBID
#     scontrol requeue ${SLURM_JOB_ID}   
#     # exit 1
# }
# trap _resubmit_on_timeout SIGTERM








accelerate launch main_pipline.py train_config.model=$MODEL \
    expert_training=True tasks="[$TASK_NAME]" \
    path="local://trained_Llama-3-8B-Instruct_experts_lora_$TASK_NAME" \
    checkpoint_path=$CHECKPOINT_PATH




while wait; test $? -gt 128; do :; done

echo "Job complete"