#!/bin/bash
#SBATCH --job-name=safety_model
#SBATCH --output=out/%A%a.out
#SBATCH --error=out/%A%a.err
#SBATCH --cpus-per-task=16 
# SBATCH --gres=gpu:a100l:2
#SBATCH --gres=gpu:l40s:4                          
# SBATCH --gres=gpu:1
#SBATCH --mem=64G 
#SBATCH --time=0:02:00
# SBATCH --partition=main
# SBATCH --constraint='ampere'
#SBATCH --partition=short-unkillable
# SBATCH --signal=USR1@90  # Preemption warning 90s before timeout
#SBATCH --signal=B:TERM@90


module load python/3.10
module load cuda/12.1.1
source /home/mila/m/maryam.hashemzadeh/projects/envs/mttl_env_1/bin/activate

TASK_NAME=$1
# CHECKPOINT_PATH="out_ckpt/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_ckpt/"
CHECKPOINT_PATH="out_ckpt/${SLURM_JOB_ID}_${TASK_NAME}_ckpt/"
# CHECKPOINT_PATH="out_ckpt/${SLURM_JOB_NAME}_ckpt.pth"

# Trap signal and requeue the job
function _resubmit_on_timeout() {
    echo "Caught signal, saving checkpoint and requeueing... ${SLURM_JOB_ID}"
    trap "" SIGTERM
    scancel -s SIGTERM --full $SLURM_JOBID
    scontrol requeue ${SLURM_JOB_ID}   
    # exit 1
}
trap _resubmit_on_timeout SIGTERM

# Check if checkpoint exists
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT_PATH"
    RESUME_FLAG="True"
else
    echo "Starting from scratch"
    RESUME_FLAG="False"
fi

# Run training
# accelerate launch main_pipline.py \
#     checkpoint_path="$CHECKPOINT_PATH" \
#     resume_from_checkpoint=$RESUME_FLAG

# trap 'echo signal recieved in BATCH!; kill -15 "${PID}"; wait "${PID}";' SIGINT SIGTERM


accelerate launch main_pipline.py \
    expert_training=True tasks="[$TASK_NAME]" \
    path="local://trained_Llama-3-8B-Instruct_experts_lora_$TASK_NAME" \
    checkpoint_path=$CHECKPOINT_PATH \
    resume_from_checkpoint=$RESUME_FLAG &

# python main_pipline.py \
#     expert_training=True tasks="[$TASK_NAME]" \
#     path="local://trained_Llama-3-8B-Instruct_experts_lora_$TASK_NAME" \
#     checkpoint_path=$CHECKPOINT_PATH \
#     resume_from_checkpoint=$RESUME_FLAG


# wait $!
# Optional cleanup
# Set the PID var so that the trap can use it
### it should be in loop
# PID="$!"
# wait "${PID}"

while wait; test $? -gt 128; do :; done

echo "Job complete"