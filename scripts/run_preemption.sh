#!/bin/bash
#SBATCH --job-name=safety_model
#SBATCH --output=out/%A%a.out
#SBATCH --error=out/%A%a.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=24G
#SBATCH --time=47:55:00
#SBATCH --partition=main

module load python/3.10
module load cuda/12.1.1
source /home/mila/m/maryam.hashemzadeh/projects/envs/mttl_env_1/bin/activate

TASK_NAME=$1
CHECKPOINT_FILE="out_ckpt/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_ckpt.pth"

exit_script() {
    echo "Preemption detected, saving state..."
    kill -- -$$
}
trap exit_script SIGTERM

if [ -f "$CHECKPOINT_FILE" ]; then
    RESUME="True"
    echo "Resuming from checkpoint: $CHECKPOINT_FILE"
else
    RESUME="False"
    echo "No checkpoint, starting fresh..."
fi

# python main_pipline.py expert_training=True tasks="[$TASK_NAME]" path="local://trained_Llama-3-8B-Instruct_experts_lora_$TASK_NAME" checkpoint_path=$CHECKPOINT_FILE resume_from_checkpoint=$( [ -f "$CHECKPOINT_FILE" ] && echo True || echo False )

accelerate launch main_pipline.py \
    expert_training=True tasks="[$TASK_NAME]" \
    path="local://trained_Llama-3-8B-Instruct_experts_lora_$TASK_NAME" \
    checkpoint_path=$CHECKPOINT_FILE \
    resume_from_checkpoint=$RESUME

echo "Job completed successfully!"
rm -f "$CHECKPOINT_FILE"
