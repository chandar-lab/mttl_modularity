- Installation

```sh
module load python/3.10
module load cuda/12.1.1
source /home/mila/m/maryam.hashemzadeh/projects/envs/mttl_env_1/bin/activate 
pip install -r requirements.txt
pip install wheel
MAX_JOBS=2 pip install flash-attn --no-build-isolation
pip uninstall torch torchvision torchaudio
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

```


- Activate the venv:

```sh
module load python/3.10
module load cuda/12.1.1
source /home/mila/m/maryam.hashemzadeh/projects/envs/mttl_env_1/bin/activate 
```


RUN:
```sh
TASK_NAME="camel-biology"
accelerate launch main_pipline.py \
    expert_training=True tasks="[$TASK_NAME]" \
    path="local://trained_Llama-3-8B-Instruct_experts_lora_$TASK_NAME"
```


accelerate configuration saved at /network/scratch/m/maryam.hashemzadeh/cache/huggingface/accelerate/default_config.yaml

deepspeed_config:
  gradient_accumulation_steps: 16
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED



deepspeed path: /home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities_main/scripts/deepspeed_config.json

/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities_main/scripts/zero_stage2_config.json


deepspeed_config:
  gradient_accumulation_steps: 16
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2