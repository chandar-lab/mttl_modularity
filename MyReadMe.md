- Installation

```sh
module load python/3.10
module load cuda/12.1.1
source ../envs/mttl_env_1/bin/activate 
pip install -r requirements.txt
MAX_JOBS=2 pip install flash-attn --no-build-isolation
pip uninstall torch torchvision torchaudio
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

```


- Activate the venv:

```sh
module load python/3.10
module load cuda/12.1.1
source ../../envs/mttl_env_1/bin/activate 
```



