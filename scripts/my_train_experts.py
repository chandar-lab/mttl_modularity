import os
import sys
sys.path.append(os.path.abspath('../'))
import mttl.arguments
from mttl.arguments import ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.expert_model import ExpertModel, ExpertModelConfig
from mttl.models.train_utils import train_model

# Set this flag to True if you want to re-train the experts!
train_library = True

train_config = ExpertConfig.from_dict(
    {
        "lora_rank": 4,
        "lora_alpha": 1.,
        "lora_dropout": 0.0,
        "weight_decay": 0.0,
        "output_dir": "/tmp/",
        "model_modifier": "lora",
        "modify_modules": ".*",
        "modify_layers": "q_proj|v_proj|k_proj",
        "trainable_param_names": ".*lora_[ab].*",
        "num_train_epochs": 5,
        "learning_rate": 1e-2,
        "micro_batch_size": 16,
        "train_batch_size": 16,
        "predict_batch_size": 8,
        "precision": "bf16",
        "model": "EleutherAI/gpt-neo-125m",
        "model_family": "gpt",
        "optimizer": "adamw",
        "dataset": "a-safety",
        "warmup_proportion": 0.,
        "max_input_length": 64,
        "max_output_length": 64,
        "truncation_side": "left",
        "padding": "max_length",
        "truncation": True,
        "max_length":64,
    }
)


library = ExpertLibrary.get_expert_library("local://trained_gpt125m_experts_camel", create=True)

for task in [
    # "a-safety",
    # "adversarial_qa_droberta_answer_the_following_q",
    "camel-biology",
]:
    # set the task name to the task we want to finetune on
    train_config.dataset = task
    train_config.finetune_task_name = task
    # initialize an expert model
    model = ExpertModel(
        ExpertModelConfig(
            base_model=train_config.model,
            expert_name=train_config.finetune_task_name,
            task_name=train_config.finetune_task_name,
            modifier_config=train_config.modifier_config,
        ),
        device_map="cuda",
        precision=train_config.precision,
    )
    # minimal training code to finetune the model on examples from the task
    train_model(train_config, model, get_datamodule(train_config))
    # add the expert to the library!
    expert_instance = model.as_expert(training_config=train_config.to_dict())
    library.add_expert(expert_instance, force=True)


# Let's see which experts are in the library... :-)
for expert_name in library.keys():
    print("Expert: ", expert_name, " with config: ", library[expert_name].expert_config)
print("Experts in library:", len(library))

