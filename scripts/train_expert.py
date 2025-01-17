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
train_library = False

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
        "dataset": "sordonia/flan-100-flat",
        "warmup_proportion": 0.,
        "max_input_length": 1024,
        "max_output_length": 128,
        "truncation_side": "left"
    }
)

if train_library:
    library = ExpertLibrary.get_expert_library("local://trained_gpt125m_experts_colab", create=True)

    for task in [
        "wiqa_what_is_the_final_step_of_the_following_process",
        "sciq_Multiple_Choice",
        "adversarial_qa_droberta_answer_the_following_q",
        "duorc_SelfRC_question_answering",
        "cos_e_v1_11_description_question_option_id",
        "race_high_Select_the_best_answer",
        "race_high_Select_the_best_answer_generate_span_",
        "wiki_qa_Is_This_True_",
        "quail_description_context_question_text",
        "wiki_hop_original_explain_relation"
    ]:
        # set the task name to the task we want to finetune on
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
else:
    library = ExpertLibrary.get_expert_library("hf://sordonia/trained_gpt125m_experts_colab").clone("local://trained_gpt125m_experts_colab")


# Let's see which experts are in the library... :-)
for expert_name in library.keys():
    print("Expert: ", expert_name, " with config: ", library[expert_name].expert_config)
print("Experts in library:", len(library))

