
import torch
import copy
from mttl.arguments import ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.models.containers.selectors import TaskNameSelectorConfig, ArrowSelectorConfig
from mttl.models.containers.selectors.base import UniformSelectorConfig
from mttl.models.containers.selectors.poly_selector import PolySelectorDirectConfigUniform
from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from mttl.evaluators.safety_evaluator import SafetyEvaluator
from mttl.models.library.expert_library import ExpertLibrary


eval_config = ExpertConfig.from_dict(
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
        "truncation_side": "left",
        "pipeline_eval_tasks": "a-safety",
    }
)

# a-safety

# let's eval on all the 8 tasks
eval_config.finetune_task_name = ",".join([
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
])
device_map="cuda" if torch.cuda.is_available() else "cpu"

### using RougeEvaluator
datamodule = get_datamodule(eval_config, for_generation=True)
evaluator = RougeEvaluator(datamodule)
datamodule.task_names

print("Test examples:", len(datamodule.test_dataset))

# the oracle model uses the "task name" in the forward pass, the task name is passed by the dataloader
model = MultiExpertModel.from_pretrained_library(
    "local://trained_gpt125m_experts_colab",
    selector_config=TaskNameSelectorConfig(),
    device_map=device_map
)

oracle_rouge = evaluator.evaluate(model, split="test")
print(oracle_rouge)


##############
########## For safety ##################
# the oracle model uses the "task name" in the forward pass, the task name is passed by the dataloader
model = MultiExpertModel.from_pretrained_library(
    "local://trained_gpt125m_experts_colab",
    selector_config=TaskNameSelectorConfig(),
    device_map=device_map
)

evaluator = SafetyEvaluator(eval_config)

oracle_safety = evaluator.evaluate(model, output_path='safety_result/', split="test")

print(oracle_safety)