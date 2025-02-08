
import torch
import copy
from mttl.arguments import ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.models.containers.selectors import TaskNameSelectorConfig, ArrowSelectorConfig
from mttl.models.containers.selectors.base import UniformSelectorConfig
from mttl.models.containers.selectors.poly_selector import PolySelectorDirectConfigUniform
from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from mttl.evaluators.loglike_evaluator import LogLikeEvaluator
from mttl.evaluators.safety_evaluator import SafetyEvaluator
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.library.library_transforms import ArrowTransformConfig, ArrowTransform
from mttl.models.containers.selectors import ArrowSelectorConfig
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import os
import sys
sys.path.append(os.path.abspath('../'))
import mttl.arguments
from mttl.arguments import ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.expert_model import ExpertModel, ExpertModelConfig
from mttl.models.train_utils import train_model
from functools import wraps

device = "cuda" if torch.cuda.is_available() else "cpu"

@hydra.main(config_name="config.yaml") #, config_name="config")
def local_train(cfg: DictConfig):
    # os.environ["WANDB_PROJECT"] = "gpv"
    # os.environ["WANDB_LOG_MODEL"] = "tests"

    wandb_run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True)  # Log entire config
    )

    # # Model and Tokenizer
    # model_id = cfg.model.model_id
    # bnb_config = BitsAndBytesConfig(**cfg.model.bnb_config)
    # peft_config = LoraConfig(**cfg.peft)

    train_config = ExpertConfig.from_dict({**cfg.train_config})

    library = ExpertLibrary.get_expert_library(cfg.path, create=True)

    for task in cfg.tasks:
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
            device_map= device,
            precision= train_config.precision,
        )
        # if not os.path.exists(cfg.path.split('local://')[1]): ###
        print("........ training .........", task)
        # minimal training code to finetune the model on examples from the task
        train_model(train_config, model, get_datamodule(train_config))
        # add the expert to the library!
        expert_instance = model.as_expert(training_config=train_config.to_dict())
        library.add_expert(expert_instance, force=True)


    # Let's see which experts are in the library... :-)
    for expert_name in library.keys():
        print("Expert: ", expert_name, " with config: ", library[expert_name].expert_config)
    print("Experts in library:", len(library))


    ## merged models
    print("........ merging .........")
    merge_experts(library_path=cfg.path, type=cfg.merging.type)
    model_merged = MultiExpertModel.from_pretrained(f"{cfg.path}_{cfg.merging.type}", device_map="cuda", precision="32")
    
    ## evaluation
    # if cfg.evaluation.rouge:
    datamodule = get_datamodule(train_config, for_generation=True)
    evaluator_domain_rouge = RougeEvaluator(datamodule)
    # if cfg.evaluation.loglikelihood:  
    datamodule = get_datamodule(train_config, for_generation=True)
    evaluator_domain_ll = LogLikeEvaluator(datamodule)

    print("........ evaluation .........")
    domain_score_rouge, predictions = evaluator_domain_rouge.evaluate(model_merged, split="test", return_predictions=True, output_path='rouge_output') ## return predictions 
    wandb_run.log({"evaluation/domain_score_rouge": domain_score_rouge})
    print("for bio mixed is", domain_score_rouge)

    domain_score_ll = evaluator_domain_ll.evaluate(model_merged, split="test", output_path='ll_output') ## save output to output path
    wandb_run.log({"evaluation/domain_score_ll": domain_score_ll})
    print("for bio mixed is", domain_score_ll)
    
    ## safety evaluation 
    evaluator_safety = SafetyEvaluator(train_config)
    safety_score = evaluator_safety.evaluate(model_merged, split="test", output_path='safety_output') ## save output to output path
    wandb_run.log({"evaluation/safety_score": safety_score})
    print("for bio mixed is", safety_score)

    wandb_run.finish()



def merge_experts(library_path=None, type='uniform'):

    if type == 'arrow':
        arrow_transform_config = ArrowTransformConfig()
        arrow_transform = ArrowTransform(arrow_transform_config)

        # # persist the prototypes in the library using arrow_transform_config.name
        arrow_transform.transform(library_path, persist=True)

        # # we inform the selector that we have to read the prototypes corresponding to our config
        selector_config = ArrowSelectorConfig(top_k=1, selector_data_id=arrow_transform_config.save_name)

    if type == 'oracle':
        selector_config = TaskNameSelectorConfig()
 
    if type== 'uniform':
        # now replace selector for lora to a uniform merging of experts during the forward pass
        # no task information is used!
        selector_config = UniformSelectorConfig()

        # model.set_selector("lora", UniformSelectorConfig())
    
    
    model = MultiExpertModel.from_pretrained_library(
        library_path,
        selector_config=selector_config,
        device_map=device
        )
    
    # save model + selector for future re-use
    model.save_pretrained(f"{library_path}_{type}")



    



if __name__ == "__main__":
    local_train()





