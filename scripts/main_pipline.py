
import torch
import copy
import os
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
print(os.getcwd())
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
from mttl.models.expert_model import ExpertModel, ExpertModelConfig
from mttl.models.train_utils import train_model, my_train_model, my_train_model_acce, train_sft_model
from functools import wraps
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from accelerate import Accelerator


print(torch.cuda.device_count())  # Should print 4
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import signal
import sys
import time





@hydra.main(config_path=".", config_name="config", version_base=None)
def local_train(cfg: DictConfig):
    # os.environ["WANDB_PROJECT"] = "gpv"
    # os.environ["WANDB_LOG_MODEL"] = "tests"

    # if cfg.path.startswith("local://"):
        # physical_path = os.path.abspath(cfg.path.replace("local://", "trained_models/"))
    os.makedirs(cfg.path, exist_ok=True)
        # cfg.path = physical_path


    wandb_run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow"
    )

    # Fetch from command line arguments or Hydra config
    checkpoint_path = cfg.checkpoint_path
    resume_from_checkpoint = cfg.resume_from_checkpoint

    if resume_from_checkpoint:
        print("Resuming from checkpoint.")
    else:
        print("Starting fresh training.")

    # # Model and Tokenizer
    # model_id = cfg.model.model_id
    # bnb_config = BitsAndBytesConfig(**cfg.model.bnb_config)
    # peft_config = LoraConfig(**cfg.peft)


    # train_config = OmegaConf.to_container(train_config, resolve=True)  # Convert to a normal dict
    # train_config["model_init_kwargs"] = train_config.get("model_init_kwargs", {})
    
    train_config = ExpertConfig.from_dict({**cfg.train_config})
    train_config.wandb = wandb_run
    
    library_path = os.path.dirname(os.path.abspath(__file__)) +'/'+ cfg.path.split('local://')[1]
    if os.path.exists(library_path):
        # cfg.path='scripts/local:/trained_Llama-3-8B-Instruct_experts_lora_beaverTails_unsafe_controversial_topics_politics_uniform'
        # path = cfg.path.replace('//', '/')
        # library_path = f'scripts/{path}'
        
        library = ExpertLibrary.get_expert_library(library_path, expert_library_type="local")
    else:
        library_path = cfg.path
        library = ExpertLibrary.get_expert_library(cfg.path, create=True)
    

    # wandb_run.log({"expert_training": cfg.expert_training})
    # wandb_run.log({"full_training": cfg.full_training})
    # wandb_run.log({"base":})
 
    for task in cfg.tasks:
        # set the task name to the task we want to finetune on
        train_config.dataset = task
        print(task)
        train_config.finetune_task_name = task

        if cfg.expert_training:
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
            # my_train_model(train_config, model, get_datamodule(train_config),  checkpoint_path=checkpoint_path, resume_from_checkpoint=resume_from_checkpoint)
            my_train_model_acce(train_config, model, get_datamodule(train_config),  checkpoint_path=checkpoint_path, resume_from_checkpoint=resume_from_checkpoint) 
            # train_model(train_config, model, get_datamodule(train_config),  checkpoint_path=checkpoint_path, resume_from_checkpoint=resume_from_checkpoint)
            # add the expert to the library!
            expert_instance = model.as_expert(training_config=train_config.to_dict())
            library.add_expert(expert_instance, force=True)
            # Let's see which experts are in the library... :-)
            for expert_name in library.keys():
                print("Expert: ", expert_name, " with config: ", library[expert_name].expert_config)
            print("Experts in library:", len(library))

        elif cfg.full_training:
            # datamodule = get_datamodule( train_config, for_generation=True)
            # train_sft(cfg, train_config, datamodule, wandb_run)
            
            # initialize an expert model
            train_config.trainable_param_names = ".*"
            model = ExpertModel(
                ExpertModelConfig(
                    base_model=train_config.model,
                    expert_name=train_config.finetune_task_name,
                    task_name=train_config.finetune_task_name,
                    modifier_config= None,
                ),
                device_map= device,
                precision= train_config.precision,
            )
            # if not os.path.exists(cfg.path.split('local://')[1]): ###
            print("........ training .........", task)
            # minimal training code to finetune the model on examples from the task
            my_train_model(train_config, model, get_datamodule(train_config))
            # add the expert to the library!
            expert_instance = model.as_expert(training_config=train_config.to_dict())
            library.add_expert(expert_instance, force=True)
            # Let's see which experts are in the library... :-)
            for expert_name in library.keys():
                print("Expert: ", expert_name, " with config: ", library[expert_name].expert_config)
            print("Experts in library:", len(library))
        
        elif cfg.no_training:
            model = AutoModelForCausalLM.from_pretrained(
                train_config.model,
                device_map=device,
                torch_dtype= torch.bfloat16,
                # quantization_config=bnb_config,
                # attn_implementation="flash_attention_2"
            )




 
    ## merged models
    if cfg.merging.do_merge:
        print("........ merging .........")
        model = merge_experts(library=library, library_path=library_path,  type=cfg.merging.type)
        # merged_model
        # model = MultiExpertModel.from_pretrained(f"{cfg.path}_{cfg.merging.type}", device_map="cuda", precision="32")
        model = MultiExpertModel.from_pretrained(f"{library_path}_{cfg.merging.type}", device_map="cuda", precision="32")
    
    ## evaluation
    # if cfg.evaluation.rouge:
    datamodule = get_datamodule(train_config, for_generation=True)
    evaluator_domain_rouge = RougeEvaluator(datamodule)
    # if cfg.evaluation.loglikelihood:  
    evaluator_domain_ll = LogLikeEvaluator(datamodule)

    print("........ evaluation .........")
    domain_score_rouge, predictions = evaluator_domain_rouge.evaluate(model, split="test", return_predictions=True, output_path='rouge_output') ## return predictions 
    wandb_run.log({"evaluation/domain_score_rouge": domain_score_rouge})
    print("for bio mixed is", domain_score_rouge)

    # domain_score_ll = evaluator_domain_ll.evaluate(model, split="test", output_path='ll_output') ## save output to output path
    # wandb_run.log({"evaluation/domain_score_ll": domain_score_ll})
    # print("LL for bio mixed is", domain_score_ll['loglike'])
    
    ## safety evaluation 
    evaluator_safety = SafetyEvaluator(train_config)
    safety_score = evaluator_safety.evaluate(model, split="test", output_path='safety_output') ## save output to output path
    wandb_run.log({"evaluation/safety_score": safety_score})
    print("for bio mixed is", safety_score)

    wandb_run.finish()

def train_sft(cfg, train_config, datamodule, wandb_run):

    model = AutoModelForCausalLM.from_pretrained(
                cfg.train_config.model,
                device_map=device,
                torch_dtype= torch.bfloat16,
                # quantization_config=bnb_config,
                # attn_implementation="flash_attention_2"
            )

    tokenizer = AutoTokenizer.from_pretrained(cfg.train_config.model,
            truncation=cfg.train_config.truncation,
            padding="max_length",
            max_length=cfg.train_config.max_length,)
    tokenizer.padding_side = cfg.train_config.truncation_side

    train_args = TrainingArguments(
                    output_dir=cfg.path,
                    per_device_train_batch_size=cfg.train_config.train_batch_size,
                    per_device_eval_batch_size=cfg.train_config.predict_batch_size,
                    num_train_epochs=cfg.train_config.num_train_epochs,
                    # save_strategy="epoch",
                    # evaluation_strategy="epoch",
                    # logging_dir="./logs",
                    # report_to="wandb",  # Enables W&B logging
                )


    # model = train_sft_model(train_args, cfg, wandb_run, model, tokenizer, datamodule)

    return model





def merge_experts(library=None, library_path=None, type='uniform'):

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
        library,
        selector_config=selector_config,
        device_map=device
        )
    
    # save model + selector for future re-use
    model.save_pretrained(f"{library_path}_{type}")
    return model



    



if __name__ == "__main__":
    local_train()





