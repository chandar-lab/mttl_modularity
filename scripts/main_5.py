import torch

assert (
    torch.cuda.get_device_capability()[0] >= 8
), "Hardware not supported for Flash Attention"


import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import setup_chat_format, SFTTrainer, SFTConfig
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel
from transformers import pipeline, DataCollatorWithPadding, DataCollatorForSeq2Seq
from tqdm import tqdm
from random import randint
import os
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
# from langchain.schema import OutputParserException
# from langchain.output_parsers.regex import RegexParser
import re
from datasets import Dataset

from mttl.datamodule.base import get_datamodule



def data_loader(tokenizer, config):
    train_dataset = load_dataset(
        "json",
        data_files=config.dataset.train_file,
        split="train[:20]", #split="train[:50]"
    ).shuffle()
    test_dataset = load_dataset(
        "json",
        data_files=config.dataset.test_file,
        split="train[:20]",
    ).shuffle()

    # Map ratings to text labels
    rating_mapping = {-1: "No", 0: "Neutral", None: "Neutral", 1: "Yes"}

    # Process Data
    def preprocess_function_completion(sample):
        inputs = []
        labels = []

        problems = [ex for ex in sample["problem"]]
        current_completions = [ex for ex in sample["current_completion"]]
        past_completions = [ex for ex in sample["past_completions"]]
        labeler = [ex for ex in sample["current_rating"]]
        labels = [rating_mapping[ex] for ex in labeler]

        for i in range(len(problems)):
            input_text = f"Problem is: {problems[i]}\n These are previous thoughts: {past_completions[i]} \n This is current thought: {current_completions[i]} \n Is that correct? Just say Yes, No, or Neutral?"
            inputs.append(input_text)

        # Only tokenize if inputs are not empty
        model_inputs = {"inputs": [], "labels": []}
        if inputs:
            model_inputs["inputs"] = inputs
            model_inputs["labels"] = labels

        return model_inputs
    
    # Process Data
    # Map ratings to text labels
    rating_mapping_finalanswer = {"found_error": "No", "give_up": "Neutral", None: "Neutral", "solution": "Yes"}
    def preprocess_function_finalanswer(sample):
        inputs = []
        labels = []

        problems = [ex for ex in sample["problem"]]
        pre_generated_answer = [ex for ex in sample["pre_generated_answer"]]
        pre_generated_steps = [ex for ex in sample["pre_generated_steps"]]
        labeler = [ex for ex in sample["finish_reason"]]
        labels = [rating_mapping_finalanswer[ex] for ex in labeler]

        for i in range(len(problems)):
            input_text = f"Problem is: {problems[i]}\n These are thoughts: {pre_generated_steps[i]} \n This is the response: {pre_generated_answer[i]} \n Is that correct? Just say Yes, No, or Neutral?"
            inputs.append(input_text)

        # Only tokenize if inputs are not empty
        model_inputs = {"inputs": [], "labels": []}
        if inputs:
            model_inputs["inputs"] = inputs
            model_inputs["labels"] = labels

        return model_inputs
    
    def preprocess_function_tokenizing(sample):
        # model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        ### with tokenizer
        model_inputs = tokenizer(
            sample["inputs"],
            max_length=config["max_seq_length"],
            truncation=True,
            # padding="max_length",
            padding=True,
        )

        labels_id = tokenizer(
            sample["labels"],
            max_length=config["max_seq_length"],
            truncation=True,
            # padding="max_length",
            padding=True,
        )

        # Return tokenized inputs and labels
        model_inputs["labels"] = labels_id["input_ids"]

        return model_inputs

    if config.finalanswer==False:
        train_data = train_dataset.map(
            preprocess_function_completion, batched=True, remove_columns=train_dataset.column_names
        )
        test_data = test_dataset.map(
            preprocess_function_completion, batched=True, remove_columns=test_dataset.column_names
        )
    elif config.finalanswer==True:
        train_data = train_dataset.map(
            preprocess_function_finalanswer, batched=True, remove_columns=train_dataset.column_names
        )
        test_data = test_dataset.map(
            preprocess_function_finalanswer, batched=True, remove_columns=test_dataset.column_names
        )

    train_data = train_data.map(
        preprocess_function_tokenizing,
        batched=True,
        remove_columns=train_data.column_names,
    )

    

    return train_data, test_data



def custom_evaluation(trainer, eval_dataset, tokenizer, regex, test_data):
    model = trainer.model
    model.eval()
    results = []

    for i, item in enumerate(eval_dataset):
        input_text = item["input"]  # Adjust to your dataset structure
        true_label = test_data["labels"][i]

        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=trainer.args.max_seq_length)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            match = re.search(regex, generated_text)
            if match:
                response = match.group(1)  # Extract the response (Yes, No, or Neutral)
            else:
                response = None
            is_correct = response == true_label
            results.append({
                "generated": generated_text,
                "response": response,
                "is_correct": is_correct,
                "true_label": true_label
            })
        except Exception as e:
            results.append({
                "generated": generated_text,
                "response": None,
                "is_correct": False,
                "error": str(e)
            })

    accuracy = sum([1 for r in results if r["is_correct"]]) / len(results)
    return {"accuracy": accuracy, "details": results}



class CustomSFTTrainer(SFTTrainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override the evaluation method to filter the dataset and run evaluation.
        """       
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        filtered_dataset = self.filter_dataset(eval_dataset)
        results = super().evaluate(
            eval_dataset=filtered_dataset, 
            ignore_keys=ignore_keys, 
            metric_key_prefix=metric_key_prefix
        )

        return results

    def filter_dataset(self, dataset):
        """
        Filter the dataset to include only samples with responses 'Yes', 'No', or 'Neutral'
        after the sentence 'Just say Yes, No, or Neutral?'.
        """
        filtered_data = []
        for example in dataset:
            response = example['completion']  # Adjust based on your dataset's structure
            if response.strip() in {"Yes", "No", "Neutral"}:
                filtered_data.append(example)
        
        # Return a filtered dataset object

        return Dataset.from_dict({key: [ex[key] for ex in filtered_data] for key in dataset.column_names})




class TrainerSetup:
    """Class to set up training arguments and initialize Trainer"""

    def __init__(
        self,
        config,
        model,
        tokenizer,
        train_data,
        test_data,
        peft_config,
    ):
        self.max_seq_length = config.max_seq_length
        self.config=config
        self.output_dir=f'{config.training.output_dir}_{config.model.model_id}_{config.peft.r}'
        self.args = SFTConfig(
            output_dir=self.output_dir,  
            **vars(config.training)  
        )


        self.trainer = CustomSFTTrainer(
            model=model,
            args=self.args,
            train_dataset=train_data,
            eval_dataset=test_data,
            peft_config=peft_config,
            max_seq_length=self.max_seq_length,
            tokenizer=tokenizer,
        )

    def train_and_save(self):
        self.trainer.train()
        # self.trainer.save_model()
        
        self.trainer.model.save_pretrained(self.output_dir)

    def merge_and_push(self, new_model_name):
        # Reload model in FP16 and merge it with LoRA weights
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_id,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map={"": torch.cuda.current_device()},
        )
        model = PeftModel.from_pretrained(base_model, self.output_dir)
        merged_model = model.merge_and_unload()

        # Reload tokenizer to save it
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_id, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"


        if self.config.push_to_hf:
            from huggingface_hub import login

            login(token="", add_to_git_credential=True)  # ADD YOUR TOKEN HERE
            merged_model.push_to_hub(new_model_name, use_temp_dir=False)
            tokenizer.push_to_hub(new_model_name, use_temp_dir=False)

        return merged_model


class ModelEvaluator:
    """Class to handle model evaluation"""

    def __init__(self, config, merged_model, tokenizer):
        self.pipe = pipeline(
            "text-generation",
            model=merged_model,
            tokenizer=tokenizer,
            # device=
        )

        self.temperature = config.evaluation.temperature
        self.top_k = config.evaluation.top_k
        self.top_p = config.evaluation.top_p
        self.config=config

    def evaluate(self, test_data, wandb_run):
        correct = 0
        total = len(test_data)

        predictions = self.pipe(
            test_data["inputs"],
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=True,
            temperature=self.temperature,
            max_length=self.config.max_seq_length,
            truncation=True,
            batch_size=4,
        )
        predicted_labels = [pred[0]["generated_text"] for pred in predictions]

        regex = r"Just say Yes, No, or Neutral?*\b(Yes|No|Neutral)\b"
        # regex_parser = RegexParser(regex=regex, output_keys=["response"])

        results = []
        for i, item in enumerate(predicted_labels):
            try:
                match = re.search(regex, item)
                if match:
                    response = match.group(1)  # Extract the response (Yes, No, or Neutral)
                else:
                    response = None
                is_correct = response == test_data["labels"][i]
                results.append({"generated": item, "response": response, "is_correct": is_correct, "true_label":test_data["labels"][i]})
            except OutputParserException:
                results.append({"generated": item, "response": None, "is_correct": False})

        with open(f"results_{self.config.model.model_id}_{self.config.peft.r}.json", "w") as f:
            json.dump(results, f)
        
        correct_predictions = sum(1 for result in results if result["is_correct"])
        accuracy = correct_predictions / total * 100
        wandb_run.log({"evaluation/accuracy": accuracy})
        print(f"Accuracy: {accuracy:.2f}%")
       
        


@hydra.main(config_name="config.yaml") #, config_name="config")
def my_train(cfg: DictConfig):
    os.environ["WANDB_PROJECT"] = "gpv"
    os.environ["WANDB_LOG_MODEL"] = "tests"

    wandb_run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True)  # Log entire config
    )

    # Model and Tokenizer
    model_id = cfg.model.model_id
    bnb_config = BitsAndBytesConfig(**cfg.model.bnb_config)
    peft_config = LoraConfig(**cfg.peft)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": torch.cuda.current_device()},
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        # attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id,
            truncation=True,
            padding="max_length",
            max_length=cfg.max_seq_length,)
    tokenizer.padding_side = "left"

    train_data, test_data = data_loader(tokenizer, cfg)
        

    # Training Setup
    trainer_setup = TrainerSetup(cfg, model, tokenizer, train_data, test_data, peft_config)
    trainer_setup.train_and_save()
    merged_model = trainer_setup.merge_and_push("finetuned-model")
    evaluator = ModelEvaluator(cfg, merged_model, tokenizer)
    evaluator.evaluate(test_data, wandb_run)

    wandb_run.finish()




if __name__ == "__main__":
    my_train()