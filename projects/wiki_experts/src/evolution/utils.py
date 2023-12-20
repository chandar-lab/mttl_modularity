import glob
import sys
import os
import re
import copy
import wandb
import numpy as np
import pandas as pd
from functools import partial
import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from mttl.utils import setup_logging, logger
from projects.wiki_experts.src.evolution.evaluators import (
    TestLossEvaluator,
    ExtendedMMLUEvaluator,
    Evaluator,
    ExtendedRougeEvaluator,
)

from mttl.utils import logger
from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary


class TableLogger:
    def __init__(self):
        self.df = pd.DataFrame()

    def from_df(self, df):
        self.df = df
        self.columns = df.columns

    def log(self, row: dict):
        if self.df is None or len(self.df) == 0:
            self.df = pd.DataFrame(columns=row.keys())
        self.df.loc[len(self.df.index)] = row

    def get_table(self):
        return self.df

    def means(self):
        # calculate mean for each row, column and diagonal of self.df
        # filter numeric columns
        df_numeric = self.df.select_dtypes(include=[np.number])
        self.df["mean"] = df_numeric.mean(axis=1)
        self.df.loc["mean"] = df_numeric.mean(axis=0)
        self.df.loc["mean", "mean"] = np.diag(df_numeric).mean()

    def log_table_wandb(self):
        if wandb.run is not None:
            wandb.log({"table": wandb.Table(data=self.get_table())})


def get_loss(model, evaluator: Evaluator, **kwargs):
    return evaluator.get_loss(model, **kwargs)


def save_new_module(output_dir, module, task_name, postfix=""):
    module_copy = copy.deepcopy(module)
    # make Loras trainable so that they are saved
    module_copy.trainable_param_names = [
        n for n, p in module_copy.named_parameters() if re.match(".*lora.*", n)
    ]
    dest = output_dir + f"/{task_name}_{postfix}"
    os.makedirs(dest, exist_ok=True)
    ckpt_path = module_copy.save_pretrained(dest)
    del module_copy
    return ckpt_path


def find_version(s):
    match = re.search(r"_v(\d+)$", s)
    return int(match.group(1)) if match else 0


def remove_outdated_experts_from_library(library: HFExpertLibrary):
    for task in library.tasks:
        experts = library.get_experts_for_task(task)
        if len(experts) <= 1:
            continue
        version = [find_version(metadatum.expert_name) for metadatum in experts]
        arg_max = np.argmax(version)
        for i, metadatum in enumerate(experts):
            if isinstance(metadatum.expert_task_name, list):
                library.remove_expert(metadatum.expert_name, soft_delete=True)
            if i != arg_max:
                library.remove_expert(metadatum.expert_name, soft_delete=True)


def get_svd_embedding(lib, expert_name: str):
    try:
        embeddings = lib.get_auxiliary_data(
            data_type="embeddings", expert_name=expert_name
        )
    except ValueError:
        return None
    return embeddings["svd"]["embedding"]


def init_wandb_logger(args):
    if args.wandb_project is None:
        args.wandb_project = os.environ.get("WANDB_PROJECT", "MMLU_ninja_merge")
    if args.wandb_project:
        run_name = os.getenv("AMLT_JOB_NAME", f"{args.run_name}")
        # wandb.init(
        #     project=args.wandb_project,
        #     name=run_name,
        #     config=args,
        # )
        logger = pl.loggers.WandbLogger(
            project=args.wandb_project,
            name=run_name,
            config=args,
        )
    return logger


def log_wandb(scores, prefix):
    if wandb.run is not None:
        for t, v in scores.items():
            wandb.log({f"{prefix}_on_{t}_test_mmlu": v["mean"]})
