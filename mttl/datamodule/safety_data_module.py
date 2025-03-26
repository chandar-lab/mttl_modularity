import os
import re
from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.models.library.dataset_library import DatasetLibrary


class SafetyDataConfig(DatasetConfig):
    pass
    # def __init__(
    #     self, config):
    #     super().__init__()

    #     self.config = config


@DataModule.register("a-safety", config_cls=SafetyDataConfig)
class SafetyDataModule(DataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = DatasetLibrary.pull_dataset("fwnlp/self-instruct-safety-alignment", name="default")

        # convert task_id to task_name and labels
        def map_example(example):
            example["dataset"] = "a-safety"
            example["task_name"] = "a-safety"
            example["source"] = example["prompt"]
            example["target"] = example["response"]
        
            return example

        self._task_to_id = {}
        self._task_names = self.config.finetune_task_name #[]
        # NEED check hard coding
        self.for_generation = True
        
        self.train_dataset = dataset["train"].select(range(200)).map(
            map_example,
            num_proc=n_proc,
        )
        self.dev_dataset = dataset["validation"].select(range(200)).map(
            map_example,
            num_proc=n_proc,
        )
        self.test_dataset = dataset["test"].select(range(200)).map(
            map_example,
            num_proc=n_proc,
        )
        # self.test_dataset = dataset["test"].select(range(20)).map(
        #     map_example,
        #     num_proc=n_proc,
        # )





