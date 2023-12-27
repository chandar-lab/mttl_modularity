from datasets import load_dataset
from mttl.datamodule.base import (
    MultiChoiceDataModule,
    DatasetConfig,
)
from dataclasses import dataclass
import os


@dataclass
class OpenbookQADataConfig(DatasetConfig):
    pass


class OpenbookQAMultiChoiceDataModule(MultiChoiceDataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = load_dataset("openbookqa", name="main")["test"]

        # convert task_id to task_name and labels
        def map_example(example):
            example["source"] = example["question_stem"]
            example["target"] = example["choices"]["text"]
            example["label_index"] = ["A", "B", "C", "D"].index(
                example["answerKey"].strip()
            )
            example["task_name"] = "openbookqa"
            example["task_source"] = "openbookqa"
            return example

        dataset = dataset.map(
            map_example,
            num_proc=n_proc,
        )

        self._task_to_id = {}
        self._task_names = []

        self.train_dataset = None
        self.dev_dataset = dataset
        self.test_dataset = dataset
        self.print_infos()
