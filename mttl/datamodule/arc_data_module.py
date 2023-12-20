from datasets import load_dataset
from mttl.datamodule.base import (
    DefaultDataModule,
    DatasetConfig,
    MultipleChoiceCollator,
)
from dataclasses import dataclass
import os

from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task


@dataclass
class ArcDataConfig(DatasetConfig):
    arc_type: str = "ARC-Easy"


class ArcMultiChoiceDataModule(DefaultDataModule):
    @property
    def collate_fn(self):
        return MultipleChoiceCollator(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
            train_on_inputs=self.config.train_on_inputs,
            task_to_id=self.task_to_id,
        )

    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = load_dataset("ai2_arc", name=self.config.arc_type)["test"]

        # convert task_id to task_name and labels
        def map_example(example):
            prompt = "Question: {}\nAnswer:"
            targets = [choice for choice in example["choices"]["text"]]

            # Prevents `label` from having wrong values due to dataset values
            # mixed between strings and integers
            answer_key_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
            answer_key = answer_key_map.get(example["answerKey"], example["answerKey"])

            example["source"] = prompt.format(example["question"])
            example["target"] = targets
            example["label_index"] = ["A", "B", "C", "D", "E"].index(answer_key)
            return example

        dataset = dataset.map(
            map_example,
            num_proc=n_proc,
        )

        self._task_to_id = {}
        self._task_names = []

        self.train_dataset = self.dev_dataset = None
        self.test_dataset = dataset
        self.print_infos()
