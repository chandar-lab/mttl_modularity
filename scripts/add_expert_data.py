from huggingface_hub import hf_hub_download
import os
import re
from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig, DefaultCollator
from mttl.models.library.dataset_library import DatasetLibrary, LocalDatasetEngine
import json
import os


def load_data(directory_path = "datasets/"):

    hf_hub_download(repo_id="camel-ai/biology", repo_type="dataset", filename="biology.zip",
                    local_dir=directory_path, local_dir_use_symlinks=False)

    all_data = []

    # Read each JSON file and append the data to the list
    for filename in os.listdir(f'{directory_path}biology'):
        if filename.endswith('.json'):
            with open(os.path.join(f'{directory_path}biology', filename), 'r') as file:
                data = json.load(file)
                all_data.append(data)

    # Write all data to a new JSON file
    with open(f'{directory_path}all_data.json', 'w') as outfile:
        json.dump(all_data, outfile, indent=4)

import json
from sklearn.model_selection import train_test_split

def train_test_split_(directory_path = "datasets/"):
    # Load your consolidated data
    with open(f'{directory_path}all_data.json', 'r') as file:
        data = json.load(file)

    train_data, temp_test_data = train_test_split(data, test_size=0.20, random_state=42, shuffle=True)  # 80% train, 20% for temp_test
    dev_data, test_data = train_test_split(temp_test_data, test_size=0.50, random_state=42)  # Splitting 50% of 20% to each

    with open(f'{directory_path}train_data.json', 'w') as file:
        json.dump(train_data, file, indent=4)

    with open(f'{directory_path}dev_data.json', 'w') as file:
        json.dump(dev_data, file, indent=4)

    with open(f'{directory_path}test_data.json', 'w') as file:
        json.dump(test_data, file, indent=4)


# load_data(directory_path = "datasets/")
# train_test_split_()

class CamelBiologyDataConfig(DatasetConfig):
    pass

@DataModule.register("camel-biology", config_cls=CamelBiologyDataConfig)
class CamelBiologyDataModule(DataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset_train = LocalDatasetEngine.pull_dataset("train_data.json", name="default")
        dataset_test = LocalDatasetEngine.pull_dataset("test_data.json", name="default")
        dataset_dev = LocalDatasetEngine.pull_dataset("dev_data.json", name="default")

        # convert task_id to task_name and labels
        def map_example(example):
            example["dataset"] = "camel-biology"
            example["task_name"] = "camel-biology"
            example["source"] = example["message_1"]
            example["target"] = example["message_2"]
        
            return example

        self._task_to_id = {}
        self._task_names = self.config.finetune_task_name #[]
        # NEED check hard coding
        self.for_generation = True
        
        self.train_dataset = dataset_train.map(
            map_example,
            num_proc=n_proc,
        )
        self.dev_dataset = dataset_dev.map(
            map_example,
            num_proc=n_proc,
        )
        self.test_dataset = dataset_test.select(range(20)).map(
            map_example,
            num_proc=n_proc,
        )



datamodule = CamelBiologyDataModule(config=None, for_generation=True)



