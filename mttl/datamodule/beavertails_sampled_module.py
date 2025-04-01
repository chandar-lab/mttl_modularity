from huggingface_hub import hf_hub_download
import os
import re
from dataclasses import dataclass
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from mttl.datamodule.base import DataModule, DatasetConfig, DefaultCollator
from mttl.models.library.dataset_library import DatasetLibrary, LocalDatasetEngine
import json
import os
import json
from sklearn.model_selection import train_test_split
import zipfile

def load_data(directory_path = "datasets/", repo_id="camel-ai/biology", name='biology'):

    hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=f"{name}.zip",
                    local_dir=directory_path, local_dir_use_symlinks=False)

    if not os.path.exists(f'{directory_path}{name}/'):
        os.makedirs(f'{directory_path}{name}/')

    with zipfile.ZipFile(f"{directory_path}{name}.zip", 'r') as zip_ref:
        # Extract all the contents into the directory
        zip_ref.extractall(f'{directory_path}{name}/')

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



def train_test_split_(directory_path = "datasets/", category=None):
    # Load your consolidated data
    with open(f'{directory_path}all_data.json', 'r') as file:
        data = json.load(file)

    train_data, temp_test_data = train_test_split(data, test_size=0.20, random_state=42, shuffle=True)  # 80% train, 20% for temp_test
    dev_data, test_data = train_test_split(temp_test_data, test_size=0.50, random_state=42)  # Splitting 50% of 20% to each

    with open(f'{category}_train_data.json', 'w') as file:
        json.dump(train_data, file, indent=4)

    with open(f'{category}_dev_data.json', 'w') as file:
        json.dump(dev_data, file, indent=4)

    with open(f'{category}_test_data.json', 'w') as file:
        json.dump(test_data, file, indent=4)


# load_data(directory_path = "datasets/")
# train_test_split_()
"/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities_main/datasets/harm/beavertails/30k_test_beaverTails_unsafe_animal_abuse.csv" 
"/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities_main/datasets/harm/beavertails/30k_train_beaverTails_unsafe_animal_abuse.csv"
"'/network/scratch/m/maryam.hashemzadeh/saftly/mttl_modularities_main/datasets/harm/beavertails//30k_train_beaverTails_unsafe_beaverTails_unsafe_animal_abuse.csv'"
class BeaverTailsSampledDataConfig(DatasetConfig):
    pass

@DataModule.register("BeaverTailsSampled", config_cls=BeaverTailsSampledDataConfig)
class BeaverTailsSampledDataModule(DataModule):
    def setup_dataset(self):

        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        complete_name = self.config.finetune_task_name # --> "camel-biology"
        folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets/sampled/beavertails/')

        file_path = os.path.join(folder_path, f'330k_test_{complete_name}.csv')
        if not file_path: 
            complete_name.split('-')[1]
            load_data(directory_path = "datasets/harm/", repo_id=f"camel-ai/{complete_name.split('-')[1]}", name=complete_name.split('-')[1])
            train_test_split_(directory_path = f"datasets/harm/", category=complete_name)

        # dataset_train = LocalDatasetEngine.pull_dataset("/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities/datasets/train_data.json")
        dataset_train = load_dataset('csv', data_files=f"{folder_path}/330k_train_{complete_name}.csv")
        dataset_test = load_dataset('csv', data_files=f"{folder_path}/330k_test_{complete_name}.csv")
        dataset_dev = load_dataset('csv', data_files=f"{folder_path}/330k_test_{complete_name}.csv")
        
        
        # convert task_id to task_name and labels
        # prompt,response,category,is_safe
        def map_example(example):
            example["dataset"] = complete_name
            example["task_name"] = complete_name
            example["source"] = example["prompt"]
            example["target"] = example["response"]
        
            return example

        self._task_to_id = {}
        self._task_names = self.config.finetune_task_name #[]
        # NEED check hard coding
        self.for_generation = True
        
        self.train_dataset = dataset_train.map(
            map_example,
            num_proc=n_proc,
        )['train'] #.select(range(32))
        self.dev_dataset = dataset_dev.map(
            map_example,
            num_proc=n_proc,
        )['train'] #.select(range(32))
        self.test_dataset = dataset_test.map(
            map_example,
            num_proc=n_proc,
        )['train'] #.select(range(32))


