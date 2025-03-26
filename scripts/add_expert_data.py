from huggingface_hub import hf_hub_download
import os
import re
from dataclasses import dataclass
import sys

sys.path.append(os.path.abspath('./'))

# import mttl
from mttl.datamodule.base import DataModule, DatasetConfig, DefaultCollator
from mttl.models.library.dataset_library import DatasetLibrary, LocalDatasetEngine
from datasets import load_dataset
import json
import os
import pandas as pd
import mttl.datamodule

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

def train_test_split_json(directory_path = "datasets/"):
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


import csv


def train_test_split_csv(directory_path="datasets/"):
    # Load your consolidated data from CSV
    with open(f'{directory_path}harm/harmful_behaviors_llmattack.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]  # Convert to a list of dictionaries

    # Splitting the data into train, dev, and test
    train_data, temp_test_data = train_test_split(data, test_size=0.20, random_state=42, shuffle=True)  # 80% train, 20% for temp_test
    dev_data, test_data = train_test_split(temp_test_data, test_size=0.50, random_state=42)  # Splitting 50% of 20% to each

    # Save train_data as CSV
    with open(f'{directory_path}harm/harmful_behaviors_llmattack_train.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=train_data[0].keys())
        writer.writeheader()  # Write headers (fieldnames)
        writer.writerows(train_data)  # Write rows

    # Save dev_data as CSV
    with open(f'{directory_path}harm/harmful_behaviors_llmattack_val.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=dev_data[0].keys())
        writer.writeheader()  # Write headers (fieldnames)
        writer.writerows(dev_data)  # Write rows

    # Save test_data as CSV
    with open(f'{directory_path}harm/harmful_behaviors_llmattack_test.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=test_data[0].keys())
        writer.writeheader()  # Write headers (fieldnames)
        writer.writerows(test_data)  # Write rows


def load_beavertrails(save_path=None):

    dataset = load_dataset('PKU-Alignment/BeaverTails', cache_dir='/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities_main/datasets/harm')
    for split in dataset.keys():
        dataset[split].to_csv(f"{save_path}/{split}.csv")

import ast
def seperate_category_beavertrail(save_path=None, split=None):
    "30k_test_beaverTails_unsafe_animal_abuse"

    df = pd.read_csv(f"{save_path}/harm/BeaverTails_{split}.csv")
    df["category"] = df["category"].apply(ast.literal_eval)
    df_filtered = df[df["is_safe"] == False]
    categories = df_filtered['category'][0].keys()

    # Separate rows by category and save them
    for category in categories:
        # Filter rows where the category is True
        subset = df_filtered[df_filtered['category'].apply(lambda x: x.get(category, False) == True)]
        print('-------------------\n')
        print(category, subset.shape[0])
        
        if not subset.empty:  # If there are rows for this category  30k_test_beaverTails_unsafe_animal_abuse
            file_name = f"{save_path}/harm/beavertails/{split}_beaverTails_unsafe_{category.replace(',', '_')}.csv"
            # subset.to_csv(file_name, index=False)

    df_filtered_safe = df[df["is_safe"] == True]
    file_name_safe = f"{save_path}/safe/{split}_beaverTails_safe.csv"
    # df_filtered_safe.to_csv(file_name_safe, index=False)
    print('safe', df_filtered_safe.shape[0])


# load_data(directory_path = "datasets/")
# train_test_split_csv()


# Load only the round 0 dataset
# dataset = load_dataset('PKU-Alignment/BeaverTails', cache_dir='/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities_main/datasets/harm')
save_path='/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities_main/datasets'
# for split in dataset.keys():
#     dataset[split].to_csv(f"{save_path}/{split}.csv")

# dataset["330k_train"].to_csv("330k_train.csv")
# dataset["330k_test"].to_csv("330k_test.csv")
# dataset["30k_train"].to_csv("30k_train.csv")
# dataset["30k_test"].to_csv("30k_test.csv")

# print("data has been splitted!")


# split = '300k_test' # '30k_test', '30k_train', '300k_train', '300k_test'

# seperate_category_beavertrail(save_path, '30k_train')
# seperate_category_beavertrail(save_path, '30k_test')

# seperate_category_beavertrail(save_path, '330k_test')
# seperate_category_beavertrail(save_path, '330k_train')



dataset = load_dataset("fwnlp/self-instruct-safety-alignment")

# Convert each split to CSV
for split in dataset.keys():  # e.g., 'train', 'test', etc.
    df = pd.DataFrame(dataset[split])  # Convert to DataFrame
    
    # Save as CSV
    df.to_csv(f"self_instruct_safety_alignment_{split}.csv", index=False)

    print(f"Saved {split} split as CSV!")



# # Load dataset
# df = pd.read_csv(f"{save_path}/BeaverTails_{split}.csv")


# # # Convert to Pandas DataFrame (assuming 'train' split)
# # df = dataset["train"].to_pandas()

# # Filter rows where is_safe is False
# df_filtered = df[df["is_safe"] == False]

# # Define save directory
# save_path = "/path/to/save/filtered_data"
# os.makedirs(save_path, exist_ok=True)

# # Group by 'category' and save each group separately
# for category, group in df_filtered.groupby("category"):
#     file_name = f"{save_path}/BeaverTails_{split}_{category}_unsafe.csv"
#     group.to_csv(file_name, index=False)


print()


# class CamelBiologyDataConfig(DatasetConfig):
#     pass

# @DataModule.register("camel-biology", config_cls=CamelBiologyDataConfig)
# class CamelBiologyDataModule(DataModule):
#     def setup_dataset(self):
#         n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
#         dataset_train = LocalDatasetEngine.pull_dataset("train_data.json", name="default")
#         dataset_test = LocalDatasetEngine.pull_dataset("test_data.json", name="default")
#         dataset_dev = LocalDatasetEngine.pull_dataset("dev_data.json", name="default")

#         # convert task_id to task_name and labels
#         def map_example(example):
#             example["dataset"] = "camel-biology"
#             example["task_name"] = "camel-biology"
#             example["source"] = example["message_1"]
#             example["target"] = example["message_2"]
        
#             return example

#         self._task_to_id = {}
#         self._task_names = self.config.finetune_task_name #[]
#         # NEED check hard coding
#         self.for_generation = True
        
#         self.train_dataset = dataset_train.map(
#             map_example,
#             num_proc=n_proc,
#         )
#         self.dev_dataset = dataset_dev.map(
#             map_example,
#             num_proc=n_proc,
#         )
#         self.test_dataset = dataset_test.select(range(20)).map(
#             map_example,
#             num_proc=n_proc,
#         )



# datamodule = CamelBiologyDataModule(config=None, for_generation=True)



