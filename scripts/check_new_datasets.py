from huggingface_hub import hf_hub_download
import os
import re
from dataclasses import dataclass
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
# from mttl.datamodule.base import DataModule, DatasetConfig, DefaultCollator
# from mttl.models.library.dataset_library import DatasetLibrary, LocalDatasetEngine
import json
import os
import json
from sklearn.model_selection import train_test_split
import zipfile
import glob

import pandas as pd

path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities_main/datasets/safe/beavertails/no_cleaned/330k_test_beaverTails_safe_gpt4_controversial_topics_politics.csv"

data_dir = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities_main/datasets/safe/beavertails/no_cleaned/"  # replace with your actual path
save_dir = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities_main/datasets/safe/beavertails/sampled/"
# Find all CSV files starting with 330k
csv_files = glob.glob(os.path.join(data_dir, "330k*.csv"))
print(f"Found {len(csv_files)} files")


unsafe_dir = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities_main/datasets/harm/beavertails/"
unsafe_save_dir = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities_main/datasets/harm/beavertails/sampled/"

for x_path in csv_files:
    filename = os.path.basename(x_path)
    unsafe_name = filename.replace("safe_gpt4", "unsafe")
    y_path = os.path.join(unsafe_dir, unsafe_name)

    # Skip if the corresponding y_file doesn't exist
    if not os.path.exists(y_path):
        print(f"⚠️ Skipping {filename}: not found in {unsafe_dir}")
        continue

    print(f"🧹 Cleaning: {filename}")

    # Load both files
    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    print(f'size of safe {len(x_df)}, size of unsafe is: {len(y_df)}')

    # Find indices where 'response' is empty or whitespace
    bad_indices = x_df[
        x_df['response'].isnull() | (x_df['response'].str.strip() == '')
    ].index

    # Drop those rows from both DataFrames
    x_clean = x_df.drop(index=bad_indices)
    y_clean = y_df.drop(index=bad_indices)
    print(f'cleaned  ------ size of safe {len(x_clean)}, size of unsafe is: {len(y_clean)}')

    # Save to parent of x_dir
    x_clean.to_csv(os.path.join(save_dir, filename), index=False)
    y_clean.to_csv(os.path.join(unsafe_save_dir, unsafe_name), index=False)

    print(f" Saved cleaned: {filename}")

# Process each file
# for file_path in csv_files:
#     print(f"Processing: {file_path}")
    
#     df = pd.read_csv(file_path)

#     # Filter out rows with empty or whitespace-only 'response'
#     empty_response_rows = df[df['response'].isnull() | (df['response'].str.strip() == '')]
#     print(f"Number of empty responses: {len(empty_response_rows)}") 

#     df_cleaned = df[~(df['response'].isnull() | (df['response'].str.strip() == ''))]
#     print(f"Number of non empty responses: {len(df_cleaned)}")
#     print(f"percentage: {len(empty_response_rows)*100/len(df)}")

#     # Save cleaned file (overwrite or to a new folder)
#     filename = os.path.basename(file_path)
#     cleaned_path = os.path.join(save_dir, filename)
#     df_cleaned.to_csv(cleaned_path, index=False)

#     print(f"Saved cleaned file: {cleaned_path}")




# # Load your CSV
# df = pd.read_csv(path)

# # Find rows where 'response' is empty or null
# empty_response_rows = df[df['response'].isnull() | (df['response'].str.strip() == '')]

# print(f"Number of empty responses: {len(empty_response_rows)}")
# print(empty_response_rows.head())

# # Remove rows where 'response' is empty or null
# df_cleaned = df[~(df['response'].isnull() | (df['response'].str.strip() == ''))]
# print(f"Number of non empty responses: {len(df_cleaned)}")

# # Save to a new CSV
# df_cleaned.to_csv("/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities_main/datasets/safe/beavertails/330k_test_beaverTails_safe_gpt4_controversial_topics_politics.csv", index=False)




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


# class BeaverTailsUnsafeDataConfig(DatasetConfig):
#     pass

# @DataModule.register("BeaverTailsUnsafe", config_cls=BeaverTailsUnsafeDataConfig)
# class BeaverTailsUsafeDataModule(DataModule):
#     def setup_dataset(self):

#         n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
#         complete_name = self.config.finetune_task_name # --> "camel-biology"
#         folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets/safe/beavertails/')

#         file_path = os.path.join(folder_path, f'330k_train_{complete_name}.csv')
#         if not file_path: 
#             complete_name.split('-')[1]
#             load_data(directory_path = "datasets/harm/", repo_id=f"camel-ai/{complete_name.split('-')[1]}", name=complete_name.split('-')[1])
#             train_test_split_(directory_path = f"datasets/harm/", category=complete_name)

#         # dataset_train = LocalDatasetEngine.pull_dataset("/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities/datasets/train_data.json")
#         dataset_train = load_dataset('csv', data_files=f"{folder_path}/330k_train_{complete_name}.csv")
#         dataset_test = load_dataset('csv', data_files=f"{folder_path}/330k_test_{complete_name}.csv")
#         dataset_dev = load_dataset('csv', data_files=f"{folder_path}/330k_test_{complete_name}.csv")
        
        
#         # convert task_id to task_name and labels
#         # prompt,response,category,is_safe
#         def map_example(example):
#             example["dataset"] = complete_name
#             example["task_name"] = complete_name
#             example["source"] = example["prompt"]
#             example["target"] = example["response"]
        
#             return example

#         self._task_to_id = {}
#         self._task_names = self.config.finetune_task_name #[]
#         # NEED check hard coding
#         self.for_generation = True
        
#         self.train_dataset = dataset_train.map(
#             map_example,
#             num_proc=n_proc,
#         )['train'] #.select(range(32))
#         self.dev_dataset = dataset_dev.map(
#             map_example,
#             num_proc=n_proc,
#         )['train'] #.select(range(32))
#         self.test_dataset = dataset_test.map(
#             map_example,
#             num_proc=n_proc,
#         )['train'] #.select(range(32))


