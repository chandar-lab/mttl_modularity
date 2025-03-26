from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.inference.prompts import PromptTemplate
import pandas as pd


project_connection_string = "<your-connection-string-goes-here>"

project = AIProjectClient.from_connection_string(
    conn_str=project_connection_string, credential=DefaultAzureCredential()
)


def get_chat_response(messages, prompt):
    # create a prompt template from an inline string (using mustache syntax)
    chat = project.inference.get_chat_completions_client()
    response = chat.complete(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        # temperature=1,
        # frequency_penalty=0.5,
        # presence_penalty=0.5,
    )

    # add the prompt messages to the user messages
    return response




if __name__ == "__main__":
    # Example usage
    file_path = '/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities_main/datasets/harm/beavertails/30k_test_beaverTails_unsafe_animal_abuse.csv'  # replace with the file path

    df = pd.read_csv(file_path)
    if 'prompt' not in df.columns or 'category' not in df.columns:
        print("CSV must have 'prompt'")

    df = df.head(1)
    df.response = df['prompt'].apply(get_chat_response)
    df['is_safe'] = True  # Set is_safe to True


    new_file_path = file_path.replace('unsafe', 'safe_gpt4.csv')
    df[['prompt', 'response', 'category', 'is_safe']].to_csv(new_file_path, index=False)
    print(f"Processed file saved as: {new_file_path}")





