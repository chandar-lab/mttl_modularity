import pandas as pd
from openai import OpenAI

client = OpenAI()
import os
import os
from openai import OpenAI
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
# openai.api_key = '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'

import time
import random

load_dotenv()

# def get_gpt_response(user_prompt):
#     client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
#     response = client.chat.completions.create(
#         model="gpt-4o-2024-11-20",
#         messages=[
#             {"role": "system", "content": "hi"},
#             {"role": "user", "content": user_prompt},
#         ],
#         max_tokens=10,
#         temperature=0.0,
#     )
#     return response.choices[0].message.content

def generate_response(prompt):
    retries = 5
    delay = 1  # Initial delay in seconds

    client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),)
    for attempt in range(retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[ {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-4o",
                )
            return chat_completion['choices'][0]['message']['content']
        
        except RateLimitError:
            print(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2 + random.uniform(0, 1)  # Exponential backoff with jitter
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    print("Failed to generate response after multiple attempts.")
    return ""



## read csv file
def process_csv(file_path):

    df = pd.read_csv(file_path)
    if 'prompt' not in df.columns or 'category' not in df.columns:
        print("CSV must have 'prompt'")
        return

    df = df.head(1)
    df.response = df['prompt'].apply(generate_response)
    df['is_safe'] = True  # Set is_safe to True


    new_file_path = file_path.replace('unsafe', 'safe_gpt4.csv')
    df[['prompt', 'response', 'category', 'is_safe']].to_csv(new_file_path, index=False)
    print(f"Processed file saved as: {new_file_path}")

# Example usage
file_path = '/home/mila/m/maryam.hashemzadeh/scratch/saftly/mttl_modularities_main/datasets/harm/beavertails/30k_test_beaverTails_unsafe_animal_abuse.csv'  # replace with the file path
process_csv(file_path)


