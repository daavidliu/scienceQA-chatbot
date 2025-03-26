import argparse
from prompts import SYSTEM_ROLE
from openai import OpenAI
import json
import threading
import time
import sys
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load the dataset

# Path to the Arrow file
# for testing

client = OpenAI(api_key=OPENAI_API_KEY)

model_name = "gpt-4o-mini"

def clean_text(text):
    # clean text by removing markdown formatting
    return text.strip("```json").strip("```").strip()

def process_text(prompt):

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {
                "role": "system", 
                "content": SYSTEM_ROLE
            },
            {   
                "role": "user", 
                "content": prompt
            }
        ]
    )

    response.choices[0].message.content

    return response.choices[0].message.content

def loading_animation():
    animation = "|/-\\"
    idx = 0
    while not stop_loading:
        print(f"\rLoading {animation[idx % len(animation)]}", end="")
        idx += 1
        time.sleep(0.1)


if __name__ == "__main__":
    print("Chat with " + model_name + "!")
    while True:
        user_input = input("\nEnter your prompt: ")
        
        stop_loading = False
        loading_thread = threading.Thread(target=loading_animation)
        loading_thread.start()
        
        result = process_text(user_input)
        
        stop_loading = True
        loading_thread.join()
        
        print(f"\n{model_name}:", result)
