import argparse
from prompts import SYSTEM_ROLE, science_QA_prompt
from openai import OpenAI
import json
import threading
import time
import sys
from tqdm import tqdm
import os
from dataset_functions import dataset_from_disk, download_dataset

from PIL import Image
import base64
from io import BytesIO

from math import exp
import numpy as np

from dotenv import load_dotenv

import pyfiglet

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load the dataset
arrow_file_path="science_qa/test/data-00000-of-00001.arrow"
client = OpenAI(api_key=OPENAI_API_KEY)

model_name = "gpt-4o-mini"


def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4o-mini",
    max_tokens=500,
    temperature=0,
    seed=123,
    tools=None,
    logprobs=None,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
    top_logprobs=None,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion

def clean_text(text):
    # clean text by removing markdown formatting
    return text.strip("```json").strip("```").strip()

def GPT_send(prompt, base64_image=None, image_format=None):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_ROLE
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    if base64_image and image_format:
        messages.append({
            "role": "user",
            "content": [
                { "type": "text", "text": prompt },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{base64_image}",
                    },
                },
            ],
        })

    response = get_completion(
        model=model_name,
        temperature=0,
        messages=messages,
        logprobs=True,
        top_logprobs=1,
    )

    answer = response.choices[0].message.content

    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    logprob = top_logprobs[0].logprob
    confidence = np.round(np.exp(logprob)*100,2)
    
    return {
        "answer": answer,
        "confidence": confidence,
        "logprob": logprob
    }

def loading_animation():
    animation = "|/-\\"
    idx = 0
    while not stop_loading:
        print(f"\rLoading {animation[idx % len(animation)]}", end="")
        idx += 1
        time.sleep(0.1)

results = []

if __name__ == "__main__":

    # get user input a number store it in a variable called entries
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", type=int, default=5)
    args = parser.parse_args()
    entries = args.entries

    # Use pyfiglet to create styled text
    ascii_banner = pyfiglet.figlet_format("test.py")
    print(ascii_banner)

    try :
        test_data = dataset_from_disk(arrow_file_path, entries)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        try:
            download_dataset()
            test_data = dataset_from_disk(arrow_file_path, entries)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Exiting...")
            sys.exit(1)

    # Evaluate an LLM on the ScienceQA dataset
    correct = 0
    total = 0

    for sample in tqdm(test_data):
        sample["index"] = total # store the index of the sample, so it's easier to find later
        prompt = science_QA_prompt(
            question=sample["question"],
            choices=sample["choices"],
            hint=sample["hint"],
        )

        image = sample["image"]
        if image:
            buffered = BytesIO()
            image.save(buffered, format=image.format)
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_format = image.format.lower()
            sample["has_image"] = True
        else:
            base64_image = None
            image_format = None
            sample["has_image"] = False

        GPT_response = GPT_send(prompt, base64_image, image_format)
        sample["GPT_response"] = GPT_response
        
        # Create a copy of the sample without the 'image' field
        sample_copy = sample.copy()
        if 'image' in sample_copy:
            del sample_copy['image']

        # Ensure both are strings before comparison
        if str(GPT_response["answer"]) == str(sample["answer"]):
            correct += 1
            sample_copy["correct"] = True
            feedback = "Correct!  "
        else :
            sample_copy["correct"] = False
            feedback = "Incorrect!"

        tqdm.write(f"{total}: {feedback} Confidence: {GPT_response["confidence"]}%")

        results.append(sample_copy)
        total += 1

    print(f"Accuracy: {correct / total * 100:.2f}%")

    # get current date and time as string in the format "MM-DD-HH-MM"
    now = time.strftime("%m-%d-%H-%M")

    # save results to file
    filename = f"results/results_{now}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")
