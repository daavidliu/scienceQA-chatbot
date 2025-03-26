import argparse
from prompts import SYSTEM_ROLE, science_QA_prompt
from openai import OpenAI
import json
import threading
import time
import sys
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
from dataset_functions import dataset_from_disk

from PIL import Image
import base64
from io import BytesIO

from math import exp
import numpy as np

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load the dataset
arrow_file_path="science_qa/test/data-00000-of-00001.arrow"

test_data = dataset_from_disk(arrow_file_path, 5)

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

# Evaluate GPT-4o mini on the test set
correct = 0
total = 0

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
    print(f"Confidence: {confidence}%")
    
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

results = {
    "correct": [

    ],
    "incorrect": [

    ]
}

if __name__ == "__main__":
    print("this is main.py...")
    # while True:
    #     user_input = input("\nEnter your prompt: ")
        
    #     stop_loading = False
    #     loading_thread = threading.Thread(target=loading_animation)
    #     loading_thread.start()
        
    #     result = process_text(user_input)
    
        
    #     print(f"\n{model_name}:", result)

    # stop_loading = False
    # loading_thread = threading.Thread(target=loading_animation)
    # loading_thread.start()

    for sample in tqdm(test_data):
        prompt = science_QA_prompt(
            question=sample["question"],
            choices=sample["choices"],
            hint=sample["hint"],
            grade=sample["grade"],
            subject=sample["subject"],
            topic=sample["topic"],
            category=sample["category"],
            skill=sample["skill"]
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
            print("Correct!")
            correct += 1
            results["correct"].append(sample_copy)
        else :
            # print(f"Incorrect!\nPredicted: {GPT_response["answer"]}\nActual: ", sample["answer"])
            print("Incorrect!")
            results["incorrect"].append(sample_copy)

        total += 1

    # stop_loading = True
    # loading_thread.join()

    print(f"Accuracy: {correct / total * 100:.2f}%")

    # save results to file
    with open("results_full.json", "w") as f:
        json.dump(results, f, indent=4)

    # actual_answer = str(sample["answer"])  # Convert to string for comparison
    # if predicted_answer == actual_answer:
    #     correct += 1
    # total += 1
