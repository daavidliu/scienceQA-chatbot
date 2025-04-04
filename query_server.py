import requests
import json

SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions" # OpenAI compatible endpoint

headers = {"Content-Type": "application/json"}

payload = {
    "model": "gemma-local", # This name is often ignored by the local server
    "messages": [
        {
            "role": "user",
            "content": "What are the first 5 prime numbers?"
        }
        # You can add more messages for conversational context:
        # {"role": "assistant", "content": "The first prime number is 2."},
        # {"role": "user", "content": "What about the next four?"}
    ],
    "max_tokens": 100,
    "temperature": 0.3,
    "stream": False, # Set to True for token-by-token streaming
    "logprobs": 1, # Set to 1 to get log probabilities
}

try:
    response = requests.post(SERVER_URL, headers=headers, json=payload)
    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

    result = response.json()

    # Print the full response for inspection
    # print("Full Response:")
    # print(json.dumps(result, indent=2))
    # print("-" * 30)

    print(result)

    # Extract and print the main content
    if "choices" in result and len(result["choices"]) > 0:
        content = result["choices"][0]["message"]["content"]
        print("Model Response:")
        print(content.strip())
    else:
        print("Could not find 'choices' in the response:")
        print(json.dumps(result, indent=2))

    # Print usage stats if available
    if "usage" in result:
        print("-" * 30)
        print("Token Usage:")
        print(json.dumps(result["usage"], indent=2))


except requests.exceptions.RequestException as e:
    print(f"Error communicating with the server: {e}")
except json.JSONDecodeError:
    print(f"Could not decode JSON response: {response.text}")