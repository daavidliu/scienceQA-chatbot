# ScienceQA Chatbot

This is a Python-based chatbot project for **ScienceQA**, designed to answer scientific and educational questions.

## Prerequisites

Make sure you have the following tools installed on your system:

- **Python**: Version 3.8 or higher
- **Poetry**: See [Poetry installation guide](https://python-poetry.org/docs/#installation) if it's not installed.

---

## Project Setup

Follow these steps to set up the project in your local environment:

### 1. Clone the Repository
Download or clone the project from GitHub (or wherever the repository is hosted):

### 2. Install Dependencies
Use Poetry to install the project's dependencies:

    poetry install

This will:

Create a virtual environment (if needed).
Install all dependencies defined in pyproject.toml.

Running the Project
To run the application, follow these steps:

    poetry run python filename.py

### 3. Add your own API keys from hugging face and OpenAI
Make a copy of the file .env_example and rename it to .env
Paste in your API keys.

### 4. Download the hugging face API
Run this command to download the scienceqa dataset to your local machine.
Run
    poetry run python dataset_functions.py