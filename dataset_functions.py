from huggingface_hub import login
import os
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
import time

# Load environment variables from .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# Just some loading animation stuff
stop_loading = True
def loading_animation():
    animation = "|/-\\"
    idx = 0
    while not stop_loading:
        print(f"\rLoading {animation[idx % len(animation)]}", end="")
        idx += 1
        time.sleep(0.1)

# Download the dataset from Hugging Face
def download_dataset(dataset_name="derek-thomas/ScienceQA"):
    print("Downloading dataset...")
    global stop_loading 
    stop_loading = False
    loading_thread = threading.Thread(target=loading_animation)
    loading_thread.start()

    ds = load_dataset(dataset_name)

    stop_loading = True
    loading_thread.join()

    print(ds)
    # Save the dataset to local files
    ds.save_to_disk("science_qa")
    print("Dataset saved to local files")

# Load the dataset from local files
def dataset_from_disk(arrow_file_path="science_qa/test/data-00000-of-00001.arrow", rows=1):
    # Load the dataset from local files
    # Path to the Arrow file
    # Get file size in MB
    
    file_size = os.path.getsize(arrow_file_path) / (1024 * 1024)
    print(f"Dataset size: {file_size:.2f} MB")
    # Load the dataset from Arrow format
    dataset = Dataset.from_file(arrow_file_path)
    print("Dataset loaded successfully")
    if (rows == -1):
        return dataset
    else:
        return dataset.select(range(rows))


if __name__ == "__main__":
    
    data = dataset_from_disk()
    
    
    

