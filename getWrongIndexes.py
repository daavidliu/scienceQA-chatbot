import json
import sys
from dataset_functions import dataset_from_disk, download_dataset

with open("results/results_FULL_4o_mini.json", "r") as file:
	results = json.load(file)
     
wrong_indexes = []

for result in results:
    if result["correct"] == False:
        wrong_indexes.append(result["index"])

# Save the wrong indexes to a file
with open("science_qa/test/4o_wrong_indexes.json", "w") as file:
    json.dump(wrong_indexes, file)

