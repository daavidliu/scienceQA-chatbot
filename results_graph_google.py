import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the results.json file into a variable named data
with open('results/results_gemini_2_flash_lite.json', 'r') as f:
    data = json.load(f)

# Prepare data for analysis
entries = []

total_entries = 0

# Parse the entries for analysis
for item in data:
    total_entries += 1
    entries.append({
        "correctness": 1 if item["correct"] else 0,  # Convert boolean to 1 (correct) or 0 (incorrect)
        "grade": item["grade"],
        "has_image": item["has_image"],
        "subject": item["subject"],  # Include subject
        "topic": item["topic"],  # Include topic
        "question": item["question"],  # Include question
    })

print(f"Total entries: {total_entries}")

# Create a DataFrame
df = pd.DataFrame(entries)

# Plot: Relationship between correctness and subject
plt.figure(figsize=(10, 6))
subject_correctness = df.groupby("subject")["correctness"].mean().reset_index()
plt.bar(subject_correctness["subject"], subject_correctness["correctness"], color="purple", alpha=0.8)
plt.title("Relationship Between Subject and Correctness", fontsize=16)
plt.xlabel("Subject", fontsize=12)
plt.ylabel("Average Correctness", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Plot: Relationship between correctness and topic
plt.figure(figsize=(12, 8))
topic_correctness = df.groupby("topic")["correctness"].mean().reset_index()
plt.bar(topic_correctness["topic"], topic_correctness["correctness"], color="teal", alpha=0.8)
plt.title("Relationship Between Topic and Correctness", fontsize=16)
plt.xlabel("Topic", fontsize=12)
plt.ylabel("Average Correctness", fontsize=12)
plt.xticks(rotation=45, ha='right')  # Rotate ticks for better readability
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Plot 2: Relationship between grade and correctness
plt.figure(figsize=(8, 5))
grade_correctness = df.groupby("grade")["correctness"].mean().reset_index()
plt.bar(grade_correctness["grade"], grade_correctness["correctness"], color="skyblue")
plt.title("Relationship: Grade vs Correctness", fontsize=16)
plt.xlabel("Grade", fontsize=12)
plt.ylabel("Average Correctness", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.show()

# Plot 3: Relationship between has_image and correctness
plt.figure(figsize=(8, 5))
image_correctness = df.groupby("has_image")["correctness"].mean().reset_index()
image_correctness["has_image"] = image_correctness["has_image"].map({True: "Has Image", False: "No Image"})
plt.bar(image_correctness["has_image"], image_correctness["correctness"], color=["orange", "green"])
plt.title("Relationship: Image Presence vs Correctness", fontsize=16)
plt.xlabel("Has Image", fontsize=12)
plt.ylabel("Average Correctness", fontsize=12)
plt.grid(axis="y")
plt.show()

# Plot 4: Relationship between question length and correctness
plt.figure(figsize=(8, 5))
df["question_length"] = df["question"].apply(lambda x: len(x.split()))
question_length_correctness = df.groupby("question_length")["correctness"].mean().reset_index()
plt.scatter(question_length_correctness["question_length"], question_length_correctness["correctness"], color="purple")
plt.title("Relationship: Question Length vs Correctness", fontsize=16)
plt.xlabel("Question Length (words)", fontsize=12)
plt.ylabel("Average Correctness", fontsize=12)
plt.grid()
plt.show()

# plot 5: correct answers vs incorrect answers percentage
plt.figure(figsize=(8, 5))
correct_count = df["correctness"].sum()
incorrect_count = len(df) - correct_count
correct_percentage = (correct_count / len(df)) * 100
incorrect_percentage = (incorrect_count / len(df)) * 100
plt.bar(["Correct", "Incorrect"], [correct_percentage, incorrect_percentage], color=["green", "red"])
plt.title("Correct vs Incorrect Answers Percentage", fontsize=16)
plt.xlabel("Answer Type", fontsize=12)
plt.ylabel("Percentage", fontsize=12)
plt.grid(axis="y")
plt.show()

# print the precentage of correct and incorrect answers
print(f"Percentage of Correct Answers: {correct_percentage:.2f}%")
print(f"Percentage of Incorrect Answers: {incorrect_percentage:.2f}%")

