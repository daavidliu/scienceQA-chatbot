import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the results.json file into a variable named data
with open('results/results_4o_wrong.json', 'r') as f:
    data = json.load(f)

# Prepare data for analysis
entries = []

total_entries = 0

# Parse the entries for analysis
for item in data:
    total_entries += 1
    entries.append({
        "correctness": 1 if item["correct"] else 0,  # Convert boolean to 1 (correct) or 0 (incorrect)
        "confidence": item["GPT_response"]["confidence"],
        "grade": item["grade"],
        "has_image": item["has_image"],
        "subject": item["subject"],  # Include subject
        "topic": item["topic"],  # Include topic
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






# Plot 1: Bar Graph - Ratio of Correct to Incorrect by Confidence Percentage
plt.figure(figsize=(12, 6))

# Group data by "confidence" and "correctness", calculate counts
confidence_correctness = df.groupby(["confidence", "correctness"]).size().unstack(fill_value=0)

# Calculate the ratio (correct / incorrect) for each confidence level
confidence_correctness["ratio"] = confidence_correctness[1] / confidence_correctness[0]
confidence_correctness["ratio"] = confidence_correctness["ratio"].replace([float('inf'), float('nan')], 0)  # Handle divide-by-zero cases

# Plot the ratios
x_indexes = confidence_correctness.index  # Confidence values
plt.bar(x_indexes, confidence_correctness["ratio"], color="blue", alpha=0.7)

# Add titles and labels
plt.title("Ratio of Correct to Incorrect by GPT Confidence", fontsize=16)
plt.xlabel("GPT Response Confidence (%)", fontsize=12)
plt.ylabel("Correct-to-Incorrect Ratio", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Plot 1: Bar Graph - Ratio of Incorrect to Correct by Confidence Percentage
plt.figure(figsize=(12, 6))

# Group data by "confidence" and "correctness", calculate counts
confidence_correctness = df.groupby(["confidence", "correctness"]).size().unstack(fill_value=0)

# Calculate the ratio (incorrect / correct) for each confidence level
confidence_correctness["ratio"] = confidence_correctness[0] / confidence_correctness[1]
confidence_correctness["ratio"] = confidence_correctness["ratio"].replace([float('inf'), float('nan')], 0)  # Handle divide-by-zero cases

# Plot the ratios
x_indexes = confidence_correctness.index  # Confidence values
plt.bar(x_indexes, confidence_correctness["ratio"], color="orange", alpha=0.7)

# Add titles and labels
plt.title("Ratio of Incorrect to Correct by GPT Confidence", fontsize=16)
plt.xlabel("GPT Response Confidence (%)", fontsize=12)
plt.ylabel("Incorrect-to-Correct Ratio", fontsize=12)
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


# plot 5: correct answers vs incorrect answers
plt.figure(figsize=(8, 5))
correct_count = df["correctness"].sum()
incorrect_count = len(df) - correct_count
plt.bar(["Correct", "Incorrect"], [correct_count, incorrect_count], color=["green", "red"])
plt.title("Correct vs Incorrect Answers", fontsize=16)
plt.xlabel("Answer Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(axis="y")
plt.show()



# Define thresholds for "confident" and "unconfident"
CONFIDENCE_THRESHOLD = 100.0

# Add a "confident" (True/False) column
df["is_confident"] = df["confidence"] >= CONFIDENCE_THRESHOLD
df["is_unconfident"] = df["confidence"] < CONFIDENCE_THRESHOLD

# Helper function to calculate probabilities
def calculate_conditional_probability(condition1, condition2):
    """Calculate P(condition1 | condition2)"""
    numerator = df[condition1 & condition2].shape[0]
    denominator = df[condition2].shape[0]
    return numerator / denominator if denominator > 0 else 0

# Define conditions
correct = df["correctness"] == 1
incorrect = df["correctness"] == 0
confident = df["is_confident"]
unconfident = df["is_unconfident"]

# Calculate probabilities
p_confident_given_correct = calculate_conditional_probability(confident, correct)
p_confident_given_incorrect = calculate_conditional_probability(confident, incorrect)
p_correct_given_confident = calculate_conditional_probability(correct, confident)
p_correct_given_unconfident = calculate_conditional_probability(correct, unconfident)
p_unconfident_given_correct = calculate_conditional_probability(unconfident, correct)
p_correct_given_unconfident = calculate_conditional_probability(correct, unconfident)
p_incorrect_given_confident = calculate_conditional_probability(incorrect, confident)
p_incorrect_given_unconfident = calculate_conditional_probability(incorrect, unconfident)
p_unconfident_given_incorrect = calculate_conditional_probability(unconfident, incorrect)

# Display results
print("\n")
print(f"P(confident | correct): {p_confident_given_correct:.4f}")
print(f"P(confident | incorrect): {p_confident_given_incorrect:.4f}")
print("\n")
print(f"P(unconfident | correct): {p_unconfident_given_correct:.4f}")
print(f"P(unconfident | incorrect): {p_unconfident_given_incorrect:.4f}")
print("\n")
print(f"P(correct | confident): {p_correct_given_confident:.4f}")
print(f"P(correct | unconfident): {p_correct_given_unconfident:.4f}")
print("\n")
print(f"P(incorrect | confident): {p_incorrect_given_confident:.4f}")
print(f"P(incorrect | unconfident): {p_incorrect_given_unconfident:.4f}")
print("\n")
