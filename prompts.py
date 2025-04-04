SYSTEM_ROLE = """
    You are an AI science knowldge bot. You are tasked with answering science questions.
    You are helpful, truthful and accurate.
"""

# Function to send prompt to GPT-4o mini
def science_QA_prompt(question, choices, hint):
    return f"""
    You are a AI answering a multiple-choice science question. Below is the question:

    **Context:** {hint}
    **Question:** {question}
    
    **Choices:**

    {chr(10).join([f"{i}. {choice}" for i, choice in enumerate(choices)])}

    Please return ONLY the choice number (0, 1, 2, etc.) as your answer.
    Do not include any other text or explanation.
    """