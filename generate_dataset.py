# Import necessary libraries
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from openai import OpenAI
import json
import config

client = OpenAI()

# Function to generate questions and answers
def generate_qa(text, temperature=0.2):    
    generation_prompt = config.GENERATION_SYSTEM_PROMPT.format(system_prompt=config.SYSTEM_PROMPT)
    print("Generation Prompt: ", generation_prompt)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": generation_prompt},
            {"role": "user", "content": text}],
        temperature=temperature,
    )
    
    print("Completion: ", response.choices[0].message.content)

    # Strip extraneous symbols from the response content
    content = response.choices[0].message.content.strip()
    
    # Remove potential JSON code block markers
    content = content.strip()
    if content.startswith('```'):
        content = content.split('\n', 1)[-1]
    if content.endswith('```'):
        content = content.rsplit('\n', 1)[0]
    content = content.strip()
    
    # Attempt to parse the cleaned content as JSON
    try:
        parsed_content = json.loads(content.strip())
        return parsed_content
    except json.JSONDecodeError:
        print("Error: Unable to parse JSON. Raw content:")
        print(content)
        return []

# Generate dataset
import json

def main():
    # Retrieve context
    with open('data/llm_bootcamp_week1_transcript.txt', 'r') as file:
        text = file.read()

    # Generate dataset if local file doesn't exist
    dataset = generate_qa(text, temperature=0.2)
    if len(dataset) == 0:
        print("Failed to generate dataset.")
        return
    else:
        print("Generated dataset size: ", len(dataset))

    # Note: we're choosing to create the dataset in Langfuse below, but it's equally easy to create it in another platform.

    from langfuse import Langfuse
    langfuse = Langfuse()

    dataset_name = config.EVAL_DATASET_NAME
    langfuse.create_dataset(
        name=dataset_name, 
        );

    for item in dataset:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=item["question"],
            expected_output=item["expected_output"],
            metadata={
                "date": "2024-09-10",
                "tags": "lecture,week1"
                }
        )
  
if __name__ == "__main__":
    main()