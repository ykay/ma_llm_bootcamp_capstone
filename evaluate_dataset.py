from langfuse import Langfuse
import openai
import json
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import config

load_dotenv()

# Load documents from a directory (you can change this path as needed)
documents = SimpleDirectoryReader("data").load_data()

# Create an index from the documents
index = VectorStoreIndex.from_documents(documents)

langfuse = Langfuse()
client = openai.OpenAI()

# we use a very simple eval here, you can use any eval library
# see https://langfuse.com/docs/scores/model-based-evals for details
def llm_as_a_judge_evaluation(input, output, expected_output):
    prompt = config.EVALUATION_PROMPT.format(input=input, output=output, expected_output=expected_output)
    print("Evaluation Prompt: ", prompt)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    evaluation = response.choices[0].message.content
    print(f"Evaluation: {evaluation}")
    try:
      result = json.loads(evaluation)  # Convert the JSON string to a Python dictionary
    except json.JSONDecodeError:
      print("Invalid JSON string")
      result["score"] = -1.0
      result["reason"] = "<Invalid Evaluation JSON string>"
    
    # Debug printout
    print(f"Output: {output}")
    print(f"Expected Output: {expected_output}")
    print(f"Evaluation Result: {result}")
    
    return result["score"], result["reason"]

def rag_query_with_openai(input, model):
  # Create a retriever to fetch relevant documents
  retriever = index.as_retriever(retrieval_mode='similarity', k=3)

  # Retrieve relevant documents
  relevant_docs = retriever.retrieve(input)

  context = ""
  context += f"Number of relevant snippets: {len(relevant_docs)}"
  context += "\n" + "="*50 + "\n"
  for i, doc in enumerate(relevant_docs):
      context += f"Snippet {i+1}:\n"
      context += f"Content: {doc.node.get_content()}...\n"
      context += f"Source: {doc.node.metadata['file_name']}\n"
      context += f"Score: {doc.score}\n"
      context += "\n" + "="*50 + "\n"
    
  rag_prompt = f"""
  Question: {input}
  Context: {context}
  """

  print("RAG Prompt: ", rag_prompt)
  
  response = client.chat.completions.create(
    model=model,
    messages=[
      {"role": "system", "content": config.SYSTEM_PROMPT},
      {"role": "user", "content": rag_prompt}
    ],
    temperature=0.2
  )
  
  return response.choices[0].message.content

from datetime import datetime
def rag_query(input):
  model = "gpt-3.5-turbo"
  generationStartTime = datetime.now()

  response = rag_query_with_openai(input, model)
  print("RAG Response: ", response)
 
  langfuse_generation = langfuse.generation(
    name=config.GENERATION_NAME,
    input=input,
    output=response,
    model=model,
    start_time=generationStartTime,
    end_time=datetime.now()
  )
 
  return response, langfuse_generation

def run_experiment(experiment_name):
  dataset = langfuse.get_dataset(config.EVAL_DATASET_NAME)
 
  for item in dataset.items:
    if item.status == "ARCHIVED":
      # No need to evaluate archived items
      continue

    completion, langfuse_generation = rag_query(item.input)
 
    item.link(langfuse_generation, experiment_name) # pass the observation/generation object or the id
 
    score, reason = llm_as_a_judge_evaluation(item.input, completion, item.expected_output)
    langfuse_generation.score(
      name="accuracy",
      value=score,
      comment=reason,
      metadata={"evaluation_prompt_template": config.EVALUATION_PROMPT.format(input="<input>", output="<output>", expected_output="<expected_output>")}
    )

def main():
  runs = langfuse.get_dataset_runs(config.EVAL_DATASET_NAME).data

  experiment_number = 1
  if len(runs) > 0:
    for run in runs:
      # Parse number from "RAG Experiment #1" and retain the highest experiment number
      experiment_number = max(experiment_number, int(run.name.split("#")[1]))
    
    # Set new experiment number
    experiment_number += 1

  print(f"Starting RAG Experiment #{experiment_number}")
  print(f"="*100)
  run_experiment(f"RAG Experiment #{experiment_number}")

if __name__ == "__main__":
  main()