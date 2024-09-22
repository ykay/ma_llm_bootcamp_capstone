import chainlit as cl
from langfuse.openai import AsyncOpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
import config
import utils
import json

# from langsmith.wrappers import wrap_openai
# from langsmith import traceable

load_dotenv()

# Retrieve context
with open('data/llm_bootcamp_week1_transcript.txt', 'r') as file:
    full_text = file.read()

# print(full_text)

# Initialize OpenAI client
endpoint_url = "https://api.openai.com/v1"
client = AsyncOpenAI()
# https://platform.openai.com/docs/models/gpt-4o
model_name = "gpt-4o-mini"
model_kwargs = {
    "model": model_name, 
    "temperature": 0.2, 
    "max_tokens": 500,
}

# Load documents from a directory (you can change this path as needed)
documents = SimpleDirectoryReader("data").load_data()

function_signatures = {
    "week1_lecture()",
    "week2_lecture()",
    "week3_lecture()",
    "need_more_info('required_information')"
}

function_call_history = [{"role": "system", "content": config.FUNCTION_SYSTEM_PROMPT}]

async def process_function_call_response(input, completion):
    function_call_history.append({"role": "assistant", "content": completion.choices[0].message.content})
    func_json = json.loads(completion.choices[0].message.content)
    function_signatures = func_json['functions']
    functions_to_call = utils.parse_function_signatures(function_signatures)
    print("Functions to Call: ", functions_to_call)
    if functions_to_call.count == 0:
        return None

    context = ""
    for func_name, params in functions_to_call:
        extracted_week = utils.extract_week(func_name)
        if extracted_week:
            for document in documents:
                file_name = document.metadata['file_name']
                if extracted_week in file_name:
                    print("Found relevant doc: ", file_name)
                    index = VectorStoreIndex.from_documents([document])

                    # Create a retriever to fetch relevant documents
                    retriever = index.as_retriever(retrieval_mode='similarity', k=3)
                    # Retrieve relevant documents
                    relevant_docs = retriever.retrieve(input)

                    context += f"Number of relevant snippets from {extracted_week}: {len(relevant_docs)}"
                    context += "\n" + "="*50 + "\n"
                    for i, doc in enumerate(relevant_docs):
                        context += f"Snippet {i+1}:\n"
                        context += f"Content: {doc.node.get_content()}...\n"
                        context += f"Source: {doc.node.metadata['file_name']}\n"
                        context += f"Score: {doc.score}\n"
                        context += "\n" + "="*50 + "\n"
        elif func_name == "need_more_info()":
            context += "Additional information is required from the user."
    
    return context

async def function_call(input, message_history):
    # Include the whole conversation history for context
    function_call_history.append({"role": "system", "content": f"Conversation Between User and Assistant: {message_history}"})
    completion = await client.chat.completions.create(messages=function_call_history, **model_kwargs)
    
    try:
        return await process_function_call_response(input, completion)
                                
    except Exception as e:
        print("Unexpected Error: ", e)
    
    return None

@cl.on_message
async def on_message(message: cl.Message):
    # Maintain an array of messages in the user session
    message_history = cl.user_session.get("message_history", [])

    # Provide the system prompt
    message_history.append({"role": "system", "content": config.SYSTEM_PROMPT})

    # Record the user's message in the history
    message_history.append({"role": "user", "content": message.content})

    # Create a message object and send a blank message to the user to show that the AI is typing
    response_message = cl.Message(content="")
    await response_message.send()

    if context := await function_call(message.content, message_history):
        message_history.append({"role": "system", "content": context})
    else:
        print("No function call made.")

    # Pass in the full message history for each request
    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **model_kwargs
    )
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()

    # Record the AI's response in the history
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)