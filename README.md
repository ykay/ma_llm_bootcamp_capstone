# Capstone Project 

## Overview

The goal of this project is to create a "Max Academy Bootcamp" LLM that provides accurate answers about the course for students who want to review topics that were taught during a lecture, upcoming assignments, deadlines, and anything else relevant to the course. The LLM will be trained with video transcripts from the recorded lectures and course documents which will allow it to provide answers along with relevant timestamps (when an answer comes from a recorded video) when this was discussed. 

Here are some various use cases where this project can be applied: 
- A "student aid" service that extracts key takeaways and things to do, by providing an LLM that evolves over time as it accumulates knowledge from past lectures and course material.

## Week 4 Updates
Last week, function calling was used to conditionally load specific transcripts into the vector store index. Metadata extraction in LlamaIndex was explored in `test_metadata_extractor.ipynb` since it seemed to address the issue of text chunks lacking the context to disambiguate one chunk from a similar chunk, both of which originate from two separate lectures. Additionally, a subquestion query engine was used to expose specific information to the LLM (e.g., week number + timestamp when a specific topic was discussed in a video), which seemed to push the LLM to include that information in their answers.

## Week 3 Updates
A RAG pipeline has been implemented in the main Chainlit app. The RAG context mainly depends on the new "function calling" functionality that uses a separate conversation history to decide what function should be called next to progress the main conversation between the user and assistant, which is provided as context in order for the AI to decide the most appropriate function call.

The function calls are able to load transcripts from various weekly lectures in order to load the most relevant documents to the vector store index (using LlamaIndex). By limiting the amount of data loaded into the indexes, the accuracy and relevancy of context retrieved will be improved. For example, if the user asked, "What was discussed in Week 1?", only loading the video transcript from Week 1's lecture increases the chance of more relevant data being retrieved compared to if transcripts from from Week 1, 2, and 3 were all loaded.


