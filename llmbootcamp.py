import dotenv

from llama_index.core.extractors import SummaryExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode
import random
import re
import openai

dotenv.load_dotenv()

# nest_asyncio.apply()

def filename_fn(filename):
    # Extract number using regular expression
    number = re.findall(r'\d+', filename)

    # Convert the list of matched numbers to an integer (or use it directly as string)
    week_number = 0
    if number:
        week_number = int(number[0])

    video_url = ""
    if week_number == 1:
        video_url = 'https://youtu.be/CvhcvM444Mw?si=gFaoEYSjpKBCdy0W'
    elif week_number == 2:
        video_url = 'https://youtu.be/oaktvOujQAY?si=wQJkO0-Nzvnl18BY'
    elif week_number == 3:
        video_url = 'https://youtu.be/AiZxP1JQTiY?si=tV9w8sceBKLO0Exw'
    elif week_number == 4:
        video_url = 'https://youtu.be/y81_wD_U3fA?si=Z80JG4Ak8X_rg4DV'
    elif week_number == 5:
        video_url = 'https://youtu.be/v0kP1SJeW_A?si=NmvOGP5BffqreAo-'

    return {
        "file_description": f"This is the transcript of Tim's lecture during Week {week_number} of the LLM Bootcamp",
        "video_url": video_url,
    }

def summary_filename_fn(filename):
    return {
        "file_description": "This contains the progress and summary of the LLM Bootcamp."
    }

class LLMBootcamp:
    def __init__(self):
        self.currentWeek = 5
        self.client = openai.OpenAI()
        self.index = None
        self.nodes = None

    async def load_data(self):
        documents = []
        for i in range(1, self.currentWeek+1):
            documents += SimpleDirectoryReader(
            input_files=[f"./data/llm_bootcamp_week{i}_transcript.txt"],
            file_metadata=filename_fn,
            ).load_data()

        documents += SimpleDirectoryReader(
        input_files=[f"./data/summary (week {self.currentWeek}).md"],
        file_metadata=summary_filename_fn,
        ).load_data()

        summary_extractor = SummaryExtractor()
        node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=100)

        print(f"Running loaded documents through ingestion pipeline...")
        pipeline = IngestionPipeline(transformations=[node_parser, summary_extractor])
        self.nodes = await pipeline.arun(documents=documents)
        print(f"Done! {len(self.nodes)} nodes created.")

        self.index = VectorStoreIndex(nodes=self.nodes)

    def raq_query(self, query):
        retriever = self.index.as_retriever(retrieval_mode='similarity', k=3)
        relevant_docs = retriever.retrieve(query)

        context = ""
        context += f"Number of relevant snippets: {len(relevant_docs)}"
        context += "\n" + "="*50 + "\n"
        for i, doc in enumerate(relevant_docs):
            context += f"Snippet {i+1}:\n"
            context += f"Text sample: {doc.node.get_content(metadata_mode=MetadataMode.LLM)[:4000]}...\n"
            context += f"Score: {doc.score}\n"
            context += "\n" + "="*50 + "\n"

        with open(f"data/summary (week {self.currentWeek}).md", "r") as f:
            summary = f.read()
            
        full_prompt = f"""
        Question: {query}
        Context: {context}
        Summary: {summary}
        """

        print("Prompt: ", full_prompt)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
            {"role": "system", "content": """
            You are a teacher's assistant who answers questions about what was discussed in a specific lecture or the LLM Bootcamp in general using the provided relevant snippets from video transcripts. Pretend that these classes are happening in real-time, so assume that the latest transcript is the most recent class. When asked about the general progression of the bootcamp, refer to the summary document. If the information for a specific week is not available, assume that the lecture hasn't happened yet.

            When answering, include the week number and timestamp of the video where the information was discussed, unless it is a general question that was not discussed in a specific week and the answer might be found in the summary. When providing the timestamp, make it into a HTML hyperlink that links to the relevant video URL.
            """},
            {"role": "user", "content": full_prompt}
            ],
            temperature=0.2
        )

        print("Response: ", response.choices[0].message.content)
        
        return response.choices[0].message.content

    def random_quiz(self):
        random_node_index = random.randint(0, len(self.nodes)-1)
        print("Random node index: ", random_node_index)

        random_node_content = (self.nodes)[random_node_index].get_content(metadata_mode=MetadataMode.LLM)

        message_history = []
        message_history.extend([
            {"role": "system", "content": 
        """
        You are a teacher's assistant who is creating a quiz for the students based on the provided snippet. Write a multiple-choice question based on the snippet provided. Make sure the question is clear and concise and has only one correct answer. If it's a question specific to a topic from a lecture week, include that context as well in the question. The correct answer should be one of the entities mentioned in the snippet. Provide 3 incorrect answers that could be plausible but are not mentioned in the snippet. Respond in JSON. You don't say anything else in the response. The JSON should follow the format:

        {
        "question": "What is the question?",
        "answer": "Correct answer",
        "choices": ["Incorrect answer 1", "Incorrect answer 2", "Incorrect answer 3", "Correct answer"],
        "explanation": "Explanation of the answer"
        }
        """
            },
            {"role": "user", "content": random_node_content},
        ])

        response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=message_history,
        temperature=0.2
        )

        return response.choices[0].message.content

    def summary_report(self):
        with open(f"data/summary (week {self.currentWeek}).md", "r") as f:
            summary = f.read()

        message_history = []
        message_history.extend([
            {"role": "system", "content": 
        """
        You are a student aid who prepares summary notes for the most current lecture week notes. Include . Respond in JSON. You don't say anything else in the response. The JSON should follow the format:

        {
            "weekly_summaries": {
                "1": "Summary of week 1",
                "2": "Summary of week 2",
            },
            "overall_summary": "Summarize the overall progress of the LLM Bootcamp. Change the speech to use second person pronouns (you, your) and address the student directly, wording in ways that set expectations for the student (e.g., you should now have a better understanding, make sure to review) and ask some hypothetical questions to get the student thinking about the course material to deepen their understand.",
        }
        """
            },
            {"role": "user", "content": summary},
        ])

        response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=message_history,
        temperature=0.2
        )

        return response.choices[0].message.content
