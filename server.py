from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from llmbootcamp import LLMBootcamp
import json

lb = LLMBootcamp()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await lb.load_data()
    yield
  
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/query")
def read_query(q: Union[str, None] = None):
    if q is None:
        return {"error": "No query provided"}

    answer = lb.raq_query(q)

    return {
        "query": q,
        "answer": answer
        }

@app.get("/random-quiz")
def read_query():
    answer = lb.random_quiz()

    return json.loads(answer)

@app.get("/summary-report")
def read_query():
    report = lb.summary_report()

    return json.loads(report)
