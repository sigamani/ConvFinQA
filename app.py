from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from rag_config import build_graph, build_vectorstore, embedding_model
from semantic_chunker import SemanticChunker
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langsmith import traceable
from judge import judge_answer
import json
import uuid

# App
app = FastAPI(title="ConvFinQA Agentic Benchmark API")

# Load vectorstore once at startup
print("🔧 Building semantic vectorstore...")
with open("dev.json") as f:
    dev_data = json.load(f)
contexts = [ex["support"] for ex in dev_data if "support" in ex]
chunker = SemanticChunker()
documents = [Document(page_content=chunk) for ctx in contexts for chunk in chunker.split_by_tokens(ctx)]
vectorstore = Chroma.from_documents(documents, embedding=embedding_model)
retriever = vectorstore.as_retriever()
graph = build_graph(retriever)

class QueryRequest(BaseModel):
    question: str
    expected_answer: Optional[str] = None
    metadata: Optional[dict] = {}

class QueryResponse(BaseModel):
    id: str
    question: str
    predicted_answer: str
    judgement: Optional[dict] = None

@app.get("/healthcheck")
def health():
    return {"status": "ok"}

@app.post("/run_benchmark", response_model=QueryResponse)
@traceable(name="run_benchmark_query")
def run_benchmark(req: QueryRequest):
    state = graph.invoke({"question": req.question})
    answer = state["answer"]
    result = {
        "id": str(uuid.uuid4()),
        "question": req.question,
        "predicted_answer": answer
    }
    if req.expected_answer:
        judgement = judge_answer(req.question, req.metadata.get("support", ""), answer, req.expected_answer)
        result["judgement"] = judgement
    return result
