from pathlib import Path
from typing import TypedDict, List
from pydantic.v1 import BaseModel
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph

from semantic_chunker import SemanticChunker
from langgraph.graph import StateGraph, StateType

# Define the graph state schema
class GraphState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    answer: str

# Semantic chunker setup (Ollama-compatible)
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
semantic_chunker = SemanticChunker()

# Node: Retrieval
def make_retrieve_node(retriever):
    def retrieve_node(state: GraphState) -> GraphState:
        question = state["question"]
        docs = retriever.invoke(question)
        return {
            **state,
            "retrieved_docs": docs
        }
    return retrieve_node

# Node: Generation
def generate_node(state: GraphState) -> GraphState:
    question = state["question"]
    docs = state["retrieved_docs"]
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a financial assistant.

Answer the following question using the provided context. If the answer cannot be found, respond with "N/A".

Question: {question}

Context:
{context}

Answer:""" 

    llm = ChatOllama(model="mistral")
    response = llm.invoke(prompt)

    return {
        **state,
        "answer": response.content.strip()
    }

# Graph wiring
def build_graph(retriever):
    builder = StateGraph(StateType(GraphState))
    builder = StateGraph(schema=GraphState)
    builder.add_node("retrieve", make_retrieve_node(retriever))
    builder.add_node("generate", generate_node)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.set_finish_point("generate")
    return builder.compile()


def build_vectorstore(contexts: List[str]):
    vectorstore_dir = Path("vectorstore")
    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    documents = []
    for context in contexts:
        # Split only if paragraph is long (e.g. 300+ tokens)
        if len(context.split()) > 200:
            chunks = semantic_chunker.split_by_tokens(context)
        else:
            chunks = [context]  # Keep entire paragraph

        for chunk in chunks:
            if chunk.strip():
                documents.append(Document(page_content=chunk))

    print(f"📚 Built semantic vectorstore with {len(documents)} chunks")
    return Chroma.from_documents(documents, embedding=embedding_model)
