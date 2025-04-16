from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph

# Optional: Replace or remove if you’re not using Pydantic at all
#from pydantic import BaseModel  
from pydantic.v1 import BaseModel

# Define the graph state schema
class GraphState(TypedDict):
    question: str
    retrieved_docs: List[Document]  # Or List[str] if you're not using LangChain Document objects
    answer: str

# Dummy example nodes (replace with your real functions)
def make_retrieve_node(retriever):
    def retrieve_node(state: GraphState) -> GraphState:
        question = state["question"]
        docs = retriever.invoke(question)  # or retriever.get_relevant_documents(question)
        return {
            **state,
            "retrieved_docs": docs
        }
    return retrieve_node


def generate_node(state: GraphState) -> GraphState:
    question = state["question"]
    docs = state["retrieved_docs"]

    # Combine retrieved docs
    context = "\n".join([doc.page_content for doc in docs])

    # Generate an answer using your LLM (e.g., via LangChain)
    prompt = f"Question: {question}\nContext:\n{context}\nAnswer:"
    
    from langchain_community.chat_models import ChatOllama
    llm = ChatOllama(model="mistral")
    response = llm.invoke(prompt)

    return {
        **state,
        "answer": response.content.strip()
    }
    return state


def build_graph(retriever):
    builder = StateGraph(schema=GraphState)
    builder.add_node("retrieve", make_retrieve_node(retriever))
    builder.add_node("generate", generate_node)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.set_finish_point("generate")
    return builder.compile()

"""
def build_graph():
    builder = StateGraph(schema=GraphState)

    # Register nodes
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    # Connect nodes
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.set_finish_point("generate")

    return builder.compile()
"""
# Other utilities (e.g., chunk_text, build_vectorstore) should be defined below
# Example placeholders:
def chunk_text(text: str) -> List[str]:
    return text.split("\n\n")  # Dummy chunker

def build_vectorstore(docs: List[Document]):
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma

    embedding = OllamaEmbeddings(model="nomic-embed-text")
    documents = [Document(page_content=doc) for doc in docs]
    return Chroma.from_documents(documents, embedding=embedding)

