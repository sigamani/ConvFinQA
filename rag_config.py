
from typing import TypedDict, List
from pydantic.v1 import BaseModel

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph

from langchain_text_splitters import RecursiveCharacterTextSplitter
from semantic_chunker import SemanticChunker

# Define the graph state schema
class GraphState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    answer: str

# Embeddings and chunkers
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
semantic_chunker = SemanticChunker()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512,
    chunk_overlap=50,
)

# RAG nodes
def make_retrieve_node(retriever):
    def retrieve_node(state: GraphState) -> GraphState:
        question = state["question"]
        docs = retriever.invoke(question)
        return {**state, "retrieved_docs": docs}
    return retrieve_node

def generate_node(state: GraphState) -> GraphState:
    question = state["question"]
    docs = state["retrieved_docs"]
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a particle physicist.

    Answer the question using only the context provided. Output your response as a JSON object with this format:

    {{
      "answer": "<your numeric or symbolic answer here>",
      "reasoning": "<your short chain-of-thought reasoning>"
    }}

    - Only include the JSON object, no extra text.
    - The "answer" field must be a number or a string like "N/A".
    - The "reasoning" field must be a one-line explanation.

    Context:
    {context}

    Question:
    {state['question']}

    Answer JSON:
    """
    llm = ChatOllama(model="mistral", temperature=0.0)
    response = llm.invoke(prompt)

    return {
        **state,
        "answer": response.content.strip()
    }

# LangGraph wiring
def build_graph(retriever):
    builder = StateGraph(GraphState)  # <- Legacy compatible constructor
    builder.add_node("retrieve", make_retrieve_node(retriever))
    builder.add_node("generate", generate_node)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.set_finish_point("generate")
    return builder.compile()

# Vectorstore builder with semantic chunking
def build_vectorstore(contexts: List[str]):
    documents = []
    for context in contexts:
        chunks = semantic_chunker.split_by_tokens(context) if len(context.split()) > 200 else [context]
        for chunk in chunks:
            if chunk.strip():
                documents.append(Document(page_content=chunk))

    print(f"📚 Built semantic vectorstore with {len(documents)} chunks")
    return Chroma.from_documents(documents, embedding=embedding_model)
