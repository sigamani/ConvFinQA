from rag_config import build_vectorstore, build_graph
from langchain_core.documents import Document

# 1. Define the test document and question
docs = [
    "Year: 2019 | Revenue: $5M | Net Income: $1M",
    "Year: 2020 | Revenue: $6M | Net Income: $1.5M"
]
question = "What is the net income in 2020?"

# 2. Convert to LangChain Documents
documents = [Document(page_content=doc) for doc in docs]

# 3. Build vectorstore and retriever
vs = build_vectorstore(docs)  # returns a Chroma object
retriever = vs.as_retriever()

# 4. Build LangGraph
graph = build_graph(retriever)

# 5. Run graph
state = graph.invoke({"question": question})
print("Answer:", state.get("answer"))

# 6. Print answer
print("Answer:", state.get("answer"))

