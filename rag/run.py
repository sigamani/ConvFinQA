from rag_config import build_graph, build_vectorstore
from langchain_core.documents import Document
from judge import judge_answer  # if you're using judge locally

# 1. Setup example
docs = [
    "Year: 2019 | Revenue: $5M | Net Income: $1M",
    "Year: 2020 | Revenue: $6M | Net Income: $1.5M"
]
question = "What is the net income in 2020?"
expected_answer = "$1.5M"

# 2. Build vector store + retriever
vs = build_vectorstore(docs)
retriever = vs.as_retriever()

# 3. Build LangGraph with retriever
graph = build_graph(retriever)

# ✅ 4. Run the graph BEFORE accessing `state`
state = graph.invoke({"question": question})

# 5. Access outputs
retrieved = state.get("retrieved_docs", [])
answer = state.get("answer", "")

# 6. Judge evaluation
evaluation = judge_answer(question, retrieved, answer, expected_answer)
print("\n🧠 Model Answer:", answer)
print("\n🧪 Judge Evaluation:", evaluation)

