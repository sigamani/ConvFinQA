
from rag_config import build_graph, build_vectorstore
from judge import judge_answer
from typing import List
import json

def run_terminal_benchmark(examples: List[dict], n: int = 5):
    retriever = build_vectorstore([
        "Year: 2020 | Revenue: $6M | Net Income: $1.5M",
        "Year: 2019 | Revenue: $5M | Net Income: $1M"
    ]).as_retriever(search_kwargs={"k": 2})

    graph = build_graph(retriever)
    correct = 0
    print("\n🔍 Running terminal benchmark preview on", n, "examples\n")

    for i, example in enumerate(examples[:n]):
        question = example["question"]
        expected = example["answer"]

        print(f"--- Example {i+1} ---")
        print("Q:", question)
        state = graph.invoke({"question": question})
        answer = state.get("answer", "").strip()
        retrieved = state.get("retrieved_docs", [])
        eval_result = judge_answer(question, retrieved, answer, expected)

        print("Predicted:", answer)
        print("Expected:", expected)
        print(eval_result)
        print()

        if "✔️ Correct" in eval_result:
            correct += 1

    print(f"✅ Accuracy: {correct}/{n} = {100 * correct / n:.1f}%\n")

if __name__ == "__main__":
    with open("dev_converted_full.json") as f:
        examples = json.load(f)
    run_terminal_benchmark(examples, n=5)
