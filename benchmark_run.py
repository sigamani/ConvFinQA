from langchain_community.vectorstores import Chroma
from rag_config import build_graph, build_vectorstore
from judge import judge_answer
from typing import List
import json

def format_context(pre_text, table, post_text):
    table_str = "\n".join(["\t".join(row) for row in table])
    context = []
    if isinstance(pre_text, list):
        context.append("\n".join(pre_text))
    else:
        context.append(pre_text)
    context.append(table_str)
    if isinstance(post_text, list):
        context.append("\n".join(post_text))
    else:
        context.append(post_text)
    return "\n\n".join([part for part in context if part.strip()])

def extract_examples(raw_data, max_examples=5):
    examples = []
    for entry in raw_data:
        try:
            question = entry["qa"]["question"]
            answer = entry["qa"]["answer"] if "answer" in entry["qa"] else str(entry["qa"].get("exe_ans", "N/A"))
            context = format_context(entry.get("pre_text", ""), entry.get("table", []), entry.get("post_text", ""))
            examples.append({
                "question": question,
                "answer": answer,
                "context": context
            })
        except Exception as e:
            continue
        if len(examples) >= max_examples:
            break
    return examples

def run_benchmark(examples: List[dict]):
    retriever = build_vectorstore([ex["context"] for ex in examples]).as_retriever(search_kwargs={"k": 2})
    graph = build_graph(retriever)

    correct = 0
    print("\n🔍 Running terminal benchmark on", len(examples), "examples\n")

    for i, example in enumerate(examples):
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

    print(f"✅ Accuracy: {correct}/{len(examples)} = {100 * correct / len(examples):.1f}%\n")

if __name__ == "__main__":
    with open("data/dev.json") as f:
        raw_data = json.load(f)
    examples = extract_examples(raw_data, max_examples=5)
    run_benchmark(examples)
