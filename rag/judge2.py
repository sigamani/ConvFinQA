def judge_answer(question, retrieved_docs, predicted, expected):
    context = "\n\n".join([
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in retrieved_docs
    ])

    prompt = f"""
You are a financial QA judge.

Compare the predicted answer to the expected answer and determine if it is correct.

Respond in this format:
✔️ Correct: <short reason>
or
❌ Incorrect: <short reason>

---

Question: {question}
Predicted Answer: {predicted}
Expected Answer: {expected}
Context:
{context}

Judgement:"""

    from langchain.chat_models import ChatOpenAI  # or ChatOllama
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)
