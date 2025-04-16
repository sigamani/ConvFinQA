from typing import List
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
llm = ChatOllama(model="mistral")

def judge_answer(question: str, retrieved_docs: List[Document], model_answer: str, expected_answer: str = "") -> str:
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
        You are evaluating a financial QA system's prediction.

        Please assign a score based on the following criteria:

        - Score 1.0: The predicted answer is numerically correct (±5% of the expected value), and the reasoning is sound.
        - Score 0.5: The reasoning shows correct logic or formula use, but the final numerical answer is incorrect.
        - Score 0.0: The answer and reasoning are both incorrect, missing, or irrelevant.

        Only output a JSON object with two keys:
        {{
          "score": <float: 0.0, 0.5, or 1.0>,
          "reason": "<brief explanation>"
        }}

        {context}:
        - Question: {question}
        - Expected Answer: {expected_answer}
        - Predicted Answer: {model_answer}
              """

    llm = ChatOllama(model="mistral")

    response = llm.invoke(prompt)
    return response.content.strip()

