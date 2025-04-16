from typing import List
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
llm = ChatOllama(model="mistral")

def judge_answer(question: str, retrieved_docs: List[Document], model_answer: str, expected_answer: str = "") -> str:
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = f"""
You are a financial QA evaluator. A model answered a question using retrieved context. Assess whether the answer is correct and explain.

Question: {question}

Retrieved Context:
{context}

Model's Answer:
{model_answer}

Expected Answer:
{expected_answer}

Respond with 'Correct' or 'Incorrect' and provide brief justification.
    """

    llm = ChatOllama(model="mistral")

    response = llm.invoke(prompt)
    return response.content.strip()

