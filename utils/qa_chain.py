from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()


def create_qa_chain(vector_db):

    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate.from_template("""
You are an AI assistant answering questions from a document.

Context:
{context}

Question:
{question}

Provide a clear and concise answer.
""")

    def qa_chain(query):

        docs = vector_db.similarity_search(query, k=2)

        context = "\n\n".join([doc.page_content for doc in docs])

        formatted_prompt = prompt.format(
            context=context,
            question=query
        )

        response = llm.invoke(formatted_prompt)

        return response.content

    return qa_chain