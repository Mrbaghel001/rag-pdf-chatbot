from utils.loader import load_documents
from utils.embeddings import get_embeddings
from utils.vectorstore import create_vectorstore
from utils.qa_chain import create_qa_chain

from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Loading PDF...")

docs = load_documents("data/sample.pdf")

print("Splitting text...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print("Creating embeddings...")

embeddings = get_embeddings()

print("Creating vector database...")

vector_db = create_vectorstore(chunks, embeddings)

print("Loading QA system...")

qa = create_qa_chain(vector_db)

print("System ready. Type 'exit' to quit.")

while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    result = qa.invoke({"query": query})

    print("\nAnswer:", result["result"])