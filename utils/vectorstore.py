from langchain_community.vectorstores import FAISS

def create_vectorstore(docs, embeddings):
    db = FAISS.from_documents(docs, embeddings)
    return db