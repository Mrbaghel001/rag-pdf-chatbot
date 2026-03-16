import streamlit as st
from utils.loader import load_documents
from utils.embeddings import get_embeddings
from utils.vectorstore import create_vectorstore
from utils.qa_chain import create_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("📄 Chat with your PDF")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache document processing
@st.cache_resource
def build_vector_db(pdf_path):

    docs = load_documents(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()

    vector_db = create_vectorstore(chunks, embeddings)

    return vector_db


uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    vector_db = build_vector_db("temp.pdf")

    qa = create_qa_chain(vector_db)

    # show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask a question about the document")

    if query:

        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        answer = qa(query)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})