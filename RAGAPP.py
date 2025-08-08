import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="📄 PDF RAG Chatbot", layout="centered")
st.title("📄 Chat with your PDF using Gemini")
st.write("Upload a PDF and ask questions about its contents.")

# PDF Upload
pdf_file = st.file_uploader("📤 Upload PDF", type=["pdf"])

if pdf_file:
    with st.spinner("🔍 Reading and splitting PDF..."):
        file_path = f"temp_{pdf_file.name}"
        with open(file_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        st.success(f"✅ Loaded and split {len(texts)} document chunks.")

        # Embedding & FAISS
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(texts, embedding_model)
        retriever = db.as_retriever()
        st.success("✅ Document embedded and stored in FAISS.")

        # Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash",  # or other model from your list
            temperature=0.4
        )

        # RAG Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type="stuff"
        )

        # Session state for chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Input
        query = st.text_input("❓ Ask a question about the PDF")

        if st.button("💬 Ask"):
            if query:
                with st.spinner("Thinking..."):
                    result = qa_chain.invoke({"query": query})
                    st.session_state.chat_history.append((query, result["result"]))
            else:
                st.warning("Please enter a question.")

        # Display Q&A History
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("📚 Chat History")
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"**❓ You:** {q}")
                st.markdown(f"**💬 Gemini:** {a}")
                st.markdown("---")

            # Clear button
            if st.button("🔄 Clear Chat History"):
                st.session_state.chat_history = []
                st.experimental_rerun()

    # Clean up temp file
    os.remove(file_path)
