# 📄 AI PDF Chatbot with Gemini & LangChain

A simple Streamlit app that lets you upload any PDF and ask natural language questions about its content — powered by **Gemini** and **LangChain**.

## ✨ Features

- Upload a PDF file
- Ask any question about the document
- Streamed, chat-style Q&A
- Persistent conversation history (within session)
- Simple, clean Web UI with Streamlit

## 🧠 Tech Stack

- [LangChain](https://www.langchain.com/)
- [Google Gemini API](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)
- [PyPDF](https://pypdf.readthedocs.io/)

---

## 🚀 Getting Started

### 1. Clone the Repo

git clone https://github.com/yourusername/pdf-chat-gemini.git
cd pdf-chat-gemini


### 2.Install Requirements
pip install -r requirements.txt



### 3.Set up API key
Set Up API Key
Create a .env file in the root directory:

.env
GOOGLE_API_KEY=your_gemini_api_key_here
💡 Get your Gemini API key from: https://ai.google.dev

### 4.Run

streamlit run app2.py

