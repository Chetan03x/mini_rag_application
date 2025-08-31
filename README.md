# 📖 MiniRAG-Gemini

MiniRAG-Gemini is a lightweight Retrieval-Augmented Generation (RAG) demo built with:

- *Streamlit* (frontend UI)  
- *Qdrant* (vector database)  
- *Cohere* (embeddings)  
- *Gemini (Google Generative AI)* (LLM for answering)  

It allows you to upload or paste documents, index them into Qdrant, and then ask natural language questions. The system retrieves the most relevant chunks and feeds them into Gemini for contextual answers.

---

## 🚀 Features
- Upload .txt, .md, or .pdf files  
- Automatic text chunking with overlap for better retrieval  
- Embedding via *Cohere*  
- Vector storage & retrieval via *Qdrant*  
- Answer generation via *Gemini* (google.generativeai)  
- Streaming answers in the UI  
- Sources and snippets displayed for transparency  

---

## 📂 Project Structure
```bash
mini_rag_app/
│── backend/
│   ├── app.py                # Streamlit entrypoint
│   ├── embeddings.py         # Extract, chunk, and embed text
│   ├── vectorstore.py        # Qdrant operations
│   ├── retriever.py          # Retrieve top-k documents
│   ├── llm_answer.py         # Prompt + Gemini answer generator
│   ├── config.py             # Configuration constants
│── requirements.txt
│── README.md