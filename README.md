# 📘 Mini RAG App  

A lightweight Retrieval-Augmented Generation (RAG) system using **Qdrant** for vector storage, **Gemini API** for generation, and a modular backend.  

---

## 🚀 Live URL(s)  
- [(https://minirag.streamlit.app/)]  

---

## 💻 Public Repository  
- GitHub Repo: [https://github.com/Chetan03x/mini_rag_application]  

---

## ⚙️ Setup Instructions  

### 1. Clone the repo  
```bash
git clone https://github.com/chetan03x/mini_rag_app.git
cd mini_rag_app
```

### 2. Create and activate virtual environment  
```bash
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)
```

### 3. Install dependencies  
```bash
pip install -r requirements.txt
```

### 4. Environment variables  
Create a `.env` file with:  
```
GEMINI_API_KEY=your_api_key_here
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here
```


### 5. Run frontend (Streamlit or FastAPI UI)  
```bash
streamlit run backend/app.py
```

---

## 🏗️ Architecture  

- **Frontend**: Streamlit UI for query & visualization  
- **Backend**: FastAPI serving embeddings & retrieval  
- **Vector DB**: Qdrant stores embeddings for semantic search  
- **LLM**: Gemini generates final grounded answers  

---

## 📊 Evaluation  

We evaluate with **recall** and **success rate**:  

- **Recall** = retrieved relevant docs ÷ total relevant docs  
- **Success Rate** = % of questions where model answer matches gold answer  

Run evaluation:  
```bash
python eval/evaluation.py
```

---

## 📑 Schema / Index Config  

**Track A (Schema):**  
- Documents stored with fields: `id`, `text`, `embedding`, `metadata`  

**Track B (Qdrant Index Config):**  
```json
{
  "collection_name": "docs",
  "vector_size": 768,
  "distance": "Cosine"
}
```

---

## 📝 Resume Link  
[Insert your resume link here]  

---

## 💡 Remarks  

- **Limits**: Gemini free-tier quota limits (50 requests/day).  
- **Trade-offs**: Mock responses used when quota exceeded to ensure evaluation runs.  
- **Next steps**:  
  - Add more evaluation metrics (precision, F1-score)  
  - Optimize chunking & retrieval  
  - Support more LLM backends (e.g., OpenAI, Llama-Index)  
