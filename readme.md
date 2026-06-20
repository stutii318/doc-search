# 🔍 DocSearch AI — Intelligent Document Parser & Semantic Search Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-brightgreen?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3-orange?style=flat-square)
![RAG](https://img.shields.io/badge/Architecture-RAG-purple?style=flat-square)

> Upload any document. Ask anything. Get precise answers with source citations — powered by RAG, FAISS, and Groq LLaMA 3.3-70b.

🚀 **Live Demo:** [docsearch-ai.streamlit.app](https://stutii318-doc-search-app-gf9bpd.streamlit.app)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 Multi-format ingestion | Upload PDF and TXT files |
| 🔍 Semantic Search | FAISS vector search with cosine similarity |
| 🤖 RAG-powered QA | Retrieval-Augmented Generation using Groq LLaMA 3.3-70b |
| 📋 Auto Summarization | Instant document summary on upload |
| 💡 Smart Suggestions | Auto-generates 5 clickable questions per document |
| 🔀 Document Comparison | Compare two documents side by side |
| 🌟 Keyword Highlighter | Highlights exact sentence that answered the query |
| 📊 Confidence Meter | Visual relevance score per answer |
| 🌍 Multi-language | Answer in English, Hindi, Spanish, French, German, Arabic |
| 💬 Chat History | Full conversation memory within session |
| 📥 PDF Export | Download complete Q&A session as PDF report |

---

## 🛠 Tech Stack

- **Frontend:** Streamlit
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **LLM:** Groq LLaMA 3.3-70b-versatile
- **PDF Parsing:** PyMuPDF
- **Text Splitting:** LangChain RecursiveCharacterTextSplitter
- **PDF Export:** fpdf2

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/stutii318/doc-search.git
cd doc-search

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
echo GROQ_API_KEY=your_key_here > .env

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure
doc-search/

├── app.py              # Main Streamlit UI

├── parser.py           # PDF/TXT document parser

├── chunker.py          # Text chunking pipeline

├── embedder.py         # FAISS index builder & retriever

├── qa_engine.py        # RAG engine (QA, summary, compare, suggest)

├── config.py           # Configuration & constants

├── requirements.txt    # Dependencies

└── .streamlit/

└── secrets.toml    # API keys (not committed)

---

## 📊 Performance

- Indexed **107 chunks** from a 2.8MB PDF in under 10 seconds
- Average query response time: **~2 seconds**
- Embedding model: `all-MiniLM-L6-v2` (384 dimensions)
- Retrieval: Top-5 chunks per query using cosine similarity

---

## 🎯 Use Cases

- Research paper analysis
- Legal document search
- Academic report QA
- Business report summarization
- Multi-document comparison

---

## 👩‍💻 Author

**Stuti Trivedi** — Data Science Undergraduate, UPES Dehradun  
[LinkedIn](https://linkedin.com/in/stutitrivedi) · [GitHub](https://github.com/stutii318)

---

## 📄 License


