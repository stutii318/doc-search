import streamlit as st
import os
from parser import load_document
from chunker import chunk_pages
from embedder import build_index, load_index
from qa_engine import ask
from config import DATA_DIR

st.set_page_config(
    page_title="DocSearch AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        color: white;
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .hero p {
        font-size: 1.1rem;
        opacity: 0.85;
        margin: 0.5rem 0 0;
    }

    /* Answer card */
    .answer-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        font-size: 1rem;
        line-height: 1.7;
    }

    /* Stat cards */
    .stat-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 12px;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .stat-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        margin-top: 4px;
    }

    /* Source badge */
    .source-badge {
        display: inline-block;
        background: rgba(103,126,234,0.3);
        border: 1px solid rgba(103,126,234,0.5);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: #c4b5fd;
        margin: 4px 4px 0 0;
    }

    /* Input styling */
    .stTextInput input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 10px !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
    }

    /* Button */
    .stButton button {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        width: 100% !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.05);
        border: 1px dashed rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 1rem;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
        color: white !important;
    }

    /* Hide default header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Section headers */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.5rem 0 0.75rem;
    }

    /* Indexed file chip */
    .file-chip {
        background: rgba(52,211,153,0.15);
        border: 1px solid rgba(52,211,153,0.3);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.82rem;
        color: #6ee7b7;
        margin: 3px 0;
        display: inline-block;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.5rem;'>🔍</div>
        <div style='font-size:1.2rem; font-weight:700; color:white;'>DocSearch AI</div>
        <div style='font-size:0.8rem; color:rgba(255,255,255,0.5); margin-top:4px;'>RAG · FAISS · Gemini</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("⚡ Index Documents"):
            all_chunks = []
            progress = st.progress(0)
            status = st.empty()
            for i, f in enumerate(uploaded_files):
                status.markdown(f"<span style='color:#a78bfa'>Processing {f.name}...</span>", unsafe_allow_html=True)
                filepath = os.path.join(DATA_DIR, f.name)
                with open(filepath, "wb") as out:
                    out.write(f.read())
                pages = load_document(filepath)
                chunks = chunk_pages(pages)
                all_chunks.extend(chunks)
                if f.name not in st.session_state.indexed_files:
                    st.session_state.indexed_files.append(f.name)
                progress.progress((i + 1) / len(uploaded_files))

            status.markdown("<span style='color:#6ee7b7'>Building FAISS index...</span>", unsafe_allow_html=True)
            index, chunks = build_index(all_chunks)
            st.session_state.index = index
            st.session_state.chunks = chunks
            status.empty()
            progress.empty()
            st.success(f"✅ {len(all_chunks)} chunks indexed!")

    if st.session_state.indexed_files:
        st.markdown('<div class="section-title">Indexed Files</div>', unsafe_allow_html=True)
        for f in st.session_state.indexed_files:
            st.markdown(f'<div class="file-chip">📄 {f}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:rgba(255,255,255,0.35); text-align:center; padding-top:0.5rem;'>
        Built with FAISS · sentence-transformers · Gemini
    </div>
    """, unsafe_allow_html=True)

# Main area
st.markdown("""
<div class="hero">
    <h1>🔍 DocSearch AI</h1>
    <p>Upload any document. Ask anything. Get precise answers with source citations.</p>
</div>
""", unsafe_allow_html=True)

# Stats row
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(st.session_state.indexed_files)}</div>
        <div class="stat-label">Documents</div>
    </div>""", unsafe_allow_html=True)
with col_b:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(st.session_state.chunks)}</div>
        <div class="stat-label">Chunks Indexed</div>
    </div>""", unsafe_allow_html=True)
with col_c:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(st.session_state.history)}</div>
        <div class="stat-label">Questions Asked</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# Query
st.markdown('<div class="section-title">Ask a Question</div>', unsafe_allow_html=True)
query = st.text_input("", placeholder="e.g. What are the key findings of this report?", label_visibility="collapsed")

if query:
    if st.session_state.index is None:
        st.warning("⚠️ Please upload and index documents first using the sidebar.")
    else:
        with st.spinner("🔎 Searching documents and generating answer..."):
            result = ask(query, st.session_state.index, st.session_state.chunks)
            st.session_state.history.append({"q": query, "a": result["answer"]})

        st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-card">{result["answer"]}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Sources</div>', unsafe_allow_html=True)
        for s in result["sources"]:
            with st.expander(f"📄 {s['source']} — Page {s['page']}  |  relevance: {s['score']}"):
                idx = next(
                    (i for i, c in enumerate(st.session_state.chunks)
                     if c["source"] == s["source"] and c["page"] == s["page"]),
                    None
                )
                if idx is not None:
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.05); border-radius:8px;
                    padding:1rem; font-size:0.88rem; color:rgba(255,255,255,0.75);
                    line-height:1.6;'>
                    {st.session_state.chunks[idx]["text"][:500]}...
                    </div>""", unsafe_allow_html=True)

# History
if st.session_state.history:
    st.markdown("---")
    st.markdown('<div class="section-title">Question History</div>', unsafe_allow_html=True)
    for i, item in enumerate(reversed(st.session_state.history[-5:])):
        with st.expander(f"Q: {item['q']}"):
            st.markdown(f'<div class="answer-card">{item["a"]}</div>', unsafe_allow_html=True)