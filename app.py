import streamlit as st
import os
from parser import load_document
from chunker import chunk_pages
from embedder import build_index
from qa_engine import ask, summarize, suggest_questions, compare_docs, highlight_answer
from config import DATA_DIR
from fpdf import FPDF
import datetime

st.set_page_config(
    page_title="DocSearch AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: white; }
    [data-testid="stSidebar"] { background: rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.1); }
    .hero { background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 2rem; color: white; }
    .hero h1 { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .hero p { font-size: 1.1rem; opacity: 0.85; margin: 0.5rem 0 0; }
    .answer-card { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.15); border-radius: 12px; padding: 1.5rem; margin: 1rem 0; color: white; font-size: 1rem; line-height: 1.7; }
    .stat-card { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12); border-radius: 12px; padding: 1.2rem; text-align: center; margin-bottom: 12px; }
    .stat-number { font-size: 2rem; font-weight: 700; color: #a78bfa; }
    .stat-label { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 4px; }
    .chat-user { background: rgba(103,126,234,0.2); border: 1px solid rgba(103,126,234,0.3); border-radius: 12px 12px 4px 12px; padding: 1rem 1.25rem; margin: 0.5rem 0; color: white; font-size: 0.95rem; }
    .chat-ai { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12); border-radius: 12px 12px 12px 4px; padding: 1rem 1.25rem; margin: 0.5rem 0; color: white; font-size: 0.95rem; line-height: 1.7; }
    .summary-card { background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.25); border-radius: 12px; padding: 1.5rem; margin: 1rem 0; color: white; font-size: 0.95rem; line-height: 1.8; }
    .compare-card { background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.25); border-radius: 12px; padding: 1.5rem; margin: 1rem 0; color: white; font-size: 0.95rem; line-height: 1.8; }
    .section-title { font-size: 1rem; font-weight: 600; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 0.1em; margin: 1.5rem 0 0.75rem; }
    .file-chip { background: rgba(52,211,153,0.15); border: 1px solid rgba(52,211,153,0.3); border-radius: 20px; padding: 4px 12px; font-size: 0.82rem; color: #6ee7b7; margin: 3px 0; display: inline-block; }
    .suggestion-btn { background: rgba(103,126,234,0.15); border: 1px solid rgba(103,126,234,0.3); border-radius: 8px; padding: 8px 14px; font-size: 0.85rem; color: #c4b5fd; margin: 4px 0; cursor: pointer; width: 100%; text-align: left; }
    .stTextInput input { background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.2) !important; border-radius: 10px !important; color: white !important; font-size: 1rem !important; padding: 0.75rem 1rem !important; }
    .stButton button { background: linear-gradient(90deg, #667eea, #764ba2) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; padding: 0.6rem 1.5rem !important; width: 100% !important; }
    .confidence-label { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-bottom: 4px; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    hr { border-color: rgba(255,255,255,0.1) !important; }
</style>
""", unsafe_allow_html=True)

# Session state
for key, default in [
    ("index", None), ("chunks", []), ("indexed_files", []),
    ("history", []), ("summary", None), ("suggestions", []),
    ("compare_result", None), ("doc_chunks_map", {}),
    ("pending_query", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# PDF export
def export_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "DocSearch AI - Session Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)
    if st.session_state.summary:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Document Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, st.session_state.summary.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(4)
    if st.session_state.history:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Q&A History", ln=True)
        for i, item in enumerate(st.session_state.history):
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 6, f"Q{i+1}: {item['q']}".encode('latin-1', 'replace').decode('latin-1'))
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, f"A: {item['a']}".encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(3)
    return pdf.output()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.5rem;'>🔍</div>
        <div style='font-size:1.2rem; font-weight:700; color:white;'>DocSearch AI</div>
        <div style='font-size:0.8rem; color:rgba(255,255,255,0.5); margin-top:4px;'>RAG · FAISS · Groq LLaMA 3.3</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Language</div>', unsafe_allow_html=True)
    language = st.selectbox("Response language", ["English", "Hindi", "Spanish", "French", "German", "Arabic"], label_visibility="collapsed")

    st.markdown('<div class="section-title">Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files:
        if st.button("⚡ Index Documents"):
            all_chunks = []
            progress = st.progress(0)
            status = st.empty()
            doc_chunks_map = {}
            for i, f in enumerate(uploaded_files):
                status.markdown(f"<span style='color:#a78bfa'>Processing {f.name}...</span>", unsafe_allow_html=True)
                filepath = os.path.join(DATA_DIR, f.name)
                with open(filepath, "wb") as out:
                    out.write(f.read())
                pages = load_document(filepath)
                chunks = chunk_pages(pages)
                doc_chunks_map[f.name] = chunks
                all_chunks.extend(chunks)
                if f.name not in st.session_state.indexed_files:
                    st.session_state.indexed_files.append(f.name)
                progress.progress((i + 1) / len(uploaded_files))

            status.markdown("<span style='color:#6ee7b7'>Building FAISS index...</span>", unsafe_allow_html=True)
            index, chunks = build_index(all_chunks)
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.doc_chunks_map = doc_chunks_map

            status.markdown("<span style='color:#6ee7b7'>Generating summary & suggestions...</span>", unsafe_allow_html=True)
            st.session_state.summary = summarize(all_chunks, language)
            st.session_state.suggestions = suggest_questions(all_chunks, language)
            status.empty()
            progress.empty()
            st.success(f"✅ {len(all_chunks)} chunks indexed!")

    if st.session_state.indexed_files:
        st.markdown('<div class="section-title">Indexed Files</div>', unsafe_allow_html=True)
        for f in st.session_state.indexed_files:
            st.markdown(f'<div class="file-chip">📄 {f}</div>', unsafe_allow_html=True)

    if len(st.session_state.doc_chunks_map) >= 2:
        st.markdown("---")
        st.markdown('<div class="section-title">Compare Documents</div>', unsafe_allow_html=True)
        doc_names = list(st.session_state.doc_chunks_map.keys())
        doc1 = st.selectbox("Document 1", doc_names, key="doc1")
        doc2 = st.selectbox("Document 2", doc_names, index=min(1, len(doc_names)-1), key="doc2")
        if st.button("🔀 Compare"):
            if doc1 != doc2:
                with st.spinner("Comparing documents..."):
                    st.session_state.compare_result = compare_docs(
                        st.session_state.doc_chunks_map[doc1],
                        st.session_state.doc_chunks_map[doc2],
                        doc1, doc2, language
                    )
            else:
                st.warning("Please select two different documents.")

    if st.session_state.history:
        st.markdown("---")
        if st.button("📥 Export PDF Report"):
            pdf_bytes = export_pdf()
            st.download_button(
                label="Download Report",
                data=bytes(pdf_bytes),
                file_name=f"docsearch_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )
        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.session_state.pending_query = None
            st.rerun()

    st.markdown("---")
    st.markdown("<div style='font-size:0.78rem; color:rgba(255,255,255,0.35); text-align:center;'>Built with FAISS · sentence-transformers · Groq</div>", unsafe_allow_html=True)

# Main
st.markdown("""
<div class="hero">
    <h1>🔍 DocSearch AI</h1>
    <p>Upload any document. Ask anything. Get precise answers with source citations.</p>
</div>
""", unsafe_allow_html=True)

# Stats
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{len(st.session_state.indexed_files)}</div><div class="stat-label">Documents</div></div>', unsafe_allow_html=True)
with col_b:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{len(st.session_state.chunks)}</div><div class="stat-label">Chunks Indexed</div></div>', unsafe_allow_html=True)
with col_c:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{len(st.session_state.history)}</div><div class="stat-label">Questions Asked</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Summary
if st.session_state.summary:
    st.markdown('<div class="section-title">📋 Document Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-card">{st.session_state.summary}</div>', unsafe_allow_html=True)

# Compare result
if st.session_state.compare_result:
    st.markdown("---")
    st.markdown('<div class="section-title">🔀 Document Comparison</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="compare-card">{st.session_state.compare_result}</div>', unsafe_allow_html=True)

st.markdown("---")

# Suggested questions
if st.session_state.suggestions:
    st.markdown('<div class="section-title">💡 Suggested Questions</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, suggestion in enumerate(st.session_state.suggestions):
        with cols[i % 2]:
            if st.button(f"💬 {suggestion}", key=f"sug_{i}"):
                st.session_state.pending_query = suggestion

# Chat history
if st.session_state.history:
    st.markdown('<div class="section-title">💬 Conversation</div>', unsafe_allow_html=True)
    for item in st.session_state.history:
        st.markdown(f'<div class="chat-user">🙋 {item["q"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-ai">🤖 {item["a"]}</div>', unsafe_allow_html=True)
        if "confidence" in item:
            st.markdown(f'<div class="confidence-label">Confidence: {item["confidence"]}%</div>', unsafe_allow_html=True)
            st.progress(item["confidence"] / 100)
        with st.expander("📄 View sources with highlights"):
            for s in item.get("sources", []):
                st.markdown(f'<span style="color:#c4b5fd; font-size:0.85rem">📄 {s["source"]} — Page {s["page"]} | score: {s["score"]}</span>', unsafe_allow_html=True)
                highlighted = highlight_answer(item["a"], s.get("text", ""))
                st.markdown(f'<div style="background:rgba(255,255,255,0.05); border-radius:8px; padding:1rem; font-size:0.85rem; color:rgba(255,255,255,0.8); line-height:1.6;">{highlighted}</div>', unsafe_allow_html=True)

st.markdown("---")

# Query input
st.markdown('<div class="section-title">💬 Ask a Question</div>', unsafe_allow_html=True)
query = st.text_input("", placeholder="e.g. What are the key findings of this report?", label_visibility="collapsed", value=st.session_state.pending_query or "")

# Process query
active_query = st.session_state.pending_query or query
if active_query and (not st.session_state.history or st.session_state.history[-1]["q"] != active_query):
    if st.session_state.index is None:
        st.warning("⚠️ Please upload and index documents first.")
    else:
        with st.spinner("🔎 Searching and generating answer..."):
            result = ask(active_query, st.session_state.index, st.session_state.chunks, language, st.session_state.history)
            st.session_state.history.append({
                "q": active_query,
                "a": result["answer"],
                "confidence": result["confidence"],
                "sources": result["sources"]
            })
        st.session_state.pending_query = None
        st.rerun()