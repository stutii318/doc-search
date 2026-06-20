from groq import Groq
from config import TOP_K
from embedder import retrieve
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

def get_client():
    try:
        key = st.secrets["GROQ_API_KEY"]
    except:
        key = os.getenv("GROQ_API_KEY", "")
    return Groq(api_key=key)

def ask(query: str, index, chunks, language: str = "English", history: list = []) -> dict:
    client = get_client()
    results = retrieve(query, index, chunks, top_k=TOP_K)
    context = "\n\n".join([r["text"] for r in results])

    avg_score = round(sum(r["score"] for r in results) / len(results), 3) if results else 0
    confidence = min(int(avg_score * 100), 100)

    history_text = ""
    if history:
        history_text = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in history[-3:]])
        history_text = f"\nPrevious conversation:\n{history_text}\n"

    prompt = f"""You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I couldn't find this in the documents."
Respond in {language} language.
{history_text}
Context:
{context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "confidence": confidence,
        "sources": [
            {"source": r["source"], "page": r["page"], "score": r["score"], "text": r["text"]}
            for r in results
        ]
    }

def summarize(chunks: list, language: str = "English") -> str:
    client = get_client()
    sample_text = "\n\n".join([c["text"] for c in chunks[:20]])
    prompt = f"""Summarize the following document content in 5-7 bullet points.
Be concise and capture the key ideas. Respond in {language} language.

Content:
{sample_text}

Summary:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    return response.choices[0].message.content.strip()

def suggest_questions(chunks: list, language: str = "English") -> list:
    client = get_client()
    sample_text = "\n\n".join([c["text"] for c in chunks[:15]])
    prompt = f"""Based on the document below, generate exactly 5 interesting and specific questions 
a user might want to ask. Return ONLY the 5 questions, one per line, no numbering, no extra text.
Respond in {language} language.

Document:
{sample_text}

Questions:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512
    )
    raw = response.choices[0].message.content.strip()
    questions = [q.strip() for q in raw.split("\n") if q.strip()]
    return questions[:5]

def compare_docs(chunks1: list, chunks2: list, name1: str, name2: str, language: str = "English") -> str:
    client = get_client()
    text1 = "\n\n".join([c["text"] for c in chunks1[:15]])
    text2 = "\n\n".join([c["text"] for c in chunks2[:15]])
    prompt = f"""Compare the two documents below. Identify:
1. Key similarities
2. Key differences
3. What each document covers that the other doesn't
Respond in {language} language. Be specific and structured.

Document 1 ({name1}):
{text1}

Document 2 ({name2}):
{text2}

Comparison:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )
    return response.choices[0].message.content.strip()

def highlight_answer(answer: str, source_text: str) -> str:
    """Find and highlight the sentence in source_text most relevant to the answer."""
    sentences = [s.strip() for s in source_text.replace("\n", " ").split(".") if len(s.strip()) > 20]
    answer_words = set(answer.lower().split())
    best_sentence = ""
    best_score = 0
    for sentence in sentences:
        words = set(sentence.lower().split())
        overlap = len(answer_words & words)
        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence
    if best_sentence:
        highlighted = source_text.replace(
            best_sentence,
            f'<mark style="background:#fbbf24; color:#1f2937; border-radius:3px; padding:1px 3px;">{best_sentence}</mark>'
        )
        return highlighted
    return source_text