from groq import Groq
from config import TOP_K
from embedder import retrieve
import os
import streamlit as st

def get_client():
    try:
        key = st.secrets["GROQ_API_KEY"]
    except:
        key = os.getenv("GROQ_API_KEY", "your-groq-key-here")
    return Groq(api_key=key)

def ask(query: str, index, chunks) -> dict:
    client = get_client()
    results = retrieve(query, index, chunks, top_k=TOP_K)
    context = "\n\n".join([r["text"] for r in results])
    prompt = f"""You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I couldn't find this in the documents."

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
        "sources": [
            {
                "source": r["source"],
                "page": r["page"],
                "score": r["score"]
            }
            for r in results
        ]
    }