import os
import streamlit as st

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-key-here")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.0-flash-lite"

for d in [DATA_DIR, EMBEDDINGS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)