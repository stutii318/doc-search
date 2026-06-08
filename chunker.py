from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_pages(pages: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for p in pages:
        splits = splitter.split_text(p["text"])
        for s in splits:
            chunks.append({
                "text": s,
                "page": p["page"],
                "source": p["source"]
            })
    return chunks