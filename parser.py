import fitz
from pathlib import Path

def parse_pdf(filepath: str) -> list:
    doc = fitz.open(filepath)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "page": i + 1,
                "text": text,
                "source": Path(filepath).name
            })
    doc.close()
    return pages

def parse_txt(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return [{"page": 1, "text": text, "source": Path(filepath).name}]

def load_document(filepath: str) -> list:
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(filepath)
    elif ext == ".txt":
        return parse_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")