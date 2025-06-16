import fitz  # PyMuPDF
import json
import re

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, max_tokens=150):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks, current = [], []
    length = 0

    for s in sentences:
        tokens = len(s.split())
        if length + tokens > max_tokens:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(s)
        length += tokens
    if current:
        chunks.append(" ".join(current))
    return chunks

pdf_path = "ARN42404-FM_5-0-000-WEB-1.pdf"
doc = fitz.open(pdf_path)

all_chunks = []
for i, page in enumerate(doc):
    text = clean_text(page.get_text())
    chunks = chunk_text(text)
    for j, chunk in enumerate(chunks):
        all_chunks.append({
            "id": f"page{i+1}_chunk{j+1}",
            "text": chunk,
            "source": f"Page {i+1}"
        })

with open("fm5_chunks.json", "w") as f:
    json.dump(all_chunks, f, indent=2)

print(f"✅ Done. Extracted {len(all_chunks)} chunks.")

