"""
Parses PDFs and plain text into clean chunks ready for embedding.

Chunking strategy: simple sliding window over sentences/paragraphs by
character count. Good enough for technical docs at hackathon scale -
no need for a fancy semantic chunker here.
"""
from pypdf import PdfReader
import io


def parse_file(filename: str, file_bytes: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    else:
        # treat anything else as plain text (txt, md, etc.)
        return file_bytes.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())  # normalise whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks
