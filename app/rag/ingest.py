from app.rag.document_parser import parse_file, chunk_text
from app.rag.embeddings import embed_texts
from app.db import sqlite_store, vector_store
from app.agents.entity_extraction_agent import extract_entities


def ingest_document(filename: str, file_bytes: bytes) -> dict:
    raw_text = parse_file(filename, file_bytes)
    chunks = chunk_text(raw_text)

    if not chunks:
        raise ValueError("No extractable text found in this file.")

    doc_type = filename.split(".")[-1].lower()
    doc_id = sqlite_store.insert_document(filename, doc_type)
    chunk_ids = sqlite_store.insert_chunks(doc_id, chunks)

    vectors = embed_texts(chunks)
    vector_store.add_vectors(vectors, chunk_ids)

    entities = extract_entities(raw_text)
    if entities:
        sqlite_store.insert_entities(doc_id, entities)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks_created": len(chunks),
        "entities_extracted": len(entities),
    }
