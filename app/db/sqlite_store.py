"""
Lightweight persistence layer.

For the hackathon demo we use SQLite + a flat numpy vector index instead of
a hosted vector DB. This keeps the project runnable with zero external infra
beyond a Groq API key. The interface is deliberately small so it can be
swapped for Supabase/pgvector later (see app/db/supabase_stub.py for the
shape that migration would take).
"""
import sqlite3
import json
import os
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "knowledge.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            doc_type TEXT,
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        );

        -- Acts as our lightweight knowledge graph: entities pulled out of
        -- each document, with the relationship back to its source doc.
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            entity_type TEXT NOT NULL,
            entity_value TEXT NOT NULL,
            context TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        );

        CREATE TABLE IF NOT EXISTS compliance_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            regulation TEXT NOT NULL,
            status TEXT NOT NULL,
            note TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        );
        """
    )
    conn.commit()
    conn.close()


def insert_document(filename: str, doc_type: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (filename, doc_type) VALUES (?, ?)",
        (filename, doc_type),
    )
    doc_id = cur.lastrowid
    conn.commit()
    conn.close()
    return doc_id


def insert_chunks(doc_id: int, chunks: list[str]) -> list[int]:
    conn = get_conn()
    cur = conn.cursor()
    ids = []
    for i, text in enumerate(chunks):
        cur.execute(
            "INSERT INTO chunks (doc_id, chunk_index, text) VALUES (?, ?, ?)",
            (doc_id, i, text),
        )
        ids.append(cur.lastrowid)
    conn.commit()
    conn.close()
    return ids


def insert_entities(doc_id: int, entities: list[dict]):
    conn = get_conn()
    cur = conn.cursor()
    for e in entities:
        cur.execute(
            "INSERT INTO entities (doc_id, entity_type, entity_value, context) VALUES (?, ?, ?, ?)",
            (doc_id, e.get("type", "unknown"), e.get("value", ""), e.get("context", "")),
        )
    conn.commit()
    conn.close()


def insert_compliance_flags(doc_id: int, flags: list[dict]):
    conn = get_conn()
    cur = conn.cursor()
    for f in flags:
        cur.execute(
            "INSERT INTO compliance_flags (doc_id, regulation, status, note) VALUES (?, ?, ?, ?)",
            (doc_id, f.get("regulation", ""), f.get("status", "unknown"), f.get("note", "")),
        )
    conn.commit()
    conn.close()


def get_chunk_by_id(chunk_id: int) -> Optional[sqlite3.Row]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT chunks.*, documents.filename FROM chunks
           JOIN documents ON documents.id = chunks.doc_id
           WHERE chunks.id = ?""",
        (chunk_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row


def list_documents():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_entities_for_doc(doc_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM entities WHERE doc_id = ?", (doc_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_chunks_for_doc(doc_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
