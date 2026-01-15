"""Database management for document storage and retrieval."""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Document:
    """Document metadata and content."""

    id: str
    path: str
    filename: str
    content: str
    file_hash: str
    chunks: list[str]
    vectors: list[list[float]]


class DocumentDatabase:
    """SQLite database for storing documents and their vectors."""

    def __init__(self, db_path: Path | None = None):
        """Initialize database connection."""
        if db_path is None:
            db_path = Path.home() / ".brainfs" / "documents.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    vector TEXT NOT NULL,  -- JSON serialized vector
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents (file_hash)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id)
            """)

    def add_document(self, document: Document) -> None:
        """Add a document to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (id, path, filename, content, file_hash)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    document.id,
                    document.path,
                    document.filename,
                    document.content,
                    document.file_hash,
                ),
            )

            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document.id,))

            for i, (chunk, vector) in enumerate(zip(document.chunks, document.vectors)):
                conn.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, text, vector)
                    VALUES (?, ?, ?, ?)
                """,
                    (document.id, i, chunk, json.dumps(vector)),
                )

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        with sqlite3.connect(self.db_path) as conn:
            doc_row = conn.execute(
                """
                SELECT id, path, filename, content, file_hash
                FROM documents WHERE id = ?
            """,
                (doc_id,),
            ).fetchone()

            if not doc_row:
                return None

            chunk_rows = conn.execute(
                """
                SELECT text, vector FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index
            """,
                (doc_id,),
            ).fetchall()

            chunks = [row[0] for row in chunk_rows]
            vectors = [json.loads(row[1]) for row in chunk_rows]

            return Document(
                id=doc_row[0],
                path=doc_row[1],
                filename=doc_row[2],
                content=doc_row[3],
                file_hash=doc_row[4],
                chunks=chunks,
                vectors=vectors,
            )

    def document_exists(self, file_hash: str) -> bool:
        """Check if document with hash already exists."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                """
                SELECT 1 FROM documents WHERE file_hash = ?
            """,
                (file_hash,),
            ).fetchone()
            return result is not None

    def list_documents(self) -> list[tuple[str, str, str]]:
        """List all documents (id, filename, path)."""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("""
                SELECT id, filename, path FROM documents
                ORDER BY filename
            """).fetchall()

    def search_chunks(self, limit: int = 100) -> list[tuple[str, str, str, list[float]]]:
        """Get all chunks for vector search (document_id, filename, text, vector)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT c.document_id, d.filename, c.text, c.vector
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY d.filename, c.chunk_index
                LIMIT ?
            """,
                (limit,),
            ).fetchall()

            return [(row[0], row[1], row[2], json.loads(row[3])) for row in rows]

    def clear_all(self) -> int:
        """Clear all documents and chunks. Returns number of documents deleted."""

        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")

            return count


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of file contents."""
    hash_sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()
