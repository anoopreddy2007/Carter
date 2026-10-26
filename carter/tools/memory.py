"""
Tool: save_note / search_notes
Stores and retrieves notes in a local SQLite database.
All notes are stored locally; no cloud sync.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from carter.utils.logging import get_logger

logger = get_logger("tools.memory")

# Schema version — increment if schema changes
SCHEMA_VERSION = 1

CREATE_NOTES_TABLE = """
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL,
    title       TEXT    NOT NULL DEFAULT '',
    content     TEXT    NOT NULL,
    tags        TEXT    NOT NULL DEFAULT ''
);
"""

CREATE_AUDIT_TABLE = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    action      TEXT    NOT NULL,
    tool        TEXT    NOT NULL DEFAULT '',
    args_json   TEXT    NOT NULL DEFAULT '{}',
    result      TEXT    NOT NULL DEFAULT '',
    error       TEXT    NOT NULL DEFAULT ''
);
"""

CREATE_META_TABLE = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _get_connection(db_path: str) -> sqlite3.Connection:
    """Open (and initialize if needed) the SQLite database."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")

    conn.execute(CREATE_NOTES_TABLE)
    conn.execute(CREATE_AUDIT_TABLE)
    conn.execute(CREATE_META_TABLE)

    # Record schema version
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    conn.commit()
    return conn


def save_note(
    content: str,
    db_path: str,
    title: str = "",
    tags: str = "",
    max_note_length: int = 4096,
) -> dict[str, Any]:
    """
    Save a note to the SQLite database.

    Args:
        content: The note body text.
        db_path: Path to the SQLite database.
        title: Optional note title.
        tags: Optional comma-separated tags.
        max_note_length: Maximum allowed note length in characters.

    Returns:
        A dict with the saved note's id and metadata.
    """
    if not content or not content.strip():
        raise ValueError("Note content cannot be empty.")

    if len(content) > max_note_length:
        raise ValueError(
            f"Note is too long ({len(content)} chars). "
            f"Maximum: {max_note_length} chars."
        )

    now = time.time()
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO notes(created_at, updated_at, title, content, tags) "
            "VALUES (?, ?, ?, ?, ?)",
            (now, now, title.strip(), content.strip(), tags.strip()),
        )
        conn.commit()
        note_id = cursor.lastrowid
    finally:
        conn.close()

    logger.info("save_note: saved note id=%d, title='%s'", note_id, title)
    return {
        "id": note_id,
        "title": title,
        "created_at": now,
        "message": f"Note saved with id={note_id}.",
    }


def search_notes(
    query: str,
    db_path: str,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Search saved notes by full-text match in content and title.

    Args:
        query: The search string.
        db_path: Path to the SQLite database.
        limit: Maximum number of results to return (capped at 50).

    Returns:
        A dict with a list of matching note dicts.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    limit = min(int(limit), 50)
    pattern = f"%{query.strip()}%"

    conn = _get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, created_at, title, content, tags
            FROM notes
            WHERE content LIKE ? OR title LIKE ? OR tags LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (pattern, pattern, pattern, limit),
        ).fetchall()
    finally:
        conn.close()

    results = [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "title": row["title"],
            "content": row["content"][:500] + ("..." if len(row["content"]) > 500 else ""),
            "tags": row["tags"],
        }
        for row in rows
    ]

    logger.info("search_notes: query='%s' returned %d results", query, len(results))
    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


def list_notes(db_path: str, limit: int = 20) -> dict[str, Any]:
    """List the most recently saved notes."""
    limit = min(int(limit), 100)
    conn = _get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, created_at, title, tags FROM notes ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    return {
        "count": len(rows),
        "notes": [dict(row) for row in rows],
    }


def delete_note(note_id: int, db_path: str) -> dict[str, Any]:
    """Delete a note by its id."""
    conn = _get_connection(db_path)
    try:
        result = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        conn.commit()
        deleted = result.rowcount
    finally:
        conn.close()

    if deleted == 0:
        raise ValueError(f"Note with id={note_id} not found.")

    logger.info("delete_note: deleted note id=%d", note_id)
    return {"deleted_id": note_id, "message": f"Note {note_id} deleted."}
