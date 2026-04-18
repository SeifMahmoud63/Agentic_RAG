"""
SQLite-based metadata store for file hash deduplication and versioning.
Tracks file identity (file_id + original_name + project_id) and content identity (SHA-256 hash).
"""

import sqlite3
import os
from logs.logger import logger
from datetime import datetime
from typing import Optional, Dict, Any


from helpers.config import get_settings

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), get_settings().METADATA_DB_NAME)


def _get_connection(check_schema: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  
    
    if check_schema:

        try:
            conn.execute("SELECT 1 FROM file_registry LIMIT 1")
        except sqlite3.OperationalError:
            logger.info("Metadata tables missing. Initializing...")
            init_db()
        
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_connection(check_schema=False)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS file_registry (
                file_id TEXT PRIMARY KEY,
                original_name TEXT NOT NULL,
                project_id TEXT NOT NULL,
                current_hash TEXT NOT NULL,
                current_version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS file_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (file_id) REFERENCES file_registry(file_id)
            );

            CREATE INDEX IF NOT EXISTS idx_file_hash ON file_registry(current_hash);
            CREATE INDEX IF NOT EXISTS idx_file_name_project ON file_registry(original_name, project_id);
            CREATE INDEX IF NOT EXISTS idx_version_file_id ON file_versions(file_id);
        """)
        conn.commit()
        logger.info(f"Metadata DB initialized at {DB_PATH}")
    finally:
        conn.close()


def hash_exists(file_hash: str) -> Optional[Dict[str, Any]]:
    """
    Check if a file with this exact hash already exists.
    Returns the file record if found, None otherwise.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM file_registry WHERE current_hash = ?",
            (file_hash,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_file_by_name(original_name: str, project_id: str) -> Optional[Dict[str, Any]]:
    """
    Find an existing file by its original name within a project.
    Used to detect updates (same file name = update of existing document group).
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM file_registry WHERE original_name = ? AND project_id = ?",
            (original_name, project_id)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_file_by_id(file_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a file record by its unique database ID."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM file_registry WHERE file_id = ?",
            (file_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def register_new_file(
    file_id: str,
    original_name: str,
    project_id: str,
    file_hash: str,
) -> Dict[str, Any]:
    """Register a brand new file in the metadata store."""
    now = datetime.utcnow().isoformat()
    conn = _get_connection()
    try:
        conn.execute(
            """INSERT INTO file_registry 
               (file_id, original_name, project_id, current_hash, current_version, created_at, updated_at)
               VALUES (?, ?, ?, ?, 1, ?, ?)""",
            (file_id, original_name, project_id, file_hash, now, now)
        )
        conn.execute(
            """INSERT INTO file_versions (file_id, file_hash, version, created_at)
               VALUES (?, ?, 1, ?)""",
            (file_id, file_hash, now)
        )
        conn.commit()
        logger.info(f"Registered new file: {original_name} (id={file_id}, hash={file_hash[:12]}...)")
        return {
            "file_id": file_id,
            "original_name": original_name,
            "project_id": project_id,
            "current_hash": file_hash,
            "current_version": 1,
        }
    finally:
        conn.close()


def register_new_version(
    file_id: str,
    file_hash: str,
) -> Dict[str, Any]:
    """Register a new version of an existing file."""
    now = datetime.utcnow().isoformat()
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT current_version FROM file_registry WHERE file_id = ?",
            (file_id,)
        ).fetchone()

        if not row:
            raise ValueError(f"file_id '{file_id}' not found in registry.")

        new_version = row["current_version"] + 1

        conn.execute(
            """UPDATE file_registry 
               SET current_hash = ?, current_version = ?, updated_at = ?
               WHERE file_id = ?""",
            (file_hash, new_version, now, file_id)
        )

        conn.execute(
            """INSERT INTO file_versions (file_id, file_hash, version, created_at)
               VALUES (?, ?, ?, ?)""",
            (file_id, file_hash, new_version, now)
        )
        conn.commit()

        logger.info(f"Registered version {new_version} for file_id='{file_id}' (hash={file_hash[:12]}...)")
        return {
            "file_id": file_id,
            "current_hash": file_hash,
            "current_version": new_version,
        }
    finally:
        conn.close()


def get_current_version(file_id: str) -> int:
    """Get the current version number of a file."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT current_version FROM file_registry WHERE file_id = ?",
            (file_id,)
        ).fetchone()
        return row["current_version"] if row else 0
    finally:
        conn.close()
