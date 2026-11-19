"""SQLite persistence for local chat conversations."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "conversations.sqlite3"
DEFAULT_TITLE = "New conversation"
MAX_TITLE_LENGTH = 60
_INITIALIZED = False


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connection():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize():
    """Create the conversation schema if it does not already exist."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                pending_json TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS messages_by_conversation
                ON messages (conversation_id, id);
            """
        )
    _INITIALIZED = True


def create_conversation(title=DEFAULT_TITLE):
    conversation_id = str(uuid.uuid4())
    now = _timestamp()
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, _normalize_title(title), now, now)
        )
    return conversation_id


def list_conversations():
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC, created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(conversation_id):
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,)
        ).fetchone()
    return dict(row) if row else None


def rename_conversation(conversation_id, title):
    title = _normalize_title(title)
    if not title:
        return False
    with _connection() as connection:
        result = connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (title, _timestamp(), conversation_id)
        )
    return result.rowcount > 0


def delete_conversation(conversation_id):
    with _connection() as connection:
        result = connection.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,)
        )
    return result.rowcount > 0


def get_messages(conversation_id):
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content, metadata_json
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (conversation_id,)
        ).fetchall()

    messages = []
    for row in rows:
        try:
            metadata = json.loads(row["metadata_json"])
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        messages.append({
            "role": row["role"],
            "content": row["content"],
            **metadata
        })
    return messages


def add_message(conversation_id, message):
    role = str(message.get("role", "assistant"))
    content = str(message.get("content", ""))
    metadata = {
        key: value
        for key, value in message.items()
        if key not in {"role", "content"}
    }
    now = _timestamp()

    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO messages (
                conversation_id, role, content, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                content,
                json.dumps(metadata, ensure_ascii=False),
                now
            )
        )

        if role == "user":
            title = _title_from_message(content)
            connection.execute(
                """
                UPDATE conversations
                SET title = CASE
                    WHEN title = ? THEN ?
                    ELSE title
                END,
                updated_at = ?
                WHERE id = ?
                """,
                (DEFAULT_TITLE, title, now, conversation_id)
            )
        else:
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id)
            )


def get_pending_request(conversation_id):
    with _connection() as connection:
        row = connection.execute(
            "SELECT pending_json FROM conversations WHERE id = ?",
            (conversation_id,)
        ).fetchone()
    if not row or not row["pending_json"]:
        return None
    try:
        return json.loads(row["pending_json"])
    except json.JSONDecodeError:
        return None


def set_pending_request(conversation_id, pending_request):
    pending_json = (
        json.dumps(pending_request, ensure_ascii=False)
        if pending_request is not None
        else None
    )
    with _connection() as connection:
        connection.execute(
            """
            UPDATE conversations
            SET pending_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (pending_json, _timestamp(), conversation_id)
        )


def _normalize_title(title):
    normalized = " ".join(str(title or "").split())
    return normalized[:MAX_TITLE_LENGTH]


def _title_from_message(content):
    normalized = " ".join(content.split())
    if len(normalized) > MAX_TITLE_LENGTH:
        normalized = normalized[:MAX_TITLE_LENGTH - 1].rstrip() + "…"
    return normalized or DEFAULT_TITLE
