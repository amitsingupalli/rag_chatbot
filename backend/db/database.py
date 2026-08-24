"""SQLite persistence for users, conversations, and messages."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    image_path TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                );

                CREATE TABLE IF NOT EXISTS user_memory (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'fact',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_user
                    ON conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_user_memory_user
                    ON user_memory(user_id);
                """
            )
            user_cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
            if "hashed_password" not in user_cols:
                try:
                    conn.execute("ALTER TABLE users ADD COLUMN hashed_password TEXT")
                except Exception:
                    pass

            cols = [r["name"] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
            if "sources" not in cols:
                try:
                    conn.execute("ALTER TABLE messages ADD COLUMN sources TEXT")
                except Exception:
                    pass
            if "web_sources" not in cols:
                try:
                    conn.execute("ALTER TABLE messages ADD COLUMN web_sources TEXT")
                except Exception:
                    pass
            if "agent_trace" not in cols:
                try:
                    conn.execute("ALTER TABLE messages ADD COLUMN agent_trace TEXT")
                except Exception:
                    pass

    @staticmethod
    def hash_password(password: str) -> str:
        import hashlib
        return hashlib.sha256(f"rag_salt_{password}".encode("utf-8")).hexdigest()

    def verify_password(self, plain_password: str, stored_hash: str | None) -> bool:
        if not stored_hash:
            return True
        return self.hash_password(plain_password) == stored_hash

    def set_user_password(self, user_id: str, plain_password: str) -> None:
        h = self.hash_password(plain_password)
        with self._connect() as conn:
            conn.execute("UPDATE users SET hashed_password = ? WHERE user_id = ?", (h, user_id))

    def authenticate_user(self, username: str, plain_password: str) -> dict[str, Any] | None:
        user = self.get_user_by_username(username)
        if not user:
            return None
        stored_hash = user.get("hashed_password")
        if stored_hash and not self.verify_password(plain_password, stored_hash):
            return None
        return user

    def create_user(self, username: str, hashed_password: str | None = None) -> dict[str, Any]:
        user_id = str(uuid.uuid4())
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO users (user_id, username, hashed_password, created_at) VALUES (?, ?, ?, ?)",
                (user_id, username, hashed_password, now),
            )
        return {"user_id": user_id, "username": username, "hashed_password": hashed_password, "created_at": now}

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        return dict(row) if row else None

    def migrate_user_data(self, old_user_id: str, new_user_id: str) -> None:
        if not old_user_id or not new_user_id or old_user_id == new_user_id:
            return
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET user_id = ? WHERE user_id = ?",
                (new_user_id, old_user_id),
            )
            try:
                conn.execute(
                    "UPDATE documents SET user_id = ? WHERE user_id = ?",
                    (new_user_id, old_user_id),
                )
            except Exception:
                pass

    def create_conversation(self, user_id: str, title: str = "New Chat") -> dict[str, Any]:
        conversation_id = str(uuid.uuid4())
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations
                (conversation_id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, user_id, title, now, now),
            )
        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }

    def list_conversations(self, user_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM conversations
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_user_daily_message_count(self, user_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as count FROM messages m
                JOIN conversations c ON m.conversation_id = c.conversation_id
                WHERE c.user_id = ? AND m.role = 'user'
                AND datetime(m.created_at) >= datetime('now', '-24 hours')
                """,
                (user_id,),
            ).fetchone()
        return row["count"] if row else 0

    def update_conversation_title(self, conversation_id: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE conversation_id = ?",
                (title, _utcnow(), conversation_id),
            )

    def touch_conversation(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (_utcnow(), conversation_id),
            )

    def delete_conversation(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
            conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        image_path: str | None = None,
        sources: list[str] | str | None = None,
        web_sources: list[str] | str | None = None,
        agent_trace: list[dict] | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import json
        if "sources" in kwargs and not sources:
            sources = kwargs["sources"]
        if "web_sources" in kwargs and not web_sources:
            web_sources = kwargs["web_sources"]
        if "agent_trace" in kwargs and not agent_trace:
            agent_trace = kwargs["agent_trace"]

        message_id = str(uuid.uuid4())
        now = _utcnow()
        src_str = json.dumps(sources) if isinstance(sources, list) else sources
        wsrc_str = json.dumps(web_sources) if isinstance(web_sources, list) else web_sources
        trace_str = json.dumps(agent_trace) if isinstance(agent_trace, (list, dict)) else agent_trace

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (message_id, conversation_id, role, content, image_path, sources, web_sources, agent_trace, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (message_id, conversation_id, role, content, image_path, src_str, wsrc_str, trace_str, now),
            )
        self.touch_conversation(conversation_id)
        return {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "image_path": image_path,
            "sources": src_str,
            "web_sources": wsrc_str,
            "agent_trace": trace_str,
            "created_at": now,
        }

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def add_user_memory(self, user_id: str, content: str, memory_type: str = "fact") -> None:
        memory_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_memory (memory_id, user_id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (memory_id, user_id, content, memory_type, _utcnow()),
            )

    def get_user_memories(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM user_memory
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_user_memories(self, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM user_memory WHERE user_id = ?", (user_id,))

    def delete_user_memory(self, memory_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM user_memory WHERE memory_id = ?", (memory_id,))
