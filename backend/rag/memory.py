"""Persistent per-user memory management."""

from __future__ import annotations

import re

from backend.db.database import Database


class PersistentMemory:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_context(self, user_id: str, limit: int = 15) -> str:
        memories = self.db.get_user_memories(user_id, limit=limit)
        if not memories:
            return ""
        lines = [f"- {m['content']}" for m in reversed(memories)]
        return "Known facts about this user:\n" + "\n".join(lines)

    def extract_and_store(self, user_id: str, user_message: str, assistant_reply: str) -> None:
        combined = f"{user_message}\n{assistant_reply}"
        patterns = [
            r"(?:my name is|i am|i'm)\s+([A-Z][a-zA-Z\s]{1,30})",
            r"(?:i work at|i work for)\s+(.{3,60})",
            r"(?:i like|i love|i prefer)\s+(.{3,80})",
            r"(?:remember that|please remember)\s+(.{5,120})",
        ]
        for pattern in patterns:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                fact = match.group(0).strip()
                existing = {m["content"] for m in self.db.get_user_memories(user_id, limit=50)}
                if fact not in existing:
                    self.db.add_user_memory(user_id, fact)

    def store_explicit(self, user_id: str, content: str) -> None:
        self.db.add_user_memory(user_id, content, memory_type="explicit")
