"""Client workspace: the kanban board and chat conversations, stored in SQLite.

This is the server side of the "Board" and "Conversations" sections of
docs/client-api.md. One process owns the file, so plain locking with the
default rollback journal is enough (and safe on Docker bind mounts, unlike WAL).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

COLUMNS: list[tuple[str, str]] = [
    ("backlog", "Backlog"),
    ("in_progress", "In progress"),
    ("review", "Review"),
    ("done", "Done"),
]
COLUMN_IDS = [c[0] for c in COLUMNS]
PRIORITIES = ("low", "medium", "high")
ROLES = ("system", "user", "assistant")
TITLE_MAX = 200
AUTO_TITLE_MAX = 48

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cards (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    col TEXT NOT NULL,
    priority TEXT NOT NULL,
    position INTEGER NOT NULL,
    conversation_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS cards_col ON cards (col, position);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_conv ON messages (conversation_id, seq);
"""


class NotFound(LookupError):
    pass


class Conflict(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


def auto_title(prompt: str) -> str:
    """First line of the first prompt, trimmed (same rule the iOS client uses)."""
    first = prompt.strip().splitlines()[0].strip() if prompt.strip() else ""
    if len(first) <= AUTO_TITLE_MAX:
        return first
    return first[:AUTO_TITLE_MAX].rstrip() + "…"


class Workspace:
    """Board cards and conversations. Thread-safe; every public call is one transaction.

    `on_change(topic, id)` is invoked after each write with topic "board" or
    "conversations", so an event feed can notify clients.
    """

    def __init__(self, path: Path | str | None, on_change: Callable[[str, str | None], None] | None = None):
        self._lock = threading.RLock()
        self._on_change = on_change or (lambda topic, id_: None)
        if path is None:
            self._db = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA synchronous = NORMAL")
        with self._db:
            self._db.executescript(_SCHEMA)
            for topic in ("board", "conversations"):
                self._db.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, '0')", (f"version:{topic}",))

    def close(self) -> None:
        self._db.close()

    # --- versions ---------------------------------------------------------

    def version(self, topic: str) -> int:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (f"version:{topic}",)).fetchone()
        return int(row["value"]) if row else 0

    def _bump(self, topic: str, id_: str | None = None) -> None:
        self._db.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = ?", (f"version:{topic}",))
        self._on_change(topic, id_)

    # --- board ------------------------------------------------------------

    @staticmethod
    def _card(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "title": row["title"],
            "notes": row["notes"],
            "column": row["col"],
            "priority": row["priority"],
            "position": row["position"],
            "conversation_id": row["conversation_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def board(self) -> dict:
        with self._lock:
            rows = self._db.execute("SELECT * FROM cards").fetchall()
            order = {c: i for i, c in enumerate(COLUMN_IDS)}
            cards = sorted((self._card(r) for r in rows), key=lambda c: (order.get(c["column"], 99), c["position"]))
            return {
                "version": self.version("board"),
                "columns": [{"id": cid, "title": title} for cid, title in COLUMNS],
                "cards": cards,
            }

    def get_card(self, card_id: str) -> dict:
        row = self._db.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
        if row is None:
            raise NotFound(f"no card '{card_id}'")
        return self._card(row)

    def create_card(self, title: str, notes: str = "", column: str = "backlog", priority: str = "medium",
                    conversation_id: str | None = None, position: str = "bottom") -> dict:
        _check_title(title)
        _check_enum("column", column, COLUMN_IDS)
        _check_enum("priority", priority, PRIORITIES)
        _check_enum("position", position, ("top", "bottom"))
        with self._lock, self._db:
            self._check_conversation(conversation_id)
            if position == "top":
                self._db.execute("UPDATE cards SET position = position + 1 WHERE col = ?", (column,))
                pos = 0
            else:
                row = self._db.execute("SELECT COALESCE(MAX(position), -1) + 1 AS p FROM cards WHERE col = ?", (column,)).fetchone()
                pos = row["p"]
            now = _now()
            card_id = _new_id()
            self._db.execute(
                "INSERT INTO cards (id, title, notes, col, priority, position, conversation_id, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (card_id, title.strip(), notes or "", column, priority, pos, conversation_id, now, now),
            )
            self._bump("board", card_id)
            return self.get_card(card_id)

    def update_card(self, card_id: str, **fields: Any) -> dict:
        """Partial update. Changing the column appends the card to that column's bottom."""
        allowed = {"title", "notes", "priority", "conversation_id", "column"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown card fields: {', '.join(sorted(unknown))}")
        with self._lock, self._db:
            card = self.get_card(card_id)
            if "title" in fields:
                _check_title(fields["title"])
                card["title"] = fields["title"].strip()
            if "notes" in fields:
                card["notes"] = fields["notes"] or ""
            if "priority" in fields:
                _check_enum("priority", fields["priority"], PRIORITIES)
                card["priority"] = fields["priority"]
            if "conversation_id" in fields:
                self._check_conversation(fields["conversation_id"])
                card["conversation_id"] = fields["conversation_id"]
            if "column" in fields and fields["column"] != card["column"]:
                _check_enum("column", fields["column"], COLUMN_IDS)
                self._place(card_id, card["column"], fields["column"], before=None, after=None)
            self._db.execute(
                "UPDATE cards SET title = ?, notes = ?, priority = ?, conversation_id = ?, updated_at = ? WHERE id = ?",
                (card["title"], card["notes"], card["priority"], card["conversation_id"], _now(), card_id),
            )
            self._bump("board", card_id)
            return self.get_card(card_id)

    def move_card(self, card_id: str, column: str, before: str | None = None, after: str | None = None) -> dict:
        _check_enum("column", column, COLUMN_IDS)
        if before and after:
            raise ValueError("give either 'before' or 'after', not both")
        with self._lock, self._db:
            card = self.get_card(card_id)
            self._place(card_id, card["column"], column, before, after)
            self._db.execute("UPDATE cards SET updated_at = ? WHERE id = ?", (_now(), card_id))
            self._bump("board", card_id)
            return self.get_card(card_id)

    def _place(self, card_id: str, from_col: str, to_col: str, before: str | None, after: str | None) -> None:
        """Put card_id into to_col at the requested spot and renumber both columns 0..n."""
        ids = [r["id"] for r in self._db.execute(
            "SELECT id FROM cards WHERE col = ? ORDER BY position", (to_col,)) if r["id"] != card_id]
        anchor = before or after
        if anchor:
            if anchor == card_id:
                raise Conflict("a card cannot be placed relative to itself")
            if anchor not in ids:
                anchor_card = self._db.execute("SELECT col FROM cards WHERE id = ?", (anchor,)).fetchone()
                if anchor_card is None:
                    raise NotFound(f"no card '{anchor}'")
                raise Conflict(f"card '{anchor}' is in column '{anchor_card['col']}', not '{to_col}'")
            at = ids.index(anchor) + (1 if after else 0)
        else:
            at = len(ids)
        ids.insert(at, card_id)
        self._db.execute("UPDATE cards SET col = ? WHERE id = ?", (to_col, card_id))
        self._db.executemany("UPDATE cards SET position = ? WHERE id = ?", [(i, cid) for i, cid in enumerate(ids)])
        if from_col != to_col:
            rest = [r["id"] for r in self._db.execute("SELECT id FROM cards WHERE col = ? ORDER BY position", (from_col,))]
            self._db.executemany("UPDATE cards SET position = ? WHERE id = ?", [(i, cid) for i, cid in enumerate(rest)])

    def delete_card(self, card_id: str) -> None:
        with self._lock, self._db:
            card = self.get_card(card_id)
            self._db.execute("DELETE FROM cards WHERE id = ?", (card_id,))
            rest = [r["id"] for r in self._db.execute("SELECT id FROM cards WHERE col = ? ORDER BY position", (card["column"],))]
            self._db.executemany("UPDATE cards SET position = ? WHERE id = ?", [(i, cid) for i, cid in enumerate(rest)])
            self._bump("board", card_id)

    def _check_conversation(self, conversation_id: str | None) -> None:
        if conversation_id is None:
            return
        if self._db.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone() is None:
            raise NotFound(f"no conversation '{conversation_id}'")

    # --- conversations ------------------------------------------------------

    def _conversation(self, row: sqlite3.Row) -> dict:
        stats = self._db.execute(
            "SELECT COUNT(*) AS n, (SELECT content FROM messages WHERE conversation_id = ? AND content != ''"
            " ORDER BY seq DESC LIMIT 1) AS preview FROM messages WHERE conversation_id = ?",
            (row["id"], row["id"]),
        ).fetchone()
        preview = (stats["preview"] or "").strip().splitlines()
        return {
            "id": row["id"],
            "title": row["title"],
            "model": row["model"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": stats["n"],
            "preview": preview[0][:200] if preview else "",
        }

    @staticmethod
    def _message(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "role": row["role"],
            "content": row["content"],
            "sources": json.loads(row["sources"]) if row["sources"] else None,
            "error": row["error"],
            "created_at": row["created_at"],
        }

    def list_conversations(self, limit: int = 50, offset: int = 0) -> dict:
        with self._lock:
            total = self._db.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()["n"]
            rows = self._db.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC, created_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
            return {
                "version": self.version("conversations"),
                "total": total,
                "conversations": [self._conversation(r) for r in rows],
            }

    def create_conversation(self, title: str = "", model: str = "") -> dict:
        if title and len(title) > TITLE_MAX:
            raise ValueError(f"title must be at most {TITLE_MAX} characters")
        with self._lock, self._db:
            now = _now()
            cid = _new_id()
            self._db.execute(
                "INSERT INTO conversations (id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (cid, (title or "").strip(), (model or "").strip(), now, now),
            )
            self._bump("conversations", cid)
            return self.get_conversation(cid, with_messages=False)

    def get_conversation(self, cid: str, with_messages: bool = True) -> dict:
        with self._lock:
            row = self._db.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
            if row is None:
                raise NotFound(f"no conversation '{cid}'")
            out = self._conversation(row)
            if with_messages:
                rows = self._db.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY seq", (cid,)).fetchall()
                out["messages"] = [self._message(r) for r in rows]
            return out

    def update_conversation(self, cid: str, **fields: Any) -> dict:
        unknown = set(fields) - {"title", "model"}
        if unknown:
            raise ValueError(f"unknown conversation fields: {', '.join(sorted(unknown))}")
        with self._lock, self._db:
            conv = self.get_conversation(cid, with_messages=False)
            if "title" in fields:
                title = (fields["title"] or "").strip()
                if len(title) > TITLE_MAX:
                    raise ValueError(f"title must be at most {TITLE_MAX} characters")
                conv["title"] = title
            if "model" in fields:
                conv["model"] = (fields["model"] or "").strip()
            self._db.execute("UPDATE conversations SET title = ?, model = ?, updated_at = ? WHERE id = ?",
                             (conv["title"], conv["model"], _now(), cid))
            self._bump("conversations", cid)
            return self.get_conversation(cid, with_messages=False)

    def delete_conversation(self, cid: str) -> None:
        with self._lock, self._db:
            self.get_conversation(cid, with_messages=False)
            linked = [r["id"] for r in self._db.execute("SELECT id FROM cards WHERE conversation_id = ?", (cid,))]
            self._db.execute("DELETE FROM conversations WHERE id = ?", (cid,))  # cascades to messages
            if linked:
                self._db.execute("UPDATE cards SET conversation_id = NULL, updated_at = ? WHERE conversation_id = ?", (_now(), cid))
                self._bump("board", None)
            self._bump("conversations", cid)

    def add_message(self, cid: str, role: str, content: str, sources: list | None = None,
                    error: str | None = None) -> dict:
        _check_enum("role", role, ROLES)
        if sources is not None and not isinstance(sources, list):
            raise ValueError("sources must be a list")
        with self._lock, self._db:
            conv = self.get_conversation(cid, with_messages=False)
            seq = self._db.execute("SELECT COALESCE(MAX(seq), -1) + 1 AS s FROM messages WHERE conversation_id = ?", (cid,)).fetchone()["s"]
            now = _now()
            mid = _new_id()
            self._db.execute(
                "INSERT INTO messages (id, conversation_id, seq, role, content, sources, error, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (mid, cid, seq, role, content or "", json.dumps(sources) if sources is not None else None, error, now),
            )
            title = conv["title"]
            if not title and role == "user" and content.strip():
                title = auto_title(content)
            self._db.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (title, now, cid))
            self._bump("conversations", cid)
            return self.get_message(cid, mid)

    def get_message(self, cid: str, mid: str) -> dict:
        row = self._db.execute("SELECT * FROM messages WHERE id = ? AND conversation_id = ?", (mid, cid)).fetchone()
        if row is None:
            raise NotFound(f"no message '{mid}' in conversation '{cid}'")
        return self._message(row)

    def update_message(self, cid: str, mid: str, **fields: Any) -> dict:
        unknown = set(fields) - {"content", "sources", "error"}
        if unknown:
            raise ValueError(f"unknown message fields: {', '.join(sorted(unknown))}")
        if "sources" in fields and fields["sources"] is not None and not isinstance(fields["sources"], list):
            raise ValueError("sources must be a list")
        with self._lock, self._db:
            msg = self.get_message(cid, mid)
            msg.update({k: v for k, v in fields.items()})
            self._db.execute(
                "UPDATE messages SET content = ?, sources = ?, error = ? WHERE id = ?",
                (msg["content"] or "", json.dumps(msg["sources"]) if msg["sources"] is not None else None, msg["error"], mid),
            )
            self._db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), cid))
            self._bump("conversations", cid)
            return self.get_message(cid, mid)

    def delete_message(self, cid: str, mid: str) -> None:
        with self._lock, self._db:
            self.get_message(cid, mid)
            self._db.execute("DELETE FROM messages WHERE id = ?", (mid,))
            self._db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), cid))
            self._bump("conversations", cid)

    # --- stats --------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            by_column = {cid: 0 for cid in COLUMN_IDS}
            for r in self._db.execute("SELECT col, COUNT(*) AS n FROM cards GROUP BY col"):
                by_column[r["col"]] = r["n"]
            return {
                "cards": sum(by_column.values()),
                "cards_by_column": by_column,
                "conversations": self._db.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()["n"],
            }


def _check_title(title: Any) -> None:
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title is required")
    if len(title.strip()) > TITLE_MAX:
        raise ValueError(f"title must be at most {TITLE_MAX} characters")


def _check_enum(name: str, value: Any, allowed) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(allowed)}")
