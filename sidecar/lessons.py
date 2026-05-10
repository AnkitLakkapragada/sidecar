"""LessonStore — persistent memory for the therapy loop.

Stores Lessons in SQLite with a full-text-search index on (pattern,
user_excerpt, rationale) so we can retrieve the K most relevant past
lessons given the current conversation context.

A "memory of being pressured before." When SupervisedAgent starts a new
session under the same constitution, it pulls the most relevant lessons
from this store and injects them as a `prior_lessons` system block. The
agent enters the new session pre-warned about the patterns it has failed
on before — the part of "therapy" that makes capacity build over time.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path

from sidecar.reflection import Lesson

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lessons (
    id TEXT PRIMARY KEY,
    constitution_hash TEXT NOT NULL,
    pattern TEXT NOT NULL,
    user_excerpt TEXT NOT NULL,
    drifted_response TEXT NOT NULL,
    correction TEXT NOT NULL,
    rationale TEXT NOT NULL,
    triggered_signals TEXT NOT NULL,
    session_id TEXT,
    turn_index INTEGER,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lessons_constitution
    ON lessons(constitution_hash, created_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS lessons_fts USING fts5(
    pattern, user_excerpt, rationale,
    content='lessons', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS lessons_ai AFTER INSERT ON lessons BEGIN
    INSERT INTO lessons_fts(rowid, pattern, user_excerpt, rationale)
    VALUES (new.rowid, new.pattern, new.user_excerpt, new.rationale);
END;

CREATE TRIGGER IF NOT EXISTS lessons_ad AFTER DELETE ON lessons BEGIN
    INSERT INTO lessons_fts(lessons_fts, rowid, pattern, user_excerpt, rationale)
    VALUES('delete', old.rowid, old.pattern, old.user_excerpt, old.rationale);
END;
"""


class LessonStore:
    def __init__(self, path: str | Path = "lessons.db"):
        self.path = str(path)
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, isolation_level=None, timeout=5.0)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _init_schema(self) -> None:
        with self._lock, self._conn() as c:
            c.executescript(_SCHEMA)

    # ---- writes -----------------------------------------------------------

    def add(self, lesson: Lesson) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO lessons
                   (id, constitution_hash, pattern, user_excerpt, drifted_response,
                    correction, rationale, triggered_signals, session_id,
                    turn_index, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    lesson.id, lesson.constitution_hash, lesson.pattern,
                    lesson.user_excerpt, lesson.drifted_response, lesson.correction,
                    lesson.rationale, json.dumps(lesson.triggered_signals),
                    lesson.session_id, lesson.turn_index, lesson.created_at,
                ),
            )

    def add_many(self, lessons: list[Lesson]) -> None:
        for l in lessons:
            self.add(l)

    # ---- reads ------------------------------------------------------------

    def all(self, constitution_hash: str | None = None) -> list[Lesson]:
        with self._lock, self._conn() as c:
            if constitution_hash:
                rows = c.execute(
                    "SELECT * FROM lessons WHERE constitution_hash=? "
                    "ORDER BY created_at DESC",
                    (constitution_hash,),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM lessons ORDER BY created_at DESC"
                ).fetchall()
            cols = [d[0] for d in c.execute("SELECT * FROM lessons LIMIT 0").description]
        return [_row_to_lesson(dict(zip(cols, r))) for r in rows]

    def retrieve(
        self,
        context: str,
        constitution_hash: str,
        k: int = 5,
    ) -> list[Lesson]:
        """Retrieve top-K most relevant lessons for the given context using FTS.
        Falls back to "all most-recent" if FTS finds nothing."""
        query = _to_fts_query(context)
        with self._lock, self._conn() as c:
            rows = []
            if query:
                rows = c.execute(
                    """SELECT lessons.*, bm25(lessons_fts) AS score
                       FROM lessons_fts
                       JOIN lessons ON lessons.rowid = lessons_fts.rowid
                       WHERE lessons_fts MATCH ? AND constitution_hash = ?
                       ORDER BY score LIMIT ?""",
                    (query, constitution_hash, k),
                ).fetchall()
            if not rows:
                rows = c.execute(
                    """SELECT lessons.*, 0.0 FROM lessons
                       WHERE constitution_hash = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (constitution_hash, k),
                ).fetchall()
            cols = [d[0] for d in
                    c.execute("SELECT * FROM lessons LIMIT 0").description]
        return [_row_to_lesson(dict(zip(cols, r[: len(cols)]))) for r in rows]

    def count(self, constitution_hash: str | None = None) -> int:
        with self._lock, self._conn() as c:
            if constitution_hash:
                row = c.execute(
                    "SELECT COUNT(*) FROM lessons WHERE constitution_hash=?",
                    (constitution_hash,),
                ).fetchone()
            else:
                row = c.execute("SELECT COUNT(*) FROM lessons").fetchone()
        return int(row[0])

    def clear(self) -> None:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM lessons")


def _row_to_lesson(d: dict) -> Lesson:
    return Lesson(
        id=d["id"],
        constitution_hash=d["constitution_hash"],
        pattern=d["pattern"],
        user_excerpt=d["user_excerpt"],
        drifted_response=d["drifted_response"],
        correction=d["correction"],
        rationale=d["rationale"],
        triggered_signals=json.loads(d["triggered_signals"]),
        session_id=d["session_id"] or "",
        turn_index=int(d["turn_index"]) if d["turn_index"] is not None else -1,
        created_at=float(d["created_at"]),
    )


def _to_fts_query(text: str) -> str:
    """Convert free text into a safe FTS5 MATCH expression. Picks distinctive
    tokens (length >= 4, alpha) and ORs them. Returns "" if nothing usable."""
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]{3,}", text or "")
    seen = set()
    keep = []
    for t in tokens:
        tl = t.lower()
        if tl in _STOP or tl in seen:
            continue
        seen.add(tl)
        keep.append(tl)
        if len(keep) >= 12:
            break
    return " OR ".join(keep)


_STOP = {
    "this", "that", "with", "from", "have", "your", "you'll", "didn't",
    "isn't", "wasn't", "aren't", "haven't", "would", "could", "should",
    "their", "there", "where", "what", "when", "which", "while", "about",
    "into", "over", "just", "than", "them", "they", "then", "some", "more",
    "much", "most", "many", "very", "well", "even", "also", "such", "still",
    "doesn't", "didnt", "isnt", "wasnt", "wont", "cant", "dont",
}
