"""SessionStore - SQLite-backed persistence for trajectories.

Two tables:
  sessions(id, started_at, constitution, payload_json)
  events(session_id, turn_index, kind, ts, payload_json)
    where kind in {turn, reading, intervention}

The payload_json column is the dataclass-as-dict for full fidelity. The
broken-out columns exist so you can SELECT WHERE constitution LIKE ...
or count tripped readings without parsing JSON.

Designed to be used as `Supervisor(..., store=SessionStore("sidecar.db"))`.
On every turn / reading / intervention, the store is upserted.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path

from sidecar.trajectory import Session

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    constitution TEXT NOT NULL,
    n_turns INTEGER NOT NULL,
    n_trips INTEGER NOT NULL,
    n_interventions INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);

CREATE TABLE IF NOT EXISTS readings (
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    threshold REAL NOT NULL,
    tripped INTEGER NOT NULL,
    rationale TEXT,
    ts REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_readings_session ON readings(session_id);
CREATE INDEX IF NOT EXISTS idx_readings_tripped ON readings(tripped, name);

CREATE TABLE IF NOT EXISTS interventions (
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    type TEXT NOT NULL,
    triggered_by TEXT NOT NULL,
    details TEXT,
    ts REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_interventions_session ON interventions(session_id);
"""


class SessionStore:
    def __init__(self, path: str | Path = "sidecar.db"):
        self.path = str(path)
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, isolation_level=None, timeout=5.0)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        return c

    def _init_schema(self) -> None:
        with self._lock, self._conn() as c:
            c.executescript(_SCHEMA)

    def upsert(self, session: Session) -> None:
        n_trips = sum(1 for r in session.readings if r.tripped)
        payload = json.dumps(asdict(session), default=str)
        ts_now = max(
            (session.started_at,
             *(t.timestamp for t in session.turns),
             *(r.timestamp for r in session.readings),
             *(i.timestamp for i in session.interventions)),
        )
        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT INTO sessions(id, started_at, constitution, n_turns, n_trips,
                                     n_interventions, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    n_turns = excluded.n_turns,
                    n_trips = excluded.n_trips,
                    n_interventions = excluded.n_interventions,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (session.id, session.started_at, session.constitution,
                 len(session.turns), n_trips, len(session.interventions),
                 payload, ts_now),
            )
            c.execute("DELETE FROM readings WHERE session_id = ?", (session.id,))
            c.execute("DELETE FROM interventions WHERE session_id = ?", (session.id,))
            c.executemany(
                """INSERT INTO readings(session_id, turn_index, name, value, threshold,
                                        tripped, rationale, ts)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [(session.id, r.turn_index, r.name, r.value, r.threshold,
                  int(r.tripped), r.rationale, r.timestamp)
                 for r in session.readings],
            )
            c.executemany(
                """INSERT INTO interventions(session_id, turn_index, type, triggered_by,
                                             details, ts)
                   VALUES (?,?,?,?,?,?)""",
                [(session.id, i.turn_index, i.type, i.triggered_by, i.details, i.timestamp)
                 for i in session.interventions],
            )

    def get(self, session_id: str) -> Session | None:
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT payload_json FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not row:
                return None
            return _session_from_dict(json.loads(row[0]))

    def list_recent(self, limit: int = 50) -> list[dict]:
        with self._lock, self._conn() as c:
            rows = c.execute(
                """SELECT id, started_at, n_turns, n_trips, n_interventions
                   FROM sessions ORDER BY started_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {"id": r[0], "started_at": r[1], "n_turns": r[2],
             "n_trips": r[3], "n_interventions": r[4]}
            for r in rows
        ]

    def signal_stats(self) -> list[dict]:
        with self._lock, self._conn() as c:
            rows = c.execute(
                """SELECT name,
                          COUNT(*) AS n_readings,
                          SUM(tripped) AS n_trips,
                          AVG(value) AS avg_value
                   FROM readings GROUP BY name""",
            ).fetchall()
        return [
            {"name": r[0], "n_readings": r[1], "n_trips": r[2] or 0,
             "trip_rate": (r[2] or 0) / max(1, r[1]), "avg_value": r[3] or 0.0}
            for r in rows
        ]


def _session_from_dict(d: dict) -> Session:
    from sidecar.trajectory import Turn, SignalReading, Intervention
    s = Session(
        id=d["id"],
        constitution=d["constitution"],
        started_at=d.get("started_at", 0.0),
    )
    s.turns = [Turn(**t) for t in d.get("turns", [])]
    s.readings = [SignalReading(**r) for r in d.get("readings", [])]
    s.interventions = [Intervention(**i) for i in d.get("interventions", [])]
    return s
