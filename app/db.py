"""SQLite storage for users, sessions and prediction history.

Note: on Render's free tier the filesystem is ephemeral, so history is reset on
each redeploy. Point DATABASE_PATH to a persistent disk, or swap this module for
Postgres, if history must survive deployments.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    inputs TEXT NOT NULL,
    quality REAL NOT NULL,
    label TEXT NOT NULL,
    model_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id, id);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Database:
    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # users
    def create_user(self, username: str, password_hash: str) -> int | None:
        try:
            with self.connect() as conn:
                cur = conn.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, password_hash, _now().isoformat()),
                )
                return cur.lastrowid
        except sqlite3.IntegrityError:
            return None

    def get_user(self, username: str):
        with self.connect() as conn:
            return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    # sessions
    def create_session(self, token: str, user_id: int, ttl_hours: int) -> None:
        expires = _now() + timedelta(hours=ttl_hours)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, expires.isoformat()),
            )

    def user_for_token(self, token: str):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT u.*, s.expires_at FROM sessions s JOIN users u ON u.id = s.user_id "
                "WHERE s.token = ?",
                (token,),
            ).fetchone()
            if row is None:
                return None
            if datetime.fromisoformat(row["expires_at"]) < _now():
                conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
                return None
            return row

    def delete_session(self, token: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))

    # history
    def add_prediction(self, user_id, inputs, quality, label, model_version) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO predictions (user_id, created_at, inputs, quality, label, model_version)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, _now().isoformat(), json.dumps(inputs), quality, label, model_version),
            )
            return cur.lastrowid

    def history(self, user_id: int, limit: int) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "inputs": json.loads(r["inputs"]),
                "quality": r["quality"],
                "label": r["label"],
                "model_version": r["model_version"],
            }
            for r in rows
        ]

    def clear_history(self, user_id: int) -> int:
        with self.connect() as conn:
            return conn.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,)).rowcount
