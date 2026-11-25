"""store.py - SQLite persistence. WAL, synchronous=FULL, unique dedup keys."""
import pathlib
import os
import sqlite3

DB = pathlib.Path(os.environ.get("CRYPTOSCAN_STATE_DIR", "./state")) / "cryptoscan.db"

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
CREATE TABLE IF NOT EXISTS alerts (
  key TEXT PRIMARY KEY, setup_id TEXT, stage TEXT, generation INTEGER,
  status TEXT, attempts INTEGER DEFAULT 0, next_at REAL, updated REAL);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, setup_id TEXT, machine TEXT,
  state TEXT, ts TEXT, price REAL, note TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS events_uniq
  ON events(setup_id, machine, state, seq);
CREATE TABLE IF NOT EXISTS setups (
  setup_id TEXT PRIMARY KEY, symbol TEXT, direction TEXT, tf TEXT,
  grade TEXT, entry REAL, stop REAL, tp1 REAL, tp2 REAL, created TEXT);
CREATE TABLE IF NOT EXISTS orders (
  order_id TEXT PRIMARY KEY, setup_id TEXT, state TEXT, qty REAL,
  filled REAL DEFAULT 0, price REAL, updated TEXT);
CREATE TABLE IF NOT EXISTS positions (
  setup_id TEXT PRIMARY KEY, state TEXT, qty REAL, entry REAL,
  stop REAL, closed REAL DEFAULT 0, updated TEXT);
"""


def conn() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, timeout=15, isolation_level=None)
    c.executescript(SCHEMA)
    return c


def healthy() -> bool:
    try:
        with conn() as c:
            c.execute("SELECT 1")
        return True
    except Exception:
        return False


def ephemeral() -> bool:
    """Cloud filesystems wipe ./state on redeploy."""
    try:
        return not DB.parent.exists() or str(DB.parent.resolve()).startswith("/app")
    except Exception:
        return True
