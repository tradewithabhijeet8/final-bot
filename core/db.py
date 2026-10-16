"""
core/db.py
==========
SQLite persistence for the paper-trading ledger, bot events, processed-candle
bookkeeping, pending orders and backtest runs.

Safety rules implemented here
-----------------------------
*  Tables are created with ``IF NOT EXISTS`` and columns are added with
   ``ALTER TABLE ... ADD COLUMN`` only when missing.  Ordinary startup NEVER
   deletes trades.
*  Every query is parameterised (no string interpolation of user data).
*  Trade entry/exit and their matching event rows are written inside one
   transaction, so a crash cannot leave a trade without its audit trail.
*  ``processed_candles`` has a UNIQUE(feed_key, candle_ts) constraint.  That is
   the database-level guarantee that a repeated dashboard refresh cannot create
   a second trade for the same signal candle.
*  Rows are validated on read; a corrupt row is reported, not silently used.
*  No secrets are ever stored.  This app uses public market data only.

Threading
---------
Streamlit reruns scripts on multiple threads.  Rather than share one connection
(and fight ``check_same_thread``), every operation opens a short-lived
connection.  WAL mode keeps reads non-blocking.

Streamlit Community Cloud caveat
--------------------------------
The cloud filesystem is EPHEMERAL.  The .db file survives a rerun but not
necessarily a restart, redeploy or app sleep.  Treat cloud history as temporary;
run locally (or wire an external managed database) if you need durable records.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("btc_db")

DEFAULT_DB_PATH = "paper_trading.db"
RESET_CONFIRM_TOKEN = "RESET"

STATUS_OPEN = "open"
STATUS_CLOSED = "closed"
STATUS_CANCELLED = "cancelled"
VALID_STATUSES = {STATUS_OPEN, STATUS_CLOSED, STATUS_CANCELLED}

PENDING = "pending"
FILLED = "filled"
EXPIRED = "expired"


class DatabaseError(Exception):
    """Raised when persistence fails.  Carries a user-safe message."""


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

_SCHEMA: Dict[str, str] = {
    "settings": """
        CREATE TABLE IF NOT EXISTS settings (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
    """,
    "paper_trades": """
        CREATE TABLE IF NOT EXISTS paper_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id        TEXT NOT NULL UNIQUE,
            exchange        TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            strategy        TEXT NOT NULL,
            direction       TEXT NOT NULL,
            signal_ts       INTEGER,
            entry_ts        INTEGER,
            entry_price     REAL,
            quantity        REAL,
            stop_loss       REAL,
            take_profit     REAL,
            initial_risk    REAL,
            entry_fee       REAL,
            exit_ts         INTEGER,
            exit_price      REAL,
            exit_fee        REAL,
            gross_pnl       REAL,
            net_pnl         REAL,
            realized_r      REAL,
            exit_reason     TEXT,
            status          TEXT NOT NULL DEFAULT 'open',
            fill_model      TEXT,
            slippage_bps    REAL,
            stop_mode       TEXT,
            notes           TEXT,
            created_at      INTEGER NOT NULL,
            updated_at      INTEGER NOT NULL
        )
    """,
    "bot_events": """
        CREATE TABLE IF NOT EXISTS bot_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         INTEGER NOT NULL,
            feed_key   TEXT,
            event_type TEXT NOT NULL,
            level      TEXT NOT NULL DEFAULT 'info',
            message    TEXT NOT NULL,
            data       TEXT
        )
    """,
    "processed_candles": """
        CREATE TABLE IF NOT EXISTS processed_candles (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            feed_key     TEXT NOT NULL,
            candle_ts    INTEGER NOT NULL,
            processed_at INTEGER NOT NULL,
            outcome      TEXT,
            UNIQUE(feed_key, candle_ts)
        )
    """,
    "pending_orders": """
        CREATE TABLE IF NOT EXISTS pending_orders (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id   TEXT NOT NULL UNIQUE,
            feed_key   TEXT NOT NULL,
            signal_ts  INTEGER NOT NULL,
            direction  TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            status     TEXT NOT NULL DEFAULT 'pending',
            params     TEXT,
            UNIQUE(feed_key, signal_ts, direction)
        )
    """,
    "backtest_runs": """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL UNIQUE,
            created_at  INTEGER NOT NULL,
            exchange    TEXT,
            symbol      TEXT,
            timeframe   TEXT,
            candles     INTEGER,
            config      TEXT,
            metrics     TEXT,
            n_trades    INTEGER,
            net_pnl     REAL,
            warnings    TEXT
        )
    """,
    "backtest_trades": """
        CREATE TABLE IF NOT EXISTS backtest_trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL,
            trade_no    INTEGER NOT NULL,
            direction   TEXT,
            signal_ts   INTEGER,
            entry_ts    INTEGER,
            entry_price REAL,
            quantity    REAL,
            stop_loss   REAL,
            take_profit REAL,
            exit_ts     INTEGER,
            exit_price  REAL,
            gross_pnl   REAL,
            net_pnl     REAL,
            realized_r  REAL,
            exit_reason TEXT,
            balance     REAL
        )
    """,
}

#: Columns added by migration if an older database is opened.
_MIGRATIONS: Dict[str, List[Tuple[str, str]]] = {
    "paper_trades": [
        ("realized_r", "REAL"),
        ("fill_model", "TEXT"),
        ("slippage_bps", "REAL"),
        ("stop_mode", "TEXT"),
        ("notes", "TEXT"),
    ],
    "bot_events": [("feed_key", "TEXT")],
    "processed_candles": [("outcome", "TEXT")],
    "backtest_runs": [("warnings", "TEXT")],
}

_INDEXES: List[str] = [
    "CREATE INDEX IF NOT EXISTS idx_trades_status ON paper_trades(status)",
    "CREATE INDEX IF NOT EXISTS idx_trades_exit_ts ON paper_trades(exit_ts)",
    "CREATE INDEX IF NOT EXISTS idx_events_ts ON bot_events(ts)",
    "CREATE INDEX IF NOT EXISTS idx_processed_feed ON processed_candles(feed_key, candle_ts)",
    "CREATE INDEX IF NOT EXISTS idx_bt_trades_run ON backtest_trades(run_id)",
]


def utc_midnight_ms(when_ms: Optional[float] = None) -> int:
    """Milliseconds since epoch of the most recent 00:00 UTC."""
    now = datetime.now(timezone.utc) if when_ms is None else datetime.fromtimestamp(when_ms / 1000.0, timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(midnight.timestamp() * 1000)


def utc_date_str(when_ms: Optional[float] = None) -> str:
    dt = datetime.now(timezone.utc) if when_ms is None else datetime.fromtimestamp(when_ms / 1000.0, timezone.utc)
    return dt.strftime("%Y-%m-%d")


def ms_to_utc(ms: Optional[float]) -> str:
    if ms is None:
        return "-"
    try:
        return datetime.fromtimestamp(float(ms) / 1000.0, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (OverflowError, OSError, ValueError):
        return "-"


# --------------------------------------------------------------------------- #
# Database wrapper
# --------------------------------------------------------------------------- #

class Database:
    """Thin, safe wrapper around a SQLite file."""

    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path
        self._initialized = False

    # -- connection management ------------------------------------------- #

    @contextmanager
    def _conn(self, commit: bool = False):
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as exc:
                raise DatabaseError(f"Cannot create database folder '{directory}': {exc}") from exc
        try:
            conn = sqlite3.connect(self.path, timeout=15.0)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot open SQLite database at '{self.path}': {exc}") from exc
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            if commit:
                conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise DatabaseError(f"SQLite error: {exc}") from exc
        finally:
            conn.close()

    @contextmanager
    def transaction(self):
        """Explicit transaction context (used for entry/exit + event atomicity)."""
        with self._conn(commit=True) as conn:
            conn.execute("BEGIN IMMEDIATE")
            yield conn

    # -- schema ----------------------------------------------------------- #

    def init_schema(self) -> Dict[str, Any]:
        """Create missing tables/indexes and apply additive migrations."""
        report = {"created": [], "migrated": [], "tables": []}
        with self._conn(commit=True) as conn:
            existing = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            for table, ddl in _SCHEMA.items():
                conn.execute(ddl)
                if table not in existing:
                    report["created"].append(table)
            # additive migrations only - never DROP or rewrite
            for table, columns in _MIGRATIONS.items():
                present = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
                for col, coltype in columns:
                    if col not in present:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
                        report["migrated"].append(f"{table}.{col}")
            for idx in _INDEXES:
                conn.execute(idx)
            report["tables"] = sorted(
                r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            )
        self._initialized = True
        if report["created"] or report["migrated"]:
            log.info("Schema init: created=%s migrated=%s", report["created"], report["migrated"])
        return report

    def ensure(self) -> None:
        if not self._initialized:
            self.init_schema()

    # -- settings --------------------------------------------------------- #

    def get_setting(self, key: str, default: Any = None) -> Any:
        self.ensure()
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return row["value"]

    def set_setting(self, key: str, value: Any) -> None:
        self.ensure()
        now = int(time.time() * 1000)
        payload = json.dumps(value)
        with self._conn(commit=True) as conn:
            conn.execute(
                """INSERT INTO settings(key, value, updated_at) VALUES(?,?,?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                (key, payload, now),
            )

    def all_settings(self) -> Dict[str, Any]:
        self.ensure()
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
        out: Dict[str, Any] = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except (json.JSONDecodeError, TypeError):
                out[r["key"]] = r["value"]
        return out

    # -- events ----------------------------------------------------------- #

    def log_event(self, event_type: str, message: str, level: str = "info",
                  feed_key: Optional[str] = None, data: Optional[Dict[str, Any]] = None,
                  conn: Optional[sqlite3.Connection] = None) -> None:
        """Append an audit event.  Pass `conn` to include it in a caller's transaction."""
        self.ensure()
        payload = json.dumps(data) if data is not None else None
        sql = ("INSERT INTO bot_events(ts, feed_key, event_type, level, message, data) "
               "VALUES(?,?,?,?,?,?)")
        args = (int(time.time() * 1000), feed_key, event_type, level, message, payload)
        if conn is not None:
            conn.execute(sql, args)
        else:
            with self._conn(commit=True) as c:
                c.execute(sql, args)

    def get_events(self, limit: int = 100, feed_key: Optional[str] = None,
                   event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        self.ensure()
        sql = "SELECT * FROM bot_events"
        clauses, args = [], []
        if feed_key:
            clauses.append("feed_key=?")
            args.append(feed_key)
        if event_type:
            clauses.append("event_type=?")
            args.append(event_type)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        args.append(int(limit))
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._row_to_event(r) for r in rows]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        raw = d.get("data")
        if raw:
            try:
                d["data"] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                d["data"] = {"_raw": str(raw)[:500]}
        else:
            d["data"] = None
        return d

    def prune_events(self, keep: int = 5000) -> int:
        self.ensure()
        with self._conn(commit=True) as conn:
            cur = conn.execute(
                "DELETE FROM bot_events WHERE id NOT IN "
                "(SELECT id FROM bot_events ORDER BY ts DESC, id DESC LIMIT ?)", (int(keep),))
            return cur.rowcount or 0

    # -- processed candles (duplicate guard) ------------------------------ #

    def is_candle_processed(self, feed_key: str, candle_ts: int) -> bool:
        self.ensure()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM processed_candles WHERE feed_key=? AND candle_ts=?",
                (feed_key, int(candle_ts))).fetchone()
        return row is not None

    def mark_candle_processed(self, feed_key: str, candle_ts: int,
                              outcome: str = "evaluated",
                              conn: Optional[sqlite3.Connection] = None) -> bool:
        """
        Record that a closed candle has been evaluated.

        Returns True if this call inserted the row, False if it was already
        there (i.e. a duplicate refresh).  The UNIQUE constraint makes this race-safe.
        """
        self.ensure()
        sql = ("INSERT OR IGNORE INTO processed_candles(feed_key, candle_ts, processed_at, outcome) "
               "VALUES(?,?,?,?)")
        args = (feed_key, int(candle_ts), int(time.time() * 1000), outcome)
        if conn is not None:
            cur = conn.execute(sql, args)
        else:
            with self._conn(commit=True) as c:
                cur = c.execute(sql, args)
        return (cur.rowcount or 0) > 0

    def last_processed_candle(self, feed_key: str) -> Optional[int]:
        self.ensure()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(candle_ts) AS m FROM processed_candles WHERE feed_key=?",
                (feed_key,)).fetchone()
        return int(row["m"]) if row and row["m"] is not None else None

    def processed_count(self, feed_key: Optional[str] = None) -> int:
        self.ensure()
        with self._conn() as conn:
            if feed_key:
                row = conn.execute("SELECT COUNT(*) AS c FROM processed_candles WHERE feed_key=?",
                                   (feed_key,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS c FROM processed_candles").fetchone()
        return int(row["c"]) if row else 0

    # -- trades ----------------------------------------------------------- #

    TRADE_COLUMNS = [
        "trade_id", "exchange", "symbol", "strategy", "direction", "signal_ts",
        "entry_ts", "entry_price", "quantity", "stop_loss", "take_profit",
        "initial_risk", "entry_fee", "exit_ts", "exit_price", "exit_fee",
        "gross_pnl", "net_pnl", "realized_r", "exit_reason", "status",
        "fill_model", "slippage_bps", "stop_mode", "notes", "created_at", "updated_at",
    ]

    def insert_trade(self, trade: Dict[str, Any],
                     conn: Optional[sqlite3.Connection] = None) -> str:
        """Insert a new trade row.  Pass `conn` to make it part of a transaction."""
        self.ensure()
        now = int(time.time() * 1000)
        row = {c: trade.get(c) for c in self.TRADE_COLUMNS}
        row["trade_id"] = row.get("trade_id") or f"T{uuid.uuid4().hex[:12].upper()}"
        row["status"] = row.get("status") or STATUS_OPEN
        if row["status"] not in VALID_STATUSES:
            raise DatabaseError(f"Invalid trade status '{row['status']}'.")
        row["created_at"] = int(row.get("created_at") or now)
        row["updated_at"] = now
        cols = self.TRADE_COLUMNS
        sql = f"INSERT INTO paper_trades({','.join(cols)}) VALUES({','.join('?' * len(cols))})"
        args = tuple(row[c] for c in cols)
        if conn is not None:
            conn.execute(sql, args)
        else:
            with self._conn(commit=True) as c:
                c.execute(sql, args)
        return row["trade_id"]

    def close_trade(self, trade_id: str, updates: Dict[str, Any],
                    conn: Optional[sqlite3.Connection] = None) -> bool:
        """Apply exit fields to an open trade."""
        self.ensure()
        allowed = {"exit_ts", "exit_price", "exit_fee", "gross_pnl", "net_pnl",
                   "realized_r", "exit_reason", "status", "notes", "updated_at"}
        payload = {k: v for k, v in updates.items() if k in allowed}
        payload["status"] = payload.get("status", STATUS_CLOSED)
        payload["updated_at"] = int(time.time() * 1000)
        sets = ", ".join(f"{k}=?" for k in payload)
        sql = f"UPDATE paper_trades SET {sets} WHERE trade_id=?"
        args = tuple(payload.values()) + (trade_id,)
        if conn is not None:
            cur = conn.execute(sql, args)
        else:
            with self._conn(commit=True) as c:
                cur = c.execute(sql, args)
        return (cur.rowcount or 0) > 0

    def get_trades(self, status: Optional[str] = None, limit: int = 500,
                   symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        self.ensure()
        sql = "SELECT * FROM paper_trades"
        clauses, args = [], []
        if status:
            clauses.append("status=?")
            args.append(status)
        if symbol:
            clauses.append("symbol=?")
            args.append(symbol)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY COALESCE(entry_ts, created_at) DESC, id DESC LIMIT ?"
        args.append(int(limit))
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            validated, problems = self._validate_trade(dict(r))
            if problems:
                log.warning("Trade %s has data problems: %s", validated.get("trade_id"), problems)
                validated["_validation_problems"] = problems
            out.append(validated)
        return out

    def get_open_trades(self, feed_key: Optional[str] = None) -> List[Dict[str, Any]]:
        exchange = symbol = None
        if feed_key and feed_key.count(":") == 2:
            exchange, symbol, _tf = feed_key.split(":")
        self.ensure()
        sql = "SELECT * FROM paper_trades WHERE status=?"
        args: List[Any] = [STATUS_OPEN]
        if exchange:
            sql += " AND exchange=? AND symbol=?"
            args += [exchange, symbol]
        sql += " ORDER BY entry_ts ASC"
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._validate_trade(dict(r))[0] for r in rows]

    def count_trades(self) -> Dict[str, int]:
        self.ensure()
        with self._conn() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS c FROM paper_trades GROUP BY status").fetchall()
        out = {"open": 0, "closed": 0, "cancelled": 0, "total": 0}
        for r in rows:
            out[r["status"]] = int(r["c"])
            out["total"] += int(r["c"])
        return out

    def realized_pnl_total(self) -> float:
        self.ensure()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(net_pnl),0) AS s FROM paper_trades WHERE status=?",
                (STATUS_CLOSED,)).fetchone()
        return float(row["s"] or 0.0)

    def realized_pnl_since(self, since_ms: int) -> float:
        """Realised P/L from trades CLOSED at or after `since_ms` (UTC midnight)."""
        self.ensure()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(net_pnl),0) AS s FROM paper_trades "
                "WHERE status=? AND exit_ts IS NOT NULL AND exit_ts>=?",
                (STATUS_CLOSED, int(since_ms))).fetchone()
        return float(row["s"] or 0.0)

    def realized_pnl_on(self, day_start_ms: int, day_end_ms: int) -> float:
        self.ensure()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(net_pnl),0) AS s FROM paper_trades "
                "WHERE status=? AND exit_ts IS NOT NULL AND exit_ts>=? AND exit_ts<?",
                (STATUS_CLOSED, int(day_start_ms), int(day_end_ms))).fetchone()
        return float(row["s"] or 0.0)

    @staticmethod
    def _validate_trade(row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Coerce and sanity-check a trade row before it is used in maths."""
        problems: List[str] = []
        numeric = ["entry_price", "quantity", "stop_loss", "take_profit", "initial_risk",
                   "entry_fee", "exit_price", "exit_fee", "gross_pnl", "net_pnl", "realized_r"]
        for col in numeric:
            v = row.get(col)
            if v is None:
                continue
            try:
                row[col] = float(v)
                if not (row[col] == row[col]):  # NaN check without importing math
                    problems.append(f"{col} is NaN")
                    row[col] = None
            except (TypeError, ValueError):
                problems.append(f"{col} is not numeric ({v!r})")
                row[col] = None
        for col in ["signal_ts", "entry_ts", "exit_ts", "created_at", "updated_at"]:
            v = row.get(col)
            if v is None:
                continue
            try:
                row[col] = int(v)
            except (TypeError, ValueError):
                problems.append(f"{col} is not an integer ({v!r})")
                row[col] = None
        if row.get("direction") not in ("long", "short"):
            problems.append(f"direction invalid ({row.get('direction')!r})")
        if row.get("status") not in VALID_STATUSES:
            problems.append(f"status invalid ({row.get('status')!r})")
        ep, sl = row.get("entry_price"), row.get("stop_loss")
        if ep and sl and row.get("direction") == "long" and sl >= ep:
            problems.append("LONG stop is not below entry")
        if ep and sl and row.get("direction") == "short" and sl <= ep:
            problems.append("SHORT stop is not above entry")
        return row, problems

    # -- pending orders (next-open fill model) ---------------------------- #

    def create_pending_order(self, feed_key: str, signal_ts: int, direction: str,
                             params: Dict[str, Any],
                             conn: Optional[sqlite3.Connection] = None) -> Optional[str]:
        """
        Queue an order to fill at the next candle's open.

        Returns the order_id, or None when an identical pending order already
        exists (UNIQUE(feed_key, signal_ts, direction)) - the duplicate guard.
        """
        self.ensure()
        order_id = f"O{uuid.uuid4().hex[:12].upper()}"
        sql = ("INSERT OR IGNORE INTO pending_orders"
               "(order_id, feed_key, signal_ts, direction, created_at, status, params) "
               "VALUES(?,?,?,?,?,?,?)")
        args = (order_id, feed_key, int(signal_ts), direction,
                int(time.time() * 1000), PENDING, json.dumps(params))
        if conn is not None:
            cur = conn.execute(sql, args)
        else:
            with self._conn(commit=True) as c:
                cur = c.execute(sql, args)
        return order_id if (cur.rowcount or 0) > 0 else None

    def get_pending_orders(self, feed_key: Optional[str] = None) -> List[Dict[str, Any]]:
        self.ensure()
        sql = "SELECT * FROM pending_orders WHERE status=?"
        args: List[Any] = [PENDING]
        if feed_key:
            sql += " AND feed_key=?"
            args.append(feed_key)
        sql += " ORDER BY signal_ts ASC"
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["params"] = json.loads(d["params"]) if d.get("params") else {}
            except (json.JSONDecodeError, TypeError):
                d["params"] = {}
            out.append(d)
        return out

    def set_pending_status(self, order_id: str, status: str,
                           conn: Optional[sqlite3.Connection] = None) -> None:
        self.ensure()
        sql = "UPDATE pending_orders SET status=? WHERE order_id=?"
        if conn is not None:
            conn.execute(sql, (status, order_id))
        else:
            with self._conn(commit=True) as c:
                c.execute(sql, (status, order_id))

    def delete_pending_order(self, order_id: str) -> None:
        self.ensure()
        with self._conn(commit=True) as conn:
            conn.execute("DELETE FROM pending_orders WHERE order_id=?", (order_id,))

    # -- backtests -------------------------------------------------------- #

    def save_backtest(self, run: Dict[str, Any], trades: Iterable[Dict[str, Any]]) -> str:
        """Persist a backtest.  NEVER touches the paper_trades table."""
        self.ensure()
        run_id = f"B{uuid.uuid4().hex[:12].upper()}"
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO backtest_runs
                   (run_id, created_at, exchange, symbol, timeframe, candles,
                    config, metrics, n_trades, net_pnl, warnings)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, int(time.time() * 1000), run.get("exchange"), run.get("symbol"),
                 run.get("timeframe"), int(run.get("candles") or 0),
                 json.dumps(run.get("config") or {}), json.dumps(run.get("metrics") or {}),
                 int(run.get("n_trades") or 0), float(run.get("net_pnl") or 0.0),
                 json.dumps(run.get("warnings") or [])),
            )
            for i, t in enumerate(trades):
                conn.execute(
                    """INSERT INTO backtest_trades
                       (run_id, trade_no, direction, signal_ts, entry_ts, entry_price,
                        quantity, stop_loss, take_profit, exit_ts, exit_price,
                        gross_pnl, net_pnl, realized_r, exit_reason, balance)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (run_id, i + 1, t.get("direction"), t.get("signal_ts"), t.get("entry_ts"),
                     t.get("entry_price"), t.get("quantity"), t.get("stop_loss"),
                     t.get("take_profit"), t.get("exit_ts"), t.get("exit_price"),
                     t.get("gross_pnl"), t.get("net_pnl"), t.get("realized_r"),
                     t.get("exit_reason"), t.get("balance")),
                )
        return run_id

    def list_backtests(self, limit: int = 20) -> List[Dict[str, Any]]:
        self.ensure()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, run_id, created_at, exchange, symbol, timeframe, candles, "
                "n_trades, net_pnl FROM backtest_runs ORDER BY created_at DESC LIMIT ?",
                (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def get_backtest_trades(self, run_id: str) -> List[Dict[str, Any]]:
        self.ensure()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM backtest_trades WHERE run_id=? ORDER BY trade_no", (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_backtest(self, run_id: str) -> None:
        self.ensure()
        with self.transaction() as conn:
            conn.execute("DELETE FROM backtest_trades WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM backtest_runs WHERE run_id=?", (run_id,))

    # -- reset ------------------------------------------------------------ #

    def reset_paper_account(self, confirm_token: str) -> Dict[str, int]:
        """
        Delete paper trades, events, processed candles and pending orders.

        Requires the exact confirmation token.  Backtest history is preserved.
        """
        if confirm_token != RESET_CONFIRM_TOKEN:
            raise DatabaseError(
                f"Reset refused: confirmation token must be exactly '{RESET_CONFIRM_TOKEN}'."
            )
        self.ensure()
        counts: Dict[str, int] = {}
        with self.transaction() as conn:
            for table in ("paper_trades", "bot_events", "processed_candles", "pending_orders"):
                cur = conn.execute(f"SELECT COUNT(*) AS c FROM {table}")
                counts[table] = int(cur.fetchone()["c"])
                conn.execute(f"DELETE FROM {table}")
            conn.execute("DELETE FROM sqlite_sequence WHERE name IN "
                         "('paper_trades','bot_events','processed_candles','pending_orders')")
        log.warning("Paper account reset by user. Deleted rows: %s", counts)
        return counts

    def vacuum(self) -> None:
        with self._conn(commit=True) as conn:
            conn.execute("VACUUM")

    def stats(self) -> Dict[str, Any]:
        """Diagnostics payload."""
        self.ensure()
        with self._conn() as conn:
            size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
            counts = {t: int(conn.execute(f"SELECT COUNT(*) AS c FROM {t}").fetchone()["c"])
                      for t in _SCHEMA}
        return {"path": os.path.abspath(self.path), "bytes": size, "rows": counts}


__all__ = [
    "Database", "DatabaseError", "DEFAULT_DB_PATH", "RESET_CONFIRM_TOKEN",
    "STATUS_OPEN", "STATUS_CLOSED", "STATUS_CANCELLED", "VALID_STATUSES",
    "PENDING", "FILLED", "EXPIRED",
    "utc_midnight_ms", "utc_date_str", "ms_to_utc",
]
