"""The storage seam — a repository interface with a SQLite implementation.

This is the production upgrade over the original flat db.py. WHY introduce an
interface: it's the single seam that makes the storage engine swappable. The rest
of the app depends on the ABSTRACT TicketRepository, never on SQLite directly, so
moving to Postgres is "write a PostgresRepository and point the factory at it" —
no handler, no graph, no test changes. (db.py is kept as a thin facade so existing
call sites keep working — strangler-fig, not a rewrite.)

Two correctness fixes over the original live here:
  - create_ticket is now race-free: it INSERTs and retries on a duplicate-key
    violation, instead of the old check-then-insert (which had a TOCTOU window
    where two concurrent requests could pick the same id and silently drop one).
  - every create and status change writes an immutable audit event in the SAME
    transaction as the change, so the history can't drift from the data.
"""

import os
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from secrets import randbelow

import config
import migrations
from config import DEFAULT_PRIORITY, SLA_HOURS, STATUS_OPEN, STATUS_RESOLVED, TICKET_STATUSES


def _now() -> str:
    # INVARIANT: every timestamp in this layer is UTC ISO-8601 to second precision,
    # produced here. That uniformity is what makes the lexicographic string compare
    # in mark_overdue_sla (sla_due_at < now) correct — same zone, same format, so
    # string order == chronological order. Don't write timestamps any other way.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sla_due(priority: str) -> str:
    """First-response SLA deadline = now + the per-priority hour budget."""
    hours = SLA_HOURS.get(priority, SLA_HOURS[DEFAULT_PRIORITY])
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat(timespec="seconds")


class TicketRepository(ABC):
    """The storage contract. Any backend (SQLite, Postgres, …) implements this."""

    @abstractmethod
    def create_ticket(self, customer_id: str, issue: str, customer_name: str | None = None,
                      priority: str = DEFAULT_PRIORITY, reason: str | None = None) -> str | None: ...

    @abstractmethod
    def get_ticket(self, ticket_id: str, customer_id: str | None = None) -> dict | None: ...

    @abstractmethod
    def list_tickets(self, customer_id: str | None = None, limit: int = 50) -> list[dict]: ...

    @abstractmethod
    def update_status(self, ticket_id: str, new_status: str, actor: str,
                     customer_id: str, note: str | None = None) -> bool: ...

    @abstractmethod
    def get_events(self, ticket_id: str, customer_id: str) -> list[dict]: ...

    @abstractmethod
    def mark_overdue_sla(self) -> int: ...

    @abstractmethod
    def load_tickets(self, rows: list[dict]) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...


class SQLiteRepository(TicketRepository):
    """SQLite-backed repository. Zero-setup, and shaped exactly like the Postgres
    implementation would be — same method bodies, only the driver/SQL dialect differ."""

    _MAX_ID_ATTEMPTS = 20

    def __init__(self, db_path: str):
        self._path = db_path
        self._ensure_schema()

    # -- connection + schema --------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        # WAL lets readers and a writer proceed concurrently (default rollback mode
        # blocks readers during a write); busy_timeout makes a contended call WAIT
        # up to 5s instead of immediately raising "database is locked".
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @contextmanager
    def _tx(self):
        """A connection scoped to one operation: commits on success, rolls back on
        error (via `with conn`), and ALWAYS closes (the plain `with conn` idiom
        commits but leaks the connection/fd — a real problem under Streamlit's
        across-thread reruns)."""
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._tx() as conn:
            migrations.migrate(conn)

    @staticmethod
    def _new_candidate_id() -> str:
        # 6-digit, CSPRNG (not predictable). Uniqueness is enforced by the PRIMARY
        # KEY + insert-retry below, NOT by a pre-check — that's the race fix.
        return f"{randbelow(900_000) + 100_000}"

    # -- writes ---------------------------------------------------------------
    def create_ticket(self, customer_id, issue, customer_name=None,
                      priority=DEFAULT_PRIORITY, reason=None):
        # `reason` is the business context for the audit trail; the CALLER owns it
        # (the storage layer shouldn't know about "negative feedback").
        ts = _now()
        sla_due = _sla_due(priority)
        for _ in range(self._MAX_ID_ATTEMPTS):
            ticket_id = self._new_candidate_id()
            try:
                with self._tx() as conn:
                    conn.execute(
                        """INSERT INTO support_tickets
                           (ticket_id, customer_id, customer_name, issue, status,
                            priority, created_at, updated_at, sla_due_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (ticket_id, customer_id, customer_name, issue, STATUS_OPEN,
                         priority, ts, ts, sla_due),
                    )
                    # Audit event in the same transaction as the insert.
                    conn.execute(
                        """INSERT INTO ticket_events
                           (ticket_id, event_type, from_status, to_status, actor, note, created_at)
                           VALUES (?, 'created', NULL, ?, ?, ?, ?)""",
                        (ticket_id, STATUS_OPEN, customer_id, reason, ts),
                    )
                return ticket_id
            except sqlite3.IntegrityError as exc:
                # Retry ONLY on a ticket_id UNIQUE/PK collision (the race-free path).
                # SQLite reports it as "UNIQUE constraint failed: support_tickets.
                # ticket_id". Any other integrity error (e.g. NOT NULL) is a real bug —
                # don't burn 20 id allocations masking it; fail fast.
                msg = str(exc).lower()
                if "unique constraint" in msg and "ticket_id" in msg:
                    continue
                return None
            except sqlite3.OperationalError as exc:
                # Transient lock contention ("database is locked"/"busy") — retry
                # rather than silently dropping a customer's complaint. busy_timeout
                # already waited; a fresh attempt usually wins. Other operational
                # errors fail fast.
                msg = str(exc).lower()
                if "locked" in msg or "busy" in msg:
                    continue
                return None
            except Exception:
                return None
        return None  # couldn't allocate a free id (table effectively full)

    def update_status(self, ticket_id, new_status, actor, customer_id, note=None):
        if new_status not in TICKET_STATUSES:
            return False
        try:
            with self._tx() as conn:
                # Scope to the owning customer: a status change for a ticket that
                # isn't theirs returns False (same as not-found), never mutates it.
                row = conn.execute(
                    "SELECT status FROM support_tickets WHERE ticket_id = ? AND customer_id = ?",
                    (ticket_id, customer_id),
                ).fetchone()
                if row is None:
                    return False
                from_status = row["status"]
                if from_status == new_status:
                    return True  # no-op, but not a failure
                ts = _now()
                # Resolving a ticket clears any SLA-breach flag — a breach is about
                # the OPEN time, so a resolved ticket shouldn't keep showing the
                # warning badge (the flag was otherwise a one-way latch).
                clear_breach = 1 if new_status == STATUS_RESOLVED else 0
                # Conditional transition: the UPDATE only fires while the row is STILL at
                # `from_status` (the value we read). If a concurrent writer changed it
                # first, rowcount is 0 and we abort WITHOUT writing an audit row — so the
                # ticket_events chain can never record a from_status we didn't transition
                # from. (Same race-free discipline as create_ticket's conditional insert.)
                cur = conn.execute(
                    "UPDATE support_tickets SET status = ?, updated_at = ?, "
                    "sla_breached = CASE WHEN ? = 1 THEN 0 ELSE sla_breached END "
                    "WHERE ticket_id = ? AND customer_id = ? AND status = ?",
                    (new_status, ts, clear_breach, ticket_id, customer_id, from_status),
                )
                if cur.rowcount != 1:
                    return False  # status changed under us between read and write
                conn.execute(
                    """INSERT INTO ticket_events
                       (ticket_id, event_type, from_status, to_status, actor, note, created_at)
                       VALUES (?, 'status_change', ?, ?, ?, ?, ?)""",
                    (ticket_id, from_status, new_status, actor, note, ts),
                )
            return True
        except Exception:
            return False

    def load_tickets(self, rows):
        ts = _now()
        with self._tx() as conn:
            for r in rows:
                if r["status"] not in TICKET_STATUSES:
                    raise ValueError(f"invalid seed status: {r['status']!r}")
                conn.execute(
                    """INSERT OR REPLACE INTO support_tickets
                       (ticket_id, customer_id, customer_name, issue, status, priority,
                        created_at, updated_at, sla_due_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (r["ticket_id"], r["customer_id"], r.get("customer_name"), r["issue"],
                     r["status"], r.get("priority", DEFAULT_PRIORITY), ts, ts, r.get("sla_due_at")),
                )

    # -- reads ----------------------------------------------------------------
    def get_ticket(self, ticket_id, customer_id=None):
        try:
            with self._tx() as conn:
                if customer_id is not None:
                    row = conn.execute(
                        "SELECT * FROM support_tickets WHERE ticket_id = ? AND customer_id = ?",
                        (ticket_id, customer_id),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT * FROM support_tickets WHERE ticket_id = ?", (ticket_id,)
                    ).fetchone()
                return dict(row) if row else None
        except Exception:
            return None

    def list_tickets(self, customer_id=None, limit=50):
        try:
            with self._tx() as conn:
                if customer_id is not None:
                    rows = conn.execute(
                        "SELECT * FROM support_tickets WHERE customer_id = ? "
                        "ORDER BY created_at DESC LIMIT ?",
                        (customer_id, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM support_tickets ORDER BY created_at DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def get_events(self, ticket_id, customer_id):
        try:
            with self._tx() as conn:
                # Ownership check first — the audit trail (actor identities, note
                # text) must not leak across customers. Not yours => empty.
                owner = conn.execute(
                    "SELECT 1 FROM support_tickets WHERE ticket_id = ? AND customer_id = ?",
                    (ticket_id, customer_id),
                ).fetchone()
                if owner is None:
                    return []
                rows = conn.execute(
                    "SELECT * FROM ticket_events WHERE ticket_id = ? ORDER BY event_id ASC",
                    (ticket_id,),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def mark_overdue_sla(self) -> int:
        """Flag unresolved tickets whose SLA deadline has passed. Returns count newly
        breached. This is the job a scheduler would run every few minutes."""
        try:
            with self._tx() as conn:
                cur = conn.execute(
                    """UPDATE support_tickets SET sla_breached = 1
                       WHERE status != ? AND sla_breached = 0
                         AND sla_due_at IS NOT NULL AND sla_due_at < ?""",
                    (STATUS_RESOLVED, _now()),
                )
                return cur.rowcount
        except Exception:
            return 0

    def reset(self):
        if os.path.exists(self._path):
            os.remove(self._path)
        self._ensure_schema()


def build_repository() -> TicketRepository:
    """Factory: choose a backend from configuration.

    DATABASE_URL switches engines. Today only SQLite is implemented; a
    postgres:// URL is where a PostgresRepository (same interface, psycopg driver)
    plugs in. The factory is the ONLY place that knows which backend is live.
    """
    if config.DATABASE_URL and config.DATABASE_URL.startswith(("postgres://", "postgresql://")):
        raise NotImplementedError(
            "PostgresRepository isn't wired yet. Implement TicketRepository against "
            "psycopg using the same method bodies as SQLiteRepository (the SQL is "
            "already portable) and return it here. This is the swap point."
        )
    return SQLiteRepository(config.DB_PATH)


# Process-wide singleton so the schema is migrated once and connections are cheap.
_repo: TicketRepository | None = None


def get_repository() -> TicketRepository:
    global _repo
    if _repo is None:
        _repo = build_repository()
    return _repo


def reset_repository_singleton() -> None:
    """Drop the cached repo so the next call rebuilds it (used by tests that
    monkeypatch the DB path)."""
    global _repo
    _repo = None
