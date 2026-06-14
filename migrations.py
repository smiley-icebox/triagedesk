"""Schema migrations — versioned, forward-only.

WHY this exists (the production upgrade over a bare CREATE TABLE): a real system's
schema changes over time, and you can't drop the table to add a column — there's
live data in it. Migrations record which schema version a database is at and apply
only the steps it hasn't seen yet. That's how you evolve a production schema safely.

Each migration is (version, list-of-SQL-statements). `migrate()` applies every
migration whose version is greater than the database's current recorded version,
inside a transaction, then bumps the version. Idempotent: running it on an
up-to-date database is a no-op.

The SQL is kept portable (no SQLite-only syntax) so the same migrations run against
Postgres once a PostgresRepository is wired in — the whole point of the repository
seam in repository.py.
"""

import sqlite3

MIGRATIONS: list[tuple[int, list[str]]] = [
    (
        1,
        [
            # The tickets table — the single source of truth for ticket facts.
            """
            CREATE TABLE IF NOT EXISTS support_tickets (
                ticket_id     TEXT PRIMARY KEY,
                customer_id   TEXT NOT NULL,
                customer_name TEXT,
                issue         TEXT NOT NULL,
                status        TEXT NOT NULL,
                priority      TEXT NOT NULL DEFAULT 'normal',
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            )
            """,
            # Index the column we filter reads by (customer scoping).
            "CREATE INDEX IF NOT EXISTS idx_tickets_customer ON support_tickets (customer_id)",
            # The audit trail: every status change is an immutable event. This is
            # what gives the system a defensible history ("who moved this to
            # Resolved, and when?") — a hard requirement in a regulated domain.
            """
            CREATE TABLE IF NOT EXISTS ticket_events (
                event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id   TEXT NOT NULL,
                event_type  TEXT NOT NULL,      -- 'created' | 'status_change'
                from_status TEXT,
                to_status   TEXT,
                actor       TEXT NOT NULL,       -- who caused it (system/agent/customer id)
                note        TEXT,
                created_at  TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_events_ticket ON ticket_events (ticket_id)",
        ],
    ),
    (
        2,
        [
            # SLA tracking (added in a later wave). Demonstrates the migration system
            # doing its job: evolving a live schema additively, version by version.
            "ALTER TABLE support_tickets ADD COLUMN sla_due_at TEXT",
            "ALTER TABLE support_tickets ADD COLUMN sla_breached INTEGER NOT NULL DEFAULT 0",
        ],
    ),
]

CURRENT_VERSION = MIGRATIONS[-1][0]


def _current_version(conn) -> int:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
    )
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    v = row["v"] if isinstance(row, dict) or hasattr(row, "keys") else row[0]
    return v or 0


def migrate(conn) -> int:
    """Apply all pending migrations to an open connection. Returns the new version.

    Takes an open connection (not a path) so the repository owns connection
    lifecycle and so this works against any DB-API-compatible engine.

    CRASH-SAFE: each statement is applied idempotently — if a prior partial apply
    already made the change (SQLite auto-commits DDL, so a crash between an
    ALTER and the version bump is possible), the "duplicate column"/"already
    exists" error is swallowed and we move on, instead of bricking every future
    boot. The version is recorded only after all of a migration's statements land.
    """
    current = _current_version(conn)
    for version, statements in MIGRATIONS:
        if version <= current:
            continue
        for sql in statements:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError as exc:
                m = str(exc).lower()
                if "duplicate column" in m or "already exists" in m:
                    continue  # this step was applied by an earlier interrupted run
                raise
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        current = version
    conn.commit()
    return current
