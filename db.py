"""Thin facade over the storage repository.

Originally db.py held the SQLite code directly. That logic now lives behind the
TicketRepository interface in repository.py (so the engine is swappable). This
module stays as a stable, function-style API — `db.create_ticket(...)`,
`db.get_ticket(...)` — so existing call sites (handlers, seed_data, tests) didn't
have to change when the storage layer was refactored. Strangler-fig: the interface
grew underneath without a rewrite at the call sites.

Every function here just delegates to the configured repository.
"""

from config import DB_PATH, DEFAULT_PRIORITY  # DB_PATH re-exported for db.DB_PATH callers
from repository import get_repository


def init_db() -> None:
    """Ensure the database exists and is migrated to the latest schema version."""
    get_repository()  # construction runs migrations


def create_ticket(customer_id: str, issue: str, customer_name: str | None = None,
                  priority: str = DEFAULT_PRIORITY, reason: str | None = None) -> str | None:
    return get_repository().create_ticket(customer_id, issue, customer_name, priority, reason)


def get_ticket(ticket_id: str, customer_id: str | None = None) -> dict | None:
    return get_repository().get_ticket(ticket_id, customer_id)


def list_tickets(customer_id: str | None = None, limit: int = 50) -> list[dict]:
    return get_repository().list_tickets(customer_id, limit)


def update_status(ticket_id: str, new_status: str, actor: str, customer_id: str,
                  note: str | None = None) -> bool:
    """Move a ticket to a new status (scoped to the owning customer), writing an
    audit event. Returns success; False if the ticket isn't this customer's."""
    return get_repository().update_status(ticket_id, new_status, actor, customer_id, note)


def get_events(ticket_id: str, customer_id: str) -> list[dict]:
    """Return the audit trail for a ticket, but only if it belongs to customer_id."""
    return get_repository().get_events(ticket_id, customer_id)


def mark_overdue_sla() -> int:
    """Flag unresolved tickets past their SLA deadline; return count newly breached."""
    return get_repository().mark_overdue_sla()


def load_tickets(rows: list[dict]) -> None:
    """Seeding helper: insert tickets with explicit ids/statuses."""
    get_repository().load_tickets(rows)


def reset_db() -> None:
    """Drop and recreate the database (tests + seed_data use this for a clean slate)."""
    get_repository().reset()
