"""Database layer: id generation, lookups, and read-scoping (the IDOR mitigation)."""

import re

import db
from seed_data import DEMO_CUSTOMER_ID, OTHER_CUSTOMER_ID


def test_create_ticket_returns_six_digit_id():
    tid = db.create_ticket(customer_id=DEMO_CUSTOMER_ID, issue="card not working")
    assert tid is not None
    assert re.fullmatch(r"\d{6}", tid), f"expected a 6-digit id, got {tid!r}"


def test_created_ticket_starts_open_and_is_readable():
    tid = db.create_ticket(customer_id=DEMO_CUSTOMER_ID, issue="app crash")
    ticket = db.get_ticket(tid, customer_id=DEMO_CUSTOMER_ID)
    assert ticket is not None
    assert ticket["status"] == "Open"
    assert ticket["issue"] == "app crash"


def test_get_ticket_found_returns_seeded_status():
    ticket = db.get_ticket("650932", customer_id=DEMO_CUSTOMER_ID)
    assert ticket is not None
    assert ticket["status"] == "Resolved"


def test_get_ticket_not_found_returns_none():
    assert db.get_ticket("000000", customer_id=DEMO_CUSTOMER_ID) is None


def test_read_is_scoped_to_owning_customer():
    # 940011 belongs to OTHER_CUSTOMER_ID. The demo customer must NOT see it —
    # this is the IDOR mitigation: you can't read a ticket that isn't yours.
    assert db.get_ticket("940011", customer_id=DEMO_CUSTOMER_ID) is None
    assert db.get_ticket("940011", customer_id=OTHER_CUSTOMER_ID) is not None


def test_ticket_ids_are_unique():
    ids = {db.create_ticket(customer_id=DEMO_CUSTOMER_ID, issue=f"issue {i}") for i in range(50)}
    assert None not in ids
    assert len(ids) == 50  # no collisions
    assert all(re.fullmatch(r"\d{6}", t) for t in ids)


def test_list_tickets_scoped_to_customer():
    demo = db.list_tickets(customer_id=DEMO_CUSTOMER_ID)
    assert all(t["customer_id"] == DEMO_CUSTOMER_ID for t in demo)
    assert "940011" not in {t["ticket_id"] for t in demo}
