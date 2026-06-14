"""Ticket lifecycle + audit trail (the production correctness additions)."""

import db
from seed_data import DEMO_CUSTOMER_ID, OTHER_CUSTOMER_ID

CID = DEMO_CUSTOMER_ID


def test_create_writes_a_created_event():
    tid = db.create_ticket(customer_id=CID, issue="card stuck")
    events = db.get_events(tid, CID)
    assert len(events) == 1
    assert events[0]["event_type"] == "created"
    assert events[0]["to_status"] == "Open"


def test_update_status_moves_and_audits():
    tid = db.create_ticket(customer_id=CID, issue="dispute")
    assert db.update_status(tid, "In Progress", actor="agent:jane", customer_id=CID, note="picked up") is True
    assert db.update_status(tid, "Resolved", actor="agent:jane", customer_id=CID) is True

    assert db.get_ticket(tid, CID)["status"] == "Resolved"
    events = db.get_events(tid, CID)
    # created + two status changes
    assert [e["event_type"] for e in events] == ["created", "status_change", "status_change"]
    assert events[1]["from_status"] == "Open" and events[1]["to_status"] == "In Progress"
    assert events[2]["to_status"] == "Resolved"
    assert events[1]["actor"] == "agent:jane"


def test_invalid_status_rejected():
    tid = db.create_ticket(customer_id=CID, issue="x")
    assert db.update_status(tid, "Bogus", actor="agent:x", customer_id=CID) is False


def test_update_nonexistent_ticket_returns_false():
    assert db.update_status("000000", "Resolved", actor="agent:x", customer_id=CID) is False


def test_same_status_is_noop_success_without_extra_event():
    tid = db.create_ticket(customer_id=CID, issue="x")
    assert db.update_status(tid, "Open", actor="agent:x", customer_id=CID) is True
    assert len(db.get_events(tid, CID)) == 1  # still just the 'created' event


def test_ticket_ids_unique_under_many_creates():
    # Exercises the race-free insert-retry path (no check-then-insert window).
    ids = {db.create_ticket(customer_id=CID, issue=f"i{i}") for i in range(60)}
    assert None not in ids and len(ids) == 60


# --- customer scoping (the IDOR fix) ----------------------------------------
def test_cannot_update_another_customers_ticket():
    # 650932 belongs to DEMO_CUSTOMER_ID; OTHER must not be able to move it.
    assert db.update_status("650932", "Open", actor="agent:evil", customer_id=OTHER_CUSTOMER_ID) is False
    # and it really wasn't changed
    assert db.get_ticket("650932", CID)["status"] == "Resolved"


def test_cannot_read_another_customers_audit_trail():
    # A ticket owned by CID with a real audit event...
    tid = db.create_ticket(customer_id=CID, issue="dispute")
    db.update_status(tid, "Resolved", actor="agent:jane", customer_id=CID, note="sensitive note")
    # ...is fully visible to its owner...
    assert len(db.get_events(tid, CID)) == 2  # created + status_change
    # ...but leaks nothing (actor identities, note text) to another customer.
    assert db.get_events(tid, OTHER_CUSTOMER_ID) == []
