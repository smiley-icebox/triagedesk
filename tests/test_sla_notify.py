"""SLA breach tracking + the notifier abstraction."""

import notifier
import db
from seed_data import DEMO_CUSTOMER_ID


# --- SLA ---------------------------------------------------------------------
def test_new_ticket_gets_future_sla_due():
    tid = db.create_ticket(customer_id=DEMO_CUSTOMER_ID, issue="x")
    t = db.get_ticket(tid, DEMO_CUSTOMER_ID)
    assert t["sla_due_at"] is not None
    assert t["sla_breached"] == 0  # just created, not overdue


def test_sla_check_flags_overdue_seeded_ticket():
    # 781205 is seeded with a deadline in 2020 -> must be flagged breached.
    assert db.get_ticket("781205", DEMO_CUSTOMER_ID)["sla_breached"] == 0
    breached = db.mark_overdue_sla()
    assert breached >= 1
    assert db.get_ticket("781205", DEMO_CUSTOMER_ID)["sla_breached"] == 1


def test_resolved_tickets_are_not_breached():
    # 650932 is Resolved -> never counts as an SLA breach even if past due.
    db.mark_overdue_sla()
    assert db.get_ticket("650932", DEMO_CUSTOMER_ID)["sla_breached"] == 0


# --- notifier (paths isolated to tmp by the conftest fixture) ----------------
def test_notify_ticket_created_writes_a_notification():
    assert notifier.notify_ticket_created("Jordan", "cust_1001", "240716") is True
    notes = notifier.read_recent()
    assert len(notes) == 1
    assert "240716" in notes[0]["subject"]
    assert notes[0]["recipient"] == "cust_1001"
