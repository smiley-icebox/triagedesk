"""Handlers: ticket-id extraction + each deterministic response path."""

import db
from handlers import (
    extract_ticket_id,
    handle_escalate,
    handle_negative,
    handle_positive,
    handle_query,
)
from seed_data import DEMO_CUSTOMER_ID


# --- extract_ticket_id -------------------------------------------------------
def test_extract_six_digit():
    assert extract_ticket_id("status of ticket 650932 please") == "650932"


def test_extract_with_hash():
    assert extract_ticket_id("update on #100428?") == "100428"


def test_extract_none_when_no_number():
    assert extract_ticket_id("where does my request stand?") is None


def test_extract_ignores_short_numbers():
    # 3-digit numbers aren't ticket ids (avoids matching "call 911").
    assert extract_ticket_id("call 911 if urgent") is None


# --- handlers ----------------------------------------------------------------
def _state(message, **extra):
    base = {"message": message, "customer_id": DEMO_CUSTOMER_ID, "customer_name": "Jordan"}
    base.update(extra)
    return base


def test_positive_uses_name_template():
    out = handle_positive(_state("thank you!"))
    assert "Jordan" in out["response"]
    assert "Thank you" in out["response"]
    assert out["db_action"] == "none"


def test_negative_creates_ticket_and_cites_it():
    out = handle_negative(_state("my card never arrived"))
    assert out["ticket_id"] is not None
    assert out["ticket_id"] in out["response"]
    # the ticket really exists in the DB now
    assert db.get_ticket(out["ticket_id"], customer_id=DEMO_CUSTOMER_ID) is not None
    assert out["db_action"].startswith("created")


def test_query_found_reports_db_status():
    out = handle_query(_state("status of ticket 650932?"))
    assert "Resolved" in out["response"]  # the seeded status, read from the DB
    assert out["ticket_id"] == "650932"


def test_query_not_found_is_graceful():
    out = handle_query(_state("status of ticket 000000?"))
    assert "couldn't find" in out["response"].lower()


def test_query_other_customers_ticket_is_not_found():
    # 940011 belongs to another customer -> handler must not reveal it.
    out = handle_query(_state("any update on 940011?"))
    assert "couldn't find" in out["response"].lower()


def test_query_without_id_asks_for_one():
    out = handle_query(_state("what's happening with my ticket?"))
    assert "ticket number" in out["response"].lower()


def test_escalate_returns_handoff_message():
    out = handle_escalate(_state("asdf qwer"))
    assert "team" in out["response"].lower()
    assert out["db_action"] == "none"
