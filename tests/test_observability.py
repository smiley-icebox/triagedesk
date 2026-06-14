"""Observability: PII redaction + metrics aggregation."""

import observability


# --- redaction ---------------------------------------------------------------
def test_redact_masks_card_numbers():
    out = observability.redact("my card 4111 1111 1111 1111 was charged")
    assert "4111" not in out
    assert "[redacted-number]" in out


def test_redact_masks_emails():
    assert observability.redact("reach me at jordan@example.com") == "reach me at [email]"


def test_redact_keeps_six_digit_ticket_ids():
    # Ticket ids are references, not PII — they must survive redaction.
    assert "650932" in observability.redact("status of ticket 650932?")


def test_redact_masks_account_numbers_but_keeps_ticket_id():
    out = observability.redact("debit from account 12345678 about ticket 650932")
    assert "12345678" not in out          # 8-digit account masked
    assert "650932" in out                # 6-digit ticket id preserved


def test_record_redacts_before_writing():
    observability.clear()
    rec = observability.record(
        message="card 4111111111111111 / email a@b.com",
        label="negative_feedback", confidence=0.9, route="negative_feedback",
        db_action="created #240716", ticket_id="240716", latency_ms=12.0, source="llm",
    )
    assert "4111111111111111" not in rec["message"]
    assert "[email]" in rec["message"]


# --- metrics -----------------------------------------------------------------
def test_metrics_aggregate_routes_and_degraded():
    observability.clear()
    common = dict(message="x", label="query", confidence=0.9, db_action="-",
                  ticket_id=None, latency_ms=10.0)
    observability.record(route="query", source="llm", **common)
    observability.record(route="escalate", source="guardrail", **common)
    observability.record(route="positive_feedback", source="heuristic", **common)

    mx = observability.metrics()
    assert mx["total"] == 3
    assert mx["escalated"] == 1
    assert mx["auto_handled"] == 2
    assert mx["degraded"] == 2  # guardrail + heuristic
    assert mx["by_route"]["query"] == 1
