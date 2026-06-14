"""Integration tests: the COMPILED graph with the real classifier seam, the RAG
route end-to-end, multi-turn folding, degraded paths, and migration idempotency.

These close the gap the unit tests left — they prove the pieces work *together*
through `build_graph()`/`respond()`, not just in isolation. All run offline.
"""

import sqlite3
from collections import namedtuple

import classifier
import db
import graph as G
import handlers
import migrations
from seed_data import DEMO_CUSTOMER_ID

Fake = namedtuple("Fake", "label confidence")


class _Boom:
    def invoke(self, *a, **k):
        raise RuntimeError("API down")


def test_graph_with_real_classify_heuristic_fallback_routes():
    # Real classifier.classify driven through the compiled graph, but with a runnable
    # that errors — exercises the keyword FALLBACK and deterministic routing together.
    g = G.build_graph(classify_fn=lambda m: classifier.classify(m, _runnable=_Boom()))
    final = G.respond("My card is broken and it's wrong",
                      customer_id=DEMO_CUSTOMER_ID, customer_name="Jordan", graph=g)
    assert final["route"] == "negative_feedback"
    assert final["source"] == "heuristic"
    assert final["ticket_id"] is not None  # a ticket was really opened


def test_general_query_route_end_to_end():
    g = G.build_graph(classify_fn=lambda _t: Fake("general_query", 0.95))
    final = G.respond("What are your ATM withdrawal limits?",
                      customer_id=DEMO_CUSTOMER_ID, customer_name="Jordan", graph=g)
    assert final["route"] == "general_query"
    assert "500" in final["response"]   # grounded extractive answer (LLM off in tests)
    assert "kb:" in final["db_action"]


def test_multiturn_history_is_folded_into_classifier_input():
    captured = {}

    def capturing(text):
        captured["text"] = text
        return Fake("query", 0.9)

    g = G.build_graph(classify_fn=capturing)
    G.respond("any update?", customer_id=DEMO_CUSTOMER_ID, customer_name="Jordan",
              history=[{"role": "user", "content": "status of ticket 100428"}], graph=g)
    assert "Conversation so far" in captured["text"]
    assert "100428" in captured["text"]
    assert "any update?" in captured["text"]


def test_handle_negative_degrades_when_create_fails(monkeypatch):
    monkeypatch.setattr(db, "create_ticket", lambda **k: None)
    out = handlers.handle_negative(
        {"message": "broken", "customer_id": DEMO_CUSTOMER_ID, "customer_name": "Jordan"}
    )
    assert out["db_action"] == "create_failed"
    assert out.get("ticket_id") is None


def test_migrations_are_idempotent():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    v1 = migrations.migrate(conn)
    v2 = migrations.migrate(conn)  # second run must be a no-op, not an error
    assert v1 == v2 == migrations.CURRENT_VERSION
    conn.close()


def test_migration_v1_to_v2_preserves_existing_data():
    # Apply only v1, insert a row, then migrate to v2 — the row must survive and the
    # new columns default sensibly (the real upgrade-on-live-data path).
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    for sql in migrations.MIGRATIONS[0][1]:
        conn.execute(sql)
    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
    conn.execute(
        "INSERT INTO support_tickets (ticket_id, customer_id, issue, status, priority,"
        " created_at, updated_at) VALUES ('650932','c1','x','Open','normal','t','t')"
    )
    conn.commit()
    assert migrations.migrate(conn) == migrations.CURRENT_VERSION
    row = conn.execute("SELECT * FROM support_tickets WHERE ticket_id='650932'").fetchone()
    assert row is not None and row["sla_due_at"] is None and row["sla_breached"] == 0
    conn.close()


def test_migration_crash_recovery_swallows_duplicate_column():
    # Simulate a partial/interrupted apply: columns already exist but the version
    # wasn't recorded. Re-running migrate() must NOT crash on "duplicate column".
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrations.migrate(conn)                 # fully migrate (columns now exist)
    conn.execute("DELETE FROM schema_version")  # forget we did it
    conn.commit()
    assert migrations.migrate(conn) == migrations.CURRENT_VERSION  # no crash
    conn.close()


def test_create_ticket_fast_fails_on_not_null_violation():
    # customer_id=None violates NOT NULL — an IntegrityError that is NOT a ticket_id
    # collision, so create_ticket returns None immediately rather than masking the bug.
    assert db.create_ticket(customer_id=None, issue="x") is None


def test_respond_redacts_pii_into_the_trace():
    import observability
    observability.clear()
    g = G.build_graph(classify_fn=lambda _t: Fake("positive_feedback", 0.95))
    G.respond("thanks! my email is jordan@example.com", customer_id=DEMO_CUSTOMER_ID,
              customer_name="Jordan", graph=g)
    rec = observability.read_recent(1)[0]
    assert "jordan@example.com" not in rec["message"]
    assert "[email]" in rec["message"]
