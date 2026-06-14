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
