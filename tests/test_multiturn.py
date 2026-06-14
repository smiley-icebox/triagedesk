"""Multi-turn context: follow-ups resolve against earlier conversation."""

from collections import namedtuple

from graph import build_graph, respond
from handlers import handle_query
from seed_data import DEMO_CUSTOMER_ID

Fake = namedtuple("Fake", "label confidence")


def test_query_resolves_ticket_id_from_history():
    state = {
        "message": "any update?",
        "customer_id": DEMO_CUSTOMER_ID,
        "customer_name": "Jordan",
        "history": [
            {"role": "user", "content": "status of ticket 650932?"},
            {"role": "assistant", "content": "Your ticket #650932 is currently Resolved."},
        ],
    }
    out = handle_query(state)
    assert "650932" in out["response"]
    assert "Resolved" in out["response"]


def test_query_without_id_or_history_asks_for_number():
    out = handle_query({"message": "any update?", "customer_id": DEMO_CUSTOMER_ID})
    assert "ticket number" in out["response"].lower()


def test_followup_routes_through_graph_with_history():
    g = build_graph(classify_fn=lambda _t: Fake("query", 0.9))
    final = respond(
        "any update on it?",
        customer_id=DEMO_CUSTOMER_ID,
        customer_name="Jordan",
        history=[{"role": "user", "content": "I opened ticket 100428 last week"}],
        graph=g,
    )
    assert final["route"] == "query"
    assert "100428" in final["response"]
