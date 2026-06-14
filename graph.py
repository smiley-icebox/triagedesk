"""The LangGraph workflow that wires TriageDesk together.

Shape:

    START -> classify -> (route_by_label) -> one of: positive / negative / query
                                                      / escalate -> END

This is the teaching contrast with NewsGenie. NewsGenie used `tools_condition`, so
the LLM itself decided what to call inside a loop (NewsGenie/graph.py:66) — an
agent. Here:

  - `classify` makes the one LLM call and writes {label, confidence, route} to state.
  - `route_by_label` is a PLAIN FUNCTION that reads state["route"] and returns the
    next node. The routing is done by code, deterministically, from a structured
    label — not by the model. That makes every path auditable and unit-testable
    without an API call.

The graph is linear after classification (no loop back), because a support triage
is a single decision, not an open-ended conversation. That's the right structure
for the job; reaching for an agent loop here would be accidental complexity.
"""

import time
from typing import Callable, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

import classifier
import observability
from config import (
    CONFIDENCE_THRESHOLD,
    LABEL_GENERAL,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    LABEL_QUERY,
    ROUTE_ESCALATE,
)
from handlers import (
    handle_escalate,
    handle_general,
    handle_negative,
    handle_positive,
    handle_query,
)


class SupportState(TypedDict, total=False):
    """The state threaded through the graph. `total=False` so nodes can return just
    the keys they set; LangGraph merges partial updates.

    Inputs:  message, customer_id, customer_name, history
    Set by classify:  label, confidence, route, source
    Set by a handler: response, ticket_id, db_action
    """

    message: str
    customer_id: str
    customer_name: Optional[str]
    history: Optional[list]   # prior [{role, content}] turns, for follow-up context
    label: Optional[str]
    confidence: Optional[float]
    route: Optional[str]
    source: Optional[str]
    response: Optional[str]
    ticket_id: Optional[str]
    db_action: Optional[str]


def route_for(label: Optional[str], confidence: float) -> str:
    """The escalation gate: take the label only if confident enough, else send to a
    human. The SINGLE source of this rule — graph.classify_node routes on it live,
    and evaluation.py scores routing against it, so the two can't drift."""
    return label if confidence >= CONFIDENCE_THRESHOLD else ROUTE_ESCALATE


def _make_classify_node(classify_fn: Callable):
    """Build the classify node around a classification function.

    classify_fn(message) -> object with .label and .confidence. Injected so tests
    can pass a fake (no API key, deterministic) — the production default is
    classifier.classify.
    """

    def classify_node(state: SupportState) -> dict:
        # Injection guardrail FIRST, on the RAW current message only — a code-side
        # routing decision (like the confidence gate below), not something the model
        # gets a say in. Suspected manipulation goes straight to a human and we don't
        # spend an LLM call. Checking the raw message (not the folded history below)
        # means a quoted prior turn can't trip it.
        if classifier.detect_injection(state["message"]):
            return {"label": None, "confidence": 0.0, "route": ROUTE_ESCALATE,
                    "source": "guardrail"}

        # Multi-turn: if there's prior conversation, give the classifier that
        # context so follow-ups ("what about my other one?", "any update?") are
        # interpreted in light of what came before. The seam stays classify_fn(text);
        # context is folded into the text so fakes/tests don't need to change.
        text = state["message"]
        history = state.get("history") or []
        if history:
            text = f"Conversation so far:\n{_render_history(history)}\n\nLatest customer message to classify: {state['message']}"
        result = classify_fn(text)
        # The escalation decision is made HERE, by code, from the confidence score —
        # it is not a class the model gets to pick. Below threshold => human.
        route = route_for(result.label, result.confidence)
        # `source` ("llm"/"heuristic"/"guardrail") is carried for observability so a
        # trace shows when we ran degraded or tripped the injection guardrail.
        source = getattr(result, "source", "llm")
        return {"label": result.label, "confidence": result.confidence,
                "route": route, "source": source}

    return classify_node


def _render_history(history: list, max_turns: int = 4) -> str:
    """Compact the last few turns into a short transcript for classifier context."""
    recent = history[-max_turns:]
    lines = []
    for turn in recent:
        who = "Customer" if turn.get("role") == "user" else "Support"
        lines.append(f"{who}: {turn.get('content', '')}")
    return "\n".join(lines)


def route_by_label(state: SupportState) -> str:
    """The conditional-edge function: deterministic routing from the structured
    route. Returns the name of the next node. No LLM, no side effects."""
    return state["route"]


def build_graph(classify_fn: Optional[Callable] = None):
    """Assemble and compile the TriageDesk StateGraph.

    classify_fn defaults to the real classifier; pass a fake to test routing and
    handlers without touching the API.
    """
    if classify_fn is None:
        from classifier import classify as classify_fn  # lazy: no API key just to import

    graph = StateGraph(SupportState)
    graph.add_node("classify", _make_classify_node(classify_fn))
    graph.add_node("positive", handle_positive)
    graph.add_node("negative", handle_negative)
    graph.add_node("query", handle_query)
    graph.add_node("general", handle_general)
    graph.add_node("escalate", handle_escalate)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_label,
        {
            LABEL_POSITIVE: "positive",
            LABEL_NEGATIVE: "negative",
            LABEL_QUERY: "query",
            LABEL_GENERAL: "general",
            ROUTE_ESCALATE: "escalate",
        },
    )
    for node in ("positive", "negative", "query", "general", "escalate"):
        graph.add_edge(node, END)

    return graph.compile()


# Lazily-built singleton so the app/eval build the graph once.
_graph_singleton = None


def _graph():
    global _graph_singleton
    if _graph_singleton is None:
        _graph_singleton = build_graph()
    return _graph_singleton


def respond(message: str, customer_id: str, customer_name: Optional[str] = None,
            history: Optional[list] = None, graph=None) -> dict:
    """Run one message through the workflow and return the final state.

    `history` is the prior [{role, content}] turns of this conversation (the UI's
    chat history minus the current message); it gives follow-ups context.

    Also times the run and writes a structured trace (the observability layer),
    so every interaction is logged for the debug view and evaluation. This is the
    single entry point the Streamlit app and the evaluator both call.
    """
    g = graph if graph is not None else _graph()
    started = time.perf_counter()
    final = g.invoke(
        {"message": message, "customer_id": customer_id, "customer_name": customer_name,
         "history": history or []}
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    observability.record(
        message=message,
        label=final.get("label"),
        confidence=final.get("confidence"),
        route=final.get("route"),
        source=final.get("source"),
        db_action=final.get("db_action"),
        ticket_id=final.get("ticket_id"),
        latency_ms=latency_ms,
    )
    final["latency_ms"] = latency_ms
    return final
