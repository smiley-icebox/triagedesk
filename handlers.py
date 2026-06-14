"""The four downstream handlers — the deterministic half of the system.

Once the classifier has labeled a message, NOTHING below here calls an LLM. These
are plain functions that fill a fixed template and, where needed, read or write the
database. That separation is the whole point: the LLM understood the language; code
does everything factual and transactional. The customer-facing wording comes only
from config.TEMPLATES, so it's auditable and consistent.

Each function is a LangGraph node: it takes the shared state dict and returns a
partial update (the keys it wants to change). LangGraph merges that into state.
"""

import re

import db
import knowledge
import notifier
import responder
from config import (
    DEFAULT_CUSTOMER_NAME,
    TEMPLATE_ESCALATE,
    TEMPLATE_QUERY_FOUND,
    TEMPLATE_QUERY_NO_ID,
    TEMPLATE_QUERY_NOT_FOUND,
)

# 6-digit ticket numbers are the scheme (config/db). Prefer an exact 6-digit token;
# fall back to any 4+ digit run so a customer who mistypes still gets a useful
# "we couldn't find that" rather than "you gave no number".
_TICKET_RE_6 = re.compile(r"\b(\d{6})\b")
_TICKET_RE_ANY = re.compile(r"\b(\d{4,8})\b")


def extract_ticket_id(message: str) -> str | None:
    """Pull a ticket number out of free text, or None if there isn't one.

    Pure string work — kept separate from the handler so it's trivially testable.
    """
    m = _TICKET_RE_6.search(message) or _TICKET_RE_ANY.search(message)
    return m.group(1) if m else None


def _name(state: dict) -> str:
    return state.get("customer_name") or DEFAULT_CUSTOMER_NAME


def _ticket_id_from_history(state: dict) -> str | None:
    """Resolve a referent: when the latest message has no ticket number ("any
    update?"), reuse the most recent ticket number mentioned earlier in the
    conversation. This is what makes follow-ups work in a multi-turn chat."""
    for turn in reversed(state.get("history") or []):
        tid = extract_ticket_id(turn.get("content", ""))
        if tid:
            return tid
    return None


def handle_positive(state: dict) -> dict:
    """Positive feedback: a warm thank-you. The responder LLM-drafts the wording
    within guardrails (falling back to the template), so it's personalized 'using a
    language model' per the brief — but still safe. No DB, no ticket."""
    response = responder.generate_thankyou(state.get("customer_name"))
    return {"response": response, "db_action": "none"}


def handle_negative(state: dict) -> dict:
    """Negative feedback: open a ticket, then apologize with its number.

    The ticket id comes from the DB (create_ticket allocates and persists it) — the
    LLM never invents a number. If the write fails, we degrade to a graceful message
    instead of crashing or fabricating a ticket that doesn't exist.
    """
    ticket_id = db.create_ticket(
        customer_id=state["customer_id"],
        issue=state["message"],
        customer_name=state.get("customer_name"),
        reason="opened from negative feedback",
    )
    if ticket_id is None:
        return {
            "response": (
                "We're sorry for the trouble. We hit a problem logging your ticket "
                "just now — please try again in a moment, and we'll get it raised."
            ),
            "db_action": "create_failed",
        }
    # Notify the customer their ticket was opened (never raises into this path).
    notifier.notify_ticket_created(_name(state), state["customer_id"], ticket_id)
    # LLM-drafted empathetic apology that cites the code-generated ticket id
    # (verified to appear, else falls back to the template).
    response = responder.generate_apology(state.get("customer_name"), ticket_id)
    return {"response": response, "ticket_id": ticket_id, "db_action": f"created #{ticket_id}"}


def handle_query(state: dict) -> dict:
    """Query: find the ticket number, read its status from the DB, report it.

    Three outcomes, all deterministic:
      - no number in the message      -> ask for one
      - number not found for this user -> graceful not-found (also the IDOR-safe
        answer when the ticket belongs to someone else)
      - found                          -> report the DB's status verbatim
    """
    # Current message first; fall back to a ticket id from earlier in the chat.
    ticket_id = extract_ticket_id(state["message"]) or _ticket_id_from_history(state)
    if ticket_id is None:
        return {"response": TEMPLATE_QUERY_NO_ID, "db_action": "lookup: no id in message"}

    # customer_id is always supplied by graph.respond — index it (not .get), so a
    # scoped read never silently degrades to an unscoped one on a missing key.
    ticket = db.get_ticket(ticket_id, customer_id=state["customer_id"])
    if ticket is None:
        return {
            "response": TEMPLATE_QUERY_NOT_FOUND.format(ticket_id=ticket_id),
            "ticket_id": ticket_id,
            "db_action": f"lookup #{ticket_id}: not found",
        }
    response = TEMPLATE_QUERY_FOUND.format(ticket_id=ticket_id, status=ticket["status"])
    return {
        "response": response,
        "ticket_id": ticket_id,
        "db_action": f"lookup #{ticket_id}: {ticket['status']}",
    }


def handle_general(state: dict) -> dict:
    """General banking question: answer via RAG over the FAQ, grounded in retrieved
    passages. If nothing relevant is found (or the model is unsure), it defers to a
    human rather than guessing — the knowledge layer enforces that."""
    result = knowledge.answer(state["message"], state.get("customer_name"))
    action = (f"kb: answered from {', '.join(result['topics'])}" if result["grounded"]
              else "kb: no confident answer → deferred")
    return {"response": result["response"], "db_action": action}


def handle_escalate(state: dict) -> dict:
    """Low-confidence / out-of-scope: hand off to a human. No DB, no guessing.

    This branch exists precisely so the system has an honest 'I'm not sure' instead
    of forcing an uncertain message into one of the happy paths."""
    response = TEMPLATE_ESCALATE.format(customer_name=_name(state))
    return {"response": response, "db_action": "none"}
