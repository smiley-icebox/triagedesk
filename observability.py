"""Structured per-interaction logging — the substrate for the debug view and eval.

Every call to graph.respond() emits one JSON record here. Two things lean on these
records:
  - the Streamlit "Debug / Logs" panel (Task 9), which shows the prompt trace,
    classification output, and ticket action for each interaction;
  - evaluation.py (Task 7), which can read routing/latency history.

WHY structured (JSONL) rather than print()/free-form logs: structured records are
queryable and aggregatable. "What was the routing success rate?" and "what's the
p50 latency?" are one-liners over JSONL, but archaeology over prose logs. Same
principle as the structured classifier output — make the data a shape, not text.

PII: the message is run through redact() before it's written (best-effort masking of
emails, SSNs, phone numbers, and card/account-length numbers — see redact() below).
This is masking-at-the-log-boundary, not a full DLP pipeline, and it does NOT cover
PII stored elsewhere (e.g. a complaint's raw text in support_tickets) — the writeup
flags storage-layer redaction + encryption-at-rest as production work.
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
TRACE_PATH = os.path.join(LOG_DIR, "traces.jsonl")


# --- PII redaction -----------------------------------------------------------
# In a real bank you NEVER let raw PII reach a log. Logs get shipped, indexed, and
# retained, so a card number in a log line is a breach waiting to happen. We mask the
# common carriers before writing. This is BEST-EFFORT pattern masking, not a
# guaranteed scrubber (a real system pairs it with a proper PII/DLP pipeline). 6-digit
# ticket ids are intentionally kept (references, not PII); SSNs, phone numbers, emails,
# and card/account-length numbers are masked. Order matters: dashed patterns (SSN,
# phone) run before the contiguous-digit rules.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_CARD_GROUPS_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")   # 13-19 digit card/account groups
_LONG_NUM_RE = re.compile(r"\b\d{7,}\b")                    # 7+ digit run (account numbers); 6-digit ticket ids preserved


def redact(text: str) -> str:
    """Best-effort masking of emails, SSNs, phone numbers, and card/account-length
    numbers. Not a vault — a real deployment adds a dedicated PII/DLP layer."""
    if not text:
        return text
    text = _EMAIL_RE.sub("[email]", text)
    text = _SSN_RE.sub("[ssn]", text)
    text = _PHONE_RE.sub("[phone]", text)
    text = _CARD_GROUPS_RE.sub("[redacted-number]", text)
    text = _LONG_NUM_RE.sub("[redacted-number]", text)
    return text


def record(
    *,
    message: str,
    label: str | None,
    confidence: float | None,
    route: str | None,
    db_action: str | None,
    ticket_id: str | None,
    latency_ms: float | None,
    source: str | None = None,
) -> dict:
    """Append one interaction trace and return the record (so callers can display it).

    Keyword-only args so a record is never assembled in the wrong column order.
    Never raises into the request path — logging must not break a conversation.
    """
    rec = {
        "trace_id": uuid.uuid4().hex[:8],
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "message": redact(message),  # PII never reaches the log
        "label": label,
        "confidence": confidence,
        "route": route,
        "source": source,
        "db_action": db_action,
        "ticket_id": ticket_id,
        "latency_ms": latency_ms,
    }
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass  # a logging failure must never surface to the customer
    return rec


def read_recent(limit: int = 50) -> list[dict]:
    """Return the most recent trace records (newest first) for the debug panel."""
    if not os.path.exists(TRACE_PATH):
        return []
    try:
        with open(TRACE_PATH, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    # Parse per line: a single corrupt/partial line (e.g. an interleaved concurrent
    # append) must not zero out the entire trace history — skip it and keep the rest.
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return list(reversed(records))[:limit]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100) * (len(s) - 1)))))
    return s[k]


def metrics(limit: int = 1000) -> dict:
    """Aggregate the trace log into operational metrics for the dashboard.

    Centralizing aggregation here (instead of in the UI) keeps the dashboard thin
    and makes the same numbers available to alerts/exports later.
    """
    traces = read_recent(limit=limit)
    total = len(traces)
    by_route: dict = {}
    by_source: dict = {}
    for t in traces:
        by_route[t.get("route")] = by_route.get(t.get("route"), 0) + 1
        by_source[t.get("source")] = by_source.get(t.get("source"), 0) + 1
    latencies = [t["latency_ms"] for t in traces if t.get("latency_ms") is not None]
    escalated = by_route.get("escalate", 0)
    # "degraded" = ran on the keyword fallback or tripped the injection guardrail.
    degraded = by_source.get("heuristic", 0) + by_source.get("guardrail", 0)
    return {
        "total": total,
        "auto_handled": total - escalated,
        "escalated": escalated,
        "escalation_rate": (escalated / total) if total else 0.0,
        "degraded": degraded,
        "by_route": by_route,
        "by_source": by_source,
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 95),
    }


def langsmith_status() -> bool:
    """Whether opt-in LangSmith tracing is configured (env-driven, no code needed).

    Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in the environment and
    LangChain auto-exports traces of every LLM call — production-grade tracing for
    free. We just surface whether it's on."""
    return os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true", "yes") and bool(
        os.getenv("LANGCHAIN_API_KEY")
    )


def clear() -> None:
    """Wipe the trace log (used by the UI's 'clear logs' control and by tests)."""
    if os.path.exists(TRACE_PATH):
        os.remove(TRACE_PATH)
