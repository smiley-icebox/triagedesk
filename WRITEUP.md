# TriageDesk — Project Writeup

Banking Customer Support AI Agent using Multi-Agent Architecture
Applied GenAI Capstone · STAR format

---

## Situation

Digital banks handle a high volume of support messages through fragmented systems
that struggle to personalize responses or give timely status updates. A customer's
"thanks, that's sorted," "my card still hasn't arrived," "what's the status of my
ticket?" and "what are your wire fees?" all land in the same inbox and all need
different handling — acknowledgement, a logged ticket, a database lookup, and a
knowledge-base answer respectively.

## Task

Build a multi-agent GenAI assistant that classifies each message, responds
appropriately per class, and tracks tickets in a support database — plus the LLMOps
layer (evaluation, a Streamlit dashboard, logging/debug). Then take it past the brief
toward something you'd actually run.

## Action

### The central design decision: workflow, not agent

I built this as a **deterministic workflow with one LLM decision**, the correct
engineering for a transactional banking flow. The LLM only **classifies** (and drafts
copy within guardrails); **code routes** from the structured label; and the LLM is
**fenced out of the data path** — ticket numbers, statuses, and customer identity are
all owned by code, never by the model. The standard LangGraph *agent* loop (binding
tools and letting the LLM choose them via `tools_condition`) was the shape I
deliberately did **not** use — you don't want a model improvising over account data.

**On "multi-agent":** the system *is* multi-agent — five specialized agents (a
classifier agent plus four handler agents: positive feedback, negative feedback,
ticket query, and a general-knowledge/RAG agent) are orchestrated as nodes in a
**LangGraph `StateGraph`** (`graph.py`). The deliberate design choice is that the
*orchestration between agents is deterministic code* (`route_by_label`) rather than an
LLM agent loop — each agent does one job, and a code-side router dispatches to the
right one based on the classifier's structured output. That's multi-agent
architecture with the control flow where it belongs in a banking system: in code, not
in a model's discretion.

### Core build (the brief)

- **LangGraph** workflow: `classify → route_by_label → {positive | negative | query |
  general_query | escalate} → END`.
- **Classifier**: Claude via `with_structured_output` against a Pydantic schema, so the
  label is constrained to the allowed classes — no parsing free text.
- **SQLite** `support_tickets`, shaped like production (status vocabulary, owning
  `customer_id`, audit timestamps).
- **Escalation branch** for low confidence — an honest "I'm not sure."
- **Streamlit** dashboard, **structured logging**, and **evaluation**.

### Production hardening (beyond the brief)

- **Authentication** — identity comes from a verified login (PBKDF2 + constant-time
  compare), not client input; lookups scope to the authenticated customer (IDOR fix).
- **Storage seam + migrations** — all DB access sits behind a `TicketRepository`
  interface with versioned migrations; Postgres is a drop-in. Ticket-id allocation is
  race-free (insert-retry, not check-then-insert — fixing a real TOCTOU bug).
- **Ticket lifecycle + audit trail** — status transitions write immutable
  `ticket_events`; **SLA** deadlines per priority with a breach check.
- **Classifier hardening** — retry/timeout, a deterministic keyword **fallback** when
  the API is down, and an **injection guardrail** that routes manipulation to a human.
- **LLM-generated empathetic replies, guardrailed** — the model drafts warmth; the
  ticket number is code-supplied and verified to appear, else it falls back to the
  approved template (closing the brief's "using a language model" gap *safely*).
- **RAG** for general questions — retrieve from a FAQ, generate a grounded answer, and
  defer to a human when the answer isn't in the knowledge base.
- **Multi-turn context**, **PII-redacted observability** with metrics + opt-in
  LangSmith, and **notifications** behind a swappable interface.

## Result

- All required flows work end-to-end in the required formats. On the labeled set, live
  **classification accuracy and routing success are 100%**, escalation of
  ambiguous/out-of-scope messages is **4/4**, and an **LLM-as-judge** scores response
  empathy and clarity (correctly rating the transactional status reply low-empathy and
  the apology high).
- **99 automated tests pass with no API call** — the LLM sits behind injectable seams,
  so guardrails, fallback, routing, lifecycle, auth, RAG, and the judge are all
  verifiable offline.
- The dashboard makes the workflow visible: per message you see the classification,
  confidence, route, **source** (llm / heuristic-fallback / guardrail), and DB action;
  plus ticket lifecycle with audit + SLA, and a metrics/notifications debug view.

## Engineering decisions worth calling out

| Decision | Why |
|----------|-----|
| Structured output, not text parsing | The label is data with a fixed shape; parsing prose for it is brittle. |
| DB as the single source of truth | The model never states or invents a ticket fact. |
| Confidence threshold → human | No "I'm not sure" branch = confidently wrong on edge cases. |
| LLM warmth, code facts | Personalized replies, but the ticket number is verified, never invented. |
| Guardrail + fallback on the one LLM call | Injection routes to a human; an API outage degrades to a keyword classifier, not a 500. |
| Repository + notifier interfaces | Postgres / SMTP / Twilio become drop-ins, not rewrites. |
| One LLM seam per concern, all injectable | The whole system is testable without the API. |

## What production would still demand (documented, not built)

The seams are in place; these are the real implementations behind them:

- A real **identity provider** behind `auth.py` (the demo seeds users in memory).
- The actual **Postgres** `TicketRepository` (the interface + portable SQL are ready).
- Real **email/SMS** adapters behind the notifier (console notifier today).
- **Embeddings + a vector store** behind the RAG retriever (keyword retrieval today).
- A **human-in-the-loop feedback loop** that retrains on escalated/misrouted cases, and
  a right-sized/fine-tuned classifier tier (e.g. Haiku) for cost and latency.
