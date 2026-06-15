"""TriageDesk — Streamlit dashboard.

Run with:  streamlit run app.py

Covers the brief's Part-2 UI requirements (accept input + simulate routing; show
classification/response/DB interaction; ticket + log views; per-role test
scenarios; debug/logs with traces and success/failure aggregates) — and the
production additions: a real login (identity from an authenticated session, not a
text box), and a ticket lifecycle / audit view on the Tickets tab.

UI patterns: cache_resource for the shared graph, session_state for auth/history,
graceful per-turn error handling (standard Streamlit).
"""

import os

import streamlit as st
import streamlit.components.v1 as components

import auth
import db
import notifier
import observability
import seed_data
from config import CONFIDENCE_THRESHOLD, CLASSIFIER_SYSTEM_PROMPT, LLM_MODEL, TICKET_STATUSES
from graph import build_graph, respond

st.set_page_config(page_title="TriageDesk", page_icon="🏦", layout="wide")

# Trim Streamlit's default top padding so the chat panel has more room. (The
# panel's height is set dynamically in JS — see _fit_and_scroll — because
# Streamlit's own styling wins over a CSS height rule here.)
st.markdown(
    """
    <style>
      /* Trim top padding for room; ZERO bottom padding. */
      .block-container { padding-top: 2.5rem !important; padding-bottom: 0 !important; }
      /* The 0-height JS helper component (an iframe) still claimed ~42px of layout
         space below the input, causing a page scrollbar. Pull it out of flow so it
         occupies no height (it still runs its script). */
      [data-testid="stElementContainer"]:has(iframe[title="st.iframe"]) {
        position: absolute !important; height: 0 !important; min-height: 0 !important;
        overflow: hidden !important; margin: 0 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_graph():
    return build_graph()


@st.cache_resource(show_spinner=False)
def _bootstrap_db():
    db.init_db()
    if not db.list_tickets():
        seed_data.seed()
    return True


_bootstrap_db()


def _md(text: str) -> str:
    return text.replace("$", "\\$")


def _fit_and_scroll():
    """Two jobs, both from a 0-height helper component:

    1. SIZE the chat panel to fill the leftover viewport height, so the PAGE never
       overflows and scrolls — only the panel scrolls internally, with the header,
       tabs, and input box staying fixed. We compute the panel height from the
       actual non-panel ("chrome") height, so it fits any window. Setting the
       height inline with `!important` beats Streamlit's own styling (a plain CSS
       rule loses to it).
    2. SCROLL the newest message into view with a single smooth glide.
    """
    components.html(
        """
        <script>
          const doc = window.parent.document;
          function fit() {
            const main = doc.querySelector('[data-testid="stMain"]');
            const panel = doc.querySelector('.st-key-triage_chat');
            if (!main || !panel) {
              // These reach into Streamlit-internal selectors; a Streamlit upgrade
              // could rename them. Warn loudly instead of silently doing nothing —
              // the chat panel would just fall back to its fixed height.
              console.warn('TriageDesk: chat-panel selectors not found ' +
                '(stMain / .st-key-triage_chat) — Streamlit internals may have changed.');
              return;
            }
            // The fixed height lives on the panel's PARENT flex item (flex: 0 0 Npx).
            const wrapper = panel.parentElement;
            // Non-circular sizing: the panel's TOP is fixed by the chrome above it,
            // and the input height is constant — so fill the rest of the viewport,
            // leaving a margin for the gap below the input. (Bottom page padding is
            // zeroed in CSS, so this no longer leaves a 14px page scrollbar.)
            const panelTop = panel.getBoundingClientRect().top;
            const input = doc.querySelector('[data-testid="stChatInput"]');
            const inputH = input ? input.getBoundingClientRect().height : 60;
            const target = Math.max(220, window.parent.innerHeight - panelTop - inputH - 40);
            wrapper.style.setProperty('flex', '0 0 ' + target + 'px', 'important');
            wrapper.style.setProperty('height', target + 'px', 'important');
            const msgs = panel.querySelectorAll('[data-testid="stChatMessage"]');
            if (msgs.length)
              msgs[msgs.length - 1].scrollIntoView({ behavior: 'smooth', block: 'end' });
          }
          requestAnimationFrame(fit);
          setTimeout(fit, 200);   // again after layout settles
        </script>
        """,
        height=0,
    )


def _render_detail(d):
    """Render the 🧭 routing expander under an assistant message (shared by the
    history loop and the live new-turn render so they can't drift)."""
    if not d:
        return
    conf = f"{d['confidence']:.2f}" if d.get("confidence") is not None else "—"
    with st.expander(
        f"🧭 routing · label={d.get('label')} · conf={conf} · "
        f"route={d.get('route')} · {d.get('latency_ms')} ms"
    ):
        st.markdown(
            f"- **Classification:** `{d.get('label')}` (confidence {conf})\n"
            f"- **Route taken:** `{d.get('route')}`"
            + ("  _(below threshold → human escalation)_"
               if d.get("route") == "escalate" else "")
            + f"\n- **Database action:** {d.get('db_action')}\n"
            + (f"- **Ticket:** #{d.get('ticket_id')}\n" if d.get("ticket_id") else "")
        )


# --- Session state ----------------------------------------------------------
st.session_state.setdefault("session", None)        # auth.Session once logged in
st.session_state.setdefault("history", [])
st.session_state.setdefault("pending", None)

# === Login gate =============================================================
# Identity is established here, ONCE, by authentication. Everything downstream
# uses session.customer_id — the client never gets to assert who it is.
if st.session_state.session is None:
    st.title("🏦 TriageDesk")
    st.caption("Banking support triage — please sign in")
    with st.form("login"):
        username = st.selectbox("Username", auth.demo_usernames())
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        sess = auth.authenticate(username, password)
        if sess is None:
            st.error("Invalid credentials.")
        else:
            st.session_state.session = sess
            st.session_state.history = []
            st.rerun()
    st.info("Demo credentials (all password `demo123`):\n"
            "- **jordan** — customer, owns the sample tickets\n"
            "- **sam** — customer, owns ticket 940011 (shows read-scoping)\n"
            "- **agent** — support agent, sees all tickets + can drive their lifecycle")
    st.stop()

session = st.session_state.session


SCENARIOS = {
    "👍 Positive feedback": "Thanks for sorting out my net banking login issue.",
    "👎 Negative feedback": "My debit card replacement still hasn't arrived.",
    "🔎 Query (your ticket)": "Could you check the status of ticket 650932?",
    "🚫 Query (someone else's)": "Any update on ticket 940011?",
    "❓ General question (RAG)": "What are your international transaction fees?",
    "🤷 Ambiguous → escalate": "asldkfj 4567 ??",
}

# --- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.title("🏦 TriageDesk")
    st.caption("Banking support triage — 5 agents (classifier + 4 handlers) "
               "orchestrated as LangGraph nodes")

    st.subheader("Signed in")
    st.write(f"**{session.customer_name}**  ·  `{session.customer_id}`  ·  _{session.role}_")
    st.caption(
        "Identity (and role) come from your authenticated session — never client input. "
        "Lookups are scoped to your own tickets; the agent role gates ticket-lifecycle "
        "actions. Query ticket 940011 (another customer's) to see read-scoping."
    )
    if st.button("Sign out", use_container_width=True):
        st.session_state.session = None
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.subheader("Try a scenario")
    for label, text in SCENARIOS.items():
        if st.button(label, use_container_width=True):
            st.session_state.pending = text
            st.rerun()

    st.divider()
    st.subheader("Status")
    key_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
    st.write(f"{'✅' if key_ok else '❌'} ANTHROPIC_API_KEY")
    st.caption(f"Model: `{LLM_MODEL}`  ·  escalate < {CONFIDENCE_THRESHOLD:.2f} conf")
    if not key_ok:
        st.warning("Set ANTHROPIC_API_KEY in a `.env` file to run live triage.")

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("↺ Reset data", use_container_width=True):
        seed_data.seed()
        st.session_state.history = []
        st.toast("Demo tickets reset.")
        st.rerun()
    if c2.button("🧹 Clear logs", use_container_width=True):
        observability.clear()
        st.toast("Trace log cleared.")
        st.rerun()

# --- Main: tabs -------------------------------------------------------------
tab_triage, tab_tickets, tab_logs = st.tabs(["💬 Triage", "🎫 Tickets", "🔬 Debug / Logs"])

# === Triage tab =============================================================
with tab_triage:
    st.caption(
        "Type a customer message (or use a sidebar scenario). The classifier labels "
        "it once; routing to the right handler is then deterministic."
    )

    # A FIXED-HEIGHT, scrollable message panel with the input pinned beneath it —
    # the chat-app layout: messages scroll inside this box, the input stays put.
    # Declaring it before the input keeps messages above the box; processing the
    # turn here (no st.rerun()) avoids the reply flashing below then above.
    msg_area = st.container(height=540, key="triage_chat")  # height overridden in JS (_fit_and_scroll) to fill viewport
    prompt = st.chat_input("Customer message…")
    if not prompt and st.session_state.pending:
        prompt = st.session_state.pending
    st.session_state.pending = None

    with msg_area:
        # 1) Existing history first, so nothing disappears while we work.
        for turn in st.session_state.history:
            with st.chat_message(turn["role"]):
                st.markdown(_md(turn["content"]) if turn["role"] == "assistant" else turn["content"])
                _render_detail(turn.get("detail"))

        # 2) The new turn, rendered in place (still above the input).
        # In-flight guard: ignore an exact-duplicate immediate re-submit (a rapid
        # double-click / double-enter), so a negative message can't open two tickets.
        hist = st.session_state.history
        is_dup = bool(prompt) and len(hist) >= 2 and hist[-2].get("role") == "user" \
            and hist[-2].get("content") == prompt and hist[-1].get("role") == "assistant"
        if prompt and not is_dup:
            # Prior turns (before this message) give follow-ups their context.
            prior = [{"role": h["role"], "content": h["content"]} for h in st.session_state.history]
            st.session_state.history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Triaging…"):
                        final = respond(
                            prompt,
                            customer_id=session.customer_id,
                            customer_name=session.customer_name,
                            history=prior,
                            graph=get_graph(),
                        )
                    answer = final.get("response", "(no response)")
                    detail = {
                        "label": final.get("label"),
                        "confidence": final.get("confidence"),
                        "route": final.get("route"),
                        "db_action": final.get("db_action"),
                        "ticket_id": final.get("ticket_id"),
                        "latency_ms": final.get("latency_ms"),
                    }
                    st.markdown(_md(answer))
                    _render_detail(detail)
                except Exception as exc:
                    # Don't presume the cause — surface the error type, hint at the
                    # common one without asserting it, and let the user retry.
                    answer = (
                        f"Sorry — something went wrong handling that ({type(exc).__name__}). "
                        "Please try again. (If this persists, check that ANTHROPIC_API_KEY "
                        "is set and the service is reachable.)"
                    )
                    detail = None
                    st.error(answer)
            st.session_state.history.append(
                {"role": "assistant", "content": answer, "detail": detail}
            )
        elif prompt and is_dup:
            # Suppression must not be silent: tell the user why their repeat didn't run,
            # and point at the answer that's already on screen (the dedup only guards
            # against a rapid double-submit opening two tickets).
            st.toast("Ignored a duplicate of your last message — see the reply above.",
                     icon="↩️")

    # Size the panel to the viewport (so the page doesn't scroll) and pin the view
    # to the newest message. Runs every render, even with an empty chat.
    _fit_and_scroll()

# === Tickets tab ============================================================
# Role-gated: an AGENT sees every ticket and can drive its lifecycle (the audit
# trail records "agent:<name>"); a CUSTOMER sees only their own tickets, read-only —
# they can't change status, clear an SLA breach, or be logged as an agent.
with tab_tickets:
    if session.is_agent:
        st.caption("**Agent console** — all tickets. Status changes are audited as "
                   f"`agent:{session.username}`.")
        if st.button("⏱ Run SLA check"):
            n = db.mark_overdue_sla()
            st.toast(f"SLA check: {n} ticket(s) newly breached.")
            st.rerun()
        tickets = db.list_tickets()  # agent: all customers
    else:
        st.caption(f"Your tickets, **{session.customer_name}** — read-only. "
                   "Our support team manages status.")
        tickets = db.list_tickets(customer_id=session.customer_id)  # customer: own only

    if not tickets:
        st.info("No tickets yet." + ("" if session.is_agent
                else " Send a negative-feedback message to open one."))
    for t in tickets:
        breach = " · ⚠️ SLA BREACHED" if t.get("sla_breached") else ""
        owner = t["customer_id"]
        with st.expander(f"#{t['ticket_id']} · {t['status']} · {t['issue']}{breach}"):
            st.write(f"**Opened:** {t['created_at']}  ·  **Last update:** {t['updated_at']}")
            if session.is_agent:
                st.write(f"**Customer:** `{owner}`")
            if t.get("sla_due_at"):
                flag = "⚠️ breached" if t.get("sla_breached") else "on track"
                st.write(f"**SLA due:** {t['sla_due_at']}  ·  {flag}  ·  priority: {t.get('priority','normal')}")

            # Lifecycle control — AGENTS ONLY. The status update is scoped to the
            # ticket's owner, and the audit records the acting agent.
            if session.is_agent:
                cols = st.columns([3, 1])
                cur_status = t["status"]
                sel_index = TICKET_STATUSES.index(cur_status) if cur_status in TICKET_STATUSES else 0
                new_status = cols[0].selectbox(
                    "Set status", TICKET_STATUSES, index=sel_index, key=f"sel_{t['ticket_id']}",
                )
                if cols[1].button("Update", key=f"upd_{t['ticket_id']}"):
                    ok = db.update_status(
                        t["ticket_id"], new_status,
                        actor=f"agent:{session.username}",
                        customer_id=owner,
                        note="changed via agent console",
                    )
                    st.toast("Status updated." if ok else "No change.")
                    st.rerun()

            events = db.get_events(t["ticket_id"], owner)
            if events:
                st.markdown("**Audit trail**")
                for e in events:
                    frm = e.get("from_status") or "—"
                    st.markdown(
                        f"- `{e['created_at']}` · {e['event_type']}: {frm} → "
                        f"**{e.get('to_status')}** by `{e['actor']}`"
                        + (f" — {e['note']}" if e.get("note") else "")
                    )

# === Debug / Logs tab =======================================================
with tab_logs:
    traces = observability.read_recent(limit=200)

    st.subheader("Agent success / failure")
    ls = "✅ on" if observability.langsmith_status() else "⚪ off (set LANGCHAIN_TRACING_V2)"
    st.caption(f"LangSmith tracing: {ls}  ·  best-effort PII masking before logging")
    if traces:
        mx = observability.metrics()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Interactions", mx["total"])
        m2.metric("Auto-handled", mx["auto_handled"])
        m3.metric("Escalated", mx["escalated"])
        m4.metric("p95 latency", f"{mx['p95_latency_ms']:.0f} ms")
        st.caption("By route: " + ", ".join(f"`{k}`={v}" for k, v in mx["by_route"].items()))
        st.caption(
            "By source: " + ", ".join(f"`{k}`={v}" for k, v in mx["by_source"].items())
            + f"  ·  degraded (fallback/guardrail): **{mx['degraded']}**"
        )
    else:
        st.info("No interactions logged yet.")

    st.subheader("Interaction traces")
    st.caption("Every message: classification output, route, ticket action, latency.")
    if traces:
        st.dataframe(
            [
                {
                    "trace": t.get("trace_id"),
                    "time": t.get("ts"),
                    "message": t.get("message"),
                    "label": t.get("label"),
                    "conf": t.get("confidence"),
                    "route": t.get("route"),
                    "source": t.get("source"),
                    "db_action": t.get("db_action"),
                    "ms": t.get("latency_ms"),
                }
                for t in traces
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Notifications sent")
    st.caption("Outbound messages (ticket-created, etc.) via the console notifier.")
    notes = notifier.read_recent(limit=50)
    if notes:
        st.dataframe(
            [{"time": n.get("ts"), "channel": n.get("channel"),
              "to": n.get("recipient"), "subject": n.get("subject")} for n in notes],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No notifications yet. Open a ticket (negative feedback) to send one.")

    with st.expander("🧾 Classifier prompt (the one place the LLM is steered)"):
        st.code(CLASSIFIER_SYSTEM_PROMPT, language="text")
