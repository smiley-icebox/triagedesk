"""RAG over a small banking FAQ — retrieval-augmented answers for general questions.

The query handler answers questions about EXISTING tickets. This handles the other
kind of question — "what are your wire fees?", "how do I reset my password?" — that
a real support assistant must field. The shape is textbook RAG:

    retrieve relevant passages  →  put them in the prompt  →  generate a GROUNDED
    answer that uses only those passages.

Two deliberate, honest choices:
  - Retrieval is a lightweight keyword/overlap scorer over a small in-memory FAQ,
    not a vector store. Anthropic doesn't provide embeddings, and pulling in a
    second provider just for a demo FAQ is accidental complexity. The retriever is
    a clean function, so swapping in embeddings + a vector DB later is a drop-in.
  - Generation is grounded and fenced: the model is told to answer ONLY from the
    retrieved context and to defer to a human if the answer isn't there. If nothing
    relevant is retrieved (or the LLM is off/unavailable), we DON'T guess — we hand
    off. A support bot that confidently invents a fee schedule is worse than one
    that says "let me connect you with someone."
"""

import re

import config
import llm
from config import DEFAULT_CUSTOMER_NAME, TEMPLATE_GENERAL_FALLBACK

# The knowledge base. In production this is a maintained content store behind a
# vector index; here it's a handful of FAQ entries with keyword tags.
FAQ: list[dict] = [
    {"topic": "branch & support hours",
     "tags": "hours open close branch when time weekend support available",
     "content": "Branches are open Monday-Friday 9am-5pm and Saturday 9am-1pm. "
                "Phone and chat support are available 24/7."},
    {"topic": "card replacement",
     "tags": "card replace replacement lost stolen new debit credit arrive delivery",
     "content": "A replacement card is issued within 1 business day and typically "
                "arrives in 5-7 business days. You can track it in the mobile app under Cards."},
    {"topic": "password reset",
     "tags": "password reset login forgot online banking access locked username",
     "content": "To reset your online banking password, choose 'Forgot password' on "
                "the login screen and follow the verification steps. For security, "
                "you may be asked to confirm a code sent to your registered phone."},
    {"topic": "international transaction fees",
     "tags": "international foreign transaction fee fees abroad currency exchange wire",
     "content": "International transactions carry a 1% foreign transaction fee. "
                "Outgoing international wires are 25 dollars; incoming are 15 dollars."},
    {"topic": "ATM withdrawal limits",
     "tags": "atm withdrawal limit cash daily max maximum",
     "content": "The standard daily ATM withdrawal limit is 500 dollars. You can "
                "request a temporary increase in the app under Card Settings."},
    {"topic": "disputing a transaction",
     "tags": "dispute transaction charge fraud unauthorized chargeback wrong",
     "content": "To dispute a transaction, open it in your statement and choose "
                "'Dispute'. Most disputes are resolved within 10 business days, and a "
                "provisional credit may be applied while we investigate."},
]

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    # Drop 1-2 char tokens ("a", "me", "is") so stop-words in FAQ topics like
    # "disputing a transaction" can't spuriously match an unrelated query.
    return {t for t in _WORD_RE.findall((text or "").lower()) if len(t) >= 3}


def retrieve(query: str, k: int = 2) -> list[dict]:
    """Return the top-k FAQ entries by keyword overlap with the query.

    A simple, transparent scorer (no embeddings): count query tokens that appear in
    each entry's tags/topic. Entries with zero overlap are dropped, so an unrelated
    question retrieves nothing — which the caller treats as "I don't know."
    """
    q = _tokens(query)
    scored = []
    for entry in FAQ:
        overlap = len(q & _tokens(entry["tags"] + " " + entry["topic"]))
        if overlap > 0:
            scored.append((overlap, entry))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [e for _, e in scored[:k]]


_GROUNDED_SYSTEM = """You are a bank's support assistant answering a general question.
Answer ONLY using the provided context passages. Rules:
- If the context doesn't contain the answer, reply EXACTLY: INSUFFICIENT_CONTEXT
- Be concise (1-3 sentences), warm, and accurate.
- Do not invent fees, limits, dates, or policies beyond the context."""

_kb_llm_singleton = None


def _kb_llm():
    global _kb_llm_singleton
    if _kb_llm_singleton is None:
        _kb_llm_singleton = llm.chat_model(200, temperature=0.2)
    return _kb_llm_singleton


_NUM_RE = re.compile(r"\d+")


def _numbers(text: str) -> set[str]:
    return set(_NUM_RE.findall(text or ""))


def _numbers_are_grounded(answer_text: str, context: str) -> bool:
    """The fact-validator: every number/amount/percent in the generated answer must
    appear in the retrieved context. This is the post-generation check the answer
    path was missing — it stops a fluent-but-wrong fee/limit from shipping as
    grounded. (Numbers are the fact most likely to be hallucinated here.)"""
    allowed = _numbers(context)
    return all(n in allowed for n in _numbers(answer_text))


def answer(query: str, customer_name: str | None = None, _llm=None) -> dict:
    """Answer a general question via RAG. Returns {response, grounded, topics}.

    grounded=True means we answered from the FAQ; False means we deferred to a human
    (nothing retrieved, the model said the context was insufficient, the answer
    introduced an unsupported number, or LLM generation is off).
    """
    name = customer_name or DEFAULT_CUSTOMER_NAME
    passages = retrieve(query, k=2)
    topics = [p["topic"] for p in passages]

    def _defer():
        return {"response": TEMPLATE_GENERAL_FALLBACK.format(customer_name=name),
                "grounded": False, "topics": topics}

    if not passages:
        return _defer()

    # Extractive fallback when LLM generation is off / no key: return the most
    # relevant passage verbatim. Inherently grounded (it IS the source text).
    if not config.USE_LLM_RESPONSES and _llm is None:
        return {"response": passages[0]["content"], "grounded": True, "topics": topics}

    context = "\n".join(f"- {p['content']}" for p in passages)
    try:
        client = _llm if _llm is not None else _kb_llm()
        msg = client.invoke([
            ("system", _GROUNDED_SYSTEM),
            ("human", f"Context:\n{context}\n\nQuestion: {query}"),
        ])
        text = llm.extract_text(getattr(msg, "content", "")).strip()
    except Exception:
        text = ""

    # Defer if: empty, the model flagged insufficient context, OR the answer
    # introduced a number that isn't in the retrieved passages (hallucination guard).
    if not text or "INSUFFICIENT_CONTEXT" in text or not _numbers_are_grounded(text, context):
        return _defer()
    return {"response": text, "grounded": True, "topics": topics}
