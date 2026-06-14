"""RAG knowledge layer: retrieval + grounded answering + graceful deferral."""

import knowledge
from handlers import handle_general
from seed_data import DEMO_CUSTOMER_ID


class _FakeLLM:
    def __init__(self, text):
        self._text = text

    def invoke(self, _messages):
        class _M:
            pass
        m = _M()
        m.content = self._text
        return m


# --- retrieval ---------------------------------------------------------------
def test_retrieve_finds_relevant_passage():
    hits = knowledge.retrieve("what are your international wire fees?")
    assert hits
    assert any("fee" in h["tags"] for h in hits)


def test_retrieve_returns_nothing_for_unrelated_query():
    assert knowledge.retrieve("what's the weather on mars") == []


# --- answer (grounding) ------------------------------------------------------
def test_answer_defers_when_nothing_retrieved():
    out = knowledge.answer("tell me a joke about penguins", "Jordan")
    assert out["grounded"] is False
    assert "connect you" in out["response"].lower()


def test_answer_extractive_fallback_when_llm_off():
    # conftest sets USE_LLM_RESPONSES=False -> extractive: returns the passage.
    out = knowledge.answer("international transaction fees", "Jordan")
    assert out["grounded"] is True
    assert "1%" in out["response"] or "foreign transaction" in out["response"].lower()


def test_answer_uses_grounded_llm_text():
    out = knowledge.answer("atm limit?", "Jordan",
                           _llm=_FakeLLM("Your daily ATM limit is 500 dollars."))
    assert out["grounded"] is True
    assert "500" in out["response"]


def test_answer_defers_when_model_says_insufficient():
    out = knowledge.answer("atm limit?", "Jordan", _llm=_FakeLLM("INSUFFICIENT_CONTEXT"))
    assert out["grounded"] is False


def test_answer_defers_when_llm_introduces_ungrounded_number():
    # The ATM passage says 500; a draft inventing "1000" must be rejected (fact-validator).
    out = knowledge.answer("atm limit?", "Jordan", _llm=_FakeLLM("Your ATM limit is 1000 dollars."))
    assert out["grounded"] is False


def test_numbers_are_grounded_helper():
    assert knowledge._numbers_are_grounded("the limit is 500", "daily limit 500 dollars")
    assert not knowledge._numbers_are_grounded("the limit is 1000", "daily limit 500 dollars")


# --- handler -----------------------------------------------------------------
def test_handle_general_reports_kb_action():
    out = handle_general({"message": "what are the branch hours?",
                          "customer_id": DEMO_CUSTOMER_ID, "customer_name": "Jordan"})
    assert "kb:" in out["db_action"]
    assert out["response"]
