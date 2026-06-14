"""LLM responder: guardrails + template fallback (offline, via a fake llm)."""

import responder
from seed_data import DEMO_CUSTOMER_NAME


class _FakeLLM:
    """Returns a fixed text as the model's reply."""

    def __init__(self, text):
        self._text = text

    def invoke(self, _messages):
        class _Msg:
            pass
        m = _Msg()
        m.content = self._text
        return m


# --- _is_safe validator ------------------------------------------------------
def test_is_safe_accepts_clean_text():
    assert responder._is_safe("Thank you so much, Jordan! We're glad to help.")


def test_is_safe_rejects_overlong():
    assert not responder._is_safe("x" * 1000)


def test_is_safe_rejects_dollar_and_urls_and_stray_digits():
    assert not responder._is_safe("We refunded you $50.")
    assert not responder._is_safe("See http://bank.example for details.")
    assert not responder._is_safe("Call 5551234 now.")  # stray digits = fabricated


def test_is_safe_allows_the_one_permitted_ticket_id():
    assert responder._is_safe("Your ticket #650932 is logged.", allowed_digits="650932")
    assert not responder._is_safe("Tickets #650932 and #111111.", allowed_digits="650932")


# --- thank-you ---------------------------------------------------------------
def test_thankyou_uses_valid_llm_draft():
    out = responder.generate_thankyou("Jordan", _llm=_FakeLLM("So glad we could help, Jordan!"))
    assert out == "So glad we could help, Jordan!"


def test_thankyou_falls_back_when_draft_invalid():
    # Draft sneaks in a dollar amount -> rejected -> template.
    out = responder.generate_thankyou("Jordan", _llm=_FakeLLM("Here's $5 for your trouble"))
    assert "Thank you for your kind words, Jordan" in out


# --- apology -----------------------------------------------------------------
def test_apology_keeps_draft_that_includes_ticket():
    draft = "So sorry, Jordan — we've opened ticket #240716 and will follow up shortly."
    out = responder.generate_apology("Jordan", "240716", _llm=_FakeLLM(draft))
    assert out == draft


def test_apology_falls_back_when_ticket_missing_from_draft():
    # Model forgot the ticket number -> we must not ship it -> template (which has it).
    out = responder.generate_apology("Jordan", "240716", _llm=_FakeLLM("So sorry for the trouble!"))
    assert "#240716" in out
    assert "apolog" in out.lower()


def test_apology_falls_back_on_invented_promise():
    # A draft promising something the system can't honor is rejected -> template.
    draft = "So sorry, Jordan — a senior manager will personally call you tomorrow about #240716."
    out = responder.generate_apology("Jordan", "240716", _llm=_FakeLLM(draft))
    assert "manager" not in out.lower() and "personally" not in out.lower()
    assert "#240716" in out  # the approved template still cites the ticket


def test_is_safe_rejects_promises_and_timelines():
    assert not responder._is_safe("We'll refund you and a manager will call you.")
    assert not responder._is_safe("We'll resolve this by Friday.")
    assert responder._is_safe("We're sorry, Jordan — the team will follow up shortly.")
