"""Evaluation harness — the project's "model evaluation & test coverage" (Task 7).

This file has two halves:

  CLASSIFICATION metrics (deterministic) — accuracy on the labeled EVAL_SET, a
  confusion matrix, routing success rate (does each message reach the route it
  should?), escalation behaviour on the ambiguous ESCALATION_SET, mean confidence,
  and per-confidence-bin calibration.

  RESPONSE-QUALITY metrics (LLM-as-judge) — empathy/clarity scoring of generated
  replies. This is a TONE sniff-test, not a correctness gate (see the honest framing
  above the judge below); factual correctness is gated deterministically elsewhere.

Run live (needs ANTHROPIC_API_KEY):  python evaluation.py
Tests inject a fake classify_fn / judge so this logic is verifiable without the API.
"""

from collections import defaultdict

from pydantic import BaseModel, Field

import config
import responder
from config import CONFIDENCE_THRESHOLD, LABELS, ROUTE_ESCALATE
from graph import route_for
from seed_data import ESCALATION_SET, EVAL_SET


def _route_for(pred) -> str:
    """Routing decision for a classification — delegates to the SINGLE source of the
    rule (graph.route_for) so eval scoring can't drift from production routing."""
    return route_for(pred.label, pred.confidence)


def evaluate(classify_fn=None) -> dict:
    """Run the eval set through the classifier and compute metrics."""
    if classify_fn is None:
        from classifier import classify as classify_fn

    confusion = defaultdict(lambda: defaultdict(int))  # expected -> predicted -> count
    correct = 0
    routing_hits = 0
    confidences = []
    # Calibration: bucket each prediction by confidence band and track accuracy, so
    # we can see whether the 0.60 escalation threshold is actually justified by the
    # model's self-reported confidence (rather than assumed).
    bins = {"low (<0.6)": [0, 0], "med (0.6-0.9)": [0, 0], "high (>=0.9)": [0, 0]}

    for message, expected in EVAL_SET:
        pred = classify_fn(message)
        confusion[expected][pred.label] += 1
        confidences.append(pred.confidence)
        hit = pred.label == expected
        correct += hit
        if _route_for(pred) == expected:  # expected route == expected label here
            routing_hits += 1
        band = "high (>=0.9)" if pred.confidence >= 0.9 else "med (0.6-0.9)" if pred.confidence >= 0.6 else "low (<0.6)"
        bins[band][0] += int(hit)
        bins[band][1] += 1

    # Ambiguous/out-of-scope: success = correctly routed to a human.
    escalation_hits = sum(1 for m in ESCALATION_SET if _route_for(classify_fn(m)) == ROUTE_ESCALATE)

    n = len(EVAL_SET)
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "routing_success_rate": routing_hits / n if n else 0.0,
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "escalation_n": len(ESCALATION_SET),
        "escalation_correct": escalation_hits,
        "calibration": {b: {"correct": c, "total": t, "accuracy": (c / t if t else 0.0)}
                        for b, (c, t) in bins.items()},
    }


# --- Response-quality evaluation (LLM-as-judge) ------------------------------
# The brief asks evaluation to cover "empathy level" and "clarity of status
# updates" — qualities of the GENERATED text that classification metrics can't
# capture. We use an LLM-as-judge: a separate model call scores each reply on empathy
# and clarity (1-5) with a rationale.
#
# HONEST FRAMING: this is a TONE sniff-test, not a correctness gate. It's the SAME
# model grading its own family of outputs against no ground truth, on a tiny sample —
# so treat the scores as a directional signal on warmth/readability, not a pass/fail
# quality bar. Factual correctness is gated separately and deterministically: the
# ticket number is code-supplied and verified (responder.py), status text is read
# from the DB (handlers.py), and RAG numbers are validated against context
# (knowledge.py). The judge complements those checks; it does not replace them.

class JudgeScore(BaseModel):
    empathy: int = Field(ge=1, le=5, description="How warm/empathetic the reply is (1-5).")
    clarity: int = Field(ge=1, le=5, description="How clear and easy to understand (1-5).")
    rationale: str = Field(description="One short sentence justifying the scores.")


_JUDGE_SYSTEM = """You are a strict QA reviewer for a bank's customer support replies.
Score the given reply on empathy (warmth, acknowledgement of the customer) and
clarity (is the message and any status/ticket info easy to understand?), each 1-5.
Give a one-sentence rationale. Be discerning — reserve 5s for genuinely excellent
replies."""

_judge_singleton = None


def _build_judge():
    import llm
    return llm.chat_model(256, temperature=0).with_structured_output(JudgeScore)


def _judge():
    global _judge_singleton
    if _judge_singleton is None:
        _judge_singleton = _build_judge()
    return _judge_singleton


def judge_response(message: str, response: str, _judge_fn=None) -> JudgeScore:
    """Score one customer-facing reply for empathy + clarity."""
    runnable = _judge_fn if _judge_fn is not None else _judge()
    return runnable.invoke([
        ("system", _JUDGE_SYSTEM),
        ("human", f"Customer message: {message}\n\nSupport reply: {response}"),
    ])


def collect_response_samples() -> list[tuple[str, str, str]]:
    """Produce representative (case, customer_message, generated_reply) tuples to
    judge — one per customer-facing path that emits prose."""
    return [
        ("positive", "Thanks for sorting out my login issue!",
         responder.generate_thankyou("Jordan")),
        ("negative", "My debit card replacement still hasn't arrived.",
         responder.generate_apology("Jordan", "650932")),
        ("query", "Status of ticket 650932?",
         config.TEMPLATE_QUERY_FOUND.format(ticket_id="650932", status="Resolved")),
        ("escalate", "asldkfj 4567 ??",
         config.TEMPLATE_ESCALATE.format(customer_name="Jordan")),
    ]


def evaluate_response_quality(judge_fn=None, samples=None) -> dict:
    """Judge each sample reply; return mean empathy/clarity + per-sample detail."""
    samples = samples if samples is not None else collect_response_samples()
    rows, emp, clr = [], [], []
    for case, message, reply in samples:
        score = judge_response(message, reply, _judge_fn=judge_fn)
        emp.append(score.empathy)
        clr.append(score.clarity)
        rows.append({"case": case, "reply": reply, "empathy": score.empathy,
                     "clarity": score.clarity, "rationale": score.rationale})
    n = len(rows) or 1
    return {
        "mean_empathy": sum(emp) / n,
        "mean_clarity": sum(clr) / n,
        "samples": rows,
    }


def print_report(report: dict) -> None:
    """Pretty-print the metrics as a readable console report."""
    print("=" * 60)
    print("TriageDesk — Classification Evaluation")
    print("=" * 60)
    print(f"Test cases:             {report['n']}")
    print(f"Classification accuracy: {report['accuracy']:.0%}")
    print(f"Routing success rate:    {report['routing_success_rate']:.0%}")
    print(f"Mean confidence:         {report['mean_confidence']:.2f}")
    print(f"Confidence threshold:    {CONFIDENCE_THRESHOLD:.2f}")
    print()
    print("Confusion matrix (rows = true label, cols = predicted):")
    header = "  " + " ".join(f"{lbl[:8]:>10}" for lbl in LABELS)
    print(f"{'':>20}{header}")
    for true_label in LABELS:
        row = report["confusion"].get(true_label, {})
        cells = " ".join(f"{row.get(pred, 0):>10}" for pred in LABELS)
        print(f"{true_label:>20}  {cells}")
    print()
    print("Confidence calibration (does the threshold hold up?):")
    for band, s in report.get("calibration", {}).items():
        if s["total"]:
            print(f"  {band:<16} accuracy {s['accuracy']:.0%}  ({s['correct']}/{s['total']})")
    print()
    print(
        f"Escalation (ambiguous/out-of-scope correctly sent to a human): "
        f"{report['escalation_correct']}/{report['escalation_n']}"
    )
    print("=" * 60)


def print_quality_report(report: dict) -> None:
    """Pretty-print the LLM-as-judge response-quality scores."""
    print()
    print("=" * 60)
    print("Response quality (LLM-as-judge, 1-5)")
    print("=" * 60)
    print(f"Mean empathy: {report['mean_empathy']:.2f}   "
          f"Mean clarity: {report['mean_clarity']:.2f}")
    print()
    for s in report["samples"]:
        print(f"[{s['case']:<8}] empathy={s['empathy']} clarity={s['clarity']}")
        print(f"           reply: {s['reply']}")
        print(f"           judge: {s['rationale']}")
    print("=" * 60)


if __name__ == "__main__":
    print_report(evaluate())
    print_quality_report(evaluate_response_quality())
