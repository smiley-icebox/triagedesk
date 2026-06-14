"""Evaluation harness: metrics compute correctly given a known classifier."""

from collections import namedtuple

from evaluation import JudgeScore, evaluate, evaluate_response_quality
from seed_data import ESCALATION_SET, EVAL_SET

Fake = namedtuple("Fake", "label confidence")


def test_perfect_classifier_scores_100_percent():
    truth = dict(EVAL_SET)

    def perfect(msg):
        if msg in truth:
            return Fake(truth[msg], 0.95)
        return Fake("query", 0.10)  # ambiguous -> low conf -> escalate

    report = evaluate(classify_fn=perfect)
    assert report["accuracy"] == 1.0
    assert report["routing_success_rate"] == 1.0
    assert report["escalation_correct"] == len(ESCALATION_SET)


def test_confusion_matrix_records_mistakes():
    # A classifier that always says "query" should be right only on the query rows.
    report = evaluate(classify_fn=lambda _m: Fake("query", 0.95))
    n_query = sum(1 for _m, lbl in EVAL_SET if lbl == "query")
    assert report["accuracy"] == n_query / len(EVAL_SET)
    # every true-positive_feedback row was predicted "query"
    assert report["confusion"]["positive_feedback"]["query"] > 0


class _FakeJudge:
    def invoke(self, _messages):
        return JudgeScore(empathy=4, clarity=5, rationale="clear and warm")


def test_response_quality_aggregates_judge_scores():
    samples = [
        ("positive", "thanks!", "So glad we could help!"),
        ("negative", "broken", "So sorry — ticket #240716 is open."),
    ]
    report = evaluate_response_quality(judge_fn=_FakeJudge(), samples=samples)
    assert report["mean_empathy"] == 4.0
    assert report["mean_clarity"] == 5.0
    assert len(report["samples"]) == 2
    assert report["samples"][0]["case"] == "positive"
