"""Archived gates for the prior-reversion protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ExposureGateReport:
    page_hit: bool
    pivot_visible: bool
    pivot_unambiguous: bool
    pivot_comprehended: bool
    visibility_judgment: dict
    comprehension_judgment: dict
    comprehension_response: str
    eligible: bool
    failure_reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_exposure(
    *,
    page_hit: bool,
    case: dict,
    pages: list[dict],
    messages: list[dict],
    tested_model: str,
    judge,
    call_model_fn,
) -> ExposureGateReport:
    """Require page hit, visible unambiguous update, and immediate comprehension."""
    failures = []
    if not page_hit:
        failures.append("page_not_hit")
        return ExposureGateReport(
            page_hit=False, pivot_visible=False, pivot_unambiguous=False,
            pivot_comprehended=False, visibility_judgment={}, comprehension_judgment={},
            comprehension_response="", eligible=False, failure_reasons=failures,
        )

    visibility = judge.judge_visibility(
        case["pk_question"], case.get("new_answer_keywords", []),
        case.get("old_answer_keywords", []), pages,
    )
    visible = visibility.decision == "visible"
    unambiguous = visibility.decision == "visible"
    if not visible:
        failures.append(f"pivot_{visibility.decision}")

    response = ""
    comprehension_dict = {}
    comprehended = False
    if visible and unambiguous:
        probe_messages = list(messages)
        probe_messages.append({
            "role": "user",
            "content": (
                "Browsing is now paused. Based on the pages you just read, answer this question "
                f"directly and briefly: {case['pk_question']}"
            ),
        })
        response = call_model_fn(tested_model, probe_messages, temperature=0.0)
        comprehension = judge.judge_answer(
            case["pk_question"], response, case.get("new_answer_keywords", []),
            case.get("old_answer_keywords", []), pages,
        )
        comprehension_dict = comprehension.to_dict()
        comprehended = comprehension.decision == "stick_new"
        if not comprehended:
            failures.append(f"pivot_not_comprehended:{comprehension.decision}")

    return ExposureGateReport(
        page_hit=page_hit, pivot_visible=visible, pivot_unambiguous=unambiguous,
        pivot_comprehended=comprehended, visibility_judgment=visibility.to_dict(),
        comprehension_judgment=comprehension_dict, comprehension_response=response,
        eligible=not failures, failure_reasons=failures,
    )


def answerability_for_item(judge, item: dict, pages: list[dict], arm: str):
    answers = (item.get("new_keywords", []) if arm == "conflict"
               else item.get("answer_keywords", []))
    return judge.judge_answerability(item["question"], answers, pages)
