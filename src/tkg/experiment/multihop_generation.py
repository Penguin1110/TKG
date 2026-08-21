"""Build auditable cutoff-relative, multi-hop Wikipedia questions.

The seed supplies semantic relations; Wikipedia supplies the revision text and
hyperlink graph.  A question-writer LLM sees the verified private entity chain
and verbalizes it without revealing the hidden answers.  Deterministic leakage
and temporal-wording gates run before an independent LLM judges whether the
evidence really expresses the claimed relation chain.  The pipeline never
invents a relation from hyperlink topology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tkg.api.openrouter import (
    UsageLedger, call_model, model_call_context, record_usage_event,
    set_usage_ledger,
)
from tkg.experiment.case_validation import validate_chain_route
from tkg.experiment.event_order import event_order_certificate_errors
from tkg.experiment.model_cutoffs import ModelCutoff, get_model_cutoff
from tkg.experiment.question_generation import (
    _fold, _strings, call_json_model,
)
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.validity_contract_v25 import (
    bounded_shortcut_diagnostic, validity_contract,
)
from tkg.wikipedia.backend import WikipediaError, WikipediaPageBackend
from tkg.wikipedia.snapshot import temporal_reverse_bfs


SCHEMA_VERSION = "wikipedia-cutoff-relative-multihop-v6"
MIN_HOPS = 2
MAX_HOPS = 6
TIME_POLICIES = {"advance_required", "same_snapshot"}
WHOLE_CHAIN_CHECKS = (
    "evidence_semantics", "relation_order_and_composition",
    "cutoff_and_snapshot_anchoring", "multi_hop_and_uniqueness",
    "no_entity_leakage", "temporal_transition_clarity",
    "natural_question_wording",
)
QUESTION_WRITER_PROMPT_VERSION = 11
QUESTION_WRITER_MAX_ATTEMPTS = 3
TENURE_ONSET_PATTERN = (
    r"\b(?:(?:begin|began) (?:holding|to hold|serving)|"
    r"(?:start|started) (?:holding|to hold|serving)|"
    r"took (?:up )?(?:the )?(?:office|position|post|role)|"
    r"assumed (?:the )?(?:office|position|post|role)|entered office|"
    r"was appointed (?:as|to)|became)\b"
)


def has_infrastructure_error(errors: list[str]) -> bool:
    """Separate retryable provider failures from semantic gate failures."""
    markers = (
        "Wikipedia fetch failed", "HTTP 429", "HTTP 5", "timed out",
        "connection", "temporarily unavailable", "API",
    )
    return any(
        marker.casefold() in error.casefold()
        for error in errors for marker in markers
    )


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _frozen_page(page: Any) -> dict[str, Any]:
    links = sorted(
        ({"target": link.target, "anchor": link.anchor} for link in page.links),
        key=lambda row: (row["target"].casefold(), row["anchor"].casefold()),
    )
    record = {
        "title": page.title, "revision_id": page.revision_id,
        "timestamp": page.timestamp, "as_of": page.as_of,
        "source_url": page.source_url, "content": page.content, "links": links,
        "content_sha256": hashlib.sha256(page.content.encode("utf-8")).hexdigest(),
        "links_sha256": _canonical_sha256(links),
        "render_contract": "wikipedia-rendered-page-v1",
    }
    record["snapshot_sha256"] = _canonical_sha256(record)
    return record


def _public_chain(chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in hop.items() if not key.startswith("_")}
        for hop in chain
    ]


@dataclass(frozen=True)
class ChainHopSeed:
    source_title: str
    target_title: str
    relation: str
    relative_clause: str
    as_of: str
    evidence: str
    target_aliases: tuple[str, ...]
    incoming_time_policy: str
    property_id: str | None = None
    relation_family: str | None = None
    structured_evidence: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any], index: int) -> "ChainHopSeed":
        required = (
            "source_title", "target_title", "relation", "relative_clause",
            "as_of", "evidence", "target_aliases",
        )
        missing = [field for field in required if not value.get(field)]
        if missing:
            raise ValueError(f"hop {index}: missing fields: {', '.join(missing)}")
        aliases = _strings(value.get("target_aliases"))
        if aliases is None:
            raise ValueError(f"hop {index}: target_aliases must be a non-empty string list")
        clause = str(value["relative_clause"]).strip()
        if clause.count("{source}") != 1:
            raise ValueError(f"hop {index}: relative_clause needs exactly one {{source}}")
        default_policy = "advance_required"
        time_policy = str(value.get("incoming_time_policy", default_policy))
        if time_policy not in TIME_POLICIES:
            raise ValueError(
                f"hop {index}: incoming_time_policy must be one of "
                f"{sorted(TIME_POLICIES)!r}"
            )
        if time_policy == "same_snapshot":
            tail_required = ("property_id", "relation_family", "structured_evidence")
            tail_missing = [field for field in tail_required if not value.get(field)]
            if tail_missing:
                raise ValueError(
                    f"hop {index}: same_snapshot attribute tail missing "
                    f"{', '.join(tail_missing)}"
                )
        _parse_time(str(value["as_of"]))
        return cls(
            source_title=str(value["source_title"]).strip(),
            target_title=str(value["target_title"]).strip(),
            relation=str(value["relation"]).strip(),
            relative_clause=clause,
            as_of=str(value["as_of"]),
            evidence=str(value["evidence"]).strip(),
            target_aliases=tuple(aliases),
            incoming_time_policy=time_policy,
            property_id=(str(value["property_id"]) if value.get("property_id") else None),
            relation_family=(
                str(value["relation_family"]) if value.get("relation_family") else None
            ),
            structured_evidence=(
                dict(value["structured_evidence"])
                if isinstance(value.get("structured_evidence"), dict) else None
            ),
        )


@dataclass(frozen=True)
class MultiHopSeed:
    id: str
    model_id: str
    cutoff: ModelCutoff
    target_as_of: str
    anchor_label: str
    answer_kind: str
    category: str
    hops: tuple[ChainHopSeed, ...]
    old_answer_keywords: tuple[str, ...] = ()
    selection_metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MultiHopSeed":
        for field in ("id", "model_id", "target_as_of", "anchor_label", "hops"):
            if not value.get(field):
                raise ValueError(f"multi-hop seed missing {field}")
        cutoff = get_model_cutoff(str(value["model_id"]))
        supplied_cutoff = value.get("cutoff_date")
        if supplied_cutoff and str(supplied_cutoff) != cutoff.cutoff_date:
            raise ValueError(
                f"{value['id']}: cutoff_date {supplied_cutoff!r} disagrees with pinned "
                f"registry date {cutoff.cutoff_date!r}"
            )
        target_as_of = str(value["target_as_of"])
        if _parse_time(target_as_of) <= _parse_time(cutoff.cutoff_date):
            raise ValueError(f"{value['id']}: target_as_of must be after model cutoff")
        raw_hops = value["hops"]
        if not isinstance(raw_hops, list) or not MIN_HOPS <= len(raw_hops) <= MAX_HOPS:
            raise ValueError(
                f"{value['id']}: reasoning chain must contain {MIN_HOPS}..{MAX_HOPS} hops"
            )
        hops = tuple(ChainHopSeed.from_dict(item, i) for i, item in enumerate(raw_hops))
        if "registered knowledge cutoff" not in hops[0].relative_clause.casefold():
            raise ValueError(
                f"{value['id']}: first relative_clause must name the registered "
                "knowledge cutoff"
            )
        if hops[0].as_of != cutoff.cutoff_date:
            raise ValueError(
                f"{value['id']}: first hop must use cutoff snapshot {cutoff.cutoff_date}"
            )
        if hops[-1].as_of != target_as_of:
            raise ValueError(f"{value['id']}: final hop must use target_as_of")
        if not any(_parse_time(hop.as_of) > _parse_time(cutoff.cutoff_date) for hop in hops[1:]):
            raise ValueError(f"{value['id']}: chain has no post-cutoff hop")
        for index, (left, right) in enumerate(zip(hops, hops[1:])):
            if left.target_title.casefold() != right.source_title.casefold():
                raise ValueError(
                    f"{value['id']}: disconnected hops {index}->{index + 1}: "
                    f"{left.target_title!r} != {right.source_title!r}"
                )
            left_time = _parse_time(left.as_of)
            right_time = _parse_time(right.as_of)
            if right.incoming_time_policy == "advance_required":
                if right_time <= left_time:
                    raise ValueError(
                        f"{value['id']}: hop {index + 1} must use a later snapshot after "
                        f"changing to entity {left.target_title!r}"
                    )
            elif right_time != left_time:
                raise ValueError(
                    f"{value['id']}: same_snapshot hop {index + 1} must retain "
                    f"snapshot {left.as_of}"
                )
        entity_path = [hops[0].source_title, *(hop.target_title for hop in hops)]
        folded_path = [title.casefold() for title in entity_path]
        if len(set(folded_path)) != len(folded_path):
            raise ValueError(f"{value['id']}: reasoning chain contains an entity cycle")
        old = value.get("old_answer_keywords", [])
        if not isinstance(old, list) or not all(isinstance(item, str) for item in old):
            raise ValueError(f"{value['id']}: old_answer_keywords must be a string list")
        return cls(
            id=str(value["id"]), model_id=cutoff.model_id, cutoff=cutoff,
            target_as_of=target_as_of, anchor_label=str(value["anchor_label"]).strip(),
            answer_kind=str(value.get("answer_kind", "Who")).strip() or "Who",
            category=str(value.get("category", "other")), hops=hops,
            old_answer_keywords=tuple(item.strip() for item in old if item.strip()),
            selection_metadata=(
                dict(value["selection_metadata"])
                if isinstance(value.get("selection_metadata"), dict) else None
            ),
        )


def compose_canonical_question(seed: MultiHopSeed) -> str:
    """Return the exact nested logical form used for machine auditing."""
    description = seed.hops[0].relative_clause.replace("{source}", seed.anchor_label)
    for hop in seed.hops[1:]:
        description = hop.relative_clause.replace("{source}", description)
    return f"At the target snapshot, {seed.answer_kind.casefold()} is {description}?"


def _sentence_clause(clause: str, source: str) -> str:
    rendered = " ".join(clause.replace("{source}", source).split()).strip()
    if rendered.casefold().startswith("the "):
        rendered = rendered[4:]
    return rendered.rstrip(".?")


def _without_ambiguous_after_that(clause: str) -> str:
    return re.sub(r"\s+after that\s*$", "", clause, flags=re.IGNORECASE).strip()


def _onset_wording_required(hop: ChainHopSeed) -> bool:
    return hop.incoming_time_policy == "advance_required" and hop.property_id == "P39"


def _order_wording_required(hop: ChainHopSeed) -> bool:
    """Whether a P39 edge claims the first/next tenure after a boundary."""
    if not _onset_wording_required(hop):
        return False
    operator = (hop.structured_evidence or {}).get("temporal_operator")
    return operator not in {"relation_after_boundary", "exact_event_date"}


def _hop_answer_kind(seed: MultiHopSeed, index: int, hop: ChainHopSeed) -> str:
    if index == len(seed.hops) - 1:
        return seed.answer_kind.title()
    structured = hop.structured_evidence or {}
    if hop.property_id == "P39":
        return "Who" if structured.get("direction") == "inverse" else "What"
    clause = _fold(hop.relative_clause)
    if any(
        token in clause
        for token in ("person", "spouse", "officeholder", "author", "successor")
    ):
        return "Who"
    return "What"


def _clarify_temporal_relation(hop: ChainHopSeed, clause: str) -> str:
    """Express P39 as a new tenure beginning, not any old role on a later page."""
    if not _onset_wording_required(hop) or re.search(
        TENURE_ONSET_PATTERN, clause, flags=re.IGNORECASE,
    ):
        return clause
    match_clause = re.sub(r"^the\s+", "", clause, flags=re.IGNORECASE)
    inverse = re.fullmatch(
        r"next person to hold (.+)", match_clause, flags=re.IGNORECASE
    )
    if inverse:
        return f"first person who began holding {inverse.group(1)}"
    forward = re.fullmatch(
        r"next (.+?position) held by (.+)", match_clause, flags=re.IGNORECASE,
    )
    if forward:
        return f"first {forward.group(1)} that {forward.group(2)} began holding"
    return clause


def _event_boundary(seed: MultiHopSeed, index: int) -> str:
    """Return a world-event boundary independent of the solver's browsing choices."""
    if index == 1:
        return "the registered knowledge cutoff"
    previous = seed.hops[index - 1]
    if previous.property_id == "P39":
        direction = (previous.structured_evidence or {}).get("direction")
        if direction == "forward":
            return "the person identified two steps earlier began holding that position"
        if direction == "inverse":
            return "the person identified in the previous step began holding that position"
        return "the previous officeholder's tenure began"
    return "the event identified in the previous step occurred"


def _has_previous_tenure_onset_boundary(text: str) -> bool:
    """Accept event language equivalent to comparing two tenure start times."""
    return bool(
        re.search(
            r"\bafter\b[^?]{0,160}\btenure\b[^?]{0,100}\b(?:began|started)\b",
            text,
        )
        or re.search(
            r"\bafter\b[^?]{0,160}\b(?:person|individual|officeholder)\b"
            r"[^?]{0,100}\b(?:began|started)\s+(?:to\s+)?"
            r"(?:hold|holding|serve|serving)\b",
            text,
        )
    )


def _has_forward_p39_event_boundary(text: str) -> bool:
    """Bind a later officeholder to the earlier person's tenure-start event."""
    reference = re.search(
        r"\b(?:two steps earlier|(?:the )?first step|step\s+1)\b", text,
    )
    event = re.search(
        r"\bafter\b[^?]{0,180}\bperson\b[^?]{0,120}"
        r"\b(?:began|started)\s+(?:to\s+)?(?:hold|holding|serve|serving)\b",
        text,
    )
    return bool(reference and event)


def _writer_source_reference(seed: MultiHopSeed, index: int) -> str:
    if index == 0:
        return seed.anchor_label
    previous = seed.hops[index - 1]
    if _hop_answer_kind(seed, index - 1, previous).casefold() == "who":
        return "the person identified in the previous step"
    if previous.property_id == "P39":
        return "the position identified in the previous step"
    return "the item identified in the previous step"


def _hidden_entity_names(seed: MultiHopSeed) -> tuple[str, ...]:
    """Return every private chain name that must not reach the tested prompt."""
    public_anchor = _fold(seed.anchor_label)
    names = {
        name.strip()
        for hop in seed.hops
        for name in (hop.target_title, *hop.target_aliases)
        if name.strip() and _fold(name) != public_anchor
    }
    return tuple(sorted(names, key=lambda value: (_fold(value), value)))


def compose_relative_question(seed: MultiHopSeed) -> str:
    """Render a temporally explicit fallback without exposing hidden entities.

    The nested canonical form remains available for audit, but is unsuitable as
    the tested-model prompt. Every advancing hop is anchored to a world event,
    never to whichever revision the solver happened to inspect.
    """
    first = _sentence_clause(seed.hops[0].relative_clause, seed.anchor_label)
    sentences = [f"Start with {seed.anchor_label}.", f"First, identify the {first}."]
    for index, hop in enumerate(seed.hops[1:-1], start=1):
        clause = _clarify_temporal_relation(
            hop, _without_ambiguous_after_that(
                _sentence_clause(
                    hop.relative_clause, _writer_source_reference(seed, index)
                )
            )
        )
        if hop.incoming_time_policy == "advance_required":
            sentences.append(
                f"After {_event_boundary(seed, index)}, identify the {clause}."
            )
        else:
            sentences.append(
                f"Without changing the snapshot, identify the {clause}."
            )
    final = _sentence_clause(
        seed.hops[-1].relative_clause,
        _writer_source_reference(seed, len(seed.hops) - 1),
    )
    final = _clarify_temporal_relation(
        seed.hops[-1], _without_ambiguous_after_that(final)
    )
    if seed.hops[-1].incoming_time_policy == "advance_required":
        boundary = _event_boundary(seed, len(seed.hops) - 1)
        sentences.append(
            f"At the target snapshot, after {boundary}, "
            f"{seed.answer_kind.casefold()} is the {final}?"
        )
    else:
        sentences.append(
            f"At the target snapshot, {seed.answer_kind.casefold()} is the {final}?"
        )
    return " ".join(sentences)


def question_wording_errors(
    seed: MultiHopSeed, question: str, *, steps: list[str] | None = None,
) -> list[str]:
    """Fail closed on ambiguous time transitions or leaked oracle details."""
    errors: list[str] = []
    folded = _fold(question)
    if not question.strip().endswith("?"):
        errors.append("question must end with a question mark")
    if seed.anchor_label.casefold() not in question.casefold():
        errors.append("question omits the public anchor label")
    if "registered knowledge cutoff" not in folded.replace("-", " "):
        errors.append("question omits the registered knowledge cutoff")
    if "target snapshot" not in folded.replace("-", " "):
        errors.append("question omits the target snapshot")
    elif folded.replace("-", " ").count("target snapshot") != 1:
        errors.append("target snapshot must appear exactly once, in the final step")
    if re.search(r"\bwho (?:was|is) the person who\b", folded):
        errors.append("question contains a redundant who-was-the-person-who construction")
    if re.search(
        r"\b(?:the )?(?:result of (?:the )?previous step|entity (?:just )?identified)\b",
        folded,
    ):
        errors.append("question contains a mechanical generic entity placeholder")
    if re.search(r"\bafter that\b", folded):
        errors.append("question uses ambiguous bare 'after that'")
    if re.search(
        r"\b(?:snapshot\s+(?:that\s+was\s+)?"
        r"(?:used|selected|chosen|viewed|loaded|inspected|opened)\s+"
        r"(?:in|during|at|for)\s+(?:the\s+)?previous step|"
        r"(?:the\s+)?previous[- ]step(?:'s|\s+)(?:used\s+)?snapshot|"
        r"snapshot\s+(?:from|of)\s+(?:the\s+)?previous step)\b",
        folded,
    ):
        errors.append("question makes the answer depend on a solver-selected snapshot")
    onset_required = sum(_onset_wording_required(hop) for hop in seed.hops[1:])
    onset_phrases = re.findall(
        TENURE_ONSET_PATTERN, folded,
    )
    if len(onset_phrases) < onset_required:
        errors.append(
            "question does not express every advancing position as a new tenure beginning"
        )
    allowed_dates = {seed.cutoff.cutoff_date, seed.target_as_of}
    leaked_dates = {
        value for value in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", question)
        if value not in allowed_dates
    }
    if leaked_dates:
        errors.append(
            "question leaks intermediate oracle dates: " + ", ".join(sorted(leaked_dates))
        )
    # A hidden title can be a literal substring of the public anchor (for
    # example "Mayor of X" inside "Deputy Mayor of X").  Mask only complete
    # public-anchor occurrences before checking hidden-name leakage.
    folded_without_anchor = folded.replace(_fold(seed.anchor_label), " ")
    for name in _hidden_entity_names(seed):
        if _fold(name) in folded_without_anchor:
            errors.append(f"question leaks hidden entity name {name!r}")
    if steps is not None:
        if len(steps) != len(seed.hops):
            errors.append("question writer must return exactly one step per relation hop")
        else:
            for index, (step, hop) in enumerate(zip(steps, seed.hops)):
                step_folded = _fold(step).replace("-", " ")
                if not step.rstrip().endswith("?"):
                    errors.append(f"writer step {index} must be a standalone question")
                wh_match = re.search(r"\b(who|what|which|where|when)\b", step_folded)
                expected_kind = _hop_answer_kind(seed, index, hop).casefold()
                actual_kind = wh_match.group(1) if wh_match else ""
                kind_matches = (
                    actual_kind == "who" if expected_kind == "who"
                    else actual_kind in {"what", "which", "where", "when"}
                )
                if not kind_matches:
                    errors.append(
                        f"writer step {index} uses {actual_kind or 'no question word'} "
                        f"instead of {expected_kind}"
                    )
                if index < len(steps) - 1 and "target snapshot" in step_folded:
                    errors.append(f"writer step {index} leaks the final target boundary early")
                if index == 0:
                    if seed.anchor_label.casefold() not in step.casefold():
                        errors.append("writer step 0 omits the public anchor")
                    if "registered knowledge cutoff" not in step_folded:
                        errors.append("writer step 0 omits the cutoff boundary")
                    if "previous step" in step_folded:
                        errors.append("writer step 0 refers to a nonexistent previous step")
                    if re.search(r"\b(?:after|later than)\b", step_folded):
                        errors.append("writer step 0 crosses into a later relation")
                elif (
                    hop.incoming_time_policy == "advance_required"
                    and not re.search(r"\b(?:after|later than)\b", step_folded)
                ):
                    errors.append(f"writer step {index} omits its event-time boundary")
                elif hop.incoming_time_policy == "advance_required":
                    if index == 1 and "registered knowledge cutoff" not in step_folded:
                        errors.append(
                            "writer step 1 does not identify the knowledge-cutoff boundary"
                        )
                    if index > 1:
                        previous = seed.hops[index - 1]
                        if not re.search(
                            r"\b(?:previous step|two steps earlier|"
                            r"(?:the )?first step|step\s+\d+)\b",
                            step_folded,
                        ):
                            errors.append(
                                f"writer step {index} does not bind its event boundary "
                                "to the previous step"
                            )
                        if previous.property_id == "P39":
                            direction = (
                                previous.structured_evidence or {}
                            ).get("direction")
                            if (
                                direction == "forward"
                                and not _has_forward_p39_event_boundary(step_folded)
                            ):
                                errors.append(
                                    f"writer step {index} does not bind the later "
                                    "officeholder to the earlier person's tenure onset"
                                )
                            elif (
                                direction != "forward"
                                and not _has_previous_tenure_onset_boundary(step_folded)
                            ):
                                errors.append(
                                    f"writer step {index} does not use the previous "
                                    "tenure onset"
                                )
                        if previous.property_id != "P39" and not (
                            "event" in step_folded
                            and re.search(r"\b(?:occurred|happened|began|started)\b", step_folded)
                        ):
                            errors.append(
                                f"writer step {index} does not use the previous world event"
                            )
                if _onset_wording_required(hop) and not re.search(
                    TENURE_ONSET_PATTERN, step_folded,
                ):
                    errors.append(f"writer step {index} omits the new-tenure onset")
                if _order_wording_required(hop) and not re.search(
                    r"\b(?:first|next)\b", step_folded,
                ):
                    errors.append(
                        f"writer step {index} does not select the first later tenure"
                    )
                if (
                    (hop.structured_evidence or {}).get("temporal_operator")
                    == "relation_after_boundary"
                    and re.search(r"\b(?:first|next)\b", step_folded)
                ):
                    errors.append(
                        f"writer step {index} invents an uncertified first/next claim"
                    )
                if _onset_wording_required(hop) and re.search(
                    r"(?:\band\b[^?]*\btenure\s+(?:begin|began|begun)\b|"
                    r"[,;]\s*(?:with|while)\b[^?]*\btenure\b[^?]*"
                    r"\b(?:begin|began|begun|beginning|start|started|starting)\b)",
                    step_folded,
                ):
                    errors.append(
                        f"writer step {index} redundantly restates that the tenure began"
                    )
            if "target snapshot" not in _fold(steps[-1]).replace("-", " "):
                errors.append("writer final step omits the target snapshot")
            elif not re.search(
                r"\b(?:at|as of|on)\s+(?:the\s+)?target snapshot\b",
                _fold(steps[-1]).replace("-", " "),
            ):
                errors.append(
                    "writer final step does not attach the answer to the target snapshot"
                )
            if re.search(
                r"\b(?:person|item|position)\s+identified\s+in\s+"
                r"(?:the\s+)?target snapshot\b",
                _fold(steps[-1]).replace("-", " "),
            ):
                errors.append(
                    "writer final step misuses the target snapshot as an entity reference"
                )
    return list(dict.fromkeys(errors))


class MultiHopQuestionWriter:
    """Use an LLM only to verbalize an already verified hidden relation chain."""

    def __init__(
        self, model: str, *, call_model_fn: Callable[..., str] = call_model,
    ):
        self.model = model
        self.call_model_fn = call_model_fn

    def write(self, seed: MultiHopSeed) -> dict[str, Any]:
        private_hops = []
        for index, hop in enumerate(seed.hops):
            source_reference = _writer_source_reference(seed, index)
            safe_clause = _clarify_temporal_relation(
                hop,
                _without_ambiguous_after_that(
                    hop.relative_clause.replace("{source}", source_reference)
                ),
            )
            private_hops.append({
                "index": index,
                "private_source_entity": hop.source_title,
                "private_target_entity": hop.target_title,
                "private_target_aliases": list(hop.target_aliases),
                "relation_semantics": hop.relation,
                "relation_clause": safe_clause,
                "time_policy": (
                    "registered_knowledge_cutoff" if index == 0
                    else hop.incoming_time_policy
                ),
                "required_boundary": (
                    "registered knowledge cutoff" if index == 0
                    else (
                        f"after {_event_boundary(seed, index)}"
                        if hop.incoming_time_policy == "advance_required"
                        else "same snapshot as the previous step"
                    )
                ),
                "is_final": index == len(seed.hops) - 1,
                "answer_kind": _hop_answer_kind(seed, index, hop),
                "new_tenure_onset_wording_required": _onset_wording_required(hop),
                "first_or_next_required": _order_wording_required(hop),
            })
        base_prompt = f"""Write one natural multi-hop Wikipedia question from this verified private relation plan.

Public starting page: {seed.anchor_label}
Final answer kind: {seed.answer_kind}
Private relation plan (entity names are supplied only so you can understand types,
references, and relation semantics; intermediate dates remain intentionally absent):
{json.dumps(private_hops, ensure_ascii=False, indent=2)}

Return one JSON object with exactly one key, "steps", containing exactly
{len(seed.hops)} strings. Each string must be a standalone question ending in "?";
joining the strings with spaces must form the complete multi-hop question.

Hard rules:
- Preserve every relation exactly once and in the supplied order.
- Use each step's supplied answer_kind: Who for a person and What/Which for an
  office, position, place, or other non-person result. Make that the first question
  word in the step; do not put a subordinate "which" before a required "who".
- Step 0 must name {seed.anchor_label} and the tested model's registered knowledge cutoff.
- Every advance_required step must compare world-event times, never observation times.
  Step 1 is after the registered knowledge cutoff. Later steps are after the previous
  relation's event occurred; explicitly identify it as the event or tenure from the
  previous step. For position relations, compare tenure start times. Do not use an
  unbound phrase such as "after its tenure began".
- Never make an answer depend on "the snapshot used in the previous step", a selected
  revision, a browsing action, or any date chosen by the solver. The solver may inspect
  any revision without changing the correct answer.
- When new_tenure_onset_wording_required is true, explicitly say that the tenure began,
  for example "began holding", "took up the position", "assumed the role", or "became",
  after the event boundary. Express this once in the main predicate; never append a
  second clause such as "and had the tenure begun?" or "and which tenure began?". For
  an inverse officeholder relation, ask for the first person who began holding the
  office after the boundary. For a forward relation, ask for the first position whose
  tenure began after the boundary. Apply that first/next rule only when
  first_or_next_required is true. A relation_after_boundary step instead preserves its
  supplied semantic scope and must not invent a first/next claim. Merely saying a
  position was "held" later is invalid.
- Copy the supplied required_boundary semantics exactly. In particular, when it says
  "the person identified two steps earlier began holding that position", retain both
  the person reference and "two steps earlier"; do not replace it with an abstract
  phrase about a tenure occurring in the previous step.
- The private source/target names are answer keys, not wording suggestions. Never copy,
  paraphrase, abbreviate, or otherwise reveal any private entity name or alias. The only
  entity name allowed in the output is the public starting page, {seed.anchor_label}.
- Never use the phrase "after that".
- Do not include exact intermediate dates. Do not tell the solver which date to choose.
- Only the final step may say "target snapshot"; earlier steps must leave their exact
  dates for the solver to discover. Attach the final answer with "at the target
  snapshot" or "as of the target snapshot"; never leave "target snapshot" as a comma
  fragment or say that a person was identified in the target snapshot. Every step must
  end with a question mark.
- Use clear pronouns or nouns such as person, office, team, or organization when the
  relation clause supports them. Never say "result of the previous step", "entity just
  identified", or another generic graph placeholder; use its supplied semantic type.
- Avoid redundant constructions such as "who was the person who".
- Keep each step to one relation and make the result read like a human-written question.
"""
        failures: list[str] = []
        responses: list[str] = []
        for attempt in range(QUESTION_WRITER_MAX_ATTEMPTS):
            prompt = base_prompt
            if failures:
                prompt += (
                    "\nThe previous draft was:\n"
                    + responses[-1]
                    + "\nIt failed only the following deterministic gates:\n- "
                    + "\n- ".join(failures)
                    + "\nMake the smallest necessary corrections and preserve every "
                    "part that already complied. Return the full JSON object again."
                )
            raw, response = call_json_model(self.model, prompt, self.call_model_fn)
            responses.append(response)
            raw_steps = raw.get("steps")
            steps = (
                [str(value).strip() for value in raw_steps]
                if isinstance(raw_steps, list)
                and all(isinstance(value, str) and value.strip() for value in raw_steps)
                else []
            )
            question = " ".join(steps)
            failures = question_wording_errors(seed, question, steps=steps)
            if not failures:
                return {
                    "schema_version": "llm-event-question-writer-v3",
                    "prompt_version": QUESTION_WRITER_PROMPT_VERSION,
                    "model": self.model,
                    "private_entity_context_supplied": True,
                    "steps": steps,
                    "question": question,
                    "raw_response": response,
                    "attempts": attempt + 1,
                }
        raise QuestionWriterGateError(
            model=self.model, failures=failures, raw_responses=responses,
        )


class QuestionWriterGateError(ValueError):
    """Preserve rejected writer drafts as private audit evidence."""

    def __init__(
        self, *, model: str, failures: list[str], raw_responses: list[str],
    ):
        self.model = model
        self.failures = list(failures)
        self.raw_responses = list(raw_responses)
        super().__init__("question writer failed gates: " + "; ".join(failures))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "llm-event-question-writer-reject-v2",
            "prompt_version": QUESTION_WRITER_PROMPT_VERSION,
            "model": self.model,
            "private_entity_context_supplied": True,
            "status": "rejected",
            "gate_errors": self.failures,
            "raw_responses": self.raw_responses,
            "attempts": len(self.raw_responses),
        }


def _validated_prior_relation_contrast(
    hop: ChainHopSeed, previous_page, later_page,
) -> bool:
    """Validate a saved semantic-contrast override for a pre-existing target link."""
    structured = hop.structured_evidence or {}
    contrast = structured.get("prior_relation_contrast")
    if not isinstance(contrast, dict):
        return False
    try:
        confidence = float(contrast.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False
    before_evidence = contrast.get("before_evidence")
    required = (
        contrast.get("method") == "independent_llm_semantic_contrast"
        and contrast.get("decision") == "pass"
        and confidence >= 0.8
        and isinstance(contrast.get("judge_model"), str)
        and bool(str(contrast.get("judge_model", "")).strip())
        and contrast.get("property_id") == hop.property_id
        and contrast.get("before_revision_id") == previous_page.revision_id
        and contrast.get("after_revision_id") == later_page.revision_id
        and isinstance(before_evidence, str)
        and bool(before_evidence.strip())
        and _fold(before_evidence) in _fold(previous_page.content)
        and _fold(hop.evidence) in _fold(later_page.content)
    )
    return bool(required)


def _page_link_targets(page: Any) -> set[str]:
    return {link.target.casefold() for link in page.links}


def _links_requested_or_canonical(
    page: Any, *, requested_title: str, canonical_title: str,
) -> bool:
    """Accept a link through a redirect only after the target page resolved it.

    Historical revisions retain the link title that was used when the revision
    was rendered, while MediaWiki resolves a page fetch through today's redirect
    table.  Both names therefore identify the same fetched page.  This remains
    an exact hyperlink gate: arbitrary aliases are not accepted.
    """
    targets = _page_link_targets(page)
    return bool(
        {requested_title.casefold(), canonical_title.casefold()} & targets
    )


def validate_chain(
    seed: MultiHopSeed,
    backend,
    *,
    validated_prefix: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Verify the chain, optionally reusing a previously verified exact prefix."""
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    question = compose_relative_question(seed)
    previous_target_page = None
    seen_state_edges: set[tuple[int, int]] = set()
    start_index = 0
    if validated_prefix:
        if len(validated_prefix) >= len(seed.hops):
            return ["validated prefix must be shorter than the candidate chain"], []
        prefix_fields = (
            "relation", "relative_clause", "as_of", "evidence",
            "incoming_time_policy", "property_id", "relation_family",
        )
        for index, record in enumerate(validated_prefix):
            hop = seed.hops[index]
            expected = {
                "relation": hop.relation,
                "relative_clause": hop.relative_clause,
                "as_of": hop.as_of,
                "evidence": hop.evidence,
                "incoming_time_policy": hop.incoming_time_policy,
                "property_id": hop.property_id,
                "relation_family": hop.relation_family,
            }
            if any(record.get(field) != expected[field] for field in prefix_fields):
                return [f"validated prefix differs from candidate at hop {index}"], []
            if record.get("requested_source_title", record.get("source_title")) != hop.source_title:
                return [f"validated prefix source differs at hop {index}"], []
            if record.get("requested_target_title", record.get("target_title")) != hop.target_title:
                return [f"validated prefix target differs at hop {index}"], []
            if list(hop.target_aliases) != record.get("target_aliases"):
                return [f"validated prefix aliases differ at hop {index}"], []
        records = [dict(record) for record in validated_prefix]
        start_index = len(records)
        seen_state_edges = {
            (int(record["source_revision_id"]), int(record["target_revision_id"]))
            for record in records
        }
        previous_hop = seed.hops[start_index - 1]
        try:
            previous_target_page = backend.fetch_page(
                previous_hop.target_title, as_of=previous_hop.as_of
            )
        except WikipediaError as exc:
            return [f"validated prefix target fetch failed: {exc}"], records

    for index in range(start_index, len(seed.hops)):
        hop = seed.hops[index]
        prior_raw_target_absent: bool | None = None
        prior_relation_contrast_verified = False
        try:
            if (
                previous_target_page is not None
                and hop.incoming_time_policy == "same_snapshot"
                and previous_target_page.title.casefold() == hop.source_title.casefold()
                and previous_target_page.as_of == hop.as_of
            ):
                source = previous_target_page
            else:
                source = backend.fetch_page(hop.source_title, as_of=hop.as_of)
            target = backend.fetch_page(hop.target_title, as_of=hop.as_of)
        except WikipediaError as exc:
            errors.append(f"hop {index}: Wikipedia fetch failed: {exc}")
            continue
        if previous_target_page is not None:
            if previous_target_page.title.casefold() != source.title.casefold():
                errors.append(f"hop {index}: fetched chain is disconnected")
            if hop.incoming_time_policy == "advance_required":
                if previous_target_page.revision_id == source.revision_id:
                    errors.append(
                        f"hop {index}: later snapshot resolves to the same source revision"
                    )
                premature_targets = _page_link_targets(previous_target_page)
                previous_content = _fold(previous_target_page.content)
                premature_aliases = [
                    alias for alias in hop.target_aliases
                    if _fold(alias) in previous_content
                ]
                prior_raw_target_absent = (
                    not {
                        target.title.casefold(), hop.target_title.casefold(),
                    } & premature_targets
                    and not premature_aliases
                )
                if not prior_raw_target_absent:
                    prior_relation_contrast_verified = _validated_prior_relation_contrast(
                        hop, previous_target_page, source
                    )
                if not prior_raw_target_absent and not prior_relation_contrast_verified:
                    errors.append(
                        f"hop {index}: target appears before the required temporal switch without "
                        "a validated prior-relation semantic contrast"
                    )
            elif previous_target_page.revision_id != source.revision_id:
                errors.append(
                    f"hop {index}: same_snapshot source does not reuse the prior target revision"
                )
        if _fold(hop.evidence) not in _fold(source.content):
            errors.append(f"hop {index}: evidence is not a verbatim source-revision substring")
        evidence_folded = _fold(hop.evidence)
        for alias in hop.target_aliases:
            if _fold(alias) not in evidence_folded:
                errors.append(f"hop {index}: alias {alias!r} is absent from evidence")
        if not _links_requested_or_canonical(
            source,
            requested_title=hop.target_title,
            canonical_title=target.title,
        ):
            errors.append(f"hop {index}: source revision has no hyperlink to target")
        edge = (source.revision_id, target.revision_id)
        if edge in seen_state_edges:
            errors.append(f"hop {index}: duplicate page-version edge")
        seen_state_edges.add(edge)
        structured = hop.structured_evidence or {}
        claims_first_or_next = bool(
            re.search(r"\b(?:first|next)\b", hop.relative_clause, re.IGNORECASE)
            or structured.get("temporal_operator") == "next_after_boundary"
        )
        if (
            index > 0
            and hop.incoming_time_policy == "advance_required"
            and structured.get("source") == "wikidata_time_qualified_statement"
            and claims_first_or_next
        ):
            selected_qid = (
                structured.get("kg_subject_qid")
                if structured.get("direction") == "inverse"
                else structured.get("kg_object_qid")
            )
            certificate_errors = event_order_certificate_errors(
                structured.get("event_order_certificate"),
                expected_boundary=_hop_event_date(seed.hops[index - 1]),
                expected_event_date=_hop_event_date(hop),
                expected_target_qid=(str(selected_qid) if selected_qid else None),
            )
            errors.extend(
                f"hop {index}: {error}" for error in certificate_errors
            )
        frozen_source = _frozen_page(source)
        frozen_target = _frozen_page(target)
        records.append({
            "index": index,
            "requested_source_title": hop.source_title,
            "source_title": source.title,
            "source_revision_id": source.revision_id,
            "source_timestamp": source.timestamp,
            "source_url": source.source_url,
            "source_content_sha256": frozen_source["content_sha256"],
            "source_links_sha256": frozen_source["links_sha256"],
            "requested_target_title": hop.target_title,
            "target_title": target.title,
            "target_revision_id": target.revision_id,
            "target_timestamp": target.timestamp,
            "target_url": target.source_url,
            "target_content_sha256": frozen_target["content_sha256"],
            "target_links_sha256": frozen_target["links_sha256"],
            "as_of": hop.as_of,
            "relation": hop.relation,
            "relative_clause": hop.relative_clause,
            "evidence": hop.evidence,
            "target_aliases": list(hop.target_aliases),
            "incoming_time_policy": hop.incoming_time_policy,
            "property_id": hop.property_id,
            "relation_family": hop.relation_family,
            "structured_evidence": hop.structured_evidence,
            "prior_snapshot_absence_verified": prior_raw_target_absent is True,
            "prior_raw_target_absent": prior_raw_target_absent,
            "prior_relation_contrast_verified": prior_relation_contrast_verified,
            "prior_as_of": seed.hops[index - 1].as_of if index > 0 else None,
            "prior_revision_id": (
                previous_target_page.revision_id
                if previous_target_page is not None else None
            ),
            "_frozen_source_snapshot": frozen_source,
            "_frozen_target_snapshot": frozen_target,
        })
        previous_target_page = target
    if records:
        canonical_path = [
            str(records[0]["source_title"]),
            *(str(record["target_title"]) for record in records),
        ]
        folded_canonical_path = [_fold(title) for title in canonical_path]
        if len(set(folded_canonical_path)) != len(folded_canonical_path):
            errors.append("fetched reasoning chain contains a canonical entity cycle")
    errors.extend(question_wording_errors(seed, question))
    final_aliases = {_fold(item) for item in seed.hops[-1].target_aliases}
    if final_aliases & {_fold(item) for item in seed.old_answer_keywords}:
        errors.append("old and target answer aliases overlap")
    return errors, records


def build_temporal_waypoints(chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand relation hops into temporal and same-snapshot graph states."""
    waypoints: list[dict[str, Any]] = []
    for index, hop in enumerate(chain):
        if index == 0 or hop.get("incoming_time_policy") != "same_snapshot":
            source = {
                "title": hop["source_title"],
                "revision_id": hop["source_revision_id"],
                "as_of": hop["as_of"],
                "incoming_edge": "start" if index == 0 else "temporal",
                "relation_hop": index,
                "role": "relation_source",
            }
            waypoints.append(source)
        target = {
            "title": hop["target_title"],
            "revision_id": hop["target_revision_id"],
            "as_of": hop["as_of"],
            "incoming_edge": "hyperlink",
            "relation_hop": index,
            "role": "relation_target",
        }
        waypoints.append(target)
    for index, waypoint in enumerate(waypoints):
        waypoint["index"] = index
    return waypoints


def _hop_event_date(hop: ChainHopSeed) -> str:
    structured = hop.structured_evidence or {}
    if structured.get("event_date"):
        return str(structured["event_date"])
    qualifiers = structured.get("qualifier_dates")
    if isinstance(qualifiers, dict):
        for property_id in ("P580", "P585"):
            if qualifiers.get(property_id):
                return str(qualifiers[property_id])
    return hop.as_of


def _bridge_probe_question(
    seed: MultiHopSeed, index: int, hop: ChainHopSeed,
) -> str:
    """Ask one direct event fact without exposing it to the later trajectory."""
    event_date = _hop_event_date(hop)
    structured = hop.structured_evidence or {}
    if hop.property_id == "P39" and structured.get("direction") == "inverse":
        previous_person = seed.hops[index - 1].source_title
        return (
            f"Who began holding {hop.source_title} on {event_date}, succeeding "
            f"{previous_person}?"
        )
    if hop.property_id == "P39":
        return (
            f"What was the first government position {hop.source_title} began "
            f"holding on {event_date}?"
        )
    direction = structured.get("direction")
    if direction == "inverse":
        inverse_questions = {
            "P1308": f"What office did {hop.source_title} begin holding on {event_date}?",
            "P169": (
                f"What organization appointed {hop.source_title} as chief executive "
                f"officer on {event_date}?"
            ),
            "P6": (
                f"What territory did {hop.source_title} become head of government "
                f"of on {event_date}?"
            ),
            "P35": (
                f"What country did {hop.source_title} become head of state of on "
                f"{event_date}?"
            ),
        }
        return inverse_questions.get(
            str(hop.property_id),
            f"What entity did {hop.source_title} begin serving as {hop.relation} of "
            f"on {event_date}?",
        )
    if direction == "forward":
        return (
            f"Who began serving as {hop.relation} of {hop.source_title} on "
            f"{event_date}?"
        )
    return (
        f"On {event_date}, what was the {hop.relation} of {hop.source_title}?"
    )


def _probe_aliases(hop: ChainHopSeed) -> list[str]:
    """Prefer the canonical target while retaining revision-visible anchors."""
    return list(dict.fromkeys((hop.target_title, *hop.target_aliases)))


def _tail_probe_question(hop: ChainHopSeed) -> str:
    if hop.property_id == "P26":
        return f"Who is {hop.source_title}'s spouse?"
    if hop.property_id == "P19":
        return f"Where was {hop.source_title} born?"
    return f"What is the {hop.relation} of {hop.source_title}?"


def build_prior_knowledge_contract(
    seed: MultiHopSeed, composed_question: str,
) -> dict[str, Any]:
    """Factor prior knowledge into bridge, tail, and composed-answer probes."""
    advancing = [
        index for index, hop in enumerate(seed.hops)
        if index > 0 and hop.incoming_time_policy == "advance_required"
    ]
    probes: list[dict[str, Any]] = [{
        "id": "anchor",
        "role": "anchor",
        "question": (
            f"Who held {seed.anchor_label} on {seed.cutoff.cutoff_date}?"
        ),
        "answer_aliases": _probe_aliases(seed.hops[0]),
        "objective": "diagnostic",
        "hop_index": 0,
    }]
    for index in advancing:
        hop = seed.hops[index]
        probes.append({
            "id": f"bridge_{index}",
            "role": "critical_bridge",
            "question": _bridge_probe_question(seed, index, hop),
            "answer_aliases": _probe_aliases(hop),
            "objective": "must_be_unknown",
            "hop_index": index,
            "event_date": _hop_event_date(hop),
        })
    final_hop = seed.hops[-1]
    composed_context_aliases = list(dict.fromkeys(
        alias
        for hop in seed.hops[:-1]
        for alias in _probe_aliases(hop)
    ))
    probes.extend([{
        "id": "tail",
        "role": "tail",
        "question": _tail_probe_question(final_hop),
        "answer_aliases": _probe_aliases(final_hop),
        "objective": "measure_known_for_composition",
        "hop_index": len(seed.hops) - 1,
    }, {
        "id": "composed",
        "role": "composed",
        "question": composed_question,
        "answer_aliases": _probe_aliases(final_hop),
        "context_aliases": composed_context_aliases,
        "objective": "diagnostic_only",
        "hop_index": None,
    }])
    primary_ids = [f"bridge_{index}" for index in advancing]
    return {
        "schema_version": "factorized-prior-knowledge-v2",
        "primary_admission_role": "critical_bridge",
        "primary_admission_probe_ids": primary_ids,
        "admission_policy": "all_designated_acquisition_edges_unknown",
        "interpretation": (
            "The final answer may be an old attribute. Admission depends on not "
            "knowing every designated post-cutoff acquisition edge; tail and "
            "composed probes are reported separately."
        ),
        "probes": probes,
    }


class MultiHopQuestionJudge:
    """Independent semantic judge after deterministic revision/link gates."""

    def __init__(
        self, model: str, *, min_confidence: float = 0.8,
        call_model_fn: Callable[..., str] = call_model,
        cache_path: str | None = None,
    ):
        self.model = model
        self.min_confidence = min_confidence
        self.call_model_fn = call_model_fn
        self._cache_lock = threading.Lock()
        self._cache_conn = None
        if cache_path:
            self._cache_conn = sqlite3.connect(cache_path, check_same_thread=False)
            self._cache_conn.execute(
                "CREATE TABLE IF NOT EXISTS judge_cache ("
                "cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, result_json TEXT NOT NULL, "
                "created_at TEXT NOT NULL)"
            )
            self._cache_conn.commit()

    def close(self) -> None:
        if self._cache_conn is not None:
            with self._cache_lock:
                self._cache_conn.close()
                self._cache_conn = None

    def _load_cached(self, cache_key: str) -> dict[str, Any] | None:
        if self._cache_conn is None:
            return None
        with self._cache_lock:
            row = self._cache_conn.execute(
                "SELECT result_json FROM judge_cache WHERE cache_key=?", (cache_key,)
            ).fetchone()
        return dict(json.loads(row[0])) if row else None

    def _store_cached(self, cache_key: str, result: dict[str, Any]) -> None:
        if self._cache_conn is None:
            return
        stored = {key: value for key, value in result.items() if key != "cache_hit"}
        with self._cache_lock:
            self._cache_conn.execute(
                "INSERT OR IGNORE INTO judge_cache("
                "cache_key,model,result_json,created_at) VALUES(?,?,?,?)",
                (
                    cache_key, self.model,
                    json.dumps(stored, ensure_ascii=False, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._cache_conn.commit()

    def judge(self, question: str, chain: list[dict[str, Any]]) -> dict[str, Any]:
        visible = [{
            "index": hop["index"], "relation": hop["relation"],
            "relative_clause": hop["relative_clause"], "as_of": hop["as_of"],
            "evidence": hop["evidence"], "source_title": hop["source_title"],
            "target_title": hop["target_title"],
            "incoming_time_policy": hop.get("incoming_time_policy"),
            "property_id": hop.get("property_id"),
            "relation_family": hop.get("relation_family"),
            "structured_evidence": hop.get("structured_evidence"),
        } for hop in chain]
        cache_key = hashlib.sha256(json.dumps({
            "prompt_version": 6,
            "schema_version": SCHEMA_VERSION,
            "model": self.model,
            "min_confidence": self.min_confidence,
            "question": question,
            "chain": visible,
        }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cached = self._load_cached(cache_key)
        if cached is not None:
            record_usage_event(
                "cache_hit", requested_model=self.model, cache_key=cache_key,
            )
            return {**cached, "cache_hit": True}
        prompt = f"""Audit this cutoff-relative multi-hop Wikipedia question.

Question: {question}
Ordered verified chain:
{json.dumps(visible, ensure_ascii=False, indent=2)}

Exact-substring and hyperlink checks already passed. Pass only if:
- every evidence excerpt semantically supports its named relation;
- the question composes all relations once, in the same order;
- "registered knowledge cutoff" anchors hop 0 and "target snapshot" anchors the answer;
- the chain is genuinely multi-hop and uniquely identifies the final target;
- no intermediate or final entity is named in the question;
- every advancing hop compares world-event times, never the solver's selected page
  revision or "snapshot used in the previous step". Evaluate temporal_transition_clarity
  from the user-facing Question, not legacy relative_clause strings. For an advancing
  position-held relation, compare tenure onsets (for example, "the next person to begin
  holding the position after the previous tenure began"), not observation timestamps.
- every step reads as a concise, grammatical, standalone question. Reject redundant
  onset clauses, run-on fragments, unclear pronouns, or wording that a human would
  need to reread. This is natural_question_wording and is independent of whether the
  underlying chain is factually correct.
Return one JSON object with decision (pass/reject), confidence (0..1), checks as an
object with exactly these boolean assertions: evidence_semantics,
relation_order_and_composition, cutoff_and_snapshot_anchoring,
multi_hop_and_uniqueness, no_entity_leakage, temporal_transition_clarity,
natural_question_wording; also return
reason, and rejected_hops (list of integers).
"""
        raw, response = call_json_model(self.model, prompt, self.call_model_fn)
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        decision = str(raw.get("decision", "reject"))
        checks = raw.get("checks")
        schema_errors = []
        if not isinstance(checks, dict):
            schema_errors.append("checks_not_object")
        else:
            for key in WHOLE_CHAIN_CHECKS:
                if checks.get(key) is not True:
                    schema_errors.append(f"check_not_true:{key}")
        rejected_hops = raw.get("rejected_hops")
        if not isinstance(rejected_hops, list) or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in rejected_hops
        ):
            schema_errors.append("rejected_hops_not_integer_list")
        elif decision == "pass" and rejected_hops:
            schema_errors.append("pass_has_rejected_hops")
        if not str(raw.get("reason", "")).strip():
            schema_errors.append("missing_reason")
        if decision != "pass" or confidence < self.min_confidence or schema_errors:
            decision = "reject"
        result = {
            **raw, "decision": decision, "confidence": confidence,
            "schema_gate_errors": schema_errors,
            "raw_response": response, "cache_hit": False,
        }
        self._store_cached(cache_key, result)
        return result


def _judge_seed(
    judge: MultiHopQuestionJudge, question: str,
    chain: list[dict[str, Any]], seed_id: str,
) -> dict[str, Any]:
    with model_call_context(role="whole_chain_judge", seed_id=seed_id):
        return judge.judge(question, chain)


def build_case(
    seed: MultiHopSeed,
    chain: list[dict[str, Any]],
    judge_result: dict[str, Any],
    *,
    question: str | None = None,
    question_generation: dict[str, Any] | None = None,
    validity: dict[str, Any] | None = None,
    search_space_diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    question = question or compose_relative_question(seed)
    canonical_question = compose_canonical_question(seed)
    required_dates = list(dict.fromkeys([hop.as_of for hop in seed.hops]))
    public_chain = _public_chain(chain)
    waypoints = build_temporal_waypoints(public_chain)
    expected_distance = len(waypoints) - 1
    frozen_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for hop in chain:
        for field in ("_frozen_source_snapshot", "_frozen_target_snapshot"):
            page = hop.get(field)
            if isinstance(page, dict):
                frozen_by_key[(str(page["title"]).casefold(), int(page["revision_id"]))] = page
    frozen_pages = sorted(
        frozen_by_key.values(),
        key=lambda page: (str(page["timestamp"]), str(page["title"]).casefold()),
    )
    frozen_manifest = {
        "schema_version": "frozen-wikipedia-evidence-v1",
        "pages": frozen_pages,
    }
    frozen_manifest["manifest_sha256"] = _canonical_sha256({
        "schema_version": frozen_manifest["schema_version"],
        "page_hashes": [page["snapshot_sha256"] for page in frozen_pages],
    })
    return {
        "id": seed.id,
        "category": seed.category,
        "wikipedia_title": chain[-1]["target_title"],
        "wikipedia_before": seed.cutoff.cutoff_date,
        "wikipedia_as_of": seed.target_as_of,
        "required_snapshot_dates": required_dates,
        "temporal_question": question,
        "start_title": chain[0]["source_title"],
        "hide_pivot_title": True,
        "reasoning_hop_count": len(chain),
        "expected_navigation_distance": expected_distance,
        "semantic_shortest_distance": expected_distance,
        "required_temporal_switches": sum(
            hop.get("incoming_time_policy", "advance_required") == "advance_required"
            for hop in chain[1:]
        ),
        "temporal_waypoints": waypoints,
        # The first-hop entity is not a stale answer to the composed final
        # question.  Until an earlier value for the *final* relation is
        # independently verified, leave stale aliases empty.
        "old_answer_keywords": [],
        "new_answer_keywords": list(seed.hops[-1].target_aliases),
        "prior_knowledge_contract": build_prior_knowledge_contract(seed, question),
        "knowledge_cutoff": {
            **seed.cutoff.to_dict(), "model_ids": [seed.model_id],
            "role": "candidate_prior_factorized_pk_gate_is_authoritative",
        },
        "reasoning_chain": public_chain,
        "frozen_wikipedia_evidence": frozen_manifest,
        "relation_families": list(dict.fromkeys(
            str(hop.get("relation_family")) for hop in chain
            if hop.get("relation_family")
        )),
        "selection_metadata": seed.selection_metadata,
        "_generation": {
            "schema_version": SCHEMA_VERSION,
            "status": "machine_pass_human_review_required",
            "question_style": (
                "llm_event_anchored_steps_v3"
                if question_generation is not None
                else "explicit_event_anchored_steps_v3"
            ),
            "canonical_question": canonical_question,
            "question_writer": (
                {
                    key: value for key, value in question_generation.items()
                    if key != "raw_response"
                }
                if question_generation is not None else None
            ),
            "judge": {key: value for key, value in judge_result.items()
                      if key != "raw_response"},
            "validity_contract_v25": validity,
            "search_space_diagnostic_v25": search_space_diagnostic,
        },
    }


def validate_shortest_arena(
    seed: MultiHopSeed,
    chain: list[dict[str, Any]],
    backend,
    *,
    branch_cap: int,
    node_cap: int = 500,
) -> dict[str, Any]:
    """Audit the semantic route and diagnose raw shortcuts in a bounded arena."""
    case = build_case(seed, chain, {"decision": "pending", "confidence": 0.0})
    navigation = temporal_reverse_bfs(
        backend,
        case["wikipedia_title"],
        seed.target_as_of,
        case["required_snapshot_dates"],
        case["expected_navigation_distance"],
        branch_cap=branch_cap,
        max_nodes=node_cap,
        required_waypoints=case["temporal_waypoints"],
    )
    contract = validate_chain_route(case, navigation)
    if contract is None:
        raise ValueError("missing reasoning-chain route contract")
    manifest_hash = hashlib.sha256(json.dumps({
        "target_key": navigation["target_key"],
        "states": navigation["states"],
        "arena_edges": navigation["arena_edges"],
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {
        "passed": True,
        "shortest_path_status": "exact",
        "shortest_distance": contract["raw_distance"],
        "raw_shortest_distance": contract["raw_distance"],
        "semantic_shortest_distance": contract["semantic_distance"],
        "raw_shortest_matches_semantic": contract["raw_shortest_match"],
        "route_keys": contract["route_keys"],
        "edge_kinds": contract["edge_kinds"],
        "arena_node_count": len(navigation["states"]),
        "arena_edge_count": len(navigation["arena_edges"]),
        "arena_node_cap": navigation["max_nodes"],
        "arena_truncated": navigation["arena_truncated"],
        "arena_discovery_mode": navigation["discovery_mode"],
        "arena_sha256": manifest_hash,
        "coverage_note": navigation["coverage_note"],
        "global_shortest_complete": not navigation["arena_truncated"],
        "frontier_incomplete": bool(navigation["arena_truncated"]),
        "bfs_explored_nodes": len(navigation["states"]),
        "admission_effect": "required_legacy_policy",
    }


def load_seeds(path: str) -> list[MultiHopSeed]:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".jsonl"):
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        values = parsed.get("seeds", []) if isinstance(parsed, dict) else parsed
    if not isinstance(values, list):
        raise ValueError("seed file must be a list or {'seeds': [...]} object")
    return [MultiHopSeed.from_dict(value) for value in values]


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate cutoff-relative multi-hop cases from verified Wikipedia links"
    )
    parser.add_argument("--seed-file", required=True)
    parser.add_argument(
        "--seed-id", action="append", default=[],
        help="process only the named seed; repeat to select multiple seeds",
    )
    parser.add_argument(
        "--generator-model",
        help="LLM question writer; omit only to use the explicit deterministic fallback",
    )
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-min-confidence", type=float, default=0.8)
    parser.add_argument(
        "--judge-workers", type=int, default=4,
        help="maximum concurrent LLM judge requests",
    )
    parser.add_argument(
        "--judge-cache-path",
        help="persistent judge cache; default is <cache-path>.judge.db",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--cache-path", default="wikipedia_multihop_generation.db")
    parser.add_argument("--backlink-branch-cap", type=int, default=25)
    parser.add_argument("--arena-node-cap", type=int, default=500)
    parser.add_argument(
        "--shortest-policy", choices=("required", "diagnostic", "skip"),
        default="required",
        help=("required preserves the legacy blocking arena; diagnostic runs "
              "only cheap non-blocking checks; skip records not_computed"),
    )
    parser.add_argument(
        "--request-interval", type=float, default=0.75,
        help="minimum seconds between Wikimedia API calls",
    )
    parser.add_argument(
        "--api-call-budget", type=int, default=2000,
        help="hard limit on actual Wikimedia HTTP attempts for this process",
    )
    parser.add_argument("--api-max-retries", type=int, default=6)
    parser.add_argument("--api-backoff-base", type=float, default=2.0)
    parser.add_argument("--api-backoff-max", type=float, default=60.0)
    parser.add_argument("--backlink-verify-workers", type=int, default=1)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default="multihop_question_packets.jsonl")
    parser.add_argument("--cases-output", default="generated_multihop_cases.json")
    parser.add_argument(
        "--usage-output",
        help="append-only judge call/cache/token/cost JSONL; default <output>.usage.jsonl",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.usage_output = args.usage_output or f"{args.output}.usage.jsonl"
    if not 0 <= args.judge_min_confidence <= 1:
        parser.error("--judge-min-confidence must be between 0 and 1")
    if args.judge_workers <= 0:
        parser.error("--judge-workers must be > 0")
    if args.backlink_branch_cap <= 0 or args.arena_node_cap <= 0:
        parser.error("--backlink-branch-cap and --arena-node-cap must be > 0")
    if args.request_interval < 0:
        parser.error("--request-interval must be >= 0")
    if (
        args.api_call_budget <= 0 or args.api_max_retries <= 0
        or args.backlink_verify_workers <= 0
    ):
        parser.error("--api-call-budget and --api-max-retries must be > 0")
    if args.api_backoff_base < 0 or args.api_backoff_max < 0:
        parser.error("API backoff values must be >= 0")
    if not args.validate_only:
        if not args.judge_model:
            parser.error("--judge-model is required unless --validate-only")
        if not os.environ.get("OPENROUTER_API_KEY"):
            parser.error("OPENROUTER_API_KEY is required for generation and judging")
        if args.generator_model and args.generator_model == args.judge_model:
            parser.error("--generator-model and --judge-model must differ")
    elif args.generator_model:
        parser.error("--generator-model cannot be used with --validate-only")
    paths = [args.output] + ([] if args.validate_only else [args.cases_output])
    for path in paths:
        assert_new_output_path(path)
        if Path(path).exists() and not args.overwrite:
            parser.error(f"refusing to overwrite {path}; pass --overwrite explicitly")
    seeds = load_seeds(args.seed_file)
    if args.seed_id:
        requested = set(args.seed_id)
        available = {seed.id for seed in seeds}
        missing = sorted(requested - available)
        if missing:
            parser.error(f"--seed-id not found in seed file: {', '.join(missing)}")
        seeds = [seed for seed in seeds if seed.id in requested]
    backend = WikipediaPageBackend(
        cache_path=args.cache_path, lang=args.lang, offline_only=args.offline,
        min_request_interval=args.request_interval,
        max_api_calls=args.api_call_budget,
        api_max_retries=args.api_max_retries,
        backoff_base_seconds=args.api_backoff_base,
        backoff_max_seconds=args.api_backoff_max,
        backlink_verify_workers=args.backlink_verify_workers,
    )
    judge_cache_path = args.judge_cache_path or f"{args.cache_path}.judge.db"
    judge = None if args.validate_only else MultiHopQuestionJudge(
        args.judge_model, min_confidence=args.judge_min_confidence,
        cache_path=judge_cache_path,
    )
    writer = (
        MultiHopQuestionWriter(args.generator_model)
        if not args.validate_only and args.generator_model else None
    )
    usage_ledger = None if args.validate_only else UsageLedger(
        args.usage_output,
        metadata={
            "experiment": "multihop_question_generation",
            "generator_model": args.generator_model,
            "judge_model": args.judge_model,
        },
    )
    set_usage_ledger(usage_ledger)
    packets: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    judge_jobs: list[tuple[int, MultiHopSeed]] = []
    validated_chains: dict[int, list[dict[str, Any]]] = {}
    try:
        for seed in seeds:
            api_before = backend.request_stats()
            errors, chain = validate_chain(seed, backend)
            shortest_path: dict[str, Any] | None = None
            question = compose_relative_question(seed)
            question_generation = None
            if not errors and args.shortest_policy == "required":
                try:
                    shortest_path = validate_shortest_arena(
                        seed, chain, backend,
                        branch_cap=args.backlink_branch_cap,
                        node_cap=args.arena_node_cap,
                    )
                except (WikipediaError, ValueError) as exc:
                    errors.append(f"shortest_path: {exc}")
            elif not errors and args.shortest_policy == "diagnostic":
                shortest_path = bounded_shortcut_diagnostic(
                    chain, final_aliases=seed.hops[-1].target_aliases,
                )
            elif not errors:
                shortest_path = {
                    "schema_version": "temporal-wikipedia-validity-v2.5",
                    "shortest_path_status": "not_computed",
                    "global_shortest_complete": False,
                    "admission_effect": "none",
                }
            if not errors and writer is not None:
                try:
                    with model_call_context(role="question_writer", seed_id=seed.id):
                        question_generation = writer.write(seed)
                    question = str(question_generation["question"])
                except QuestionWriterGateError as exc:
                    question_generation = exc.to_dict()
                    errors.append(f"question writer: {exc}")
                except Exception as exc:
                    errors.append(f"question writer: {exc}")
            leakage_errors = [
                error for error in question_wording_errors(seed, question)
                if "leaks hidden entity" in error
            ]
            validity = validity_contract(
                seed_id=seed.id, chain=chain, question=question,
                deterministic_errors=errors,
                question_leakage_errors=leakage_errors,
                cutoff_date=seed.cutoff.cutoff_date,
                critical_event_dates=[
                    _hop_event_date(hop) for index, hop in enumerate(seed.hops)
                    if index > 0 and hop.incoming_time_policy == "advance_required"
                ],
            )
            packet_index = len(packets)
            validated_chains[packet_index] = chain
            packet: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "packet_id": hashlib.sha256(
                    f"{seed.id}|{seed.model_id}|{seed.target_as_of}".encode()
                ).hexdigest()[:16],
                "seed_id": seed.id,
                "question": question,
                "question_generation": question_generation,
                "knowledge_cutoff": seed.cutoff.to_dict(),
                "category": seed.category,
                "selection_metadata": seed.selection_metadata,
                "chain": _public_chain(chain),
                "shortest_path": shortest_path,
                "validity_contract_v25": validity,
                "deterministic_errors": errors,
                "wikipedia_api_usage": {
                    "calls_this_seed": (
                        int(backend.request_stats()["api_calls_used"] or 0)
                        - int(api_before["api_calls_used"] or 0)
                    ),
                    **backend.request_stats(),
                    "cache_path": args.cache_path,
                    "request_interval_seconds": args.request_interval,
                    "max_retries": args.api_max_retries,
                    "backoff_base_seconds": args.api_backoff_base,
                    "backoff_max_seconds": args.api_backoff_max,
                },
            }
            if errors:
                packet["status"] = (
                    "infrastructure_error"
                    if has_infrastructure_error(errors)
                    else "deterministic_reject"
                )
            elif args.validate_only:
                packet["status"] = "deterministic_pass"
            else:
                packet["status"] = "judge_pending"
                judge_jobs.append((len(packets), seed))
            packets.append(packet)
            print(
                f"[{packet['status']}] {seed.id}: {len(chain)} verified hops",
                flush=True,
            )

        accepted_by_index: dict[int, dict[str, Any]] = {}
        if judge_jobs:
            assert judge is not None
            with ThreadPoolExecutor(max_workers=args.judge_workers) as executor:
                futures = {
                    executor.submit(
                        _judge_seed,
                        judge,
                        str(packets[index]["question"]),
                        list(packets[index]["chain"]),
                        seed.id,
                    ): (index, seed)
                    for index, seed in judge_jobs
                }
                for future in as_completed(futures):
                    index, seed = futures[future]
                    packet = packets[index]
                    try:
                        judged = future.result()
                    except Exception as exc:
                        packet["status"] = "infrastructure_error"
                        packet["judge_error"] = str(exc)
                        print(
                            f"[infrastructure_error] {seed.id}: judge request failed",
                            flush=True,
                        )
                        continue
                    packet["judge"] = judged
                    packet["status"] = (
                        "machine_pass_human_review_required"
                        if judged["decision"] == "pass" else "judge_reject"
                    )
                    if judged["decision"] == "pass":
                        accepted_by_index[index] = build_case(
                            seed, validated_chains[index], judged,
                            question=str(packet["question"]),
                            question_generation=packet.get("question_generation"),
                            validity=packet.get("validity_contract_v25"),
                            search_space_diagnostic=packet.get("shortest_path"),
                        )
                    print(
                        f"[{packet['status']}] {seed.id}: judge "
                        f"cache_hit={judged.get('cache_hit', False)}",
                        flush=True,
                    )
        accepted = [accepted_by_index[index] for index in sorted(accepted_by_index)]
    finally:
        set_usage_ledger(None)
        if usage_ledger is not None:
            usage_ledger.close()
        backend.close()
        if judge is not None:
            judge.close()
    _write_jsonl(args.output, packets)
    if not args.validate_only:
        with open(args.cases_output, "w", encoding="utf-8") as fh:
            json.dump({"cases": accepted}, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    failed_infrastructure = any(
        row.get("status") == "infrastructure_error" for row in packets
    )
    return 0 if (
        all(not row["deterministic_errors"] for row in packets)
        and not failed_infrastructure
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
