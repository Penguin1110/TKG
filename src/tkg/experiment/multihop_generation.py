"""Build auditable cutoff-relative, multi-hop Wikipedia questions.

The seed supplies semantic relations; Wikipedia supplies the revision text and
hyperlink graph.  The program composes the relative question deterministically,
then an independent LLM judges whether the evidence really expresses the
claimed relation chain.  It never invents a relation from hyperlink topology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tkg.api.openrouter import call_model
from tkg.experiment.case_validation import validate_chain_route
from tkg.experiment.model_cutoffs import ModelCutoff, get_model_cutoff
from tkg.experiment.question_generation import (
    _fold, _strings, call_json_model,
)
from tkg.experiment.results import assert_new_output_path
from tkg.wikipedia.backend import WikipediaError, WikipediaPageBackend
from tkg.wikipedia.snapshot import temporal_reverse_bfs


SCHEMA_VERSION = "wikipedia-cutoff-relative-multihop-v1"
MIN_HOPS = 2
MAX_HOPS = 6


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class ChainHopSeed:
    source_title: str
    target_title: str
    relation: str
    relative_clause: str
    as_of: str
    evidence: str
    target_aliases: tuple[str, ...]

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
        _parse_time(str(value["as_of"]))
        return cls(
            source_title=str(value["source_title"]).strip(),
            target_title=str(value["target_title"]).strip(),
            relation=str(value["relation"]).strip(),
            relative_clause=clause,
            as_of=str(value["as_of"]),
            evidence=str(value["evidence"]).strip(),
            target_aliases=tuple(aliases),
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
        )


def compose_relative_question(seed: MultiHopSeed) -> str:
    """Nest relation clauses while keeping every target entity hidden."""
    description = seed.hops[0].relative_clause.replace("{source}", seed.anchor_label)
    for hop in seed.hops[1:]:
        description = hop.relative_clause.replace("{source}", description)
    return f"At the target snapshot, {seed.answer_kind.casefold()} is {description}?"


def validate_chain(seed: MultiHopSeed, backend) -> tuple[list[str], list[dict[str, Any]]]:
    """Verify exact evidence, real revision links, connectivity, and no leakage."""
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    question = compose_relative_question(seed)
    hidden_aliases: list[str] = []
    previous_target_page = None
    seen_state_edges: set[tuple[int, int]] = set()
    for index, hop in enumerate(seed.hops):
        try:
            source = backend.fetch_page(hop.source_title, as_of=hop.as_of)
            target = backend.fetch_page(hop.target_title, as_of=hop.as_of)
        except WikipediaError as exc:
            errors.append(f"hop {index}: Wikipedia fetch failed: {exc}")
            continue
        if source.title.casefold() != hop.source_title.casefold():
            errors.append(f"hop {index}: source canonical title changed to {source.title!r}")
        if target.title.casefold() != hop.target_title.casefold():
            errors.append(f"hop {index}: target canonical title changed to {target.title!r}")
        if previous_target_page is not None:
            if previous_target_page.title.casefold() != source.title.casefold():
                errors.append(f"hop {index}: fetched chain is disconnected")
        if _fold(hop.evidence) not in _fold(source.content):
            errors.append(f"hop {index}: evidence is not a verbatim source-revision substring")
        evidence_folded = _fold(hop.evidence)
        for alias in hop.target_aliases:
            if _fold(alias) not in evidence_folded:
                errors.append(f"hop {index}: alias {alias!r} is absent from evidence")
        targets = {link.target.casefold() for link in source.links}
        if target.title.casefold() not in targets:
            errors.append(f"hop {index}: source revision has no hyperlink to target")
        edge = (source.revision_id, target.revision_id)
        if edge in seen_state_edges:
            errors.append(f"hop {index}: duplicate page-version edge")
        seen_state_edges.add(edge)
        hidden_aliases.extend([hop.target_title, *hop.target_aliases])
        records.append({
            "index": index,
            "source_title": source.title,
            "source_revision_id": source.revision_id,
            "source_timestamp": source.timestamp,
            "source_url": source.source_url,
            "target_title": target.title,
            "target_revision_id": target.revision_id,
            "target_timestamp": target.timestamp,
            "target_url": target.source_url,
            "as_of": hop.as_of,
            "relation": hop.relation,
            "relative_clause": hop.relative_clause,
            "evidence": hop.evidence,
            "target_aliases": list(hop.target_aliases),
        })
        previous_target_page = target
    folded_question = _fold(question)
    for alias in hidden_aliases:
        if _fold(alias) in folded_question:
            errors.append(f"question leaks hidden entity alias {alias!r}")
    if seed.anchor_label.casefold() not in question.casefold():
        errors.append("question omits the public anchor label")
    final_aliases = {_fold(item) for item in seed.hops[-1].target_aliases}
    if final_aliases & {_fold(item) for item in seed.old_answer_keywords}:
        errors.append("old and target answer aliases overlap")
    return errors, records


class MultiHopQuestionJudge:
    """Independent semantic judge after deterministic revision/link gates."""

    def __init__(
        self, model: str, *, min_confidence: float = 0.8,
        call_model_fn: Callable[..., str] = call_model,
    ):
        self.model = model
        self.min_confidence = min_confidence
        self.call_model_fn = call_model_fn

    def judge(self, question: str, chain: list[dict[str, Any]]) -> dict[str, Any]:
        visible = [{
            "index": hop["index"], "relation": hop["relation"],
            "relative_clause": hop["relative_clause"], "as_of": hop["as_of"],
            "evidence": hop["evidence"], "source_title": hop["source_title"],
            "target_title": hop["target_title"],
        } for hop in chain]
        prompt = f"""Audit this cutoff-relative multi-hop Wikipedia question.

Question: {question}
Ordered verified chain:
{json.dumps(visible, ensure_ascii=False, indent=2)}

Exact-substring and hyperlink checks already passed. Pass only if:
- every evidence excerpt semantically supports its named relation;
- the question composes all relations once, in the same order;
- "registered knowledge cutoff" anchors hop 0 and "target snapshot" anchors the answer;
- the chain is genuinely multi-hop and uniquely identifies the final target;
- no intermediate or final entity is named in the question.
Return one JSON object with decision (pass/reject), confidence (0..1), checks,
reason, and rejected_hops (list of integers).
"""
        raw, response = call_json_model(self.model, prompt, self.call_model_fn)
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        decision = str(raw.get("decision", "reject"))
        if decision != "pass" or confidence < self.min_confidence:
            decision = "reject"
        return {**raw, "decision": decision, "confidence": confidence,
                "raw_response": response}


def build_case(
    seed: MultiHopSeed, chain: list[dict[str, Any]], judge_result: dict[str, Any]
) -> dict[str, Any]:
    question = compose_relative_question(seed)
    required_dates = list(dict.fromkeys([hop.as_of for hop in seed.hops]))
    expected_distance = len(chain) + sum(
        left["target_revision_id"] != right["source_revision_id"]
        for left, right in zip(chain, chain[1:])
    )
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
        "old_answer_keywords": list(seed.old_answer_keywords),
        "new_answer_keywords": list(seed.hops[-1].target_aliases),
        "knowledge_cutoff": {
            **seed.cutoff.to_dict(), "model_ids": [seed.model_id],
            "role": "candidate_prior_only_pk_gate_is_authoritative",
        },
        "reasoning_chain": chain,
        "_generation": {
            "schema_version": SCHEMA_VERSION,
            "status": "machine_pass_human_review_required",
            "judge": {key: value for key, value in judge_result.items()
                      if key != "raw_response"},
        },
    }


def validate_shortest_arena(
    seed: MultiHopSeed,
    chain: list[dict[str, Any]],
    backend,
    *,
    branch_cap: int,
) -> dict[str, Any]:
    """Build the bounded arena and reject semantic paths with a graph shortcut."""
    case = build_case(seed, chain, {"decision": "pending", "confidence": 0.0})
    navigation = temporal_reverse_bfs(
        backend,
        case["wikipedia_title"],
        seed.target_as_of,
        case["required_snapshot_dates"],
        case["expected_navigation_distance"],
        branch_cap=branch_cap,
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
        "shortest_distance": contract["distance"],
        "route_keys": contract["route_keys"],
        "edge_kinds": contract["edge_kinds"],
        "arena_node_count": len(navigation["states"]),
        "arena_edge_count": len(navigation["arena_edges"]),
        "arena_sha256": manifest_hash,
        "coverage_note": navigation["coverage_note"],
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
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-min-confidence", type=float, default=0.8)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--cache-path", default="wikipedia_multihop_generation.db")
    parser.add_argument("--backlink-branch-cap", type=int, default=25)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default="multihop_question_packets.jsonl")
    parser.add_argument("--cases-output", default="generated_multihop_cases.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.judge_min_confidence <= 1:
        parser.error("--judge-min-confidence must be between 0 and 1")
    if args.backlink_branch_cap <= 0:
        parser.error("--backlink-branch-cap must be > 0")
    if not args.validate_only:
        if not args.judge_model:
            parser.error("--judge-model is required unless --validate-only")
        if not os.environ.get("OPENROUTER_API_KEY"):
            parser.error("OPENROUTER_API_KEY is required for judging")
    paths = [args.output] + ([] if args.validate_only else [args.cases_output])
    for path in paths:
        assert_new_output_path(path)
        if Path(path).exists() and not args.overwrite:
            parser.error(f"refusing to overwrite {path}; pass --overwrite explicitly")
    seeds = load_seeds(args.seed_file)
    backend = WikipediaPageBackend(
        cache_path=args.cache_path, lang=args.lang, offline_only=args.offline
    )
    judge = None if args.validate_only else MultiHopQuestionJudge(
        args.judge_model, min_confidence=args.judge_min_confidence
    )
    packets: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            errors, chain = validate_chain(seed, backend)
            shortest_path = None
            if not errors:
                try:
                    shortest_path = validate_shortest_arena(
                        seed, chain, backend, branch_cap=args.backlink_branch_cap
                    )
                except (WikipediaError, ValueError) as exc:
                    errors.append(f"shortest_path: {exc}")
            packet: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "packet_id": hashlib.sha256(
                    f"{seed.id}|{seed.model_id}|{seed.target_as_of}".encode()
                ).hexdigest()[:16],
                "seed_id": seed.id,
                "question": compose_relative_question(seed),
                "knowledge_cutoff": seed.cutoff.to_dict(),
                "chain": chain,
                "shortest_path": shortest_path,
                "deterministic_errors": errors,
            }
            if errors:
                packet["status"] = "deterministic_reject"
            elif args.validate_only:
                packet["status"] = "deterministic_pass"
            else:
                assert judge is not None
                judged = judge.judge(packet["question"], chain)
                packet["judge"] = judged
                packet["status"] = (
                    "machine_pass_human_review_required"
                    if judged["decision"] == "pass" else "judge_reject"
                )
                if judged["decision"] == "pass":
                    accepted.append(build_case(seed, chain, judged))
            packets.append(packet)
            print(f"[{packet['status']}] {seed.id}: {len(chain)} verified hops")
    finally:
        backend.close()
    _write_jsonl(args.output, packets)
    if not args.validate_only:
        with open(args.cases_output, "w", encoding="utf-8") as fh:
            json.dump({"cases": accepted}, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    return 0 if all(not row["deterministic_errors"] for row in packets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
