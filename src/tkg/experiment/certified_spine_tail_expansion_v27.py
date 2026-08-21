"""Expand tail questions from an already promoted temporal-spine certificate.

The first three hops are immutable and hash-bound to a successful promotion.
Only the candidate-specific tail hyperlink is fetched here.  The downstream v6
whole-chain validator remains responsible for per-question composition judgment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tkg.experiment.candidate_seed_promotion import TAILS, TAIL_FAMILIES, evidence_block
from tkg.experiment.ready_candidate_curation_v27 import _spine_key
from tkg.experiment.results import assert_new_output_path
from tkg.wikipedia.backend import WikipediaPageBackend, normalize_title


SCHEMA_VERSION = "certified-spine-tail-expansion-v2.7"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest()


def expand_tail(
    candidate: dict[str, Any], certificate_seed: dict[str, Any],
    backend: WikipediaPageBackend,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    certificate_hops = certificate_seed.get("hops", [])[:3]
    certificate_sha = _canonical_hash(certificate_hops)
    packet: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate.get("id"),
        "spine_certificate_sha256": certificate_sha,
        "bridge_review_reused": True,
        "status": "rejected",
        "errors": [],
    }
    if len(certificate_hops) != 3:
        packet["errors"].append("certificate seed must contain three bridge hops")
        return None, packet
    chain = candidate.get("private_chain", [])
    if not isinstance(chain, list) or len(chain) != 4:
        packet["errors"].append("candidate must contain four private-chain edges")
        return None, packet
    source_candidate_id = str(
        certificate_seed.get("selection_metadata", {}).get("source_candidate_id", "")
    )
    if not source_candidate_id:
        packet["errors"].append("certificate seed lacks source candidate identity")
        return None, packet

    tail = chain[-1]
    property_id = str(tail.get("property_id"))
    if property_id not in TAILS:
        packet["errors"].append(f"unsupported tail property: {property_id}")
        return None, packet
    target_as_of = str(candidate["target_as_of"])
    try:
        page = backend.fetch_page(str(tail["source_title"]), as_of=target_as_of)
        label, clause, answer_kind = TAILS[property_id]
        block, aliases = evidence_block(
            page, str(tail["target_title"]), preferred_terms=label.split(),
        )
    except Exception as exc:
        packet["status"] = "infrastructure_error"
        packet["errors"].append(str(exc))
        return None, packet
    packet["tail_snapshot"] = {
        "source_title": page.title,
        "revision_id": page.revision_id,
        "revision_date": page.timestamp,
        "as_of": target_as_of,
        "target_title": tail["target_title"],
        "evidence_found": block is not None,
    }
    if block is None:
        packet["errors"].append("no revision-present tail hyperlink evidence")
        return None, packet

    tail_hop = {
        "source_title": tail["source_title"],
        "target_title": tail["target_title"],
        "relation": label,
        "relative_clause": clause,
        "as_of": target_as_of,
        "evidence": block,
        "target_aliases": aliases,
        "incoming_time_policy": "same_snapshot",
        "property_id": property_id,
        "relation_family": candidate.get("tail_family") or TAIL_FAMILIES[property_id],
        "structured_evidence": {
            "source": "wikidata_statement",
            "direction": "forward_attribute",
            "kg_subject_qid": tail["source_qid"],
            "kg_object_qid": tail["target_qid"],
        },
    }
    seed = {
        "id": "promoted_" + str(candidate["id"]).removeprefix("kgcand_"),
        "model_id": candidate.get("_model_id") or certificate_seed.get("model_id"),
        "cutoff_date": candidate["knowledge_cutoff"],
        "target_as_of": target_as_of,
        "anchor_label": candidate["public_anchor"],
        "answer_kind": answer_kind,
        "category": candidate["domain_family"],
        "old_answer_keywords": [],
        "hops": [*copy.deepcopy(certificate_hops), tail_hop],
        "selection_metadata": {
            "source_candidate_id": candidate["id"],
            "promotion_schema": SCHEMA_VERSION,
            "private_chain_inference_visibility": "forbidden",
            "spine_certificate_source_candidate_id": source_candidate_id,
            "spine_certificate_sha256": certificate_sha,
        },
    }
    packet["status"] = "promoted_pending_v6_validation"
    return seed, packet


def _claim_qids(entity: dict[str, Any], property_id: str) -> list[str]:
    qids = []
    for claim in entity.get("claims", {}).get(property_id, []):
        if not isinstance(claim, dict) or claim.get("rank") == "deprecated":
            continue
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        qid = value.get("id") if isinstance(value, dict) else None
        if isinstance(qid, str) and qid.startswith("Q"):
            qids.append(qid)
    return list(dict.fromkeys(qids))


def synthesize_linked_tails(
    certificate_seed: dict[str, Any], backend: WikipediaPageBackend,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enumerate KG attributes that are hyperlinks in the exact target revision.

    These remain pending seeds: the downstream whole-chain semantic judge must
    still establish that the cited passage expresses the requested relation.
    """
    bridge_hops = certificate_seed.get("hops", [])[:3]
    if len(bridge_hops) != 3:
        return [], [{"status": "rejected", "errors": ["certificate lacks bridge hops"]}]
    final_hop = bridge_hops[-1]
    structured = final_hop.get("structured_evidence", {})
    subject_qid = str(structured.get("kg_object_qid") or "")
    subject_title = str(final_hop.get("target_title") or "")
    target_as_of = str(certificate_seed.get("target_as_of") or "")
    certificate_sha = _canonical_hash(bridge_hops)
    try:
        page = backend.fetch_page(subject_title, as_of=target_as_of)
        subject = backend.get_wikidata_entities([subject_qid])[subject_qid]
        claims = [
            (property_id, qid)
            for property_id in sorted(TAILS)
            for qid in _claim_qids(subject, property_id)
        ]
        entities = backend.get_wikidata_entities(
            [qid for _, qid in claims], props="labels|sitelinks",
        ) if claims else {}
    except Exception as exc:
        return [], [{
            "schema_version": SCHEMA_VERSION,
            "spine_certificate_sha256": certificate_sha,
            "status": "infrastructure_error", "errors": [str(exc)],
        }]

    linked_targets = {normalize_title(link.target).casefold() for link in page.links}
    seeds, packets = [], []
    seen: set[tuple[str, str]] = set()
    for property_id, target_qid in claims:
        entity = entities.get(target_qid, {})
        target_title = entity.get("sitelinks", {}).get("enwiki", {}).get("title")
        identity = (property_id, target_qid)
        if not isinstance(target_title, str) or identity in seen:
            continue
        if normalize_title(target_title).casefold() not in linked_targets:
            continue
        seen.add(identity)
        label, clause, answer_kind = TAILS[property_id]
        block, aliases = evidence_block(
            page, target_title, preferred_terms=label.split(),
        )
        packet: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": None,
            "spine_certificate_sha256": certificate_sha,
            "bridge_review_reused": True,
            "tail_source": "wikidata_claim_intersect_exact_revision_hyperlink",
            "tail_property": property_id,
            "target_qid": target_qid,
            "status": "rejected",
            "errors": [],
        }
        if block is None:
            packet["errors"].append("linked target lacked extractable evidence block")
            packets.append(packet)
            continue
        entity_path = [
            str(bridge_hops[0].get("source_title") or ""),
            *(str(hop.get("target_title") or "") for hop in bridge_hops),
            target_title,
        ]
        nonempty_path = [title for title in entity_path if title]
        if len({title.casefold() for title in nonempty_path}) != len(nonempty_path):
            packet["errors"].append("tail would create an entity cycle")
            packets.append(packet)
            continue
        suffix = _canonical_hash({
            "certificate": certificate_sha, "property": property_id, "target": target_qid,
        })[:16]
        synthetic_id = f"linkedtail_{suffix}"
        tail_hop = {
            "source_title": subject_title, "target_title": target_title,
            "relation": label, "relative_clause": clause, "as_of": target_as_of,
            "evidence": block, "target_aliases": aliases,
            "incoming_time_policy": "same_snapshot", "property_id": property_id,
            "relation_family": TAIL_FAMILIES[property_id],
            "structured_evidence": {
                "source": "wikidata_statement", "direction": "forward_attribute",
                "kg_subject_qid": subject_qid, "kg_object_qid": target_qid,
            },
        }
        seed = {
            "id": synthetic_id,
            "model_id": certificate_seed.get("model_id"),
            "cutoff_date": certificate_seed.get("cutoff_date"),
            "target_as_of": target_as_of,
            "anchor_label": certificate_seed.get("anchor_label"),
            "answer_kind": answer_kind,
            "category": certificate_seed.get("category"),
            "old_answer_keywords": [],
            "hops": [*copy.deepcopy(bridge_hops), tail_hop],
            "selection_metadata": {
                "source_candidate_id": synthetic_id,
                "promotion_schema": SCHEMA_VERSION,
                "private_chain_inference_visibility": "forbidden",
                "spine_certificate_sha256": certificate_sha,
                "tail_source": packet["tail_source"],
            },
        }
        packet["candidate_id"] = synthetic_id
        packet["status"] = "promoted_pending_v6_validation"
        packet["tail_snapshot"] = {
            "source_title": page.title, "revision_id": page.revision_id,
            "revision_date": page.timestamp, "as_of": target_as_of,
            "target_title": target_title, "evidence_found": True,
        }
        seeds.append(seed)
        packets.append(packet)
    return seeds, packets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", action="append", required=True,
        help="candidate batch; repeat to expand certified spines across batches",
    )
    parser.add_argument("--certificate-seeds", action="append", required=True)
    parser.add_argument(
        "--exclude-seeds", action="append", default=[],
        help="exclude already emitted exact spine/property/target tail identities",
    )
    parser.add_argument("--model-id", default="openai/gpt-4.1-mini")
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--request-interval", type=float, default=1.0)
    parser.add_argument("--api-call-budget", type=int, default=5000)
    parser.add_argument(
        "--synthesize-linked-tails", action="store_true",
        help="also enumerate every supported KG attribute linked in the exact revision",
    )
    parser.add_argument(
        "--synthesis-only", action="store_true",
        help="skip candidate tails and emit only exact-revision linked-tail synthesis",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--packets-output", required=True)
    args = parser.parse_args()
    if args.synthesis_only and not args.synthesize_linked_tails:
        parser.error("--synthesis-only requires --synthesize-linked-tails")
    assert_new_output_path(args.output)
    assert_new_output_path(args.packets_output)

    candidates: list[dict[str, Any]] = []
    for candidate_path in args.candidates:
        payload = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
        candidates.extend(
            dict(row, _model_id=args.model_id)
            for row in payload.get("questions", []) if isinstance(row, dict)
        )
    by_id = {str(row["id"]): row for row in candidates}
    certificates: dict[str, dict[str, Any]] = {}
    certified_candidate_ids: set[str] = set()
    for seed_path in args.certificate_seeds:
        seed_payload = json.loads(Path(seed_path).read_text(encoding="utf-8"))
        for seed in seed_payload.get("seeds", []):
            source_id = str(seed.get("selection_metadata", {}).get("source_candidate_id", ""))
            source = by_id.get(source_id)
            if source is None:
                continue
            certificates[_spine_key(source)] = dict(seed)
            certified_candidate_ids.add(source_id)

    selected = [] if args.synthesis_only else [
        row for row in candidates
        if _spine_key(row) in certificates and row["id"] not in certified_candidate_ids
    ]
    backend = WikipediaPageBackend(
        cache_path=args.cache_path,
        min_request_interval=args.request_interval,
        max_api_calls=args.api_call_budget,
    )
    seeds: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []
    excluded_tail_keys: set[tuple[str, str, str]] = set()
    for seed_path in args.exclude_seeds:
        payload = json.loads(Path(seed_path).read_text(encoding="utf-8"))
        for seed in payload.get("seeds", []):
            hops = seed.get("hops", []) if isinstance(seed, dict) else []
            if len(hops) < 4:
                continue
            excluded_tail_keys.add((
                _canonical_hash(hops[:3]), str(hops[3].get("property_id")),
                str(hops[3].get("structured_evidence", {}).get("kg_object_qid")),
            ))
    try:
        for row in selected:
            seed, packet = expand_tail(row, certificates[_spine_key(row)], backend)
            packets.append(packet)
            if seed is not None:
                seeds.append(seed)
        if args.synthesize_linked_tails:
            existing_ids = {str(seed.get("id")) for seed in seeds}
            existing_tail_keys = {
                (
                    _canonical_hash(seed.get("hops", [])[:3]),
                    str(seed.get("hops", [{}, {}, {}, {}])[3].get("property_id")),
                    str(seed.get("hops", [{}, {}, {}, {}])[3].get(
                        "structured_evidence", {}
                    ).get("kg_object_qid")),
                )
                for seed in seeds if len(seed.get("hops", [])) >= 4
            } | excluded_tail_keys
            for certificate in certificates.values():
                synthesized, synthetic_packets = synthesize_linked_tails(
                    certificate, backend,
                )
                packets.extend(synthetic_packets)
                for seed in synthesized:
                    tail_key = (
                        _canonical_hash(seed.get("hops", [])[:3]),
                        str(seed["hops"][3].get("property_id")),
                        str(seed["hops"][3].get("structured_evidence", {}).get(
                            "kg_object_qid"
                        )),
                    )
                    if (
                        str(seed.get("id")) not in existing_ids
                        and tail_key not in existing_tail_keys
                    ):
                        seeds.append(seed)
                        existing_ids.add(str(seed.get("id")))
                        existing_tail_keys.add(tail_key)
    finally:
        backend.close()
    Path(args.output).write_text(json.dumps({
        "schema_version": "multihop-seed-batch-v1",
        "seeds": seeds,
        "provenance": {
            "expansion_schema": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "certified_spine_count": len(certificates),
            "selected_tail_candidate_count": len(selected),
            "synthesize_linked_tails": args.synthesize_linked_tails,
            "promoted": len(seeds),
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.packets_output).write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in packets
    ), encoding="utf-8")
    print(json.dumps({
        "certified_spines": len(certificates),
        "selected_tail_candidates": len(selected),
        "promoted": len(seeds),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
