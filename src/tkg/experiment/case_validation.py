"""Static preflight for Wikipedia experiment cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime

from tkg.wikipedia.snapshot import page_version_key
from tkg.experiment.event_order import event_order_certificate_errors


def _canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def validate_chain_route(
    case: dict,
    navigation: dict,
    *,
    require_raw_shortest: bool | None = None,
) -> dict | None:
    """Prove the semantic route exists and separately measure raw shortcuts.

    New temporal-waypoint cases intentionally use a product-graph contract: a
    later Wikipedia revision may retain old links, so raw hyperlink distance is
    diagnostic rather than an admission gate.  Legacy cases retain the old
    raw-shortest requirement by default.
    """
    chain = case.get("reasoning_chain")
    if not chain:
        return None
    arena_edges = {
        (edge["source"], edge["target"], edge["kind"])
        for edge in navigation["arena_edges"]
    }
    route: list[str] = []
    edge_kinds: list[str] = []
    for index, hop in enumerate(chain):
        source_key = page_version_key(hop["source_title"], hop["source_revision_id"])
        target_key = page_version_key(hop["target_title"], hop["target_revision_id"])
        if index == 0:
            route.append(source_key)
        elif route[-1] != source_key:
            if (route[-1], source_key, "temporal") not in arena_edges:
                raise ValueError(
                    f"{case['id']}: intended temporal transition before hop {index} "
                    "is absent from navigation arena"
                )
            route.append(source_key)
            edge_kinds.append("temporal")
        if (source_key, target_key, "hyperlink") not in arena_edges:
            raise ValueError(
                f"{case['id']}: intended hyperlink hop {index} is absent from navigation arena"
            )
        route.append(target_key)
        edge_kinds.append("hyperlink")
    if route[-1] != navigation["target_key"]:
        raise ValueError(f"{case['id']}: reasoning chain does not end at pivot revision")
    expected = int(case["expected_navigation_distance"])
    route_distance = len(edge_kinds)
    if expected != route_distance:
        raise ValueError(
            f"{case['id']}: expected_navigation_distance={expected}, route={route_distance}"
        )
    raw_shortest = navigation["distances"].get(route[0])
    if raw_shortest is None:
        raise ValueError(f"{case['id']}: reasoning-chain anchor is absent from arena")
    if require_raw_shortest is None:
        require_raw_shortest = not bool(case.get("temporal_waypoints"))
    if require_raw_shortest and raw_shortest != route_distance:
        raise ValueError(
            f"{case['id']}: declared chain is not shortest; chain={route_distance}, "
            f"arena minimum={raw_shortest}"
        )
    return {
        "start_title": case["start_title"], "start_key": route[0],
        "distance": route_distance,
        "semantic_distance": route_distance,
        "raw_distance": raw_shortest,
        "raw_shortest_match": raw_shortest == route_distance,
        "route_keys": route, "edge_kinds": edge_kinds,
    }


def validate_case(case: dict, *, allow_legacy: bool = False) -> list[str]:
    errors = []
    prefix = case.get("id", "<missing-id>")
    if not case.get("id"):
        errors.append(f"{prefix}: missing id")
    if not case.get("temporal_question") and not (
        allow_legacy and case.get("pk_question")
    ):
        errors.append(f"{prefix}: missing temporal_question")
    if not case.get("wikipedia_before") and not allow_legacy:
        errors.append(f"{prefix}: missing wikipedia_before")
    for date_field in ("wikipedia_before", "wikipedia_as_of"):
        if not case.get(date_field):
            continue
        try:
            datetime.fromisoformat(str(case[date_field]).replace("Z", "+00:00"))
        except ValueError:
            errors.append(f"{prefix}: invalid {date_field}")
    if case.get("wikipedia_before") and case.get("wikipedia_as_of"):
        try:
            before = datetime.fromisoformat(str(case["wikipedia_before"]).replace("Z", "+00:00"))
            after = datetime.fromisoformat(str(case["wikipedia_as_of"]).replace("Z", "+00:00"))
            if before >= after:
                errors.append(f"{prefix}: wikipedia_before must precede wikipedia_as_of")
        except ValueError:
            pass
    if not (case.get("wikipedia_title") or case.get("pivot_qid")):
        errors.append(f"{prefix}: missing wikipedia_title/pivot_qid")
    if not case.get("new_answer_keywords"):
        errors.append(f"{prefix}: empty new_answer_keywords")
    is_multihop = bool(case.get("reasoning_chain"))
    if not case.get("old_answer_keywords") and not is_multihop:
        errors.append(f"{prefix}: empty old_answer_keywords")
    if is_multihop:
        question = str(case.get("temporal_question") or "")
        if re.search(
            r"\b(?:snapshot\s+(?:that\s+was\s+)?"
            r"(?:used|selected|chosen|viewed|loaded|inspected|opened)\s+"
            r"(?:in|during|at|for)\s+(?:the\s+)?previous step|"
            r"(?:the\s+)?previous[- ]step(?:'s|\s+)(?:used\s+)?snapshot|"
            r"snapshot\s+(?:from|of)\s+(?:the\s+)?previous step)\b",
            question.casefold(),
        ):
            errors.append(
                f"{prefix}: answer depends on a solver-selected previous snapshot"
            )
        chain = case.get("reasoning_chain")
        if not isinstance(chain, list) or len(chain) < 2:
            errors.append(f"{prefix}: reasoning_chain must contain at least two hops")
            chain = []
        else:
            if case.get("reasoning_hop_count") != len(chain):
                errors.append(f"{prefix}: reasoning_hop_count does not match chain")
            if not case.get("start_title"):
                errors.append(f"{prefix}: multi-hop case missing start_title")
            elif str(case["start_title"]).casefold() != str(
                chain[0].get("source_title", "")
            ).casefold():
                errors.append(f"{prefix}: start_title does not match first hop")
            if not case.get("hide_pivot_title"):
                errors.append(f"{prefix}: multi-hop case must hide pivot title")
            if not isinstance(case.get("expected_navigation_distance"), int):
                errors.append(f"{prefix}: missing expected_navigation_distance")
            for index, hop in enumerate(chain):
                required = {
                    "source_title", "source_revision_id", "target_title",
                    "target_revision_id", "as_of", "relation", "evidence",
                    "target_aliases",
                }
                missing = sorted(required - set(hop)) if isinstance(hop, dict) else sorted(required)
                if missing:
                    errors.append(
                        f"{prefix}: hop {index} missing fields: {', '.join(missing)}"
                    )
            for index, (left, right) in enumerate(zip(chain, chain[1:])):
                if str(left.get("target_title", "")).casefold() != str(
                    right.get("source_title", "")
                ).casefold():
                    errors.append(f"{prefix}: disconnected reasoning hops {index}->{index + 1}")
                try:
                    left_time = datetime.fromisoformat(
                        str(left.get("as_of", "")).replace("Z", "+00:00")
                    )
                    right_time = datetime.fromisoformat(
                        str(right.get("as_of", "")).replace("Z", "+00:00")
                    )
                    policy = right.get("incoming_time_policy", "advance_required")
                    if policy == "advance_required" and right_time <= left_time:
                        errors.append(
                            f"{prefix}: hop {index + 1} must use a later snapshot"
                        )
                    elif policy == "same_snapshot" and right_time != left_time:
                        errors.append(
                            f"{prefix}: hop {index + 1} must retain the same snapshot"
                        )
                    elif policy not in {"advance_required", "same_snapshot"}:
                        errors.append(
                            f"{prefix}: hop {index + 1} has invalid incoming_time_policy"
                        )
                except ValueError:
                    errors.append(f"{prefix}: invalid reasoning-hop timestamp")
        waypoints = case.get("temporal_waypoints")
        expected_edges = []
        temporal_switches = 0
        for index, hop in enumerate(chain):
            policy = hop.get("incoming_time_policy", "advance_required")
            if index == 0 or policy != "same_snapshot":
                expected_edges.append((
                    hop.get("source_title"), hop.get("source_revision_id"),
                    "start" if index == 0 else "temporal",
                ))
                if index > 0:
                    temporal_switches += 1
            expected_edges.append((
                hop.get("target_title"), hop.get("target_revision_id"), "hyperlink",
            ))
        if not isinstance(waypoints, list) or len(waypoints) != len(expected_edges):
            errors.append(
                f"{prefix}: temporal_waypoints do not match temporal/same-snapshot hops"
            )
        else:
            for index, (waypoint, expected) in enumerate(zip(waypoints, expected_edges)):
                title, revision_id, incoming = expected
                if not isinstance(waypoint, dict):
                    errors.append(f"{prefix}: waypoint {index} is not an object")
                    continue
                if waypoint.get("index") != index:
                    errors.append(f"{prefix}: waypoint {index} has wrong index")
                if str(waypoint.get("title", "")).casefold() != str(title).casefold():
                    errors.append(f"{prefix}: waypoint {index} title does not match chain")
                if waypoint.get("revision_id") != revision_id:
                    errors.append(f"{prefix}: waypoint {index} revision does not match chain")
                if waypoint.get("incoming_edge") != incoming:
                    errors.append(f"{prefix}: waypoint {index} incoming edge is invalid")
            semantic_distance = len(waypoints) - 1
            if case.get("semantic_shortest_distance") != semantic_distance:
                errors.append(f"{prefix}: semantic_shortest_distance does not match waypoints")
            if case.get("expected_navigation_distance") != semantic_distance:
                errors.append(f"{prefix}: expected_navigation_distance does not match waypoints")
            if case.get("required_temporal_switches") != temporal_switches:
                errors.append(f"{prefix}: required_temporal_switches does not match chain")
        cutoff = case.get("knowledge_cutoff")
        if not isinstance(cutoff, dict) or not cutoff.get("cutoff_date"):
            errors.append(f"{prefix}: multi-hop case missing knowledge_cutoff")
        required_dates = case.get("required_snapshot_dates")
        if not isinstance(required_dates, list) or len(required_dates) < 2:
            errors.append(f"{prefix}: multi-hop case needs at least two snapshot dates")
        generation = case.get("_generation", {})
        if generation.get("schema_version") == "wikipedia-cutoff-relative-multihop-v5":
            errors.append(
                f"{prefix}: generation v5 is invalidated; regenerate under v6 "
                "with event-order and frozen-evidence contracts"
            )
            contract = case.get("prior_knowledge_contract")
            if not isinstance(contract, dict) or contract.get(
                "schema_version"
            ) != "factorized-prior-knowledge-v1":
                errors.append(
                    f"{prefix}: v5 generation requires factorized prior knowledge"
                )
            elif not any(
                isinstance(probe, dict)
                and probe.get("role") == contract.get("primary_admission_role")
                and probe.get("objective") == "must_be_unknown"
                for probe in contract.get("probes", [])
            ):
                errors.append(
                    f"{prefix}: factorized prior knowledge lacks a critical bridge"
                )
        if generation.get("schema_version") == "wikipedia-cutoff-relative-multihop-v6":
            contract = case.get("prior_knowledge_contract")
            if not isinstance(contract, dict) or contract.get(
                "schema_version"
            ) != "factorized-prior-knowledge-v2":
                errors.append(
                    f"{prefix}: v6 generation requires factorized prior knowledge v2"
                )
            else:
                probes = contract.get("probes", [])
                primary_ids = contract.get("primary_admission_probe_ids")
                objective_ids = {
                    str(probe.get("id")) for probe in probes
                    if isinstance(probe, dict)
                    and probe.get("objective") == "must_be_unknown"
                }
                if (
                    not isinstance(primary_ids, list) or not primary_ids
                    or set(map(str, primary_ids)) != objective_ids
                ):
                    errors.append(
                        f"{prefix}: v2 PK primary IDs do not match acquisition edges"
                    )
            for index, hop in enumerate(chain):
                for field in (
                    "source_content_sha256", "source_links_sha256",
                    "target_content_sha256", "target_links_sha256",
                ):
                    if not re.fullmatch(r"[0-9a-f]{64}", str(hop.get(field, ""))):
                        errors.append(f"{prefix}: hop {index} missing valid {field}")
                structured = hop.get("structured_evidence")
                claims_first_or_next = bool(
                    re.search(
                        r"\b(?:first|next)\b",
                        str(hop.get("relative_clause", "")), re.IGNORECASE,
                    )
                    or (
                        isinstance(structured, dict)
                        and structured.get("temporal_operator")
                        == "next_after_boundary"
                    )
                )
                if (
                    index > 0
                    and hop.get("incoming_time_policy") == "advance_required"
                    and isinstance(structured, dict)
                    and structured.get("source") == "wikidata_time_qualified_statement"
                    and claims_first_or_next
                ):
                    previous = chain[index - 1]
                    previous_structured = previous.get("structured_evidence")
                    boundary = (
                        previous_structured.get("event_date")
                        if isinstance(previous_structured, dict)
                        and previous_structured.get("event_date")
                        else previous.get("as_of")
                    )
                    selected_qid = (
                        structured.get("kg_subject_qid")
                        if structured.get("direction") == "inverse"
                        else structured.get("kg_object_qid")
                    )
                    for error in event_order_certificate_errors(
                        structured.get("event_order_certificate"),
                        expected_boundary=str(boundary),
                        expected_event_date=str(structured.get("event_date")),
                        expected_target_qid=(
                            str(selected_qid) if selected_qid else None
                        ),
                    ):
                        errors.append(f"{prefix}: hop {index} {error}")

            frozen = case.get("frozen_wikipedia_evidence")
            if not isinstance(frozen, dict) or frozen.get(
                "schema_version"
            ) != "frozen-wikipedia-evidence-v1":
                errors.append(f"{prefix}: missing frozen Wikipedia evidence")
            else:
                frozen_pages = frozen.get("pages")
                if not isinstance(frozen_pages, list) or not frozen_pages:
                    errors.append(f"{prefix}: frozen Wikipedia evidence has no pages")
                    frozen_pages = []
                frozen_index = {}
                page_hashes = []
                for page_index, page in enumerate(frozen_pages):
                    if not isinstance(page, dict):
                        errors.append(
                            f"{prefix}: frozen evidence page {page_index} is invalid"
                        )
                        continue
                    content_hash = hashlib.sha256(
                        str(page.get("content", "")).encode("utf-8")
                    ).hexdigest()
                    links_hash = _canonical_sha256(page.get("links"))
                    snapshot_payload = {
                        key: value for key, value in page.items()
                        if key != "snapshot_sha256"
                    }
                    snapshot_hash = _canonical_sha256(snapshot_payload)
                    if page.get("content_sha256") != content_hash:
                        errors.append(
                            f"{prefix}: frozen page {page_index} content hash mismatch"
                        )
                    if page.get("links_sha256") != links_hash:
                        errors.append(
                            f"{prefix}: frozen page {page_index} link hash mismatch"
                        )
                    if page.get("snapshot_sha256") != snapshot_hash:
                        errors.append(
                            f"{prefix}: frozen page {page_index} snapshot hash mismatch"
                        )
                    page_hashes.append(page.get("snapshot_sha256"))
                    frozen_index[(
                        str(page.get("title", "")).casefold(),
                        page.get("revision_id"),
                    )] = page
                manifest_hash = _canonical_sha256({
                    "schema_version": "frozen-wikipedia-evidence-v1",
                    "page_hashes": page_hashes,
                })
                if frozen.get("manifest_sha256") != manifest_hash:
                    errors.append(f"{prefix}: frozen evidence manifest hash mismatch")
                for index, hop in enumerate(chain):
                    for role in ("source", "target"):
                        page = frozen_index.get((
                            str(hop.get(f"{role}_title", "")).casefold(),
                            hop.get(f"{role}_revision_id"),
                        ))
                        if page is None:
                            errors.append(
                                f"{prefix}: hop {index} {role} revision is not frozen"
                            )
                        elif page.get("content_sha256") != hop.get(
                            f"{role}_content_sha256"
                        ):
                            errors.append(
                                f"{prefix}: hop {index} {role} frozen hash mismatch"
                            )
        if generation.get("question_style") == "short_ordered_steps_v1":
            question = str(case.get("temporal_question", ""))
            sentences = [
                value.strip() for value in re.split(r"(?<=[.!?])\s+", question)
                if value.strip()
            ]
            if len(sentences) < len(chain) + 1:
                errors.append(
                    f"{prefix}: short-step question must separate relation hops"
                )
            if any(len(sentence.split()) > 32 for sentence in sentences):
                errors.append(
                    f"{prefix}: short-step question contains a sentence over 32 words"
                )
            canonical = generation.get("canonical_question")
            if not isinstance(canonical, str) or not canonical.strip():
                errors.append(f"{prefix}: missing canonical audit question")
    return errors


def validate_cases(cases: list[dict], *, allow_legacy: bool = False) -> list[str]:
    errors = []
    seen = set()
    for case in cases:
        case_id = case.get("id")
        if case_id in seen:
            errors.append(f"duplicate case id: {case_id}")
        seen.add(case_id)
        errors.extend(validate_case(case, allow_legacy=allow_legacy))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="generated_cases.json")
    parser.add_argument(
        "--allow-legacy", action="store_true",
        help="accept pk_question and missing wikipedia_before in archived cases",
    )
    args = parser.parse_args()
    with open(args.cases, "r", encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]
    errors = validate_cases(cases, allow_legacy=args.allow_legacy)
    if errors:
        print("\n".join(f"[ERROR] {error}" for error in errors))
        raise SystemExit(1)
    print(f"[OK] {len(cases)} cases passed static validation")


if __name__ == "__main__":
    main()
