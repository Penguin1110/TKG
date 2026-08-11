"""Static preflight for Wikipedia experiment cases."""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from tkg.wikipedia.snapshot import page_version_key


def validate_chain_route(case: dict, navigation: dict) -> dict | None:
    """Prove the declared relation path exists and is shortest in the arena."""
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
    shortest = navigation["distances"].get(route[0])
    if shortest is None:
        raise ValueError(f"{case['id']}: reasoning-chain anchor is absent from arena")
    if shortest != route_distance:
        raise ValueError(
            f"{case['id']}: declared chain is not shortest; chain={route_distance}, "
            f"arena minimum={shortest}"
        )
    return {
        "start_title": case["start_title"], "start_key": route[0],
        "distance": shortest, "route_keys": route, "edge_kinds": edge_kinds,
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
        chain = case.get("reasoning_chain")
        if not isinstance(chain, list) or len(chain) < 2:
            errors.append(f"{prefix}: reasoning_chain must contain at least two hops")
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
        cutoff = case.get("knowledge_cutoff")
        if not isinstance(cutoff, dict) or not cutoff.get("cutoff_date"):
            errors.append(f"{prefix}: multi-hop case missing knowledge_cutoff")
        required_dates = case.get("required_snapshot_dates")
        if not isinstance(required_dates, list) or len(required_dates) < 2:
            errors.append(f"{prefix}: multi-hop case needs at least two snapshot dates")
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
