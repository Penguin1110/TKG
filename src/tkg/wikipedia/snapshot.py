"""Build a reproducible Wikipedia hyperlink snapshot around configured pivots."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from tkg.wikipedia.backend import WikipediaError, WikipediaPageBackend


def configured_pivot(case: dict, backend: WikipediaPageBackend) -> str | None:
    """Resolve the single pivot used by the temporal exploration task."""
    if case.get("wikipedia_title"):
        return case["wikipedia_title"]
    if case.get("pivot_qid"):
        return backend.resolve_qid_title(case["pivot_qid"])
    return None


def configured_target(case: dict, arm: str, backend: WikipediaPageBackend) -> str | None:
    """Compatibility resolver for the archived conflict/control experiment."""
    if arm == "conflict":
        return configured_pivot(case, backend)
    if case.get("control_wikipedia_title"):
        return case["control_wikipedia_title"]
    if case.get("control_pivot_qid"):
        return backend.resolve_qid_title(case["control_pivot_qid"])
    # Default to the same page with stable control questions.  This gives the
    # closest structural/content-length match; the answerability gate still
    # rejects any control question that is not supported by the viewed pages.
    return configured_pivot(case, backend)


def outgoing_bfs(
    backend: WikipediaPageBackend,
    target: str,
    max_depth: int,
    *,
    as_of: str | None,
    branch_cap: int,
) -> dict[int, list[str]]:
    """Cache the exact outgoing hyperlink graph around a pivot revision."""
    pivot = backend.fetch_page(target, as_of=as_of)
    frontier = {0: [pivot.title]}
    seen = {pivot.title.casefold()}
    current = [pivot.title]
    for depth in range(1, max_depth + 1):
        next_titles = []
        for title in current:
            page = backend.fetch_page(title, as_of=as_of)
            for link in page.links[:branch_cap]:
                if link.target.casefold() in seen:
                    continue
                try:
                    linked = backend.fetch_page(link.target, as_of=as_of)
                except WikipediaError:
                    continue
                seen.add(linked.title.casefold())
                next_titles.append(linked.title)
        if not next_titles:
            break
        frontier[depth] = next_titles
        current = next_titles
    return frontier


def page_version_key(title: str, revision_id: int) -> str:
    """Stable identity for one graph node: canonical page title + revision."""
    return f"{int(revision_id)}:{' '.join(title.split()).casefold()}"


def temporal_reverse_bfs(
    backend: WikipediaPageBackend,
    pivot_title: str,
    target_as_of: str | None,
    allowed_as_of: list[str | None],
    max_depth: int,
    *,
    branch_cap: int = 25,
) -> dict:
    """Compute exact minimum navigation distance to a target page revision.

    Hyperlink predecessors come from verified backlinks in the same snapshot.
    Temporal predecessors are other allowed revisions of the same page.  The
    raw Wikipedia graph may contain cycles; first discovery in this reverse BFS
    assigns the minimum directed distance to the target revision.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    if branch_cap <= 0:
        raise ValueError("branch_cap must be > 0")
    dates: list[str | None] = []
    for value in allowed_as_of:
        if value not in dates:
            dates.append(value)
    if not dates:
        raise ValueError("allowed_as_of must not be empty")
    if target_as_of not in dates:
        raise ValueError("target_as_of must be one of allowed_as_of")

    states: dict[str, dict] = {}
    distances: dict[str, int] = {}
    shortest_next: dict[str, str] = {}
    shortest_edge_kind: dict[str, str] = {}
    frontier_keys: dict[int, list[str]] = {0: []}

    def add_state(
        page,
        as_of: str | None,
        distance: int,
        *,
        next_key: str | None = None,
        edge_kind: str | None = None,
    ) -> tuple[str, bool]:
        key = page_version_key(page.title, page.revision_id)
        if key in states:
            if as_of not in states[key]["as_of_tokens"]:
                states[key]["as_of_tokens"].append(as_of)
            return key, False
        states[key] = {
            "key": key,
            "title": page.title,
            "revision_id": page.revision_id,
            "timestamp": page.timestamp,
            "as_of": as_of,
            "as_of_tokens": [as_of],
            "distance_to_pivot": distance,
        }
        distances[key] = distance
        frontier_keys.setdefault(distance, []).append(key)
        if next_key is not None and edge_kind is not None:
            shortest_next[key] = next_key
            shortest_edge_kind[key] = edge_kind
        return key, True

    target_page = backend.fetch_page(pivot_title, as_of=target_as_of)
    target_key, _ = add_state(target_page, target_as_of, 0)

    for depth in range(max_depth):
        current = list(frontier_keys.get(depth, []))
        if not current:
            break
        for current_key in current:
            state = states[current_key]
            state_as_of = state["as_of"]

            for other_as_of in dates:
                if other_as_of == state_as_of:
                    continue
                try:
                    predecessor = backend.fetch_page(state["title"], as_of=other_as_of)
                    # A renamed/redirected page is a temporal predecessor only
                    # when switching its actual title lands on this revision.
                    landing = backend.fetch_page(predecessor.title, as_of=state_as_of)
                except WikipediaError:
                    continue
                if page_version_key(landing.title, landing.revision_id) != current_key:
                    continue
                predecessor_key = page_version_key(
                    predecessor.title, predecessor.revision_id
                )
                if predecessor_key == current_key:
                    if other_as_of not in state["as_of_tokens"]:
                        state["as_of_tokens"].append(other_as_of)
                    continue
                add_state(
                    predecessor, other_as_of, depth + 1,
                    next_key=current_key, edge_kind="temporal",
                )

            # The same revision may be selected by several allowed dates. Its
            # content is identical, but predecessor page revisions can differ,
            # so verify backlinks independently for every such date token.
            for link_as_of in list(state["as_of_tokens"]):
                sources = backend.find_backlinks(
                    state["title"], as_of=link_as_of, max_results=branch_cap
                )
                for source_title in sorted(sources, key=str.casefold):
                    try:
                        source = backend.fetch_page(source_title, as_of=link_as_of)
                    except WikipediaError:
                        continue
                    add_state(
                        source, link_as_of, depth + 1,
                        next_key=current_key, edge_kind="hyperlink",
                    )

    # Rebuild every real edge among the discovered page-version states, then
    # recompute distances.  This makes the result exact for the constructed
    # bounded arena even when discovery used a backlink branch cap.
    state_for_title_token: dict[tuple[str, str | None], str] = {}
    for key, state in states.items():
        for token in state["as_of_tokens"]:
            state_for_title_token[(state["title"].casefold(), token)] = key
    arena_edges: set[tuple[str, str, str]] = set()
    by_title: dict[str, list[str]] = {}
    for key, state in states.items():
        by_title.setdefault(state["title"].casefold(), []).append(key)
        page = backend.fetch_page(state["title"], as_of=state["as_of"])
        for token in state["as_of_tokens"]:
            for link in page.links:
                destination = state_for_title_token.get(
                    (link.target.casefold(), token)
                )
                if destination is not None:
                    arena_edges.add((key, destination, "hyperlink"))
    for keys in by_title.values():
        for source_key in keys:
            for destination_key in keys:
                if source_key != destination_key:
                    arena_edges.add((source_key, destination_key, "temporal"))

    reverse_edges: dict[str, list[tuple[str, str]]] = {}
    for source_key, destination_key, edge_kind in arena_edges:
        reverse_edges.setdefault(destination_key, []).append((source_key, edge_kind))
    distances = {target_key: 0}
    shortest_next = {}
    shortest_edge_kind = {}
    frontier_keys = {0: [target_key]}
    for depth in range(max_depth):
        next_keys: list[str] = []
        for destination_key in frontier_keys.get(depth, []):
            for source_key, edge_kind in sorted(
                reverse_edges.get(destination_key, []),
                key=lambda item: (states[item[0]]["title"].casefold(), item[1]),
            ):
                if source_key in distances:
                    continue
                distances[source_key] = depth + 1
                shortest_next[source_key] = destination_key
                shortest_edge_kind[source_key] = edge_kind
                next_keys.append(source_key)
        if not next_keys:
            break
        frontier_keys[depth + 1] = next_keys
    # Discovery only adds states with a verified route, but retain this filter
    # defensively if a redirect changed identity while the arena was assembled.
    states = {key: state for key, state in states.items() if key in distances}
    for key, state in states.items():
        state["distance_to_pivot"] = distances[key]
    arena_edges = {
        edge for edge in arena_edges if edge[0] in states and edge[1] in states
    }

    title_min_distance: dict[str, int] = {}
    title_display: dict[str, str] = {}
    for state in states.values():
        folded = state["title"].casefold()
        title_display.setdefault(folded, state["title"])
        title_min_distance[folded] = min(
            title_min_distance.get(folded, max_depth + 1),
            state["distance_to_pivot"],
        )
    candidate_titles: dict[str, list[str]] = {}
    for folded, distance in title_min_distance.items():
        candidate_titles.setdefault(str(distance), []).append(title_display[folded])
    for values in candidate_titles.values():
        values.sort(key=str.casefold)
    allowed_titles_by_snapshot: dict[str, list[str]] = {}
    for state in states.values():
        for token in state["as_of_tokens"]:
            label = token or "CURRENT"
            allowed_titles_by_snapshot.setdefault(label, []).append(state["title"])
    for label, values in allowed_titles_by_snapshot.items():
        allowed_titles_by_snapshot[label] = sorted(set(values), key=str.casefold)

    return {
        "target_key": target_key,
        "target": states[target_key],
        "allowed_as_of": dates,
        "max_depth": max_depth,
        "branch_cap": branch_cap,
        "states": states,
        "distances": distances,
        "frontiers": {
            str(depth): [states[key] for key in keys]
            for depth, keys in sorted(frontier_keys.items())
        },
        "candidate_titles_by_distance": candidate_titles,
        "shortest_next": shortest_next,
        "shortest_edge_kind": shortest_edge_kind,
        "arena_edges": [
            {"source": source, "target": target, "kind": kind}
            for source, target, kind in sorted(arena_edges)
        ],
        "allowed_version_keys": sorted(states),
        "allowed_titles_by_snapshot": allowed_titles_by_snapshot,
        "coverage_note": (
            "Historical backlink candidates come from current MediaWiki backlinks and are "
            "verified against each selected revision; removed historical-only links may be missed."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build a temporal reverse-BFS snapshot toward each target pivot revision"
    )
    parser.add_argument("--cases", default="generated_cases.json")
    parser.add_argument("--case-ids")
    parser.add_argument("--as-of", help="YYYY-MM-DD or ISO timestamp; defaults to each case/current")
    parser.add_argument(
        "--snapshot-dates",
        help="comma-separated dates to cache for temporal switching; CURRENT is allowed",
    )
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--branch-cap", type=int, default=25)
    parser.add_argument("--cache-path", default="wikipedia_snapshot.db")
    parser.add_argument("--manifest", default="wikipedia_snapshot_manifest.json")
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()

    if args.max_depth < 0:
        parser.error("--max-depth must be >= 0")
    if args.branch_cap <= 0:
        parser.error("--branch-cap must be > 0")

    with open(args.cases, "r", encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]
    if args.case_ids:
        wanted = set(args.case_ids.split(","))
        cases = [case for case in cases if case["id"] in wanted]
    backend = WikipediaPageBackend(cache_path=args.cache_path, lang=args.lang)
    records = []
    try:
        for case in cases:
            target = configured_pivot(case, backend)
            if not target:
                print(f"[skip] {case['id']}: no configured Wikipedia target")
                continue
            raw_dates = [
                item.strip() for item in (args.snapshot_dates or "").split(",") if item.strip()
            ]
            if args.as_of:
                raw_dates = [args.as_of]
            if not raw_dates:
                generation = case.get("_generation", {})
                raw_dates = [value for value in (
                    case.get("wikipedia_before")
                    or generation.get("before", {}).get("requested_as_of"),
                    case.get("wikipedia_as_of")
                    or generation.get("after", {}).get("requested_as_of"),
                ) if value]
            dates = list(dict.fromkeys(
                None if str(value).upper() == "CURRENT" else str(value)
                for value in raw_dates
            ))
            if not dates:
                dates = [None]
            generation = case.get("_generation", {})
            target_as_of = (
                case.get("wikipedia_as_of")
                or generation.get("after", {}).get("requested_as_of")
                or dates[-1]
            )
            print(
                f"[snapshot] {case['id']}: target={target!r}, "
                f"target_as_of={target_as_of!r}, dates={dates!r}"
            )
            try:
                navigation = temporal_reverse_bfs(
                    backend, target, target_as_of, dates, args.max_depth,
                    branch_cap=args.branch_cap,
                )
            except (WikipediaError, ValueError) as exc:
                print(f"[error] {case['id']}: {exc}")
                continue
            records.append({
                "case_id": case["id"],
                "target": navigation["target"],
                "allowed_as_of": navigation["allowed_as_of"],
                "frontier_sizes": {
                    depth: len(values) for depth, values in navigation["frontiers"].items()
                },
                "frontiers": navigation["frontiers"],
                "candidate_titles_by_distance": navigation[
                    "candidate_titles_by_distance"
                ],
                "shortest_next": navigation["shortest_next"],
                "shortest_edge_kind": navigation["shortest_edge_kind"],
                "arena_edges": navigation["arena_edges"],
                "state_count": len(navigation["states"]),
                "coverage_note": navigation["coverage_note"],
            })
    finally:
        backend.close()

    manifest = {
        "schema_version": "wikipedia-temporal-shortest-path-v3",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "cache_path": args.cache_path, "lang": args.lang,
        "max_depth": args.max_depth, "branch_cap": args.branch_cap,
        "graph_policy": (
            "reverse BFS over verified same-snapshot hyperlinks plus repeatable temporal switches; "
            "first discovery is minimum directed distance to the target page revision"
        ),
        "targets": records,
    }
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {args.manifest} with {len(records)} target snapshots")


if __name__ == "__main__":
    main()
