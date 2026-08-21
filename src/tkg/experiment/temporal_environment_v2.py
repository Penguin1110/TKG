"""Paginated environment and explicit solver funnel for open-world eval v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from tkg.experiment.contracts import PageSnapshot
from tkg.wikipedia.backend import normalize_title


ENVIRONMENT_SCHEMA_V2 = "temporal-wikipedia-environment-v2"
ACTION_FUNNEL_SCHEMA_V2 = "temporal-solver-action-funnel-v2"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _fold(value: str) -> str:
    return " ".join(value.casefold().split())


@dataclass(frozen=True)
class EnvironmentActionV2:
    kind: str
    params: dict[str, Any]
    label: str
    environment_order: int | None = None

    @property
    def action_id(self) -> str:
        return f"{self.kind.casefold()}:{_canonical_hash([self.kind, self.params])[:16]}"

    def to_dict(self) -> dict[str, Any]:
        return {"action_id": self.action_id, **asdict(self)}


@dataclass(frozen=True)
class SolverPolicyV2:
    link_page_size: int = 50
    revision_page_size: int = 50
    max_environment_queries_per_state: int = 4
    dense_action_limit: int = 30
    max_expansions: int = 40
    max_actions_per_state: int = 4

    def __post_init__(self) -> None:
        for field_name in (
            "link_page_size", "revision_page_size",
            "max_environment_queries_per_state", "dense_action_limit",
            "max_expansions", "max_actions_per_state",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.link_page_size > 500 or self.revision_page_size > 500:
            raise ValueError("environment page sizes cannot exceed 500")


@dataclass(frozen=True)
class EnvironmentActionPageV2:
    action_kind: str
    cursor: str | None
    next_cursor: str | None
    page_size: int
    items: tuple[EnvironmentActionV2, ...]
    full_count: int | None
    full_sha256: str | None
    source_node: tuple[str, int]
    time_window: tuple[str, str] | None = None
    schema_version: str = ENVIRONMENT_SCHEMA_V2

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["items"] = [row.to_dict() for row in self.items]
        result["source_node"] = list(self.source_node)
        return result


class RevisionMetadataBackend(Protocol):
    def fetch_revision(self, revision_id: int) -> PageSnapshot:
        ...

    def list_revision_metadata_page(
        self, title: str, from_date: str, to_date: str, *,
        cursor: str | None = None, page_size: int = 50,
    ) -> dict:
        ...


class TemporalWikipediaEnvironmentV2:
    """Complete legal graph access; solver limits never enter this layer."""

    def __init__(self, backend: RevisionMetadataBackend):
        self.backend = backend

    @staticmethod
    def _unique_links(snapshot: PageSnapshot) -> list[tuple[str, str]]:
        links: dict[str, tuple[str, str]] = {}
        for link in snapshot.links:
            target = normalize_title(link.target)
            links.setdefault(_fold(target), (target, link.anchor))
        return list(links.values())

    def list_links(
        self, snapshot: PageSnapshot, *, cursor: str | None = None,
        page_size: int = 50,
    ) -> EnvironmentActionPageV2:
        if isinstance(page_size, bool) or page_size <= 0:
            raise ValueError("page_size must be positive")
        try:
            offset = int(cursor or "0")
        except ValueError as exc:
            raise ValueError("link cursor must be a non-negative integer") from exc
        if offset < 0:
            raise ValueError("link cursor must be a non-negative integer")
        links = self._unique_links(snapshot)
        all_actions = [
            EnvironmentActionV2(
                kind="FOLLOW_LINK",
                params={"page_title": target},
                label=f'Follow hyperlink "{anchor}" to {target}',
                environment_order=index,
            )
            for index, (target, anchor) in enumerate(links)
        ]
        selected = tuple(all_actions[offset:offset + page_size])
        next_offset = offset + len(selected)
        next_cursor = str(next_offset) if next_offset < len(all_actions) else None
        full_payload = [row.to_dict() for row in all_actions]
        return EnvironmentActionPageV2(
            action_kind="FOLLOW_LINK",
            cursor=cursor,
            next_cursor=next_cursor,
            page_size=page_size,
            items=selected,
            full_count=len(all_actions),
            full_sha256=_canonical_hash(full_payload),
            source_node=(snapshot.title, snapshot.revision_id),
        )

    def list_revisions(
        self, snapshot: PageSnapshot, *, from_date: str, to_date: str,
        cursor: str | None = None, page_size: int = 50,
    ) -> EnvironmentActionPageV2:
        raw = self.backend.list_revision_metadata_page(
            snapshot.title, from_date, to_date, cursor=cursor, page_size=page_size,
        )
        title = str(raw.get("title") or snapshot.title)
        if _fold(title) != _fold(snapshot.title):
            raise ValueError("revision pagination changed page title")
        items = tuple(
            EnvironmentActionV2(
                kind="SWITCH_SNAPSHOT",
                params={
                    "revision_id": int(row["revision_id"]),
                    "revision_timestamp": str(row["timestamp"]),
                },
                label=(
                    f"Switch {snapshot.title} to revision {int(row['revision_id'])} "
                    f"({str(row['timestamp'])})"
                ),
                environment_order=None,
            )
            for row in raw.get("revisions", [])
            if int(row["revision_id"]) != snapshot.revision_id
        )
        return EnvironmentActionPageV2(
            action_kind="SWITCH_SNAPSHOT",
            cursor=cursor,
            next_cursor=raw.get("next_cursor"),
            page_size=page_size,
            items=items,
            full_count=None,
            full_sha256=None,
            source_node=(snapshot.title, snapshot.revision_id),
            time_window=(from_date, to_date),
        )

    def enumerate_revisions(
        self, snapshot: PageSnapshot, *, from_date: str, to_date: str,
        page_size: int = 50,
    ) -> tuple[tuple[EnvironmentActionV2, ...], dict[str, Any]]:
        actions: list[EnvironmentActionV2] = []
        cursor: str | None = None
        pages = 0
        seen_cursors: set[str | None] = set()
        while cursor not in seen_cursors:
            seen_cursors.add(cursor)
            page = self.list_revisions(
                snapshot, from_date=from_date, to_date=to_date,
                cursor=cursor, page_size=page_size,
            )
            actions.extend(page.items)
            pages += 1
            cursor = page.next_cursor
            if cursor is None:
                break
        else:
            raise ValueError("revision pagination cursor cycle")
        unique: dict[str, EnvironmentActionV2] = {}
        for action in actions:
            unique.setdefault(action.action_id, action)
        ordered = tuple(unique.values())
        payload = [row.to_dict() for row in ordered]
        return ordered, {
            "complete": True,
            "count": len(ordered),
            "sha256": _canonical_hash(payload),
            "pages": pages,
            "page_size": page_size,
        }

    def verify_action(self, snapshot: PageSnapshot, action: EnvironmentActionV2) -> bool:
        if action.kind == "FOLLOW_LINK":
            requested = _fold(normalize_title(str(action.params.get("page_title", ""))))
            return requested in {
                _fold(normalize_title(link.target)) for link in snapshot.links
            }
        if action.kind == "SWITCH_SNAPSHOT":
            revision_id = int(action.params.get("revision_id", 0))
            page = self.backend.fetch_revision(revision_id)
            return _fold(page.title) == _fold(snapshot.title)
        return False


def initial_environment_queries_v2(
    *, link_page_size: int, revision_page_size: int,
    from_date: str, to_date: str,
) -> tuple[EnvironmentActionV2, EnvironmentActionV2]:
    """Shared forced-state/end-to-end entry into the same environment builder."""
    return (
        EnvironmentActionV2(
            kind="LIST_LINKS",
            params={"cursor": None, "page_size": link_page_size},
            label="List the first rendered-document hyperlink page",
        ),
        EnvironmentActionV2(
            kind="LIST_REVISIONS",
            params={
                "cursor": None,
                "page_size": revision_page_size,
                "time_window": [from_date, to_date],
            },
            label="List the first revision-metadata page",
        ),
    )


def execute_environment_query_v2(
    environment: TemporalWikipediaEnvironmentV2,
    snapshot: PageSnapshot,
    query: EnvironmentActionV2,
) -> EnvironmentActionPageV2:
    """The sole v2 query dispatcher used by every solver/diagnostic mode."""
    if query.kind == "LIST_LINKS":
        return environment.list_links(
            snapshot,
            cursor=query.params.get("cursor"),
            page_size=int(query.params["page_size"]),
        )
    if query.kind == "LIST_REVISIONS":
        window = query.params.get("time_window")
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError("LIST_REVISIONS requires a two-date time_window")
        return environment.list_revisions(
            snapshot,
            from_date=str(window[0]),
            to_date=str(window[1]),
            cursor=query.params.get("cursor"),
            page_size=int(query.params["page_size"]),
        )
    raise ValueError(f"not an environment pagination query: {query.kind}")


def continuation_action(page: EnvironmentActionPageV2) -> EnvironmentActionV2 | None:
    if page.next_cursor is None:
        return None
    if page.action_kind == "FOLLOW_LINK":
        return EnvironmentActionV2(
            kind="LIST_LINKS",
            params={"cursor": page.next_cursor, "page_size": page.page_size},
            label=f"List next hyperlink page from cursor {page.next_cursor}",
        )
    if page.action_kind == "SWITCH_SNAPSHOT":
        if page.time_window is None:
            raise ValueError("revision page lacks time window")
        return EnvironmentActionV2(
            kind="LIST_REVISIONS",
            params={
                "cursor": page.next_cursor,
                "page_size": page.page_size,
                "time_window": list(page.time_window),
            },
            label=f"List next revision page from cursor {page.next_cursor}",
        )
    raise ValueError(f"unsupported paginated action kind {page.action_kind}")


def compact_solver_actions_v2(
    retrieved_pages: list[EnvironmentActionPageV2], *, dense_limit: int = 30,
) -> tuple[list[EnvironmentActionV2], dict[str, Any]]:
    """Gold-free policy over explicitly retrieved environment pages."""
    if dense_limit <= 0:
        raise ValueError("dense_limit must be positive")
    transitions: list[EnvironmentActionV2] = []
    pagination: list[EnvironmentActionV2] = []
    seen: set[str] = set()
    for page in retrieved_pages:
        for action in page.items:
            if action.action_id not in seen:
                transitions.append(action)
                seen.add(action.action_id)
        followup = continuation_action(page)
        if followup is not None and followup.action_id not in seen:
            pagination.append(followup)
            seen.add(followup.action_id)
    reserved = pagination[:dense_limit]
    kept = transitions[:max(0, dense_limit - len(reserved))]
    compacted = [*kept, *reserved]
    return compacted, {
        "schema_version": ACTION_FUNNEL_SCHEMA_V2,
        "policy": "retrieval_order_first_n_reserving_pagination_v2",
        "dense_limit": dense_limit,
        "solver_retrieved_actions": [row.to_dict() for row in [*transitions, *pagination]],
        "compacted_ranker_actions": [row.to_dict() for row in compacted],
    }


def action_funnel_record_v2(
    *, environment_actions: list[EnvironmentActionV2] | None,
    retrieved_pages: list[EnvironmentActionPageV2], dense_limit: int,
    ranker_scores: dict[str, float] | None,
    expanded_action_ids: list[str], artifact_reference: str | None = None,
    environment_manifest: dict[str, Any] | None = None,
    parent_node: tuple[str, int] | None = None,
) -> dict[str, Any]:
    compacted, partial = compact_solver_actions_v2(
        retrieved_pages, dense_limit=dense_limit,
    )
    environment_payload = [row.to_dict() for row in (environment_actions or [])]
    manifest = environment_manifest or {
        "count": len(environment_payload),
        "sha256": _canonical_hash(environment_payload),
        "complete": True,
    }
    if environment_actions is None and not (
        manifest.get("complete") is True
        and isinstance(manifest.get("count"), int)
        and isinstance(manifest.get("sha256"), str)
        and artifact_reference
    ):
        raise ValueError(
            "externalized environment legal set needs complete count/hash/artifact"
        )
    compacted_ids = {row.action_id for row in compacted}
    scores = ranker_scores or {}
    ranker_valid = set(scores) == compacted_ids and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in scores.values()
    )
    return {
        **partial,
        "parent_page": parent_node[0] if parent_node else None,
        "parent_revision_id": parent_node[1] if parent_node else None,
        "environment_legal_actions": environment_payload,
        "environment_legal_action_count": manifest["count"],
        "environment_legal_actions_sha256": manifest["sha256"],
        "environment_legal_actions_inline": environment_actions is not None,
        "environment_legal_actions_artifact_reference": artifact_reference,
        "ranker_scores": scores if ranker_valid else {},
        "ranker_contract_valid": ranker_valid,
        "expanded_actions": list(expanded_action_ids),
    }
