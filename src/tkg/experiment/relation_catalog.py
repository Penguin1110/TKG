"""Auditable Wikidata relation whitelist for diverse person follow-up hops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationSpec:
    property_id: str
    family: str
    relation: str
    relative_clause: str
    answer_kind: str
    selector: str
    target_kind: str
    guessability: str


PERSON_RELATIONS: tuple[RelationSpec, ...] = (
    RelationSpec(
        "P19", "geography", "place of birth", "the place of birth of {source}",
        "What", "single", "place", "easy",
    ),
    RelationSpec(
        "P27", "geography", "country of citizenship",
        "the country of citizenship of {source}", "What", "single", "country", "easy",
    ),
    RelationSpec(
        "P69", "education", "educational institution attended",
        "the educational institution attended by {source}",
        "What", "single", "organization", "hard",
    ),
    RelationSpec(
        "P108", "career", "current employer", "the current employer of {source}",
        "What", "current", "organization", "hard",
    ),
    RelationSpec(
        "P39", "career", "current position held",
        "the current position held by {source}", "What", "current", "position", "hard",
    ),
    RelationSpec(
        "P54", "sports", "most recent sports team represented as a player",
        "the most recent sports team represented as a player by {source}",
        "What", "latest_end", "organization", "hard",
    ),
    RelationSpec(
        "P102", "politics", "current political party",
        "the current political party of {source}", "What", "current", "organization", "hard",
    ),
    RelationSpec(
        "P26", "family", "current spouse", "the current spouse of {source}",
        "Who", "current", "person", "medium",
    ),
    RelationSpec(
        "P166", "awards", "most recent award received",
        "the most recent award received by {source}", "What", "latest_point", "award", "hard",
    ),
    RelationSpec(
        "P463", "affiliation", "current organization membership",
        "the current organization of which {source} is a member",
        "What", "current", "organization", "hard",
    ),
    RelationSpec(
        "P103", "language", "native language", "the native language of {source}",
        "What", "single", "language", "easy",
    ),
    RelationSpec(
        "P22", "family", "father", "the father of {source}",
        "Who", "single", "person", "medium",
    ),
    RelationSpec(
        "P25", "family", "mother", "the mother of {source}",
        "Who", "single", "person", "medium",
    ),
    RelationSpec(
        "P413", "sports", "playing position",
        "the playing position of {source}", "What", "single", "position", "medium",
    ),
)


RELATION_BY_PROPERTY = {spec.property_id: spec for spec in PERSON_RELATIONS}
