"""Generate diverse, auditable person-attribute tails for temporal seeds."""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from tkg.experiment.multihop_generation import MultiHopSeed, validate_chain
from tkg.experiment.question_generation import _fold, _plain_text
from tkg.experiment.relation_catalog import PERSON_RELATIONS, RelationSpec
from tkg.wikipedia.backend import WikipediaError
from tkg.wikipedia.pageviews import PopularitySnapshot


@dataclass(frozen=True)
class SelectedClaim:
    property_id: str
    target_qid: str
    selector: str
    qualifier_dates: dict[str, str]


def _snak_date(snak: dict[str, Any]) -> date | None:
    value = snak.get("datavalue", {}).get("value", {})
    raw = value.get("time") if isinstance(value, dict) else None
    if not isinstance(raw, str):
        return None
    match = re.match(r"^\+?(\d{4})-(\d{2})-(\d{2})T", raw)
    if not match:
        return None
    try:
        return date.fromisoformat("-".join(match.groups()))
    except ValueError:
        return None


def _qualifier_date(claim: dict[str, Any], property_id: str) -> date | None:
    values = claim.get("qualifiers", {}).get(property_id, [])
    parsed = [_snak_date(value) for value in values if isinstance(value, dict)]
    dates = [value for value in parsed if value is not None]
    return max(dates) if dates else None


def _claim_target_qid(claim: dict[str, Any]) -> str | None:
    value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
    qid = value.get("id") if isinstance(value, dict) else None
    return qid if isinstance(qid, str) and re.fullmatch(r"Q\d+", qid) else None


def _eligible_claims(
    claims: list[dict[str, Any]], as_of: date, *, active_only: bool
) -> list[tuple[dict[str, Any], str]]:
    eligible = []
    for claim in claims:
        if claim.get("rank") == "deprecated":
            continue
        qid = _claim_target_qid(claim)
        if qid is None:
            continue
        start = _qualifier_date(claim, "P580")
        end = _qualifier_date(claim, "P582")
        if active_only and start is None and end is None and claim.get("rank") != "preferred":
            continue
        if start is not None and start > as_of:
            continue
        if active_only and end is not None and end < as_of:
            continue
        eligible.append((claim, qid))
    preferred = [value for value in eligible if value[0].get("rank") == "preferred"]
    return preferred or eligible


def select_claim(
    claims: list[dict[str, Any]], spec: RelationSpec, as_of: str
) -> SelectedClaim | None:
    """Apply a declared cardinality/time selector; ambiguous properties fail closed."""
    selected_date = date.fromisoformat(as_of)
    eligible = _eligible_claims(
        claims, selected_date, active_only=spec.selector == "current"
    )
    if not eligible:
        return None
    qualifier_property = None
    if spec.selector == "single":
        candidates = eligible
    elif spec.selector == "current":
        candidates = eligible
    elif spec.selector == "latest_end":
        qualifier_property = "P582"
        dated: list[tuple[dict[str, Any], str, date]] = []
        for claim, qid in eligible:
            when = _qualifier_date(claim, qualifier_property)
            if when is not None:
                dated.append((claim, qid, when))
        if not dated:
            return None
        latest = max(value[2] for value in dated)
        candidates = [(claim, qid) for claim, qid, when in dated if when == latest]
    elif spec.selector == "latest_point":
        qualifier_property = "P585"
        dated = []
        for claim, qid in eligible:
            when = _qualifier_date(claim, qualifier_property)
            if when is not None and when <= selected_date:
                dated.append((claim, qid, when))
        if not dated:
            return None
        latest = max(value[2] for value in dated)
        candidates = [(claim, qid) for claim, qid, when in dated if when == latest]
    else:
        raise ValueError(f"unknown relation selector: {spec.selector}")
    by_qid = {qid: claim for claim, qid in candidates}
    if len(by_qid) != 1:
        return None
    qid, claim = next(iter(by_qid.items()))
    qualifiers = {}
    for property_id in ("P580", "P582", "P585"):
        value = _qualifier_date(claim, property_id)
        if value is not None:
            qualifiers[property_id] = value.isoformat()
    if qualifier_property and qualifier_property not in qualifiers:
        return None
    return SelectedClaim(
        property_id=spec.property_id, target_qid=qid,
        selector=spec.selector, qualifier_dates=qualifiers,
    )


def _target_title(entity: dict[str, Any], lang: str) -> str | None:
    sitelink = entity.get("sitelinks", {}).get(f"{lang}wiki", {})
    title = sitelink.get("title") if isinstance(sitelink, dict) else None
    return str(title).strip() if title else None


def _evidence_window(page, target_title: str) -> tuple[str, str] | None:
    aliases = [target_title]
    aliases.extend(
        link.anchor for link in page.links
        if link.target.casefold() == target_title.casefold()
    )
    plain = _plain_text(page.content)
    present = [alias for alias in aliases if alias and _fold(alias) in _fold(plain)]
    if not present:
        return None
    alias = max(present, key=len)
    position = plain.casefold().find(alias.casefold())
    start = max(0, position - 100)
    end = min(len(plain), position + len(alias) + 180)
    return plain[start:end].strip(), alias


def generate_attribute_tail_variants(
    seed_value: dict[str, Any],
    *,
    backend,
    popularity: PopularitySnapshot | None = None,
    relations: tuple[RelationSpec, ...] = PERSON_RELATIONS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Append one verified same-snapshot attribute hop to a temporal seed."""
    seed = MultiHopSeed.from_dict(seed_value)
    if len(seed.hops) >= 6:
        return [], [{"status": "reject", "reason": "seed already has maximum hops"}]
    prefix_errors, validated_prefix = validate_chain(seed, backend)
    if prefix_errors:
        return [], [{
            "status": "reject",
            "reason": "base temporal spine failed validation",
            "errors": prefix_errors,
        }]
    source_title = seed.hops[-1].target_title
    source_path = {seed.hops[0].source_title.casefold()}
    source_path.update(hop.target_title.casefold() for hop in seed.hops)
    packets: list[dict[str, Any]] = []
    try:
        source_qid = backend.resolve_title_qid(source_title)
        source_entity = backend.get_wikidata_entities([source_qid]).get(source_qid)
        source_page = backend.fetch_page(source_title, as_of=seed.target_as_of)
    except (WikipediaError, ValueError) as exc:
        return [], [{"status": "reject", "reason": f"source lookup failed: {exc}"}]
    if not source_entity:
        return [], [{"status": "reject", "reason": "missing source Wikidata entity"}]

    selections: list[tuple[RelationSpec, SelectedClaim]] = []
    claims_by_property = source_entity.get("claims", {})
    for spec in relations:
        raw_claims = claims_by_property.get(spec.property_id, [])
        if not isinstance(raw_claims, list):
            continue
        selected = select_claim(raw_claims, spec, seed.target_as_of)
        if selected is not None:
            selections.append((spec, selected))
    target_qids = [selected.target_qid for _, selected in selections]
    target_entities: dict[str, dict] = {}
    try:
        for start in range(0, len(target_qids), 50):
            target_entities.update(
                backend.get_wikidata_entities(target_qids[start:start + 50])
            )
    except (WikipediaError, ValueError) as exc:
        return [], [{
            "status": "infrastructure_error",
            "source_title": source_title,
            "source_qid": source_qid,
            "reason": f"target Wikidata batch failed: {exc}",
        }]

    variants = []
    lang = str(getattr(backend, "lang", "en"))
    source_views = popularity.views(source_title) if popularity else 0
    for spec, selected in selections:
        title = _target_title(target_entities.get(selected.target_qid, {}), lang)
        packet = {
            "source_title": source_title,
            "source_qid": source_qid,
            "property_id": spec.property_id,
            "relation_family": spec.family,
            "selector": spec.selector,
            "target_qid": selected.target_qid,
            "target_title": title,
            "guessability": spec.guessability,
        }
        if not title:
            packets.append({**packet, "status": "reject", "reason": "no Wikipedia sitelink"})
            continue
        if title.casefold() in source_path:
            packets.append({**packet, "status": "reject", "reason": "entity cycle"})
            continue
        evidence = _evidence_window(source_page, title)
        if evidence is None or not any(
            link.target.casefold() == title.casefold() for link in source_page.links
        ):
            packets.append({
                **packet, "status": "reject",
                "reason": "target revision lacks visible target hyperlink/evidence",
            })
            continue
        try:
            backend.fetch_page(title, as_of=seed.target_as_of)
        except WikipediaError as exc:
            packets.append({**packet, "status": "reject", "reason": str(exc)})
            continue
        excerpt, alias = evidence
        target_views = popularity.views(title) if popularity else 0
        variant = copy.deepcopy(seed_value)
        variant["id"] = (
            f"{seed.id}_{spec.property_id.casefold()}_"
            + re.sub(r"[^a-z0-9]+", "_", title.casefold()).strip("_")
        )[:220]
        variant["answer_kind"] = spec.answer_kind
        variant.setdefault("hops", []).append({
            "source_title": source_title,
            "target_title": title,
            "relation": spec.relation,
            "relative_clause": spec.relative_clause,
            "as_of": seed.target_as_of,
            "evidence": excerpt,
            "target_aliases": [alias],
            "incoming_time_policy": "same_snapshot",
            "property_id": spec.property_id,
            "relation_family": spec.family,
            "structured_evidence": {
                "source": "wikidata_statement",
                "source_qid": source_qid,
                "target_qid": selected.target_qid,
                "property_id": spec.property_id,
                "selector": selected.selector,
                "qualifier_dates": selected.qualifier_dates,
            },
        })
        variant["selection_metadata"] = {
            "relation_family": spec.family,
            "property_id": spec.property_id,
            "source_entity": source_title,
            "target_entity": title,
            "popularity_month": popularity.month if popularity else None,
            "source_pageviews": source_views,
            "target_pageviews": target_views,
            "answer_guessability": spec.guessability,
        }
        try:
            parsed = MultiHopSeed.from_dict(variant)
            errors, _ = validate_chain(
                parsed, backend, validated_prefix=validated_prefix
            )
        except ValueError as exc:
            errors = [str(exc)]
        if errors:
            packets.append({**packet, "status": "reject", "reason": errors})
            continue
        variants.append(variant)
        packets.append({
            **packet, "status": "pass", "source_pageviews": source_views,
            "target_pageviews": target_views,
        })
    variants.sort(key=lambda item: (
        -int(item.get("selection_metadata", {}).get("source_pageviews", 0)),
        -int(item.get("selection_metadata", {}).get("target_pageviews", 0)),
        str(item.get("selection_metadata", {}).get("relation_family", "")),
        str(item.get("id", "")),
    ))
    return variants, packets


def select_diverse_variants(
    variants: list[dict[str, Any]],
    *,
    max_total: int,
    max_per_family: int,
    max_per_source: int = 2,
    max_per_property: int = 1,
) -> list[dict[str, Any]]:
    """Greedy popularity ranking with hard family/source diversity caps."""
    remaining = list(variants)
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    property_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    while remaining and len(selected) < max_total:
        eligible = []
        for value in remaining:
            meta = value.get("selection_metadata", {})
            family = str(meta.get("relation_family", "other"))
            source = str(meta.get("source_entity", ""))
            if family_counts.get(family, 0) >= max_per_family:
                continue
            property_id = str(meta.get("property_id", ""))
            if property_counts.get(property_id, 0) >= max_per_property:
                continue
            if source_counts.get(source, 0) >= max_per_source:
                continue
            views = int(meta.get("source_pageviews", 0)) + int(
                meta.get("target_pageviews", 0)
            ) + int((meta.get("spine_popularity") or {}).get("score", 0))
            diversity_penalty = (
                1 + 2 * family_counts.get(family, 0)
                + property_counts.get(property_id, 0)
                + source_counts.get(source, 0)
            )
            score = math.log1p(max(1, views)) / diversity_penalty
            eligible.append((score, views, str(value.get("id", "")), value))
        if not eligible:
            break
        _, _, _, chosen = max(eligible, key=lambda item: (item[0], item[1], item[2]))
        selected.append(chosen)
        remaining.remove(chosen)
        meta = chosen.get("selection_metadata", {})
        family = str(meta.get("relation_family", "other"))
        property_id = str(meta.get("property_id", ""))
        source = str(meta.get("source_entity", ""))
        family_counts[family] = family_counts.get(family, 0) + 1
        property_counts[property_id] = property_counts.get(property_id, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
    return selected
