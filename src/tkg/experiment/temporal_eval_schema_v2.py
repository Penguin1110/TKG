"""Append-only schemas for open-world temporal graph evaluation v2.

The public inference projection and private evaluation case are deliberately
different types.  A private reference route is a witness of feasibility and a
post-hoc diagnostic only; it is never part of ``PublicTemporalCaseV2``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CASE_SCHEMA_V2 = "open-world-temporal-evaluation-case-v2"
PUBLIC_CASE_SCHEMA_V2 = "open-world-temporal-public-case-v2"
SUBMISSION_SCHEMA_V2 = "structured-temporal-evidence-submission-v2"
WITNESS_SCHEMA_V2 = "temporal-claim-witness-set-v2"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized(value: Any) -> str:
    return " ".join(str(value).casefold().split()).strip(" .?!")


@dataclass(frozen=True)
class PublicTemporalCaseV2:
    case_id: str
    model_id: str
    question: str
    start_page: str
    cutoff_date: str
    target_date: str
    schema_version: str = PUBLIC_CASE_SCHEMA_V2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimWitnessV2:
    page_title: str
    revision_id: int
    evidence_excerpt: str
    evidence_hash: str
    support_type: str = "semantic"
    validation_status: str = "machine_pass_human_review_required"
    semantic_validation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.support_type != "semantic":
            raise ValueError("v2 claim witnesses must use support_type=semantic")
        if not re.fullmatch(r"[0-9a-f]{64}", self.evidence_hash):
            raise ValueError("witness evidence_hash must be SHA-256")
        if self.evidence_hash != hashlib.sha256(
            self.evidence_excerpt.encode("utf-8")
        ).hexdigest():
            raise ValueError("witness evidence_hash does not match excerpt")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RequiredClaimV2:
    claim_id: str
    subject: str
    relation: str
    object: str
    event_time: str | None
    witnesses: tuple[ClaimWitnessV2, ...]
    schema_version: str = WITNESS_SCHEMA_V2

    def __post_init__(self) -> None:
        if not self.claim_id or not self.subject or not self.relation or not self.object:
            raise ValueError("required claim fields must be non-empty")
        if not self.witnesses:
            raise ValueError(f"{self.claim_id}: at least one witness is required")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["witnesses"] = [row.to_dict() for row in self.witnesses]
        return result


@dataclass(frozen=True)
class ReferenceActionV2:
    kind: str
    parent_page: str
    parent_revision_id: int
    page_title: str | None = None
    revision_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrivateReferenceRouteV2:
    route_id: str
    actions: tuple[ReferenceActionV2, ...]
    distance: int
    usage: str = "feasibility_witness_and_posthoc_diagnostics_only"

    def __post_init__(self) -> None:
        if self.distance != len(self.actions):
            raise ValueError("reference distance must equal action count")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["actions"] = [row.to_dict() for row in self.actions]
        return result


@dataclass(frozen=True)
class EvaluationCaseV2:
    case_id: str
    model_id: str
    question: str
    start_page: str
    cutoff_date: str
    target_date: str
    accepted_final_answer_aliases: tuple[str, ...]
    critical_claims: tuple[RequiredClaimV2, ...]
    tail_relation: RequiredClaimV2
    validated_evidence_requirements: dict[str, Any]
    event_time_constraints: dict[str, Any]
    reference_routes: tuple[PrivateReferenceRouteV2, ...] = ()
    source_case_schema: str = ""
    source_case_sha256: str = ""
    schema_version: str = CASE_SCHEMA_V2

    def __post_init__(self) -> None:
        if not self.accepted_final_answer_aliases:
            raise ValueError("accepted final-answer aliases cannot be empty")
        if not self.critical_claims:
            raise ValueError("at least one critical claim is required")
        ids = [claim.claim_id for claim in (*self.critical_claims, self.tail_relation)]
        if len(ids) != len(set(ids)):
            raise ValueError("claim IDs must be unique")

    def public_view(self) -> PublicTemporalCaseV2:
        return PublicTemporalCaseV2(
            case_id=self.case_id,
            model_id=self.model_id,
            question=self.question,
            start_page=self.start_page,
            cutoff_date=self.cutoff_date,
            target_date=self.target_date,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["critical_claims"] = [row.to_dict() for row in self.critical_claims]
        result["tail_relation"] = self.tail_relation.to_dict()
        result["reference_routes"] = [row.to_dict() for row in self.reference_routes]
        return result


@dataclass(frozen=True)
class SubmittedClaimV2:
    subject: str
    relation: str
    object: str
    event_time: str | None
    supporting_evidence_ids: tuple[str, ...]
    claim_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuredSubmissionV2:
    answer: str
    critical_claims: tuple[SubmittedClaimV2, ...]
    tail_claim: SubmittedClaimV2
    schema_version: str = SUBMISSION_SCHEMA_V2

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["critical_claims"] = [row.to_dict() for row in self.critical_claims]
        result["tail_claim"] = self.tail_claim.to_dict()
        return result


def _legacy_reference_route(case: dict[str, Any]) -> PrivateReferenceRouteV2:
    actions: list[ReferenceActionV2] = []
    chain = case.get("reasoning_chain")
    if not isinstance(chain, list):
        chain = []
    for hop in chain:
        if not isinstance(hop, dict):
            continue
        source = str(hop.get("source_title") or "")
        raw_source_revision = hop.get("source_revision_id")
        if raw_source_revision is None:
            raise ValueError("legacy reference hop lacks source revision")
        source_revision = int(raw_source_revision)
        prior_revision = hop.get("prior_revision_id")
        if prior_revision is not None:
            actions.append(ReferenceActionV2(
                kind="SWITCH_SNAPSHOT",
                parent_page=source,
                parent_revision_id=int(prior_revision),
                revision_id=source_revision,
            ))
        actions.append(ReferenceActionV2(
            kind="FOLLOW_LINK",
            parent_page=source,
            parent_revision_id=source_revision,
            page_title=str(hop.get("target_title") or ""),
        ))
    return PrivateReferenceRouteV2(
        route_id="legacy_reference_route_1",
        actions=tuple(actions),
        distance=len(actions),
    )


def _legacy_claim(
    *, claim_id: str, hop: dict[str, Any], event_time: str | None,
) -> RequiredClaimV2:
    excerpt = str(hop.get("evidence") or "")
    raw_revision_id = hop.get("source_revision_id")
    if raw_revision_id is None:
        raise ValueError(f"{claim_id}: legacy hop lacks source revision")
    witness = ClaimWitnessV2(
        page_title=str(hop.get("source_title") or ""),
        revision_id=int(raw_revision_id),
        evidence_excerpt=excerpt,
        evidence_hash=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
        semantic_validation={
            "source": "legacy_v6_machine_validation",
            "complete_judge_io_available": False,
            "limitation": "legacy case does not embed complete semantic judge I/O",
            "formal_v2_eligible": False,
        },
    )
    return RequiredClaimV2(
        claim_id=claim_id,
        subject=str(hop.get("source_title") or ""),
        relation=str(hop.get("relation") or ""),
        object=str(hop.get("target_title") or ""),
        event_time=event_time,
        witnesses=(witness,),
    )


def legacy_v6_case_to_v2(case: dict[str, Any]) -> EvaluationCaseV2:
    """Read a legacy v6 case without mutating or rewriting its artifact."""
    generation = case.get("_generation")
    source_schema = (
        str(generation.get("schema_version") or "")
        if isinstance(generation, dict) else ""
    )
    if source_schema != "wikipedia-cutoff-relative-multihop-v6":
        raise ValueError("legacy conversion requires a v6 multihop case")
    chain = case.get("reasoning_chain")
    contract = case.get("prior_knowledge_contract")
    if not isinstance(chain, list) or not isinstance(contract, dict):
        raise ValueError("legacy v6 case lacks chain or PK contract")
    probes = contract.get("probes")
    if not isinstance(probes, list):
        probes = []
    primary_ids = {str(value) for value in contract.get(
        "primary_admission_probe_ids", []
    )}
    critical: list[RequiredClaimV2] = []
    for probe in probes:
        if not isinstance(probe, dict) or str(probe.get("id")) not in primary_ids:
            continue
        hop_index = probe.get("hop_index")
        if not isinstance(hop_index, int) or not 0 <= hop_index < len(chain):
            raise ValueError("critical probe has invalid hop index")
        hop = chain[hop_index]
        structured = hop.get("structured_evidence")
        event_time = (
            str(structured.get("event_date"))
            if isinstance(structured, dict) and structured.get("event_date")
            else str(probe.get("event_date") or hop.get("as_of") or "")
        )
        critical.append(_legacy_claim(
            claim_id=str(probe["id"]), hop=hop, event_time=event_time,
        ))
    if not chain:
        raise ValueError("legacy v6 case has no reasoning chain")
    tail_hop = chain[-1]
    tail = _legacy_claim(
        claim_id="tail",
        hop=tail_hop,
        event_time=None,
    )
    model_ids = case.get("knowledge_cutoff", {}).get("model_ids", [])
    model_id = str(model_ids[0]) if model_ids else str(
        case.get("knowledge_cutoff", {}).get("model_id") or ""
    )
    return EvaluationCaseV2(
        case_id=str(case.get("id") or ""),
        model_id=model_id,
        question=str(case.get("temporal_question") or ""),
        start_page=str(case.get("start_title") or ""),
        cutoff_date=str(case.get("wikipedia_before") or ""),
        target_date=str(case.get("wikipedia_as_of") or ""),
        accepted_final_answer_aliases=tuple(
            str(value) for value in case.get("new_answer_keywords", [])
        ),
        critical_claims=tuple(critical),
        tail_relation=tail,
        validated_evidence_requirements={
            "critical_support": "semantic_witness_or_auditable_judge",
            "tail_support": "semantic_witness_or_auditable_judge",
            "answer_literal_support": True,
            "legacy_validation_status": "machine_pass_human_review_required",
        },
        event_time_constraints={
            claim.claim_id: {"operator": "equals", "event_time": claim.event_time}
            for claim in critical
        },
        reference_routes=(_legacy_reference_route(case),),
        source_case_schema=source_schema,
        source_case_sha256=canonical_sha256(case),
    )


def load_cases_v2(path: str | Path) -> list[EvaluationCaseV2]:
    """Load native v2 or legacy v6 case manifests read-only."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("case manifest must contain a cases list")
    if payload.get("schema_version") == CASE_SCHEMA_V2:
        return [evaluation_case_from_dict(row) for row in rows]
    return [legacy_v6_case_to_v2(row) for row in rows]


def validate_evaluation_case_v2(
    case: EvaluationCaseV2, *, require_formal_witness_records: bool = False,
) -> list[str]:
    errors: list[str] = []
    if case.cutoff_date >= case.target_date:
        errors.append("cutoff_date must precede target_date")
    for claim in (*case.critical_claims, case.tail_relation):
        for witness in claim.witnesses:
            if witness.validation_status == "human_approved":
                continue
            record = witness.semantic_validation
            required = {
                "judge_input", "judge_output", "model", "version", "confidence",
                "deterministic_guards",
            }
            if require_formal_witness_records and not required.issubset(record):
                errors.append(
                    f"{claim.claim_id}@{witness.revision_id}: incomplete semantic "
                    "judge record for formal v2"
                )
            if require_formal_witness_records and witness.validation_status != (
                "machine_pass_human_review_required"
            ):
                errors.append(
                    f"{claim.claim_id}@{witness.revision_id}: invalid machine review status"
                )
    return errors


def _witness_from_dict(value: dict[str, Any]) -> ClaimWitnessV2:
    return ClaimWitnessV2(**value)


def _claim_from_dict(value: dict[str, Any]) -> RequiredClaimV2:
    data = dict(value)
    data.pop("schema_version", None)
    data["witnesses"] = tuple(_witness_from_dict(row) for row in data["witnesses"])
    return RequiredClaimV2(**data)


def evaluation_case_from_dict(value: dict[str, Any]) -> EvaluationCaseV2:
    data = dict(value)
    if data.pop("schema_version", None) != CASE_SCHEMA_V2:
        raise ValueError(f"native case must use {CASE_SCHEMA_V2}")
    data["accepted_final_answer_aliases"] = tuple(
        data["accepted_final_answer_aliases"]
    )
    data["critical_claims"] = tuple(
        _claim_from_dict(row) for row in data["critical_claims"]
    )
    data["tail_relation"] = _claim_from_dict(data["tail_relation"])
    routes = []
    for raw in data.get("reference_routes", []):
        route = dict(raw)
        route["actions"] = tuple(ReferenceActionV2(**row) for row in route["actions"])
        routes.append(PrivateReferenceRouteV2(**route))
    data["reference_routes"] = tuple(routes)
    return EvaluationCaseV2(**data)


def structured_submission_from_dict(value: dict[str, Any]) -> StructuredSubmissionV2:
    data = dict(value)
    if data.pop("schema_version", SUBMISSION_SCHEMA_V2) != SUBMISSION_SCHEMA_V2:
        raise ValueError(f"submission must use {SUBMISSION_SCHEMA_V2}")

    def submitted(raw: dict[str, Any]) -> SubmittedClaimV2:
        row = dict(raw)
        row["supporting_evidence_ids"] = tuple(row.get("supporting_evidence_ids", []))
        return SubmittedClaimV2(**row)

    return StructuredSubmissionV2(
        answer=str(data.get("answer") or ""),
        critical_claims=tuple(submitted(row) for row in data.get("critical_claims", [])),
        tail_claim=submitted(data.get("tail_claim", {})),
    )
