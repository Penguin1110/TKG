"""Explicit human-review admission contract for generated experiment cases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


REVIEW_SCHEMA = "tkg-human-review-v1"
REVIEW_REQUIRED_STATUSES = {
    "machine_pass_human_review_required",
    "machine_accepted_human_review_required",
}


def case_review_sha256(case: dict[str, Any]) -> str:
    """Hash every fact a reviewer is approving, excluding prior review metadata."""
    payload = copy.deepcopy(case)
    generation = payload.get("_generation")
    if isinstance(generation, dict):
        generation.pop("human_review", None)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def requires_human_review(case: dict[str, Any]) -> bool:
    status = str(case.get("_generation", {}).get("status", ""))
    return status in REVIEW_REQUIRED_STATUSES


def _case_value(case: dict[str, Any], direct: str, generated: str) -> Any:
    return case.get(direct) or case.get("_generation", {}).get(generated, {}).get(
        "requested_as_of"
    )


def _parse_review_time(value: Any, case_id: str) -> str:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{case_id}: reviewed_at must be an ISO-8601 timestamp") from exc
    if parsed.utcoffset() is None:
        raise ValueError(f"{case_id}: reviewed_at must include a timezone offset")
    return text


def load_review_file(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError(f"human review file must use schema {REVIEW_SCHEMA!r}")
    values = payload.get("reviews")
    if not isinstance(values, list):
        raise ValueError("human review file needs a reviews list")
    reviews: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(values):
        if not isinstance(raw, dict):
            raise ValueError(f"review {index} must be an object")
        case_id = str(raw.get("case_id", "")).strip()
        if not case_id or case_id in reviews:
            raise ValueError(f"review {index} has a missing or duplicate case_id")
        decision = str(raw.get("decision", "")).strip()
        if decision not in {"approved", "rejected", "waived_by_user"}:
            raise ValueError(f"{case_id}: invalid human review decision {decision!r}")
        reviewer = str(raw.get("reviewer", "")).strip()
        if not reviewer:
            raise ValueError(f"{case_id}: reviewer must not be empty")
        case_hash = str(raw.get("case_sha256", "")).strip()
        if not re.fullmatch(r"[0-9a-f]{64}", case_hash):
            raise ValueError(f"{case_id}: case_sha256 must be a SHA-256 hex digest")
        record = {
            "schema_version": REVIEW_SCHEMA,
            "case_id": case_id,
            "case_sha256": case_hash,
            "decision": decision,
            "reviewer": reviewer,
            "reviewed_at": _parse_review_time(raw.get("reviewed_at"), case_id),
            "notes": str(raw.get("notes", "")).strip(),
            "source": str(path),
        }
        if not record["notes"]:
            raise ValueError(f"{case_id}: human review needs notes")
        reviews[case_id] = record
    return reviews


def resolve_human_reviews(
    cases: list[dict[str, Any]], *, review_file: str | None,
    waive_human_review: bool,
) -> dict[str, dict[str, Any]]:
    """Resolve every case to approved, explicitly waived, or not-required."""
    if review_file and waive_human_review:
        raise ValueError("use either --human-review-file or --waive-human-review, not both")
    supplied = load_review_file(review_file) if review_file else {}
    case_ids = {str(case.get("id")) for case in cases}
    extras = sorted(set(supplied) - case_ids)
    if extras:
        raise ValueError(f"human review file contains unknown case IDs: {extras!r}")
    resolved: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case.get("id"))
        case_hash = case_review_sha256(case)
        if not requires_human_review(case):
            resolved[case_id] = {
                "schema_version": REVIEW_SCHEMA,
                "case_id": case_id,
                "case_sha256": case_hash,
                "decision": "not_required",
                "source": "case_generation_status",
            }
            continue
        record = supplied.get(case_id)
        if record is not None:
            if record["case_sha256"] != case_hash:
                raise ValueError(
                    f"{case_id}: human review hash does not match the current case"
                )
            if record["decision"] == "rejected":
                raise ValueError(f"{case_id}: case was rejected by human review")
            resolved[case_id] = record
            continue
        if waive_human_review:
            resolved[case_id] = {
                "schema_version": REVIEW_SCHEMA,
                "case_id": case_id,
                "case_sha256": case_hash,
                "decision": "waived_by_user",
                "reviewer": "user_via_cli",
                "reviewed_at": datetime.now().astimezone().isoformat(),
                "notes": "Explicit --waive-human-review CLI authorization.",
                "source": "cli_flag",
            }
            continue
        raise ValueError(
            f"{case_id}: human review is required; provide --human-review-file or "
            "explicitly pass --waive-human-review"
        )
    return resolved


def review_template(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a non-admissible draft that a human must complete explicitly."""
    return {
        "schema_version": REVIEW_SCHEMA,
        "instructions": (
            "Review every question, answer aliases, evidence chain, and dates. Replace "
            "pending with approved or rejected, identify the reviewer, set reviewed_at "
            "to ISO-8601, and add notes. Do not change case_sha256."
        ),
        "reviews": [
            {
                "case_id": str(case.get("id")),
                "case_sha256": case_review_sha256(case),
                "decision": "pending",
                "reviewer": "",
                "reviewed_at": "",
                "notes": "",
                "question": case.get("temporal_question"),
                "before": _case_value(case, "wikipedia_before", "before"),
                "target": _case_value(case, "wikipedia_as_of", "after"),
                "old_answer_keywords": case.get("old_answer_keywords", []),
                "new_answer_keywords": case.get("new_answer_keywords", []),
                "reasoning_hop_count": case.get("reasoning_hop_count"),
            }
            for case in cases if requires_human_review(case)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a hash-bound draft human-review artifact for generated cases"
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite {output}; pass --overwrite explicitly")
    payload = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        parser.error("case file must be an object containing a cases list")
    output.write_text(
        json.dumps(review_template(cases), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote pending review draft: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
