"""Evidence-conditioned LLM refinement for v2.5 shortcut audit unknowns.

This is a post-hoc judge only. It never enters the solver context or scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tkg.api.openrouter import UsageLedger, model_call_context, set_usage_ledger
from tkg.experiment.question_generation import call_json_model
from tkg.experiment.shortcut_semantic_audit_v25 import _direct_link_candidate
from tkg.wikipedia.backend import WikipediaPageBackend


SCHEMA_VERSION = "shortcut-semantic-llm-refinement-v2.5"


def _fold(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _excerpt(content: str, needles: list[str], *, limit: int = 6000) -> str:
    folded = content.casefold()
    windows: list[str] = []
    for needle in needles:
        start = folded.find(needle.casefold())
        if start < 0:
            continue
        left, right = max(0, start - 500), min(len(content), start + len(needle) + 900)
        window = content[left:right].strip()
        if window and window not in windows:
            windows.append(window)
    if not windows:
        windows.append(content[:1200])
    return "\n...\n".join(windows)[:limit]


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                result[str(row["prompt_sha256"])] = row
    return result


def _claims(case: dict[str, Any]) -> list[dict[str, Any]]:
    probe_by_hop = {
        int(probe["hop_index"]): probe
        for probe in case["prior_knowledge_contract"]["probes"]
        if probe.get("role") in {"critical_bridge", "tail"}
    }
    claims = []
    for index, probe in sorted(probe_by_hop.items()):
        hop = case["reasoning_chain"][index]
        claims.append({
            "id": f"hop_{index}", "role": probe["role"],
            "source": hop["source_title"], "relation": hop["relation"],
            "target": hop["target_title"], "target_aliases": hop["target_aliases"],
            "event_date": probe.get("event_date"),
        })
    return claims


def _visible_pages(
    candidate: dict[str, Any], backend: WikipediaPageBackend, claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    needles: list[str] = []
    for claim in claims:
        needles.extend([claim["source"], claim["target"], *claim["target_aliases"]])
    pages: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for node in candidate.get("shortcut_path", []):
        key = (str(node["page"]).casefold(), int(node["revision_id"]))
        if key in seen:
            continue
        seen.add(key)
        page = backend.fetch_page(str(node["page"]), as_of=node.get("as_of"))
        pages.append({
            "node_id": f"node_{len(pages)}", "page": page.title,
            "revision_id": page.revision_id, "as_of": node.get("as_of"),
            "excerpt": _excerpt(page.content, needles),
        })
    return pages


def _prompt(case: dict[str, Any], claims: list[dict[str, Any]], pages: list[dict[str, Any]]) -> str:
    return f"""Audit whether each claim is semantically supported by the visible Wikipedia revision excerpts.

This is a post-hoc benchmark audit. Do not use outside knowledge. A person/entity name merely
appearing is not support. For a temporal bridge, the excerpt must express the stated relation and,
when an event date is specified, a compatible onset/event. Historical lists count only when they
unambiguously encode that relation. If wording is insufficient, return does_not_support; use unclear
only for genuinely ambiguous evidence.

Case cutoff: {case['knowledge_cutoff']['cutoff_date']}
Claims:
{json.dumps(claims, ensure_ascii=False, indent=2)}

Visible pages:
{json.dumps(pages, ensure_ascii=False, indent=2)}

Return exactly one JSON object:
{{
  "results": [
    {{"id": "hop_N", "verdict": "supports|does_not_support|unclear",
      "evidence_node_ids": ["node_N"], "reason": "short explanation"}}
  ]
}}
Return every claim ID exactly once, no extra IDs. Evidence node IDs must come from the supplied pages.
"""


def _validate(raw: dict[str, Any], claims: list[dict[str, Any]], pages: list[dict[str, Any]]) -> dict[str, Any]:
    rows = raw.get("results")
    if not isinstance(rows, list):
        raise ValueError("results must be a list")
    expected = {claim["id"] for claim in claims}
    page_ids = {page["node_id"] for page in pages}
    returned = [row.get("id") for row in rows if isinstance(row, dict)]
    if len(returned) != len(set(returned)) or set(returned) != expected:
        raise ValueError("judge claim-ID coverage failure")
    for row in rows:
        if row.get("verdict") not in {"supports", "does_not_support", "unclear"}:
            raise ValueError("invalid verdict")
        evidence_ids = row.get("evidence_node_ids")
        if not isinstance(evidence_ids, list) or not set(evidence_ids) <= page_ids:
            raise ValueError("invalid evidence node IDs")
    return {"results": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-cases", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--usage-output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    summary_path = Path(args.summary)
    if output.exists() or summary_path.exists():
        parser.error("refinement outputs already exist; refuse overwrite")
    cases = {
        str(case["id"]): case
        for case in json.loads(Path(args.frozen_cases).read_text(encoding="utf-8"))["cases"]
    }
    audits = [json.loads(line) for line in Path(args.audit_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    cache_path = Path(args.judge_cache)
    cache = _load_cache(cache_path)
    backend = WikipediaPageBackend(cache_path=args.cache_path, offline_only=False)
    usage = UsageLedger(args.usage_output, metadata={"experiment": SCHEMA_VERSION, "model": args.model})
    set_usage_ledger(usage)
    refined = []
    try:
        for audit in audits:
            case = cases[str(audit["case_id"])]
            claims = _claims(case)
            candidates = []
            for candidate_index, candidate in enumerate(audit["candidate_shortcuts"]):
                candidate = dict(candidate)
                if (
                    candidate["classification"] == "SEMANTIC_UNKNOWN"
                    and not candidate.get("shortcut_path")
                    and candidate.get("signal", {}).get("kind") == "non_adjacent_route_link"
                ):
                    candidate.update(_direct_link_candidate(
                        case, candidate["signal"], backend,
                    ))
                if candidate["classification"] != "SEMANTIC_UNKNOWN" or not candidate.get("shortcut_path"):
                    candidates.append(candidate)
                    continue
                try:
                    pages = _visible_pages(candidate, backend, claims)
                    prompt = _prompt(case, claims, pages)
                    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    cached = cache.get(prompt_hash)
                    if cached:
                        judged = cached["judged"]
                        cache_hit = True
                    else:
                        with model_call_context(
                            role="shortcut_semantic_judge", case_id=case["id"],
                            candidate_index=candidate_index,
                        ):
                            raw, _ = call_json_model(args.model, prompt)
                        judged = _validate(raw, claims, pages)
                        cache_row = {"prompt_sha256": prompt_hash, "judged": judged}
                        with cache_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(cache_row, ensure_ascii=False) + "\n")
                        cache[prompt_hash] = cache_row
                        cache_hit = False
                    by_id = {row["id"]: row for row in judged["results"]}
                    bridge_ids = [claim["id"] for claim in claims if claim["role"] == "critical_bridge"]
                    tail_ids = [claim["id"] for claim in claims if claim["role"] == "tail"]
                    bridge_verdicts = [by_id[item]["verdict"] for item in bridge_ids]
                    tail_verdicts = [by_id[item]["verdict"] for item in tail_ids]
                    cutoff = str(case["knowledge_cutoff"]["cutoff_date"])[:10]
                    node_by_id = {page["node_id"]: page for page in pages}
                    pre_cutoff_bridge = any(
                        row["verdict"] == "supports" and any(
                            str(node_by_id[node_id].get("as_of") or "")[:10] <= cutoff
                            for node_id in row["evidence_node_ids"]
                        )
                        for claim_id, row in by_id.items() if claim_id in bridge_ids
                    )
                    if pre_cutoff_bridge:
                        classification = "PRE_CUTOFF_LEAKAGE"
                    elif "unclear" in [*bridge_verdicts, *tail_verdicts]:
                        classification = "SEMANTIC_UNKNOWN"
                    elif all(value == "supports" for value in bridge_verdicts) and all(value == "supports" for value in tail_verdicts) and candidate.get("post_cutoff_switch"):
                        classification = "TEMPORAL_VALID_ALTERNATIVE"
                    else:
                        classification = "ANSWER_ONLY_SHORTCUT"
                    candidate.update({
                        "classification": classification,
                        "semantic_judge": {"model": args.model, "cache_hit": cache_hit, **judged},
                        "structured_evaluator_status": (
                            "semantic_prerequisites_passed" if classification == "TEMPORAL_VALID_ALTERNATIVE"
                            else "must_fail_without_complete_semantic_bridge_and_tail_support"
                        ),
                    })
                except Exception as exc:
                    candidate["semantic_judge_error"] = str(exc)
                candidates.append(candidate)
            labels = {candidate["classification"] for candidate in candidates}
            if "PRE_CUTOFF_LEAKAGE" in labels:
                disposition = "reject"
            elif "REDIRECT_ALIAS_ARTIFACT" in labels:
                disposition = "reject_or_fix"
            elif "SEMANTIC_UNKNOWN" in labels:
                disposition = "quarantine"
            else:
                disposition = "retain"
            refined.append({
                **audit, "schema_version": SCHEMA_VERSION,
                "candidate_shortcuts": candidates,
                "classification_counts": {label: sum(c["classification"] == label for c in candidates) for label in sorted(labels)},
                "disposition": disposition,
            })
    finally:
        set_usage_ledger(None)
        usage.close()
        backend.close()
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in refined), encoding="utf-8")
    summary_labels = sorted({label for row in refined for label in row["classification_counts"]})
    dispositions = {label: sum(row["disposition"] == label for row in refined) for label in ("retain", "reject", "reject_or_fix", "quarantine")}
    summary = {
        "schema_version": SCHEMA_VERSION, "case_count": len(refined),
        "classification_counts": {label: sum(row["classification_counts"].get(label, 0) for row in refined) for label in summary_labels},
        "dispositions": dispositions,
        "engineering_abcd_authorized": dispositions == {"retain": len(refined), "reject": 0, "reject_or_fix": 0, "quarantine": 0},
        "formal_abcd_authorized": False,
        "formal_blocker": "human evidence review waived_not_performed",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
