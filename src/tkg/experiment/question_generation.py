"""Generate auditable temporal QA candidates from two Wikipedia revisions.

The generator is intentionally a proposal system, not an authority.  MediaWiki
selects the evidence, one model converts changed text into questions, hard
deterministic checks reject unsupported fields, and a different model judges
the temporal semantics.  Accepted candidates are written to a new staging file
for human review; this command never edits ``cases.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from tkg.api.openrouter import OpenRouterError, call_model
from tkg.experiment.results import assert_new_output_path
from tkg.wikipedia.backend import WikipediaError, WikipediaPageBackend


SCHEMA_VERSION = "wikipedia-temporal-question-v2"
GENERIC_ANSWERS = {
    "yes", "no", "none", "unknown", "n/a", "not known", "no information",
    "true", "false",
}
@dataclass(frozen=True)
class GenerationSeed:
    title: str
    before: str
    after: str
    category: str = "other"
    id_prefix: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationSeed":
        missing = [key for key in ("title", "before", "after") if not value.get(key)]
        if missing:
            raise ValueError(f"seed missing fields: {', '.join(missing)}")
        before = str(value["before"])
        after = str(value["after"])
        if _parse_time(before) >= _parse_time(after):
            raise ValueError(f"seed {value['title']!r}: before must be earlier than after")
        return cls(
            title=str(value["title"]), before=before, after=after,
            category=str(value.get("category", "other")),
            id_prefix=str(value.get("id_prefix", "")),
        )


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _plain_text(value: str) -> str:
    """Turn backend link annotations into the anchor text the reader saw."""
    value = re.sub(r"\[([^\[\]]+?)\s+->\s+[^\[\]]+\]", r"\1", value)
    return " ".join(value.split())


def _fold(value: str) -> str:
    return _plain_text(value).casefold()


def _blocks(content: str, min_chars: int = 20) -> list[str]:
    values = []
    for raw in re.split(r"\n\s*\n", content):
        block = _plain_text(raw)
        if len(block) >= min_chars:
            values.append(block)
    return values


def revision_diff(
    before_content: str,
    after_content: str,
    *,
    max_changed_blocks: int = 24,
    max_stable_blocks: int = 16,
) -> dict[str, list[str]]:
    """Return removed, added, and unchanged visible blocks in document order."""
    before = _blocks(before_content)
    after = _blocks(after_content)
    matcher = SequenceMatcher(a=before, b=after, autojunk=False)
    removed: list[str] = []
    added: list[str] = []
    stable: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "delete"}:
            removed.extend(before[i1:i2])
        if tag in {"replace", "insert"}:
            added.extend(after[j1:j2])
        if tag == "equal":
            stable.extend(after[j1:j2])
    return {
        "before_changed": removed[:max_changed_blocks],
        "after_changed": added[:max_changed_blocks],
        "stable": stable[:max_stable_blocks],
    }


def _numbered(values: list[str], max_chars: int) -> str:
    lines = []
    used = 0
    for index, value in enumerate(values, start=1):
        item = f"[{index}] {value[:1600]}"
        if used + len(item) > max_chars:
            break
        lines.append(item)
        used += len(item)
    return "\n".join(lines) or "(none)"


def generation_prompt(seed: GenerationSeed, diff: dict[str, list[str]], max_candidates: int) -> str:
    return f"""Create up to {max_candidates} temporal QA case candidates from Wikipedia evidence.

Page: {seed.title}
Older snapshot: {seed.before}
Newer snapshot: {seed.after}

OLDER CHANGED BLOCKS
{_numbered(diff['before_changed'], 12000)}

NEWER CHANGED BLOCKS
{_numbered(diff['after_changed'], 12000)}

Return exactly one JSON object with key "candidates" (a list).  Every candidate must contain:
- id_suffix, category, question, old_answer_keywords, new_answer_keywords,
  old_evidence, new_evidence, rationale

Strict rules:
1. The question must ask the same time-sensitive property in both snapshots.  The older page
   must explicitly support the old answer and the newer page must explicitly support a different
   new answer.  Reject announcements, speculation, and historical mentions presented as current.
2. Evidence fields must be short verbatim substrings from the supplied blocks.  Every answer alias
   must literally occur in its corresponding evidence string.
3. The question must not contain either answer.  Use invariant wording that works for
   both dates.  Answers must be concise entities, dates, numbers, titles, or names.
4. Fewer valid questions are better than plausible-sounding unsupported questions.  Return an empty
   candidates list if no clean current-value A-to-B change exists.
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if match:
        value = json.loads(match.group(1))
        if isinstance(value, dict):
            return value
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(text[start:end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError("model did not return one JSON object")


def call_json_model(
    model: str,
    prompt: str,
    call_model_fn: Callable[..., str] = call_model,
) -> tuple[dict[str, Any], str]:
    messages = [
        {"role": "system", "content": (
            "You generate auditable research data. Use only supplied evidence. "
            "Return one JSON object and no prose."
        )},
        {"role": "user", "content": prompt},
    ]
    last_response = ""
    for attempt in range(2):
        last_response = call_model_fn(model, messages, temperature=0.0)
        try:
            return _extract_json_object(last_response), last_response
        except (ValueError, json.JSONDecodeError):
            if attempt == 0:
                messages.extend([
                    {"role": "assistant", "content": last_response},
                    {"role": "user", "content": "Return only one valid JSON object."},
                ])
    raise ValueError("model returned invalid JSON twice")


def _strings(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        return None
    return [item.strip() for item in value]


def _check_evidence(
    *,
    label: str,
    evidence: Any,
    aliases: Any,
    page_content: str,
    errors: list[str],
) -> None:
    if not isinstance(evidence, str) or not evidence.strip():
        errors.append(f"{label}: missing evidence")
        return
    if _fold(evidence) not in _fold(page_content):
        errors.append(f"{label}: evidence is not a verbatim page substring")
    parsed = _strings(aliases)
    if parsed is None:
        errors.append(f"{label}: aliases must be a non-empty string list")
        return
    evidence_folded = _fold(evidence)
    for alias in parsed:
        if _fold(alias) not in evidence_folded:
            errors.append(f"{label}: alias {alias!r} is absent from evidence")
    if all(_fold(alias) in GENERIC_ANSWERS for alias in parsed):
        errors.append(f"{label}: generic yes/no/unknown answers are forbidden")


def _check_question(
    label: str,
    question: Any,
    paraphrases: Any,
    aliases: list[str],
    errors: list[str],
    *,
    require_paraphrases: bool = True,
) -> None:
    if not isinstance(question, str) or not question.strip().endswith("?"):
        errors.append(f"{label}: question must be a non-empty question")
        return
    variants = _strings(paraphrases) if require_paraphrases else []
    if require_paraphrases and (variants is None or len(variants) != 2):
        errors.append(f"{label}: exactly two paraphrases are required")
        variants = []
    for variant in [question, *(variants or [])]:
        folded = _fold(variant)
        if any(_fold(alias) in folded for alias in aliases):
            errors.append(f"{label}: question/paraphrase leaks an answer")


def validate_temporal_candidate(
    candidate: dict[str, Any], before_content: str, after_content: str
) -> list[str]:
    """Hard evidence checks for the single-question temporal exploration task."""
    errors: list[str] = []
    old = _strings(candidate.get("old_answer_keywords")) or []
    new = _strings(candidate.get("new_answer_keywords")) or []
    _check_evidence(
        label="before", evidence=candidate.get("old_evidence"), aliases=old,
        page_content=before_content, errors=errors,
    )
    _check_evidence(
        label="after", evidence=candidate.get("new_evidence"), aliases=new,
        page_content=after_content, errors=errors,
    )
    if not old or not new:
        errors.append("distinct before and after answers are required")
    elif {_fold(item) for item in old} & {_fold(item) for item in new}:
        errors.append("before and after answer aliases overlap")
    _check_question(
        "question", candidate.get("question"), [], [*old, *new], errors,
        require_paraphrases=False,
    )
    return errors


class TemporalQuestionJudge:
    """Independent whole-candidate semantic judge."""

    def __init__(
        self,
        model: str,
        *,
        min_confidence: float = 0.8,
        call_model_fn: Callable[..., str] = call_model,
    ):
        self.model = model
        self.min_confidence = min_confidence
        self.call_model_fn = call_model_fn

    def judge(self, candidate: dict[str, Any]) -> dict[str, Any]:
        prompt = f"""Audit this temporal Wikipedia QA candidate.

Candidate:
{json.dumps(candidate, ensure_ascii=False, indent=2)}

The evidence strings already passed exact-substring checks against their named revisions.
Decide pass only when all are true:
- The old/new evidence describes the same current-valued property at two dates and the value changed.
- Neither side is merely speculation, an announcement of a future event, or a historical mention.
- The single question is clear, uniquely answerable at either date, and does not reveal its answers.

Return one JSON object with keys decision (pass or reject), confidence (0..1), checks (object),
reason, and rejected_items (list of field names).
"""
        raw, response = call_json_model(self.model, prompt, self.call_model_fn)
        decision = str(raw.get("decision", "reject"))
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        if decision != "pass" or confidence < self.min_confidence:
            decision = "reject"
        return {**raw, "decision": decision, "confidence": confidence, "raw_response": response}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug[:60] or "wikipedia_case"


def candidate_case(
    candidate: dict[str, Any],
    seed: GenerationSeed,
    before_page: dict[str, Any],
    after_page: dict[str, Any],
    judge_result: dict[str, Any],
    index: int,
    generator_model: str,
    judge_model: str,
) -> dict[str, Any]:
    suffix = _slug(str(candidate.get("id_suffix", "")))
    prefix = _slug(seed.id_prefix or seed.title)
    case_id = prefix if suffix in {"", prefix} else f"{prefix}_{suffix}"
    if index:
        case_id = f"{case_id}_{index + 1}"
    case = {
        "id": case_id,
        "category": (
            seed.category if seed.category != "other"
            else candidate.get("category") or "other"
        ),
        "wikipedia_title": after_page["title"],
        "wikipedia_before": seed.before,
        "wikipedia_as_of": seed.after,
        "temporal_question": candidate["question"],
        "old_answer_keywords": candidate["old_answer_keywords"],
        "new_answer_keywords": candidate["new_answer_keywords"],
        "_generation": {
            "schema_version": SCHEMA_VERSION,
            "status": "machine_pass_human_review_required",
            "generator_model": generator_model,
            "judge_model": judge_model,
            "judge": {key: value for key, value in judge_result.items() if key != "raw_response"},
            "before": {
                "requested_as_of": seed.before,
                "revision_id": before_page["revision_id"],
                "timestamp": before_page["timestamp"],
                "source_url": before_page.get("source_url"),
            },
            "after": {
                "requested_as_of": seed.after,
                "revision_id": after_page["revision_id"],
                "timestamp": after_page["timestamp"],
                "source_url": after_page.get("source_url"),
            },
            "old_evidence": candidate["old_evidence"],
            "new_evidence": candidate["new_evidence"],
            "rationale": candidate.get("rationale", ""),
        },
    }
    return case


def load_seeds(path: str) -> list[GenerationSeed]:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".jsonl"):
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        values = value.get("seeds", []) if isinstance(value, dict) else value
    if not isinstance(values, list):
        raise ValueError("seed file must contain a list or {'seeds': [...]} object")
    return [GenerationSeed.from_dict(item) for item in values]


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _page_reference(page) -> dict[str, Any]:
    return {
        "title": page.title,
        "page_id": page.page_id,
        "revision_id": page.revision_id,
        "timestamp": page.timestamp,
        "as_of": page.as_of,
        "source_url": page.source_url,
        "content_sha256": hashlib.sha256(page.content.encode("utf-8")).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate staged temporal QA cases from two Wikipedia revisions"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seed-file", help="JSON/JSONL rows: title, before, after, category")
    source.add_argument("--title", help="single Wikipedia page title")
    parser.add_argument("--before", help="required with --title")
    parser.add_argument("--after", help="required with --title")
    parser.add_argument("--category", default="other")
    parser.add_argument("--id-prefix", default="")
    parser.add_argument("--generator-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-min-confidence", type=float, default=0.8)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--cache-path", default="wikipedia_question_generation.db")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--diff-only", action="store_true", help="fetch/diff only; no LLM calls")
    parser.add_argument("--output", default="generated_question_packets.jsonl")
    parser.add_argument("--cases-output", default="generated_cases.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.title:
        if not args.before or not args.after:
            parser.error("--title requires --before and --after")
        seeds = [GenerationSeed.from_dict({
            "title": args.title, "before": args.before, "after": args.after,
            "category": args.category, "id_prefix": args.id_prefix,
        })]
    else:
        seeds = load_seeds(args.seed_file)
    if not args.diff_only:
        if not args.generator_model or not args.judge_model:
            parser.error("--generator-model and --judge-model are required")
        if args.generator_model == args.judge_model:
            parser.error("generator and judge models must be different")
        if not os.environ.get("OPENROUTER_API_KEY"):
            parser.error("OPENROUTER_API_KEY is required for generation")
    if not 0 <= args.judge_min_confidence <= 1:
        parser.error("--judge-min-confidence must be between 0 and 1")
    if args.max_candidates <= 0:
        parser.error("--max-candidates must be > 0")
    output_paths = [args.output] + ([] if args.diff_only else [args.cases_output])
    if not args.diff_only and Path(args.cases_output).resolve() == Path("cases.json").resolve():
        parser.error("the generator never writes cases.json; use a staging --cases-output")
    for path in output_paths:
        try:
            assert_new_output_path(path)
        except ValueError as exc:
            parser.error(str(exc))
        if Path(path).exists() and not args.overwrite:
            parser.error(f"refusing to overwrite {path}; pass --overwrite explicitly")

    backend = WikipediaPageBackend(
        cache_path=args.cache_path, lang=args.lang, offline_only=args.offline
    )
    packets: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    judge = None if args.diff_only else TemporalQuestionJudge(
        args.judge_model, min_confidence=args.judge_min_confidence
    )
    try:
        for seed in seeds:
            print(f"[source] {seed.title}: {seed.before} -> {seed.after}")
            try:
                before = backend.fetch_page(seed.title, as_of=seed.before)
                after = backend.fetch_page(seed.title, as_of=seed.after)
                diff = revision_diff(before.content, after.content)
                source_record = {
                    "seed": seed.__dict__,
                    "before": _page_reference(before),
                    "after": _page_reference(after),
                    "diff": diff,
                }
                if args.diff_only:
                    packets.append({
                        "schema_version": SCHEMA_VERSION, "status": "diff_only", **source_record,
                    })
                    continue
                if not diff["before_changed"] or not diff["after_changed"]:
                    packets.append({
                        "schema_version": SCHEMA_VERSION, "status": "no_changed_blocks",
                        **source_record,
                    })
                    continue
                prompt = generation_prompt(seed, diff, args.max_candidates)
                generated, raw_response = call_json_model(args.generator_model, prompt)
                candidates = generated.get("candidates", [])
                if not isinstance(candidates, list):
                    raise ValueError("generator candidates must be a list")
                for index, candidate in enumerate(candidates[:args.max_candidates]):
                    if not isinstance(candidate, dict):
                        continue
                    errors = validate_temporal_candidate(
                        candidate, before.content, after.content,
                    )
                    packet: dict[str, Any] = {
                        "schema_version": SCHEMA_VERSION,
                        "packet_id": hashlib.sha256(
                            f"{seed.title}|{before.revision_id}|{after.revision_id}|{index}".encode()
                        ).hexdigest()[:16],
                        "seed": seed.__dict__,
                        "source_revisions": {
                            "before": {"revision_id": before.revision_id,
                                       "timestamp": before.timestamp, "url": before.source_url},
                            "after": {"revision_id": after.revision_id,
                                      "timestamp": after.timestamp, "url": after.source_url},
                        },
                        "candidate": candidate,
                        "generator_model": args.generator_model,
                        "generator_response": raw_response,
                        "deterministic_errors": errors,
                    }
                    if errors:
                        packet["status"] = "deterministic_reject"
                        packets.append(packet)
                        continue
                    assert judge is not None
                    judge_result = judge.judge(candidate)
                    packet["judge_model"] = args.judge_model
                    packet["judge"] = judge_result
                    packet["status"] = (
                        "machine_pass_human_review_required"
                        if judge_result["decision"] == "pass" else "judge_reject"
                    )
                    packets.append(packet)
                    if judge_result["decision"] == "pass":
                        accepted.append(candidate_case(
                            candidate, seed, before.to_dict(), after.to_dict(), judge_result,
                            index, args.generator_model, args.judge_model,
                        ))
            except (WikipediaError, OpenRouterError, ValueError, json.JSONDecodeError) as exc:
                print(f"[error] {seed.title}: {exc}", file=sys.stderr)
                packets.append({
                    "schema_version": SCHEMA_VERSION, "status": "error",
                    "seed": seed.__dict__, "error": str(exc),
                })
    finally:
        backend.close()

    _write_jsonl(args.output, packets)
    if not args.diff_only:
        with open(args.cases_output, "w", encoding="utf-8") as fh:
            json.dump({
                "_schema_notes": (
                    "Machine-generated staging cases. Every row requires human factual review "
                    "before merging into cases.json."
                ),
                "cases": accepted,
            }, fh, ensure_ascii=False, indent=2)
        print(f"[done] packets={len(packets)} accepted={len(accepted)} -> {args.cases_output}")
    else:
        print(f"[done] diff packets={len(packets)} -> {args.output}")
    return 0 if args.diff_only or accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
