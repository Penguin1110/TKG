"""New multi-path synthetic live smoke using real open-weight action logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tkg.experiment.contracts import PageLink, PageSnapshot
from tkg.experiment.open_weight_answer_generator_v24 import (
    EvidenceConditionedAnswerGeneratorV24,
)
from tkg.experiment.open_weight_action_scorer_v24 import (
    HuggingFaceCausalLMBackendV24, OpenWeightConditionalActionScorerV24,
)
from tkg.experiment.open_weight_live_controller_v24 import (
    HierarchicalOpenWeightLiveControllerV24,
)
from tkg.experiment.results import assert_new_output_path
from tkg.experiment.temporal_environment_v2 import TemporalWikipediaEnvironmentV2
from tkg.experiment.temporal_eval_schema_v2 import PublicTemporalCaseV2
from tkg.experiment.temporal_live_runner_v23 import LiveSearchConfigV23
from tkg.experiment.temporal_live_runner_v24 import run_live_temporal_search_v24


def _page(
    title: str, revision_id: int, timestamp: str, content: str,
    links: list[str] | None = None,
) -> PageSnapshot:
    return PageSnapshot(
        title=title, page_id=revision_id, revision_id=revision_id,
        timestamp=timestamp, as_of=timestamp, content=content,
        links=[PageLink(target=target, anchor=target) for target in (links or [])],
        source_url=f"synthetic-open-weight-v24://revision/{revision_id}",
    )


class OpenWeightMultiPathBackendV24:
    def __init__(self) -> None:
        pages = [
            _page(
                "Aurora Hub", 701, "2024-05-01T00:00:00Z", "Aurora index.",
                ["Aurora Museum leadership timeline", "Aurora Museum curator archive", *[
                    f"Aurora Distractor {index}" for index in range(8)
                ]],
            ),
            _page("Aurora Museum leadership timeline", 710, "2024-05-01T00:00:00Z", "Old roster."),
            _page(
                "Aurora Museum leadership timeline", 711, "2025-04-04T00:00:00Z",
                "On 4 April 2025, Mira Sol became curator of Aurora Museum.",
                ["Mira Sol"],
            ),
            _page("Aurora Museum curator archive", 720, "2024-05-01T00:00:00Z", "Old archive."),
            _page(
                "Aurora Museum curator archive", 721, "2025-04-04T00:00:00Z",
                "Aurora Museum appointed Mira Sol as curator on 4 April 2025.",
                ["Mira Sol"],
            ),
            _page(
                "Mira Sol", 730, "2023-01-01T00:00:00Z",
                "Mira Sol was born in Lumen Bay.",
            ),
        ]
        pages.extend(
            _page(
                f"Aurora Distractor {index}", 800 + index,
                "2024-01-01T00:00:00Z", "Unrelated archive.",
            ) for index in range(8)
        )
        self.by_revision = {page.revision_id: page for page in pages}
        self.by_title: dict[str, list[PageSnapshot]] = {}
        for page in pages:
            self.by_title.setdefault(page.title.casefold(), []).append(page)
        for values in self.by_title.values():
            values.sort(key=lambda page: page.timestamp)

    def fetch_page(self, title: str, as_of: str | None = None) -> PageSnapshot:
        values = self.by_title[title.casefold()]
        if as_of is None:
            return values[-1]
        eligible = [page for page in values if page.timestamp[:10] <= as_of[:10]]
        if not eligible:
            raise ValueError(f"no synthetic revision for {title} at {as_of}")
        return eligible[-1]

    def fetch_revision(self, revision_id: int) -> PageSnapshot:
        return self.by_revision[revision_id]

    def list_revision_metadata_page(
        self, title: str, from_date: str, to_date: str, *,
        cursor: str | None = None, page_size: int = 50,
    ) -> dict[str, Any]:
        if cursor is not None:
            return {"title": title, "revisions": [], "next_cursor": None}
        revisions = [
            {"revision_id": page.revision_id, "timestamp": page.timestamp}
            for page in self.by_title[title.casefold()]
            if from_date <= page.timestamp[:10] <= to_date
        ]
        return {"title": title, "revisions": revisions[:page_size],
                "next_cursor": None}


def public_case_v24() -> PublicTemporalCaseV2:
    return PublicTemporalCaseV2(
        case_id="open-weight-multipath-synthetic-v24",
        model_id="open-weight-development",
        question=(
            "Where was the person who became curator of Aurora Museum after "
            "the cutoff born?"
        ),
        start_page="Aurora Hub", cutoff_date="2024-06-01",
        target_date="2025-12-31",
    )


def run_open_weight_synthetic_v24(
    *, model: str, device: str = "cuda", dtype: str = "float16",
) -> dict[str, Any]:
    backend = OpenWeightMultiPathBackendV24()
    logits = HuggingFaceCausalLMBackendV24(
        model, device=device, dtype=dtype,
    )
    controller = HierarchicalOpenWeightLiveControllerV24(
        OpenWeightConditionalActionScorerV24(logits),
        compact_payload_proposer=EvidenceConditionedAnswerGeneratorV24(logits),
        payload_proposer_name=(
            "open_weight_evidence_conditioned_answer_generator_v2.4"
        ),
    )
    result = run_live_temporal_search_v24(
        public_case=public_case_v24(), backend=backend,
        environment=TemporalWikipediaEnvironmentV2(backend),
        controller=controller,
        config=LiveSearchConfigV23(
            beam_width=3, max_expansions=40, max_actions_per_state=3,
            dense_action_limit=30, seed=47,
        ),
    )
    payload = result.to_dict()
    success = (
        result.final_state.submitted is not None
        and result.final_state.submitted.get("answer") == "Lumen Bay"
        and not result.final_state.error
    )
    return {
        "schema_version": "open-weight-live-multipath-smoke-v2.4",
        "model": model, "search": payload,
        "success": success,
        "payload_generation_is_model_based": True,
        "payload_fixture_used": False,
        "action_ranking_uses_real_model_logits": True,
        "formal_conclusion_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert_new_output_path(args.output)
    result = run_open_weight_synthetic_v24(
        model=args.model, device=args.device, dtype=args.dtype,
    )
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "output": args.output, "success": result["success"],
        "stop_reason": result["search"]["stop_reason"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
