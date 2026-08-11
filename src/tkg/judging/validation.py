"""Compare the configured LLM judge with a human-labelled JSONL gold set."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from tkg.judging.llm import LLMJudge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold", required=True,
        help=("JSONL: question,response,target_snapshot_as_of,after_answers,"
              "before_answers,pages,human_label"),
    )
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-min-confidence", type=float, default=0.8)
    parser.add_argument("--min-agreement", type=float, default=0.9)
    args = parser.parse_args()

    judge = LLMJudge(args.judge_model, min_confidence=args.judge_min_confidence)
    confusion: Counter[tuple[str, str]] = Counter()
    total = correct = 0
    with open(args.gold, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            decision = judge.judge_temporal_answer(
                item["question"], item["response"], item.get("after_answers", []),
                item.get("before_answers", []), item.get("pages", []),
                target_snapshot_as_of=item.get("target_snapshot_as_of"),
            )
            expected = item["human_label"]
            total += 1
            correct += int(decision.decision == expected)
            confusion[(expected, decision.decision)] += 1
            print(f"line={line_number} human={expected} judge={decision.decision} "
                  f"confidence={decision.confidence:.2f}")
    agreement = correct / total if total else 0.0
    print(f"agreement={agreement:.3f} ({correct}/{total})")
    for (human, predicted), count in sorted(confusion.items()):
        print(f"  human={human:12s} judge={predicted:12s} n={count}")
    if total == 0 or agreement < args.min_agreement:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
