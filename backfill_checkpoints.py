"""
backfill_checkpoints.py
------------------------
一次性工具：run_experiment.py 加上 checkpoint 機制之前跑出來的 results.jsonl
沒有 checkpoint 紀錄，run_experiment.py 沒辦法知道哪些 (case, model, repeat, arm)
已經跑過，重跑就會整組重花錢。這個腳本掃過現有資料，用「精確重算當初的
round_schedule、比對實際列數是否剛好對得上」的方式判斷一個 arm 是否真的完整跑完，
完整的話就把對應的 checkpoint 紀錄補寫回 results.jsonl。

判斷邏輯：
  - conflict arm 完整 = 有 1 筆 pk_probe（label 不是 error）+ 5 筆 explore + 1 筆
    exposure + round_schedule 展開後預期的 ripple/distractor 列數，一列不多不少
  - control arm 完整 = 5 筆 explore + 1 筆 exposure + round_schedule 展開後預期的
    control/distractor 列數，一列不多不少
  - round_schedule 用跟 run_experiment.py 一樣的 build_round_schedule() 重算
    （需要跟當初那次 run_experiment.py 一樣的 --seed/--rounds/--distractor-every，
    預設值 42/6/3，跟目前 README 範例指令一致；如果你當初跑的時候有另外指定這幾個
    參數，這裡也要對應改）

執行：
    python3 backfill_checkpoints.py --input results.jsonl
"""

import argparse
import json
import random
from collections import defaultdict

from run_experiment import build_round_schedule, available_distances, _write_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results.jsonl")
    parser.add_argument("--cases", type=str, default="cases.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--distractor-every", type=int, default=3)
    args = parser.parse_args()

    with open(args.cases, "r", encoding="utf-8") as f:
        cases_by_id = {c["id"]: c for c in json.load(f)["cases"]}

    with open(args.input, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    already_checkpointed = {
        (r["case_id"], r["model"], r["repeat"], r["arm"])
        for r in rows if r["slot"] == "checkpoint"
    }

    # groups[(case_id, model, repeat, arm)][slot] = 實際列數
    groups = defaultdict(lambda: defaultdict(int))
    pk_ok = set()  # (case_id, model, repeat) 有沒有一筆 label != error 的 pk_probe
    for r in rows:
        if r["slot"] == "checkpoint":
            continue
        key = (r["case_id"], r["model"], r["repeat"], r["arm"])
        groups[key][r["slot"]] += 1
        if r["slot"] == "pk_probe" and r["label"] != "error":
            pk_ok.add((r["case_id"], r["model"], r["repeat"]))

    to_backfill = []
    for (case_id, model, repeat, arm), slot_counts in groups.items():
        if (case_id, model, repeat, arm) in already_checkpointed:
            continue
        case = cases_by_id.get(case_id)
        if case is None:
            continue

        distances = available_distances(case)
        rng = random.Random(args.seed + repeat * 1000 + hash(case_id) % 997)
        schedule = build_round_schedule(distances, args.rounds, args.distractor_every, rng)

        fact_pool = case["ripples"] if arm == "conflict" else case["control"]
        expected_distractor = sum(1 for s in schedule if s == "distractor")
        expected_fact = sum(1 for s in schedule if s != "distractor" and fact_pool.get(str(s)))
        expected_slot_name = "ripple" if arm == "conflict" else "control"

        ok = (slot_counts.get("explore", 0) == 5
              and slot_counts.get("exposure", 0) == 1
              and slot_counts.get("distractor", 0) == expected_distractor
              and slot_counts.get(expected_slot_name, 0) == expected_fact)
        if arm == "conflict":
            ok = ok and (case_id, model, repeat) in pk_ok

        if ok:
            to_backfill.append((case_id, model, repeat, arm))
        else:
            print(f"[skip] {case_id}/{model}/repeat={repeat}/{arm} 看起來不完整，不補 checkpoint"
                  f"（實際 {dict(slot_counts)}，預期 explore=5 exposure=1 distractor={expected_distractor} "
                  f"{expected_slot_name}={expected_fact}）")

    if not to_backfill:
        print("沒有需要補的 checkpoint。")
        return

    with open(args.input, "a", encoding="utf-8") as fh:
        for case_id, model, repeat, arm in to_backfill:
            _write_checkpoint(fh, case_id, model, arm, repeat)

    print(f"補回 {len(to_backfill)} 筆 checkpoint 到 {args.input}：")
    for k in to_backfill:
        print("  ", k)


if __name__ == "__main__":
    main()
