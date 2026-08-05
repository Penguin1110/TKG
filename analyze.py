"""
analyze.py
----------
讀 run_experiment.py 產生的 results.jsonl，算出：

  1. 每個 (model, arm, distance, occurrence) 格子的正確率 + bootstrap 信賴區間
     -> summary.csv，橫軸 occurrence（這個距離被問第幾次）、縱軸 distance，
        conflict arm 就是報告要畫的「輪數(occurrence) x ripple距離」反悔熱力圖
  2. 半衰輪數（half-life occurrence）：conflict arm 正確率第一次掉到 50% 以下
     是第幾次被問到
  3. 有效傳播半徑：固定 occurrence，conflict arm 正確率仍 >= 門檻的最大距離
  4. conflict vs control 的差異顯著性檢定（bootstrap，取代 McNemar——因為兩組
     問的不是同一題，不是配對資料）-> summary_conflict_vs_control.csv，這是
     kill gate 的實際判準：只有 conflict 顯著低於 control，才能說是先驗反悔，
     不是純粹的 lost-in-the-middle

Bootstrap 用純 stdlib random，不引入 numpy/scipy，維持 MVP 少依賴的風格。
"""

import argparse
import csv
import json
import random
from collections import defaultdict

N_BOOTSTRAP = 2000
CI_LOW, CI_HIGH = 0.025, 0.975


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_correct(row) -> bool:
    if row["arm"] == "conflict":
        return row["label"] == "stick_new"
    return row["label"] == "correct"


def bootstrap_ci(successes: int, n: int, rng: random.Random, b: int = N_BOOTSTRAP):
    if n == 0:
        return None, None
    p_hat = successes / n
    samples = []
    for _ in range(b):
        s = sum(1 for _ in range(n) if rng.random() < p_hat)
        samples.append(s / n)
    samples.sort()
    lo = samples[int(CI_LOW * b)]
    hi = samples[min(int(CI_HIGH * b), b - 1)]
    return round(100 * lo, 1), round(100 * hi, 1)


def bootstrap_diff_ci(succ_a: int, n_a: int, succ_b: int, n_b: int, rng: random.Random, b: int = N_BOOTSTRAP):
    """回傳 (a的正確率 - b的正確率) 的 bootstrap 差異信賴區間，用來判斷差異是否顯著（CI 不含 0）"""
    if n_a == 0 or n_b == 0:
        return None, None
    p_a, p_b = succ_a / n_a, succ_b / n_b
    diffs = []
    for _ in range(b):
        sa = sum(1 for _ in range(n_a) if rng.random() < p_a) / n_a
        sb = sum(1 for _ in range(n_b) if rng.random() < p_b) / n_b
        diffs.append(sa - sb)
    diffs.sort()
    lo = diffs[int(CI_LOW * b)]
    hi = diffs[min(int(CI_HIGH * b), b - 1)]
    return round(100 * lo, 1), round(100 * hi, 1)


def pct(successes, n):
    if n == 0:
        return None
    return round(100 * successes / n, 1)


def summarize_cells(rows):
    """key: (model, arm, distance, occurrence) -> (successes, n)"""
    cells = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["slot"] not in ("ripple", "control"):
            continue
        key = (r["model"], r["arm"], r["distance"], r["occurrence"])
        cells[key][1] += 1
        if is_correct(r):
            cells[key][0] += 1
    return cells


def summarize_pk(rows):
    """key: (model, case_id) -> (stick_old_count, n)"""
    pk = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["slot"] != "pk_probe":
            continue
        key = (r["model"], r["case_id"])
        pk[key][1] += 1
        if r["label"] == "stick_old":
            pk[key][0] += 1
    return pk


def half_life_occurrence(cells, model):
    """對每個 distance，找 conflict arm 正確率第一次 < 50% 的 occurrence"""
    by_distance = defaultdict(dict)
    for (m, arm, distance, occurrence), (succ, n) in cells.items():
        if m != model or arm != "conflict" or distance is None:
            continue
        by_distance[distance][occurrence] = pct(succ, n)

    result = {}
    for distance, occ_map in by_distance.items():
        half_life = None
        for occ in sorted(occ_map):
            if occ_map[occ] is not None and occ_map[occ] < 50.0:
                half_life = occ
                break
        result[distance] = half_life if half_life is not None else f">{max(occ_map)}（pilot 內未觀察到反悔）"
    return result


def effective_propagation_radius(cells, model, occurrence_target, threshold=50.0):
    """固定 occurrence，conflict arm 正確率仍 >= threshold 的最大 distance"""
    reachable = []
    for (m, arm, distance, occurrence), (succ, n) in cells.items():
        if m != model or arm != "conflict" or occurrence != occurrence_target or distance is None:
            continue
        p = pct(succ, n)
        if p is not None and p >= threshold:
            reachable.append(distance)
    return max(reachable) if reachable else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results.jsonl")
    parser.add_argument("--output-csv", type=str, default="summary.csv")
    parser.add_argument("--diff-output-csv", type=str, default="summary_conflict_vs_control.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        print(f"{args.input} 裡沒有資料，請先跑 run_experiment.py")
        return

    rng = random.Random(args.seed)
    cells = summarize_cells(rows)
    pk = summarize_pk(rows)
    models = sorted({r["model"] for r in rows})

    print("\n=== PK 探針（舊先驗夠不夠強，kill gate 門檻通常 80%）===")
    print(f"{'model':30s} {'case_id':25s} {'stick_old%':>10s} {'n':>4s}")
    for (model, case_id), (succ, n) in sorted(pk.items()):
        print(f"{model:30s} {case_id:25s} {pct(succ, n) or 0:>10} {n:>4d}")

    print("\n=== 輪數(occurrence) x ripple距離：正確率 + bootstrap 95% CI ===")
    print(f"{'model':25s} {'arm':10s} {'dist':>5s} {'occ':>4s} {'acc%':>6s} {'CI':>15s} {'n':>4s}")
    csv_rows = []
    for (model, arm, distance, occurrence), (succ, n) in sorted(
            cells.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or 0, x[0][3] or 0)):
        acc = pct(succ, n)
        lo, hi = bootstrap_ci(succ, n, rng)
        ci_str = f"[{lo},{hi}]" if lo is not None else "n/a"
        print(f"{model:25s} {arm:10s} {distance or 0:>5} {occurrence or 0:>4} {acc or 0:>6} {ci_str:>15s} {n:>4d}")
        csv_rows.append({"model": model, "arm": arm, "distance": distance, "occurrence": occurrence,
                          "accuracy_pct": acc, "ci_low": lo, "ci_high": hi, "n": n})

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "arm", "distance", "occurrence",
                                                "accuracy_pct", "ci_low", "ci_high", "n"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n已輸出：{args.output_csv}（畫熱力圖：橫軸 occurrence，縱軸 distance，篩 arm==conflict 看反悔曲線）")

    print("\n=== 半衰輪數（conflict arm 正確率第一次掉到 50% 以下是第幾次被問到）===")
    for model in models:
        hl = half_life_occurrence(cells, model)
        for distance in sorted(hl):
            print(f"{model:25s} distance={distance}: half-life occurrence = {hl[distance]}")

    print("\n=== 有效傳播半徑（固定 occurrence=1 / occurrence=max，conflict acc >= 50% 的最大距離）===")
    for model in models:
        occs = sorted({occ for (m, arm, d, occ) in cells if m == model and arm == "conflict" and occ is not None})
        if not occs:
            continue
        r1 = effective_propagation_radius(cells, model, occs[0])
        r_last = effective_propagation_radius(cells, model, occs[-1])
        print(f"{model:25s} occurrence={occs[0]}: radius={r1}   occurrence={occs[-1]}: radius={r_last}")

    print("\n=== conflict vs control 顯著性檢定（kill gate 判準：conflict 顯著低於 control 才算先驗反悔）===")
    print(f"{'model':25s} {'dist':>5s} {'occ':>4s} {'conflict%':>10s} {'control%':>9s} "
          f"{'diff_CI':>16s} {'significant':>12s}")
    diff_rows = []
    keys = {(m, d, o) for (m, arm, d, o) in cells if arm == "conflict" and d is not None}
    for model, distance, occurrence in sorted(keys):
        c_succ, c_n = cells.get((model, "conflict", distance, occurrence), [0, 0])
        k_succ, k_n = cells.get((model, "control", distance, occurrence), [0, 0])
        if c_n == 0 or k_n == 0:
            continue
        c_acc, k_acc = pct(c_succ, c_n), pct(k_succ, k_n)
        lo, hi = bootstrap_diff_ci(c_succ, c_n, k_succ, k_n, rng)
        significant = (lo is not None) and (hi is not None) and (lo > 0 or hi < 0) and hi < 0
        print(f"{model:25s} {distance:>5} {occurrence:>4} {c_acc:>10} {k_acc:>9} "
              f"[{lo},{hi}]{'':>4} {str(significant):>12}")
        diff_rows.append({"model": model, "distance": distance, "occurrence": occurrence,
                           "conflict_acc_pct": c_acc, "control_acc_pct": k_acc,
                           "diff_ci_low": lo, "diff_ci_high": hi, "significant_reversion": significant})

    with open(args.diff_output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "distance", "occurrence", "conflict_acc_pct",
                                                "control_acc_pct", "diff_ci_low", "diff_ci_high",
                                                "significant_reversion"])
        writer.writeheader()
        writer.writerows(diff_rows)
    print(f"已輸出：{args.diff_output_csv}")


if __name__ == "__main__":
    main()
