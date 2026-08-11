"""
legacy.analyze
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
  5. occ1→occ2 配對轉移分析：同一個 (case, model, repeat, distance) 軌跡裡，
     occurrence=1 答對、occurrence=2 答錯的比例，才是真正「held then dropped」
     的反悔事件——單看某個 occurrence 的正確率跟理論上限（100%）的差距，沒辦法
     分辨「曝光當下就沒記住」跟「記住了但後來反悔」這兩種情況；occ1→occ2 是
     同一段對話裡的配對資料，這裡才真的能用 McNemar exact test（reversion vs
     recovery 兩種不對稱轉移的差異）-> summary_paired_transitions.csv

Bootstrap 用純 stdlib random，不引入 numpy/scipy，維持 MVP 少依賴的風格。
"""

import argparse
import csv
import json
import math
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


def summarize_pre_seen(rows):
    """
    給選項 B（自由探索）的 results_free_exploration.jsonl 用：在 hit 的探索
    軌跡裡，模型是不是在正式被問到某個距離的 ripple 問題之前，探索過程中就
    已經自己看過那個距離的答案關鍵字了（見 run_free_exploration_batch.py 的
    find_pre_seen_ripple_distances()）。這不是要排除或懲罰，是記錄一個
    「這條軌跡的追問結果，有多少可能只是重複探索時已經看過的東西」分析維度。

    只有 slot == "free_explore_summary" 且 label == "hit" 的資料列才有這個
    欄位；讀舊版 results.jsonl（劇本式路徑）不會有這種列，回傳空 dict。

    key: (model, case_id) -> {"n_hits": int, "n_with_pre_seen": int,
                               "pre_seen_by_distance": {distance: count}}
    """
    result = defaultdict(lambda: {"n_hits": 0, "n_with_pre_seen": 0, "pre_seen_by_distance": defaultdict(int)})
    for r in rows:
        if r.get("slot") != "free_explore_summary" or r.get("label") != "hit":
            continue
        key = (r["model"], r["case_id"])
        result[key]["n_hits"] += 1
        pre_seen = r.get("pre_seen_ripple_distances") or []
        if pre_seen:
            result[key]["n_with_pre_seen"] += 1
            for d in pre_seen:
                result[key]["pre_seen_by_distance"][d] += 1
    return result


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


def paired_transitions(rows):
    """
    對每個 (case_id, model, repeat, distance) 軌跡，把 conflict arm 依 occurrence
    排序後，看「連續兩次被問到同一個距離」之間的轉移，分類成四種：
      stable_correct   這次對、下次也對
      reversion        這次對、下次錯（真正的「持有過又反悔」）
      recovery         這次錯、下次對（反過來，修正）
      stable_incorrect 這次錯、下次也錯
    軌跡若被問到超過 2 次（例如 occurrence 1~5），會產生多筆相鄰轉移
    （1→2、2→3、3→4、4→5 各算一筆），不是只看 occurrence=1 跟 2——round
    schedule 拉長之後，這是配對樣本數的主要來源。這些觀測不是完全獨立
    （同一條軌跡貢獻好幾筆），是為了在小樣本 pilot 下盡量用滿現有資料的
    簡化，解讀時要記得這一點。

    回傳 {model: {"stable_correct": n, "reversion": n, "recovery": n, "stable_incorrect": n}}
    """
    traj = defaultdict(dict)
    for r in rows:
        if r["slot"] != "ripple":
            continue
        key = (r["case_id"], r["model"], r["repeat"], r["distance"])
        traj[key][r["occurrence"]] = (r["label"] == "stick_new")

    result = defaultdict(lambda: defaultdict(int))
    for (case_id, model, repeat, distance), occs in traj.items():
        model_result = result[model]
        occ_sorted = sorted(occs)
        for prev_occ, next_occ in zip(occ_sorted, occ_sorted[1:]):
            c1, c2 = occs[prev_occ], occs[next_occ]
            if c1 and c2:
                t = "stable_correct"
            elif c1 and not c2:
                t = "reversion"
            elif not c1 and c2:
                t = "recovery"
            else:
                t = "stable_incorrect"
            model_result[t] += 1
    return result


def _binomial_pmf(k, n, p=0.5):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def mcnemar_exact_p(n_reversion, n_recovery):
    """
    McNemar exact test：只看 reversion / recovery 這兩種不對稱轉移（discordant
    pairs），檢定「occ1→occ2 往壞的方向掉」是不是顯著多於「往好的方向修正」。
    stable_correct / stable_incorrect 這兩種一致的轉移不影響檢定，跟標準
    McNemar 定義一致。虛無假設：兩種轉移發生機率各半（Binomial(n, 0.5)）。
    """
    n = n_reversion + n_recovery
    if n == 0:
        return None
    k = min(n_reversion, n_recovery)
    pk = _binomial_pmf(k, n)
    total = sum(_binomial_pmf(i, n) for i in range(n + 1) if _binomial_pmf(i, n) <= pk + 1e-12)
    return min(total, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results.jsonl")
    parser.add_argument("--output-csv", type=str, default="summary.csv")
    parser.add_argument("--diff-output-csv", type=str, default="summary_conflict_vs_control.csv")
    parser.add_argument("--paired-output-csv", type=str, default="summary_paired_transitions.csv")
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

    print("\n=== occ1→occ2 配對轉移分析（同一段對話裡，真正的 held-then-dropped 反悔事件）===")
    print(f"{'model':25s} {'stable_correct':>14s} {'reversion':>10s} {'recovery':>9s} "
          f"{'stable_incorrect':>16s} {'reversion_rate%':>16s} {'mcnemar_p':>10s}")
    trans = paired_transitions(rows)
    paired_rows = []
    for model in models:
        t = trans.get(model, {})
        sc, rev, rec, si = t.get("stable_correct", 0), t.get("reversion", 0), t.get("recovery", 0), t.get("stable_incorrect", 0)
        held = sc + rev  # occ1 答對的軌跡數，反悔率的分母
        rev_rate = round(100 * rev / held, 1) if held else None
        p = mcnemar_exact_p(rev, rec)
        p_str = f"{p:.4f}" if p is not None else "n/a"
        print(f"{model:25s} {sc:>14d} {rev:>10d} {rec:>9d} {si:>16d} "
              f"{rev_rate if rev_rate is not None else 0:>16} {p_str:>10s}")
        paired_rows.append({"model": model, "stable_correct": sc, "reversion": rev, "recovery": rec,
                             "stable_incorrect": si, "reversion_rate_pct": rev_rate, "mcnemar_p": p})

    with open(args.paired_output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "stable_correct", "reversion", "recovery",
                                                "stable_incorrect", "reversion_rate_pct", "mcnemar_p"])
        writer.writeheader()
        writer.writerows(paired_rows)
    print(f"已輸出：{args.paired_output_csv}（reversion_rate% = occ1 答對的軌跡裡，occ2 又答錯的比例；"
          f"mcnemar_p 是 reversion vs recovery 兩種轉移是否對稱的 exact test，p 越小代表越不對稱、"
          f"往反悔方向掉得越顯著）")

    pre_seen = summarize_pre_seen(rows)
    if pre_seen:
        print("\n=== 選項B專用：hit 軌跡裡，模型是否在被正式問到之前就已經探索看過答案 ===")
        print(f"{'model':25s} {'case_id':20s} {'n_hits':>7s} {'n_with_pre_seen':>16s} {'by_distance':>20s}")
        for (model, case_id), stats in sorted(pre_seen.items()):
            by_dist = dict(stats["pre_seen_by_distance"])
            print(f"{model:25s} {case_id:20s} {stats['n_hits']:>7d} "
                  f"{stats['n_with_pre_seen']:>16d} {str(by_dist):>20s}")


if __name__ == "__main__":
    main()
