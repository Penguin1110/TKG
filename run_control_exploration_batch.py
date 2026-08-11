"""
run_control_exploration_batch.py
-----------------------------------
選項 B 的 control arm。跟 run_free_exploration_batch.py 共用幾乎所有機制
（起點池抽樣、hit/miss 判斷、checkpoint、接上 Line B），這裡直接 import
重用那些函式（run_case_model()/run_one_exploration()/checkpoint 相關），
不重新寫一份——兩邊唯一該不一樣的地方只有「pivot 怎麼選」跟「要不要做
洩漏檢查」，其他都是同一套機制。

**跟舊設計最大的差異：control arm 現在是「結構配對」，不是「路徑配對」**。
舊版（劇本式路徑）可以讓 conflict/control 走一模一樣的 5 步、逐字對應；
自由探索下沒辦法讓 control arm 走「完全一樣的路徑」——因為兩邊的起點、圖
結構本來就不同。改成配對兩邊的「探索預算與起點距離結構」：一樣的
--n-starts-per-distance、--repeats-per-start、--max-steps、一樣的距離
1/2/3 起點池大小。

**為什麼這樣配對仍然合理**：kill gate 要排除的是「距離／步數造成的通用
衰退」（lost-in-the-middle），不是要證明兩邊問了逐字相同的問題。只要兩邊
的「曝光要走幾步」「追問距離分佈」是對稱的，這個排除目的就達到了——
control arm 如果也在同樣的步數/距離結構下衰退，就代表看到的是通用衰退，
不是先驗反悔專屬現象；如果 control 沒有衰退而 conflict 有，才能宣稱是
先驗反悔。這跟「兩邊問一模一樣的題目」是兩個不同的必要條件，此處放棄的
是後者（自由探索下做不到），保留的是前者（kill gate 真正需要的）。

control pivot 選取邏輯（fetch_wikidata_pivots.py）：
    找「同類型、瀏覽量相近、但近期沒有更替」的穩定實體。目前的實作是在
    case["control_pivot_candidates"]（人工整理的候選 QID 清單）裡面篩選，
    不是自動查詢全 Wikidata——這是誠實的取捨，見 fetch_wikidata_pivots.py
    開頭的說明。也可以用 --control-pivot-qid 直接指定，跳過自動篩選。
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

# 見 run_free_exploration_batch.py 同一行的說明：強制 stdout 一行一行 flush，
# 避免 uv run 之類的包裝工具把 stdout 變成整塊緩衝，導致進度訊息卡在緩衝區
# 不會即時顯示。
sys.stdout.reconfigure(line_buffering=True)

from fetch_wikidata_pivots import find_stable_control_pivot, get_wikipedia_title, fetch_pageviews
from openrouter_client import call_model
from run_free_exploration_batch import (
    load_completed_explorations, run_case_model, write_hit_rate_csv,
    build_dryrun_fixtures, make_plain_mock_call_model,
)
from wikidata_graph_backend import WikidataGraphBackend


def resolve_control_pivot(backend, case: dict, conflict_position_label: str,
                           tolerance_pct: float, min_stable_years: float, override_qid: str = None):
    """
    回傳 control pivot 的 qid，找不到回傳 None。override_qid 有給就直接用，
    不做自動篩選（適合已經人工驗證過的案例）。
    """
    if override_qid:
        return override_qid

    candidates = case.get("control_pivot_candidates", [])
    if not candidates:
        print(f"[skip] case={case['id']} 沒有 control_pivot_candidates，也沒給 --control-pivot-qid，"
              f"沒辦法選 control pivot")
        return None

    pivot_qid = case.get("pivot_qid")
    conflict_title = get_wikipedia_title(pivot_qid) if pivot_qid else None
    target_pageviews = fetch_pageviews(conflict_title) if conflict_title else None
    if target_pageviews is None:
        print(f"[warn] case={case['id']} 抓不到 conflict pivot 的瀏覽量，"
              f"control pivot 篩選會跳過瀏覽量比對這一關（is_pageview_similar 對 None 回傳 True）")

    return find_stable_control_pivot(
        backend, candidates, target_pageviews, conflict_position_label,
        tolerance_pct=tolerance_pct, min_stable_years=min_stable_years,
    )


def main():
    parser = argparse.ArgumentParser(description="選項 B：自由探索版 control arm（結構配對，非路徑配對）")
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--cases", type=str, default="cases.json")
    parser.add_argument("--case-ids", type=str, default=None)
    parser.add_argument("--output", type=str, default="results_free_exploration.jsonl",
                         help="預設跟 conflict arm 寫同一份檔案，這樣 analyze.py 才能同時看到兩邊"
                              "（checkpoint key 有含 arm，不會互相覆蓋，見 run_free_exploration_batch.py）")
    parser.add_argument("--hit-rate-output", type=str, default="hit_rate_by_distance_control.csv")
    parser.add_argument("--distances", type=str, default="1,2,3")
    parser.add_argument("--n-starts-per-distance", type=int, default=5)
    parser.add_argument("--repeats-per-start", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--distractor-every", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-path", type=str, default="tkg_cache.db")
    parser.add_argument("--offline", action="store_true",
                         help="圖探索只用本地快取，快取沒有的節點直接報錯（見 build_tkg_snapshot.py）。"
                              "注意：--control-pivot-qid 沒指定時的自動篩選（pageviews 比對）不算在"
                              "圖探索快取範圍內，仍然需要連線；要完全離線請搭配 --control-pivot-qid")
    parser.add_argument("--restrict-subgraph-k", type=int, default=0)
    parser.add_argument("--control-pivot-qid", type=str, default=None,
                         help="直接指定某個 case 的 control pivot（單一 case 用；多 case 批次跑時"
                              "請改在 cases.json 的 control_pivot_candidates 裡準備候選池）")
    parser.add_argument("--pageview-tolerance-pct", type=float, default=50.0)
    parser.add_argument("--min-stable-years", type=float, default=3.0)
    parser.add_argument("--position-property-label", type=str, default="chief executive officer",
                         help="fallback 用：case 沒有自己標 control_position_property_label 時才會用"
                              "這個值。例如 \"chief executive officer\"、\"head coach\"、\"chairperson\"、"
                              "\"head of state\"，用來判斷 control 候選是不是「近期沒有更替」。多 case "
                              "批次跑時每個 case 的職位屬性通常不一樣，建議在 cases.json 標好，不要"
                              "只靠這個全域參數")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.dry_run:
        import run_experiment
        cases, backend, walker_factory = build_dryrun_fixtures()
        args.distractors = ["mock distractor question 1?", "mock distractor question 2?"]
        plain_mock = make_plain_mock_call_model()
        run_experiment.call_model = plain_mock
        plain_call_model_fn = plain_mock
        # dry-run 的 mock case 直接把 pivot_qid 借來當 control pivot 用（mock 圖本來
        # 就不區分 conflict/control 語意，重點是驗證機制，不是驗證 pageviews 篩選邏輯）
        for c in cases:
            c["_dryrun_control_pivot_qid"] = c["pivot_qid"]
    else:
        with open(args.cases, "r", encoding="utf-8") as f:
            data = json.load(f)
        cases = data["cases"]
        if args.case_ids:
            wanted = set(args.case_ids.split(","))
            cases = [c for c in cases if c["id"] in wanted]
        args.distractors = data["distractor_questions"]
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("[error] 請先在 .env 或環境變數設定 OPENROUTER_API_KEY", file=sys.stderr)
            sys.exit(1)
        backend = WikidataGraphBackend(cache_path=args.cache_path, offline_only=args.offline)
        walker_factory = None
        plain_call_model_fn = call_model

    completed = load_completed_explorations(args.output)
    if completed:
        print(f"[checkpoint] 從 {args.output} 讀到 {len(completed)} 個已完成的探索組合，會跳過")

    hit_counts = defaultdict(lambda: {"attempts": 0, "hits": 0, "from_checkpoint": 0})

    with open(args.output, "a", encoding="utf-8") as log_fh:
        for model in models:
            for case in cases:
                if args.dry_run:
                    control_pivot_qid = case["_dryrun_control_pivot_qid"]
                else:
                    # 不同 case 的 pivot 職位屬性不一樣（CEO/head coach/chairperson/
                    # head of state...），所以 property label 優先讀 case 自己標好的
                    # control_position_property_label，--position-property-label 只是
                    # 沒標的 case 的 fallback 預設值，不能所有 case 共用同一個當真值
                    position_label = case.get("control_position_property_label",
                                               args.position_property_label)
                    control_pivot_qid = resolve_control_pivot(
                        backend, case, position_label,
                        args.pageview_tolerance_pct, args.min_stable_years,
                        override_qid=args.control_pivot_qid,
                    )
                if not control_pivot_qid:
                    continue

                rng = random.Random(args.seed + hash(case["id"]) % 997 + hash(model) % 991)
                run_case_model(case, model, backend, args, log_fh, completed, rng, hit_counts,
                                arm="control", pivot_qid=control_pivot_qid,
                                walker_factory=walker_factory, skip_leak_check=True,
                                plain_call_model_fn=plain_call_model_fn)

    write_hit_rate_csv(args.hit_rate_output, hit_counts)
    print(f"完成，結果寫入 {args.output}")


if __name__ == "__main__":
    main()
