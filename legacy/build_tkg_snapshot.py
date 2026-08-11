"""
build_tkg_snapshot.py
------------------------
把 cases.json 裡的 pivot_qid / control_pivot_candidates 當種子，對 Wikidata
做有界 BFS，把探索會用到的節點鄰域預先抓下來、存進本地 sqlite 快取
（tkg_cache.db）。目的：

  1. 正式跑 run_free_exploration_batch.py / run_control_exploration_batch.py
     時可以完全離線（配合 WikidataGraphBackend(offline_only=True)），不受
     Wikidata 即時 API 的速率限制、連線問題、或請求當下網站抽風影響
  2. 同一份快照可以重複用在多個模型的實驗上，確保「同一次 pilot 裡每個模型
     看到的圖是同一份」，不會因為 Wikidata 在兩次實驗之間被編輯過而讓不同
     模型實際上測的不是同一個問題

**快照不是自動保鮮的**：Wikidata 隨時有人在編輯，快照建好那一刻就開始跟
「現在的 Wikidata」產生落差。這個工具用 --verify 模式處理這個問題：對
種子節點（pivot_qid，不含衍生出來的鄰居）重新即時查一次，比對跟快照裡的
版本有沒有差異，把落差印出來讓你自己判斷要不要重建——不會自動幫你決定
「差異不大所以沒關係」，因為多大算「不大」是研究判斷，不是工程判斷。

用法：
    # 建快照（種子 = 所有 case 的 pivot_qid + control_pivot_candidates）
    uv run python -m legacy.build_tkg_snapshot --max-depth 3 --branch-cap 25

    # 只驗新鮮度，不重建（比對種子節點現在 vs 快照裡的版本）
    uv run python -m legacy.build_tkg_snapshot --verify
"""

import argparse
import json
import time
from datetime import datetime, timezone

from legacy.wikidata_graph_backend import DEFAULT_BRANCH_CAP, WikidataGraphBackend, bfs_frontier

DEFAULT_MANIFEST_PATH = "tkg_snapshot_manifest.json"


def collect_seed_qids(cases: list, include_control_candidates: bool = True) -> list:
    seeds = []
    for case in cases:
        if case.get("pivot_qid"):
            seeds.append(case["pivot_qid"])
        if include_control_candidates:
            seeds.extend(case.get("control_pivot_candidates", []))
    # 保留原始順序但去重
    seen = set()
    deduped = []
    for qid in seeds:
        if qid not in seen:
            seen.add(qid)
            deduped.append(qid)
    return deduped


def build(args):
    with open(args.cases, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"]
    if args.case_ids:
        wanted = set(args.case_ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    seeds = collect_seed_qids(cases, include_control_candidates=not args.pivots_only)
    if not seeds:
        print("[error] 沒有找到任何 pivot_qid / control_pivot_candidates，沒有種子可以建快照")
        return

    print(f"種子節點（{len(seeds)} 個）：{seeds}")
    backend = WikidataGraphBackend(cache_path=args.cache_path)

    before = _count_cached(args.cache_path)
    t0 = time.time()
    per_seed_counts = {}
    for qid in seeds:
        print(f"\n=== 展開 {qid}（max_depth={args.max_depth}, branch_cap={args.branch_cap}）===")
        frontier = bfs_frontier(backend.fetch_node, qid, args.max_depth, branch_cap=args.branch_cap)
        total_this_seed = sum(len(v) for v in frontier.values())
        per_seed_counts[qid] = {str(d): len(v) for d, v in frontier.items()}
        for depth, qids in sorted(frontier.items()):
            print(f"  distance={depth}: {len(qids)} 個節點")
        print(f"  這個種子總共走訪 {total_this_seed} 個節點")

        # bfs_frontier() 只有在「展開一個節點去找它的鄰居」時才會呼叫
        # fetch_node()，最外層（distance == max_depth）的節點只是被鄰居清單
        # 提到、本身沒被抓過——如果不額外補抓，之後離線探索時模型一走到這層
        # 就會因為快取沒有而報錯。這裡把最外層也明確抓一次，確保整個 frontier
        # 涵蓋的節點都真的有自己的 facts/neighbors 進快取。
        outermost = frontier.get(args.max_depth, [])
        if outermost:
            print(f"  補抓最外層（distance={args.max_depth}）的 {len(outermost)} 個節點本身的內容...")
            for nb_qid in outermost:
                try:
                    backend.fetch_node(nb_qid)
                except Exception as e:  # noqa: BLE001 - 個別節點抓失敗不該讓整個 snapshot 中斷
                    print(f"    [warn] {nb_qid} 抓失敗，跳過：{e}")

    elapsed = time.time() - t0
    after = _count_cached(args.cache_path)
    backend.close()

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "cache_path": args.cache_path,
        "cases_file": args.cases,
        "seed_qids": seeds,
        "max_depth": args.max_depth,
        "branch_cap": args.branch_cap,
        "nodes_in_cache_before": before,
        "nodes_in_cache_after": after,
        "nodes_newly_cached": after - before,
        "build_seconds": round(elapsed, 1),
        "per_seed_frontier_sizes": per_seed_counts,
    }
    with open(args.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n完成。快取裡現在有 {after} 個節點（這次新增 {after - before} 個），"
          f"花了 {elapsed:.1f} 秒。manifest 寫入 {args.manifest_path}")
    print("接下來可以用 WikidataGraphBackend(offline_only=True) 完全離線跑實驗，"
          "或用 --verify 檢查種子節點現在有沒有變動。")


def verify(args):
    try:
        with open(args.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        print(f"[error] 找不到 {args.manifest_path}，請先跑一次不帶 --verify 的建快照流程")
        return

    seeds = manifest["seed_qids"]
    built_at = manifest["built_at"]
    print(f"快照建立於 {built_at}，種子節點 {len(seeds)} 個，逐一重新即時查證：\n")

    backend = WikidataGraphBackend(cache_path=manifest["cache_path"])
    changed_qids = []
    for qid in seeds:
        result = backend.refresh_node(qid)
        status = "沒有變動" if not result["changed"] else "⚠ 有變動"
        print(f"  {qid}: {status}"
              f"（快照裡 {result['old_record_count']} 條 claim -> 現在 {result['new_record_count']} 條）")
        if result["changed"]:
            changed_qids.append(qid)
    backend.close()

    print()
    if changed_qids:
        print(f"⚠ {len(changed_qids)}/{len(seeds)} 個種子節點自快照建立後有變動："
              f"{changed_qids}")
        print("這幾個節點的快取已經被 refresh_node() 更新成最新版本了（其他還沒重新"
              "驗證過的鄰居節點還是舊快照）。如果變動的是 pivot_qid 本身，建議重新確認"
              "對應案例的 old/new_answer_keywords 是否還準確，必要時重跑"
              "build_tkg_snapshot.py 整個重建。")
    else:
        print(f"全部 {len(seeds)} 個種子節點都沒有變動，快照仍然準確。")


def _count_cached(cache_path: str) -> int:
    import sqlite3
    try:
        conn = sqlite3.connect(cache_path)
        count = conn.execute("SELECT COUNT(*) FROM node_cache").fetchone()[0]
        conn.close()
        return count
    except sqlite3.OperationalError:
        return 0


def main():
    parser = argparse.ArgumentParser(description="用真實 Wikidata 建立本地 TKG 快照")
    parser.add_argument("--cases", type=str, default="cases.json")
    parser.add_argument("--case-ids", type=str, default=None,
                         help="只用指定 case 的 pivot 當種子（逗號分隔），不填就用全部")
    parser.add_argument("--pivots-only", action="store_true",
                         help="種子只用 pivot_qid，不含 control_pivot_candidates")
    parser.add_argument("--cache-path", type=str, default="tkg_cache.db")
    parser.add_argument("--manifest-path", type=str, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--max-depth", type=int, default=3,
                         help="從每個種子往外展開幾層（對應 ripple distance 1-3 的需求）")
    parser.add_argument("--branch-cap", type=int, default=DEFAULT_BRANCH_CAP,
                         help="每個節點展開下一層時最多走幾個鄰居，避免熱門節點"
                              "（國家/大型組織）組合爆炸，見 wikidata_graph_backend.py 說明")
    parser.add_argument("--verify", action="store_true",
                         help="不重建快照，只對 manifest 裡的種子節點重新即時查證有沒有變動")
    args = parser.parse_args()

    if args.verify:
        verify(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
