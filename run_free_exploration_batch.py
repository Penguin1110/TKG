"""
run_free_exploration_batch.py
-------------------------------
選項 B 的 conflict arm 主流程：模型在真實 Wikidata 衍生的圖上「自由」探索
（不是舊版的劇本式固定路徑，見 graph_exploration_agent.py），事後才判斷有
沒有「碰巧」走到 pivot 節點。

核心邏輯（對每個 case 的每個 model）：
  1. 對每個 case["pivot_qid"]，用 backend.find_nodes_at_distance() 找出跟
     pivot 剛好距離 1/2/3 步的候選起點池
  2. 每個距離抽 --n-starts-per-distance 個起點
  3. 每個起點跑 --repeats-per-start 趟自由探索（temperature > 0 才有意義，
     同一起點才可能走出不同路徑）
  4. 每趟結束後判斷 hit（有沒有走到 pivot_qid）/ miss：
       hit  -> 從走到 pivot 那一刻的對話記錄開始，接上 run_experiment.py 的
               run_round_schedule()（Line B），跑 ripple/control 追問排程
       miss -> 存檔但不進入 Line B，只用來算自然命中率
  5. 輸出 hit_rate_by_distance.csv——這是要在報告裡呈現的描述性統計，不是
     可以丟掉的副產品

checkpoint 的粒度是「探索批次」層級：(case_id, model, distance, start_qid,
repeat_idx) 這個組合，不是等一整個 case 跑完才記一筆——探索比劇本式貴很多
（多輪 tool-calling + oversample），跑到一半斷線不該整批重來。

輸出寫進獨立的檔案（預設 results_free_exploration.jsonl），不會跟舊版
results.jsonl 混在一起——曝光機制完全不同（自由探索 vs 劇本式固定路徑），
混在一起分析會搞不清楚衰退是先驗反悔還是曝光方式改變造成的（這個教訓
在 README「已踩過的坑」已經有記錄過一次）。
"""

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone

from graph_exploration_agent import run_free_exploration
from run_experiment import build_round_schedule, run_round_schedule, available_distances
from wikidata_graph_backend import WikidataGraphBackend

NEUTRAL_TASK_PROMPT_TEMPLATE = (
    "你正在瀏覽一個知識圖譜。你可以使用工具查看目前節點的資訊、列出相鄰節點、"
    "移動到任何一個相鄰節點，或是覺得逛得差不多了就結束探索。看看你覺得有趣的"
    "資訊，最多可以走 {max_steps} 步。"
)
# 注意：這個 prompt 刻意寫成中性、探索導向，不能改成任何暗示「往哪個方向走」
# 的文字（例如提到 case 的主題、公司名、職位名）。一旦暗示，就不再是「碰巧
# 路過」，整個自然命中率的量測就沒有意義了。


class RestrictedSubgraphBackend:
    """
    --restrict-subgraph-k 用：包一層在真正的 backend 外面，把 list_neighbors()
    看得到、move_to() 走得到的範圍限制在 pivot 的 k-hop 鄰域內，其餘節點不會
    出現。模型還是完全自由選路，只是能走的地圖範圍被縮小——跟「強迫走某條
    路徑」性質不同（模型在縮小後的地圖裡要走哪條路仍然是自己決定的），但要在
    報告裡誠實交代這個選項的取捨（自然命中率會被人為墊高，見 README）。
    """

    def __init__(self, inner_backend, allowed_qids: set):
        self._inner = inner_backend
        self._allowed = allowed_qids

    def fetch_node(self, qid: str) -> dict:
        node = self._inner.fetch_node(qid)
        filtered_neighbors = [nb for nb in node["neighbors"] if nb["qid"] in self._allowed]
        return {**node, "neighbors": filtered_neighbors}


def build_restricted_backend(backend, pivot_qid: str, k: int):
    allowed = {pivot_qid}
    for depth in range(1, k + 1):
        allowed.update(backend.find_nodes_at_distance(pivot_qid, depth, max_depth=k))
    return RestrictedSubgraphBackend(backend, allowed)


def check_pivot_leak(pivot_facts: list, case: dict) -> dict:
    """
    洩漏檢查的新定義（選項 B 版）：只檢查 pivot 節點自己的 facts 有沒有直接
    寫出 ripple 答案，不檢查模型探索到的其他節點——模型自己走到別的節點、
    自然看到下游事實，是正常的多跳推理行為，不是洩漏（見 README 說明）。

    回傳 {distance: [命中的關鍵字,...]}，空 dict 代表沒查到洩漏。
    """
    text = " ".join(pivot_facts).lower()
    leaks = {}
    for distance, items in case.get("ripples", {}).items():
        hits = []
        for item in items:
            for kw in item.get("new_keywords", []) + item.get("old_keywords", []):
                if kw and len(kw) > 2 and kw.lower() in text:
                    hits.append(kw)
        if hits:
            leaks[distance] = hits
    return leaks


def find_pre_seen_ripple_distances(trajectory: list, case: dict, pivot_qid: str) -> list:
    """
    掃探索過程中所有 view_current_node() 的結果（排除 pivot 節點自己，那個已經
    用 check_pivot_leak 另外把關），看模型是不是在正式被問 ripple 問題之前，
    自己就已經在探索中看過某個距離的答案關鍵字了。這不是要阻止或懲罰，是
    要記錄下來當一個分析維度（見 README「洩漏檢查的語意重新定義」）。
    """
    seen = set()
    for step in trajectory:
        if step["action"] != "view_current_node" or step["from_qid"] == pivot_qid:
            continue
        text = step["result"].lower()
        for distance, items in case.get("ripples", {}).items():
            for item in items:
                for kw in item.get("new_keywords", []):
                    if kw and len(kw) > 2 and kw.lower() in text:
                        seen.add(int(distance))
    return sorted(seen)


def _write_row(fh, case_id, model, arm, round_idx, slot, distance, occurrence,
               question, response, label, repeat_idx, extra=None):
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id, "model": model, "arm": arm, "repeat": repeat_idx,
        "round": round_idx, "slot": slot, "distance": distance, "occurrence": occurrence,
        "question": question, "response": response, "label": label,
    }
    if extra:
        row.update(extra)
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()


def _write_explore_checkpoint(fh, case_id, model, arm, distance, start_qid, repeat_idx):
    _write_row(fh, case_id, model, arm=arm, round_idx=0, slot="free_explore_checkpoint",
               distance=distance, occurrence=None, question=None, response=None,
               label="done", repeat_idx=repeat_idx, extra={"start_qid": start_qid})


def load_completed_explorations(path: str) -> set:
    """回傳已經完成的 (case_id, model, arm, distance, start_qid, repeat_idx) 組合集合。
    conflict/control 兩個 arm 的 checkpoint 都用同一份檔案存也不會撞號——arm 是
    key 的一部分。"""
    completed = set()
    if not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("slot") == "free_explore_checkpoint":
                completed.add((row["case_id"], row["model"], row["arm"], row["distance"],
                               row["start_qid"], row["repeat"]))
    return completed


def run_one_exploration(model, backend, start_qid, pivot_qid, max_steps, temperature,
                         case, arm, distance, repeat_idx, log_fh, call_model_fn=None):
    """跑一趟自由探索 + 記錄逐步 trajectory，回傳這趟的結果 dict（含 hit/miss）。"""
    kwargs = {}
    if call_model_fn is not None:
        kwargs["call_model_fn"] = call_model_fn
    result = run_free_exploration(
        model=model, backend=backend, start_qid=start_qid, max_steps=max_steps,
        task_prompt=NEUTRAL_TASK_PROMPT_TEMPLATE.format(max_steps=max_steps),
        target_qid=pivot_qid, temperature=temperature, **kwargs,
    )

    for step in result["trajectory"]:
        _write_row(log_fh, case["id"], model, arm=arm, round_idx=0, slot="free_explore",
                   distance=distance, occurrence=step["step"], question=None,
                   response=step.get("free_text"), label=step["action"], repeat_idx=repeat_idx,
                   extra={"start_qid": start_qid, "from_qid": step["from_qid"],
                          "args": step.get("args"), "result": step.get("result")})

    # pre-seen 分析只對 conflict arm 有意義（control 沒有 ripples 可以被「提早看到」）
    pre_seen = (find_pre_seen_ripple_distances(result["trajectory"], case, pivot_qid)
                if (arm == "conflict" and result["hit"]) else [])
    _write_row(log_fh, case["id"], model, arm=arm, round_idx=0, slot="free_explore_summary",
               distance=distance, occurrence=None, question=None, response=None,
               label="hit" if result["hit"] else "miss", repeat_idx=repeat_idx,
               extra={"start_qid": start_qid, "final_qid": result["final_qid"],
                      "stop_reason": result["stop_reason"],
                      "n_steps": len(result["trajectory"]),
                      "pre_seen_ripple_distances": pre_seen})
    return result


def run_case_model(case, model, backend, args, log_fh, completed, rng, hit_counts,
                    arm="conflict", pivot_qid=None, walker_factory=None, skip_leak_check=False):
    """
    arm="conflict" 時用 case["pivot_qid"]、接 Line B 的 case["ripples"]；
    arm="control" 時呼叫端要另外算好 control pivot_qid 傳進來（見
    run_control_exploration_batch.py），接 Line B 的 case["control"]，
    也不需要做洩漏檢查（control 事實本來就不是秘密，見該檔案說明）。

    walker_factory(backend, start_qid, rng) -> call_model_fn：只有 --dry-run 會傳，
    每個 (start_qid, repeat_idx) 都要重新 call 一次拿全新實例（各自獨立的隨機
    路徑），不能在多趟探索之間共用同一個 walker——它有內部狀態（目前在哪個節點）。
    正式模式（非 dry-run）不傳，run_one_exploration 就會用真的
    openrouter_client.call_model_with_tools。
    """
    case_id = case["id"]
    pivot_qid = pivot_qid or case.get("pivot_qid")
    if not pivot_qid:
        print(f"[skip] case={case_id} arm={arm} 沒有 pivot_qid，跳過")
        return

    if not skip_leak_check:
        pivot_node = backend.fetch_node(pivot_qid)
        leaks = check_pivot_leak(pivot_node["facts"], case)
        if leaks:
            print(f"[error] case={case_id} 的 pivot 節點 facts 裡直接含有 ripple 答案關鍵字：{leaks}"
                  f"——這代表 Wikidata 上這個節點的敘述本身就洩題，不能用來測傳播推論，跳過這個 case")
            return

    explore_backend = backend
    if args.restrict_subgraph_k:
        explore_backend = build_restricted_backend(backend, pivot_qid, args.restrict_subgraph_k)

    distances = [int(d) for d in args.distances.split(",") if d.strip()]
    for distance in distances:
        candidates = backend.find_nodes_at_distance(pivot_qid, distance, max_depth=max(distances))
        if not candidates:
            print(f"[warn] case={case_id} arm={arm} distance={distance} 找不到候選起點，跳過這個距離")
            continue
        n_pick = min(args.n_starts_per_distance, len(candidates))
        starts = rng.sample(candidates, n_pick)
        if n_pick < args.n_starts_per_distance:
            print(f"[warn] case={case_id} arm={arm} distance={distance} 候選起點只有 {len(candidates)} 個，"
                  f"少於要求的 {args.n_starts_per_distance} 個")

        for start_qid in starts:
            for repeat_idx in range(args.repeats_per_start):
                key = (case_id, model, arm, distance, start_qid, repeat_idx)
                if key in completed:
                    print(f"[checkpoint] 跳過 {key}（已完成）")
                    hit_counts[(model, case_id, arm, distance)]["from_checkpoint"] += 1
                    continue

                call_model_fn = walker_factory(explore_backend, start_qid, rng) if walker_factory else None
                result = run_one_exploration(
                    model, explore_backend, start_qid, pivot_qid, args.max_steps,
                    args.temperature, case, arm, distance, repeat_idx, log_fh,
                    call_model_fn=call_model_fn,
                )
                hit_counts[(model, case_id, arm, distance)]["attempts"] += 1
                if result["hit"]:
                    hit_counts[(model, case_id, arm, distance)]["hits"] += 1
                    composite_repeat = f"d{distance}_{start_qid}_{repeat_idx}"
                    round_schedule = build_round_schedule(
                        available_distances(case), args.rounds, args.distractor_every, rng)
                    run_round_schedule(case, model, arm, args.distractors, round_schedule,
                                        log_fh, rng, composite_repeat, result["messages"])

                _write_explore_checkpoint(log_fh, case_id, model, arm, distance, start_qid, repeat_idx)


def write_hit_rate_csv(path, hit_counts):
    rows = []
    for (model, case_id, arm, distance), counts in sorted(hit_counts.items()):
        hits = counts["hits"]  # 只有這次新跑的才知道 hit/miss；checkpoint 裡沒重算舊的
        rate = round(100 * hits / counts["attempts"], 1) if counts["attempts"] else None
        rows.append({"model": model, "case_id": case_id, "arm": arm, "distance": distance,
                      "n_attempts_this_run": counts["attempts"], "n_hits_this_run": hits,
                      "hit_rate_pct_this_run": rate,
                      "n_skipped_via_checkpoint": counts["from_checkpoint"]})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "case_id", "arm", "distance", "n_attempts_this_run",
                                                "n_hits_this_run", "hit_rate_pct_this_run",
                                                "n_skipped_via_checkpoint"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"已輸出：{path}")


def main():
    parser = argparse.ArgumentParser(description="選項 B：自由探索版 conflict arm")
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--cases", type=str, default="cases.json")
    parser.add_argument("--case-ids", type=str, default=None)
    parser.add_argument("--output", type=str, default="results_free_exploration.jsonl")
    parser.add_argument("--hit-rate-output", type=str, default="hit_rate_by_distance.csv")
    parser.add_argument("--distances", type=str, default="1,2,3")
    parser.add_argument("--n-starts-per-distance", type=int, default=5)
    parser.add_argument("--repeats-per-start", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--distractor-every", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-path", type=str, default="tkg_cache.db")
    parser.add_argument("--restrict-subgraph-k", type=int, default=0,
                         help="0 表示關閉（預設，完全自由探索）。>0 時只把 pivot 的 k-hop "
                              "鄰域開放給模型探索，會人為墊高自然命中率，報告裡要誠實交代")
    parser.add_argument("--dry-run", action="store_true",
                         help="不打真的 Wikidata/OpenRouter API，用內建的 mock 圖跟 mock 模型"
                              "跑一次完整流程（起點選擇→探索→命中判斷→接上 Line B→checkpoint）")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.dry_run:
        import run_experiment
        cases, backend, walker_factory = build_dryrun_fixtures()
        args.distractors = ["mock distractor question 1?", "mock distractor question 2?"]
        run_experiment.call_model = make_plain_mock_call_model()  # Line B 的純文字問答也要 mock 掉
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
        backend = WikidataGraphBackend(cache_path=args.cache_path)
        walker_factory = None

    completed = load_completed_explorations(args.output)
    if completed:
        print(f"[checkpoint] 從 {args.output} 讀到 {len(completed)} 個已完成的探索組合，會跳過")

    from collections import defaultdict
    hit_counts = defaultdict(lambda: {"attempts": 0, "hits": 0, "from_checkpoint": 0})

    with open(args.output, "a", encoding="utf-8") as log_fh:
        for model in models:
            for case in cases:
                rng = random.Random(args.seed + hash(case["id"]) % 997 + hash(model) % 991)
                run_case_model(case, model, backend, args, log_fh, completed, rng, hit_counts,
                                walker_factory=walker_factory)

    write_hit_rate_csv(args.hit_rate_output, hit_counts)
    print(f"完成，結果寫入 {args.output}")


def build_dryrun_fixtures():
    """
    --dry-run 用的自足 fixture：不碰真的 cases.json/Wikidata/OpenRouter，
    回傳 (cases, backend, walker_factory)。

    Mock 圖刻意設計成分支因子小、pivot 在中心，讓「距離 1 的起點隨機走幾步就
    走到 pivot」的機率明顯高於「距離 3 的起點」——這是驗收標準要求的
    「距離1命中率應該高於距離3」的結構性保證，不是碰運氣。
    """
    from mock_graph_fixtures import MockGraphBackend, build_mock_graph, build_mock_case

    graph = build_mock_graph()
    backend = MockGraphBackend(graph)
    case = build_mock_case()

    def walker_factory(backend_, start_qid, rng):
        return ScriptedRandomWalker(backend_, start_qid, rng)

    return [case], backend, walker_factory


def make_plain_mock_call_model():
    """dry-run 時拿來蓋掉 run_experiment.call_model 的純文字 mock，讓 Line B 的
    ripple/control 追問也能在不打 API 的情況下跑完整條流程。"""
    def _mock(model, messages, temperature=0.0, **kwargs):
        return "(dry-run mock response，沒有真的打 API)"
    return _mock


class ScriptedRandomWalker:
    """
    dry-run 專用的假「模型」：每步用 rng 隨機選一個鄰居移動過去，模擬自由探索
    （不是真的 LLM，純粹讓流程機制可以在沒有 API 的情況下被驗證）。每次探索
    嘗試都要建一個新的實例（各自獨立的隨機路徑），不要在多趟探索之間共用。
    """

    def __init__(self, backend, start_qid, rng, stop_probability=0.05, min_steps_before_stop=2):
        self.backend = backend
        self.current_qid = start_qid
        self.rng = rng
        self.stop_probability = stop_probability
        self.min_steps_before_stop = min_steps_before_stop
        self.call_count = 0

    def __call__(self, model, messages, tools, temperature=0.7):
        self.call_count += 1
        if self.call_count > self.min_steps_before_stop and self.rng.random() < self.stop_probability:
            return {"role": "assistant", "content": "逛得差不多了。",
                     "tool_calls": [{"id": f"c{self.call_count}", "type": "function",
                                      "function": {"name": "stop_exploring", "arguments": "{}"}}]}
        node = self.backend.fetch_node(self.current_qid)
        if not node["neighbors"]:
            return {"role": "assistant", "content": "沒有路可以走了。",
                     "tool_calls": [{"id": f"c{self.call_count}", "type": "function",
                                      "function": {"name": "stop_exploring", "arguments": "{}"}}]}
        target = self.rng.choice(node["neighbors"])["qid"]
        self.current_qid = target
        return {"role": "assistant", "content": f"移動到 {target}。",
                 "tool_calls": [{"id": f"c{self.call_count}", "type": "function",
                                  "function": {"name": "move_to",
                                               "arguments": json.dumps({"neighbor_id": target})}}]}


if __name__ == "__main__":
    main()
