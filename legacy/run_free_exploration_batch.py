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

# uv run 之類的包裝工具會讓子行程的 stdout 變成 pipe，不是真的 tty，Python
# 偵測到不是 tty 就會切成整塊緩衝（block buffering）——print() 呼叫的內容
# 會一直卡在緩衝區裡，不會即時顯示，看起來就像整個程式卡住了（實測發生過）。
# 這裡強制轉成一行一行 flush，不管在哪種執行環境下都能即時看到進度。
sys.stdout.reconfigure(line_buffering=True)

from legacy.graph_exploration_agent import run_free_exploration
from legacy.run_experiment import (
    available_distances,
    build_round_schedule,
    run_pk_probe,
    run_round_schedule,
)
from legacy.wikidata_graph_backend import WikidataGraphBackend, bfs_frontier
from tkg.api.openrouter import OpenRouterError, call_model

NEUTRAL_TASK_PROMPT_TEMPLATE = (
    "You are browsing a knowledge graph. You can use tools to view the current "
    "node's information, list its neighboring nodes, move to any neighboring "
    "node, or end the exploration whenever you feel you've seen enough. Look "
    "around for anything you find interesting. You may take at most {max_steps} "
    "steps.\n\n"
    "Important: this is a fully automated process — there is no human on the "
    "other end reading your messages or replying to you. Each turn, you must "
    "call a tool to keep acting. Do not describe what you see in plain text, "
    "and do not ask what to do next or wait for confirmation — decide for "
    "yourself what to do and just call the corresponding tool. Only call "
    "stop_exploring() when you genuinely want to end the exploration."
)
# 注意：這個 prompt 刻意寫成中性、探索導向，不能改成任何暗示「往哪個方向走」
# 的文字（例如提到 case 的主題、公司名、職位名）。一旦暗示，就不再是「碰巧
# 路過」，整個自然命中率的量測就沒有意義了。
#
# 踩過的坑（第一次大規模跑之前的小規模驗證發現）：
#   1. 拿掉「這是全自動流程、不用反問」這段之前，gpt-4.1-mini 常常在看完一個
#      節點的內容後，不呼叫任何工具，反而用文字描述剛剛看到什麼、然後反問
#      「你想看哪個？」——這其實是它把這個任務當成一般對話在回應，以為有
#      真人會回覆。因為沒有真人接話，run_free_exploration() 判定成
#      no_tool_call 提早結束探索，實測占了將近一半的失敗案例，直接拖累命中率、
#      浪費大量 API 額度在 2-3 步就中止的短軌跡上。
#   2. 整個探索過程（這個 prompt、下面的 TRANSITION_PROMPT、
#      graph_exploration_agent.py 裡所有工具描述跟回傳訊息）原本是中文寫的，
#      Line B 的 ripple/control 問題則是英文——實測發現模型因為前面大量中文
#      context，Line B 常常整段用中文回答，導致 judge.py 全英文的關鍵字／
#      hedge 判斷完全比對不到，一大堆回應被誤判成 "other"（明明內容看得出來
#      是先驗反悔或正確回答，只是judge抓不到）。改成整個探索過程、轉場訊息、
#      工具描述都用英文之後，跟 Line B 的英文問題語言一致，才解決。

TRANSITION_PROMPT = (
    "Your exploration has ended and the exploration tools are no longer "
    "available. I'm now going to ask you a few questions directly — please "
    "answer using everything you know, not only what you personally saw while "
    "exploring the graph. You can also draw on your own prior knowledge, "
    "inference, or your best guess if you're not certain. You don't need to "
    "have seen something in the graph to answer it."
)
# 踩過的坑 1（第一次真的跑）：探索階段是 tool-calling（call_model_with_tools），
# Line B 的追問是純文字（call_model()，不帶 tools 參數）。如果直接把探索結束時
# 的 messages 原封不動接上 Line B 的第一個問題，模型會因為看到自己前面一直在
# 用工具、現在卻沒有 tools 可用而卡住，回答變成「請讓我先查看 XX 節點的資訊」
# 這種話，不會真的回答問題。
#
# 踩過的坑 2（修完坑 1 之後又踩到）：加了轉場訊息之後，模型不再要求用工具了，
# 但變成回答「根據目前已知的資訊，我沒有相關資料，因此無法回答」——這不是
# 先驗反悔訊號，是轉場訊息的措辭「根據你目前已知的資訊」被模型理解成「只能用
# 探索時親眼看到的內容」，把自己原本的訓練知識也一起關掉了。這違背了 Line B
# 的設計初衷（跟選項 A 一樣，ripple 問題本來就是要模型用自己的知識＋剛學到的
# pivot fact 去推論，不是要它把答案寫在探索過程裡讓模型照抄）。改成明確告訴
# 模型「不是只能用圖上看到的，你自己原本知道的知識都可以用」之後才解決。


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
    掃探索過程中所有「有揭露 facts 內容」的步驟結果（view_current_node()，
    以及 move_to()——現在 move_to 也會自動帶出目的地節點的 facts，模擬真的
    點連結會看到頁面內容，見 graph_exploration_agent.py 的說明），排除 pivot
    節點自己被看到的那次（那個已經用 check_pivot_leak() 另外把關），看模型
    是不是在正式被問 ripple 問題之前，自己就已經在探索中看過某個距離的答案
    關鍵字了。這不是要阻止或懲罰，是要記錄下來當一個分析維度（見 README
    「洩漏檢查的語意重新定義」）。

    排除 pivot 節點的判斷：view_current_node() 看的是 step["from_qid"]（那一步
    當下所在的節點）；move_to() 揭露的是「移動過去的目的地」，也就是
    step["args"]["neighbor_id"]，不是 from_qid（from_qid 在 move_to 的情況下
    是移動前的起點，跟被揭露內容的節點不是同一個，混用會把 pivot 自己的內容
    誤判成別的節點洩漏，或反過來漏判）。
    """
    seen = set()
    for step in trajectory:
        if step["action"] == "view_current_node":
            revealed_qid = step["from_qid"]
        elif step["action"] == "move_to" and "錯誤" not in step["result"]:
            revealed_qid = step.get("args", {}).get("neighbor_id")
        else:
            continue
        if revealed_qid == pivot_qid:
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


def _apply_line_b_transition(model: str, messages: list, plain_call_model_fn) -> bool:
    """
    探索結束、要接上 Line B 之前，用一輪純文字明確告訴模型「探索結束了，
    接下來直接用你已知的資訊回答，沒有工具可以用」，並取得它的回覆——這一輪
    會被附加進 messages，讓 Line B 的第一個問題不是緊接在最後一次 tool 回應
    後面。回傳 False 代表這輪呼叫失敗，呼叫端不該再接 Line B。

    沒有這一步的話，模型會因為看到自己前面一直在用工具、現在卻沒有 tools
    可用而卡住，回答變成「請讓我先查看 XX 節點的資訊」這種話，不會真的
    回答問題——這是實測第一次真的跑選項 B 時發現的，見這個檔案開頭的
    TRANSITION_PROMPT 說明。
    """
    messages.append({"role": "user", "content": TRANSITION_PROMPT})
    try:
        response = plain_call_model_fn(model, messages)
    except OpenRouterError as e:
        print(f"[error] Line B 轉場失敗：{e}")
        return False
    messages.append({"role": "assistant", "content": response})
    return True


def run_one_exploration(model, backend, start_qid, pivot_qid, max_steps, temperature,
                         case, arm, distance, repeat_idx, log_fh, call_model_fn=None,
                         plain_call_model_fn=call_model):
    """跑一趟自由探索 + 記錄逐步 trajectory，回傳這趟的結果 dict（含 hit/miss/line_b_ready）。"""
    kwargs = {}
    if call_model_fn is not None:
        kwargs["call_model_fn"] = call_model_fn
    result = run_free_exploration(
        model=model, backend=backend, start_qid=start_qid, max_steps=max_steps,
        task_prompt=NEUTRAL_TASK_PROMPT_TEMPLATE.format(max_steps=max_steps),
        target_qid=pivot_qid, temperature=temperature, **kwargs,
    )

    result["line_b_ready"] = False
    if result["hit"]:
        result["line_b_ready"] = _apply_line_b_transition(model, result["messages"], plain_call_model_fn)

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


PK_PROBE_REPEATS = 5


def run_pk_probes_if_needed(case, model, log_fh, completed, arm):
    """
    選項 A（run_experiment.py）本來就有 PK 探針當 kill gate；選項 B 一開始
    漏掉了這一步（第一次真的跑才發現：沒有 PK 探針，完全不知道這次測到的
    模型對這個 pivot 事實的先驗夠不夠強，反悔宣稱沒有立足點）。這裡補上，
    只在 conflict arm 跑（PK 探針測的是對 pivot fact 本身的先驗，跟 control
    arm 無關）。用跟 free_explore_checkpoint 一樣的機制記錄完成度，重跑時
    已經測過的不會重測。
    """
    if arm != "conflict":
        return
    case_id = case["id"]
    pk_labels = []
    for repeat_idx in range(PK_PROBE_REPEATS):
        key = (case_id, model, arm, "pk", "pk", repeat_idx)
        if key in completed:
            print(f"[checkpoint] 跳過 PK 探針 {key}（已完成）")
            continue
        label = run_pk_probe(case, model, log_fh, repeat_idx)
        pk_labels.append(label)
        _write_explore_checkpoint(log_fh, case_id, model, arm, "pk", "pk", repeat_idx)
    if pk_labels:
        rate = pk_labels.count("stick_old") / len(pk_labels)
        threshold = case.get("pk_threshold", 0.8)
        verdict = "OK" if rate >= threshold else "WARN 先驗不夠強，建議從分析中剔除"
        print(f"[kill-gate:pk] case={case_id} model={model} stick_old_rate={rate:.2f} "
              f"(threshold={threshold}, 只算這次新跑的) -> {verdict}")


def run_case_model(case, model, backend, args, log_fh, completed, rng, hit_counts,
                    arm="conflict", pivot_qid=None, walker_factory=None, skip_leak_check=False,
                    plain_call_model_fn=call_model):
    """
    arm="conflict" 時用 case["pivot_qid"]、接 Line B 的 case["ripples"]；
    arm="control" 時呼叫端要另外算好 control pivot_qid 傳進來（見
    run_control_exploration_batch.py），接 Line B 的 case["control"]，
    也不需要做洩漏檢查（control 事實本來就不是秘密，見該檔案說明）。

    walker_factory(backend, start_qid, rng) -> call_model_fn：只有 --dry-run 會傳，
    每個 (start_qid, repeat_idx) 都要重新 call 一次拿全新實例（各自獨立的隨機
    路徑），不能在多趟探索之間共用同一個 walker——它有內部狀態（目前在哪個節點）。
    正式模式（非 dry-run）不傳，run_one_exploration 就會用真的
    openrouter_client.call_model_with_tools。plain_call_model_fn 同理，是給
    PK 探針/Line B 轉場這種純文字呼叫用的（--dry-run 會換成 mock）。
    """
    case_id = case["id"]
    pivot_qid = pivot_qid or case.get("pivot_qid")
    if not pivot_qid:
        print(f"[skip] case={case_id} arm={arm} 沒有 pivot_qid，跳過")
        return

    run_pk_probes_if_needed(case, model, log_fh, completed, arm)

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
    # 一次把 BFS 展開到 max(distances)，再從同一份 frontier 切各個距離的候選池，
    # 不要對每個距離各自呼叫 find_nodes_at_distance()——那樣每次都會重新展開到
    # max(distances) 深度，雖然有 sqlite 快取讓後面幾次變快，但仍然是三倍重工，
    # 也會讓 bfs 進度訊息重複印三次，徒增困惑。
    print(f"[explore] case={case_id} arm={arm} 開始找候選起點池（distances={distances}）...")
    frontier = bfs_frontier(explore_backend.fetch_node, pivot_qid, max(distances))
    for distance in distances:
        candidates = frontier.get(distance, [])
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

                print(f"=== case={case_id} model={model} arm={arm} distance={distance} "
                      f"start={start_qid} repeat={repeat_idx} ===")
                call_model_fn = walker_factory(explore_backend, start_qid, rng) if walker_factory else None
                result = run_one_exploration(
                    model, explore_backend, start_qid, pivot_qid, args.max_steps,
                    args.temperature, case, arm, distance, repeat_idx, log_fh,
                    call_model_fn=call_model_fn, plain_call_model_fn=plain_call_model_fn,
                )
                hit_counts[(model, case_id, arm, distance)]["attempts"] += 1
                if result["hit"]:
                    hit_counts[(model, case_id, arm, distance)]["hits"] += 1
                    if result["line_b_ready"]:
                        composite_repeat = f"d{distance}_{start_qid}_{repeat_idx}"
                        round_schedule = build_round_schedule(
                            available_distances(case), args.rounds, args.distractor_every, rng)
                        run_round_schedule(case, model, arm, args.distractors, round_schedule,
                                            log_fh, rng, composite_repeat, result["messages"])
                    else:
                        print(f"[warn] {case_id}/{model}/{arm} 這趟 hit 了但 Line B 轉場失敗，"
                              f"沒有追問資料（探索本身的 hit/miss 統計仍然有效）")

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
    parser.add_argument("--offline", action="store_true",
                         help="只用本地快取（build_tkg_snapshot.py 先建好），快取沒有的節點"
                              "直接報錯、不會悄悄即時打 Wikidata API。跑正式實驗前建議先開這個，"
                              "確保整組模型看到的是同一份固定快照，不受 Wikidata 中途被編輯影響")
    parser.add_argument("--restrict-subgraph-k", type=int, default=0,
                         help="0 表示關閉（預設，完全自由探索）。>0 時只把 pivot 的 k-hop "
                              "鄰域開放給模型探索，會人為墊高自然命中率，報告裡要誠實交代")
    parser.add_argument("--dry-run", action="store_true",
                         help="不打真的 Wikidata/OpenRouter API，用內建的 mock 圖跟 mock 模型"
                              "跑一次完整流程（起點選擇→探索→命中判斷→接上 Line B→checkpoint）")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.dry_run:
        from legacy import run_experiment
        cases, backend, walker_factory = build_dryrun_fixtures()
        args.distractors = ["mock distractor question 1?", "mock distractor question 2?"]
        plain_mock = make_plain_mock_call_model()
        run_experiment.call_model = plain_mock  # PK 探針/Line B 的純文字問答也要 mock 掉
        plain_call_model_fn = plain_mock
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

    from collections import defaultdict
    hit_counts = defaultdict(lambda: {"attempts": 0, "hits": 0, "from_checkpoint": 0})

    with open(args.output, "a", encoding="utf-8") as log_fh:
        for model in models:
            for case in cases:
                rng = random.Random(args.seed + hash(case["id"]) % 997 + hash(model) % 991)
                run_case_model(case, model, backend, args, log_fh, completed, rng, hit_counts,
                                walker_factory=walker_factory, plain_call_model_fn=plain_call_model_fn)

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
    from legacy.mock_graph_fixtures import MockGraphBackend, build_mock_graph, build_mock_case

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
