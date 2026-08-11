"""
graph_exploration_agent.py
----------------------------
取代舊版 run_experiment.run_graph_exploration()（劇本式固定路徑）：這裡讓模型
用 OpenRouter 的 tool-calling 介面，在圖上真的自己決定要走去哪個鄰居節點，
事後才判斷這趟軌跡有沒有「碰巧」走到 pivot 節點（見 run_free_exploration_batch.py
的 hit/miss 篩選邏輯）。

跟舊版最大的差別：模型的選擇會真的影響下一步看到什麼。這是刻意的架構轉向
（見 README「選項 A vs 選項 B」章節），不是延續舊版「排除導航能力」的設計。

四個工具：
    list_neighbors()      列出目前節點可以移動過去的相鄰節點（已訪問過的會標註）
    view_current_node()   查看目前節點的 facts
    move_to(neighbor_id)  移動過去（必須是真的相鄰節點，否則回傳錯誤讓模型重選，
                           不會靜默失敗）
    stop_exploring()      模型主動結束這趟探索

任務指令（task_prompt）刻意寫成中性、探索導向，不能暗示要往哪裡走——一旦
暗示，就不再是「碰巧路過」，整個「自然命中率」的量測就沒有意義了。這個
prompt 由呼叫端（run_free_exploration_batch.py）傳入，這裡不寫死，但呼叫端
的 prompt 也必須遵守這條規則（見那個檔案的說明）。

終止條件（三選一，都會記錄在回傳的 stop_reason）：
    "stop_exploring" - 模型自己呼叫 stop_exploring()
    "pivot_reached"  - 移動到 target_qid（通常是 pivot 節點）
                        --一旦碰到就停止，不繼續往後走，原因：(1) 省 API 呼叫
                        （繼續走的軌跡反正事後會被截斷、丟掉）(2) 讓「從碰到
                        pivot 的那個時間點接上 Line B」這件事有明確、一致的
                        訊息邊界，不會因為模型碰到後又多繞幾步而讓不同軌跡的
                        「曝光後過了幾輪」不可比
    "max_steps"      - 走到 --max-steps 上限（環狀圖的保護，避免無窮繞圈）
    "no_tool_call"   - 模型這輪沒有呼叫任何工具（視為它選擇不再行動）
    "error"          - API 呼叫失敗
"""

import json

from tkg.api.openrouter import OpenRouterError, call_model_with_tools

MAX_NEIGHBORS_SHOWN = 20  # 真實 Wikidata 節點的鄰居可能有上百個，全部列出會爆 token，這裡設實務上限
MAX_FACTS_SHOWN = 30      # 同理，view_current_node() 也設上限

TOOLS = [
    {"type": "function", "function": {
        "name": "list_neighbors",
        "description": "List all neighboring nodes you can move to from the current node.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "view_current_node",
        "description": "View detailed information about the current node (its known facts).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "move_to",
        "description": "Move to the specified neighboring node. neighbor_id must be a node id "
                       "previously returned by list_neighbors().",
        "parameters": {
            "type": "object",
            "properties": {"neighbor_id": {"type": "string", "description": "The id of the node to move to"}},
            "required": ["neighbor_id"],
        },
    }},
    {"type": "function", "function": {
        "name": "stop_exploring",
        "description": "End this exploration session; stop moving to other nodes.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
]


def _format_neighbors(neighbors: list, visited: set) -> str:
    if not neighbors:
        return "(This node has no outgoing neighbors.)"
    shown = neighbors[:MAX_NEIGHBORS_SHOWN]
    lines = []
    for nb in shown:
        tag = " (visited)" if nb["qid"] in visited else ""
        lines.append(f"- id={nb['qid']} | {nb['label']} | relation: {nb['property']}{tag}")
    truncated_note = ""
    if len(neighbors) > MAX_NEIGHBORS_SHOWN:
        truncated_note = f"\n(This node actually has {len(neighbors)} neighbors; only the first {MAX_NEIGHBORS_SHOWN} are shown.)"
    return "\n".join(lines) + truncated_note


def _format_facts(facts: list) -> str:
    if not facts:
        return "(No known facts for this node.)"
    shown = facts[:MAX_FACTS_SHOWN]
    note = f"\n(There are {len(facts)} facts in total; only the first {MAX_FACTS_SHOWN} are shown.)" if len(facts) > MAX_FACTS_SHOWN else ""
    return "\n".join(f"- {f}" for f in shown) + note


def run_free_exploration(model: str, backend, start_qid: str, max_steps: int,
                          task_prompt: str, target_qid: str = None,
                          temperature: float = 0.7,
                          call_model_fn=call_model_with_tools,
                          verbose: bool = True) -> dict:
    """
    跑一趟自由探索。backend 只要有 fetch_node(qid) -> {"qid","label","facts","neighbors"}
    這個介面就行（WikidataGraphBackend 或測試用的 MockGraphBackend 都可以）。

    verbose=True（預設）會逐步印出每一步呼叫了哪個工具、結果摘要——真的打
    Wikidata/OpenRouter 時，單一趟探索可能要跑到 max_steps 步，中間完全沒有
    輸出會讓人以為程式卡住了（實測發生過），所以預設開著；測試/dry-run 場景
    輸出多幾行也無傷大雅，不需要另外關掉。

    回傳：
        {
          "start_qid": str, "final_qid": str, "hit": bool,
          "stop_reason": "stop_exploring"|"pivot_reached"|"max_steps"|"no_tool_call"|"error",
          "visited_qids": [qid,...]（含重複，走過幾次記幾次，用來看有沒有走回頭路）,
          "trajectory": [{"step":int,"from_qid":str,"action":str,"args":dict,
                           "free_text":str|None,"result":str}],
          "messages": [...] （完整對話歷史，hit 的話可以直接接上 Line B 的 round schedule）
        }
    """
    current_qid = start_qid
    visited = {start_qid}
    visited_path = [start_qid]
    trajectory = []

    messages = [
        {"role": "system", "content": task_prompt},
        {"role": "user", "content": "Start exploring. You can use tools to view the current node, "
                                      "list its neighbors, move to a neighbor, or end the exploration."},
    ]

    def _log(msg):
        if verbose:
            print(msg)

    _log(f"[explore] 開始，起點={start_qid}，最多 {max_steps} 步")

    stop_reason = "max_steps"
    for step in range(1, max_steps + 1):
        _log(f"[explore step {step}/{max_steps}] 目前在 {current_qid}，等待模型回應...")
        try:
            assistant_msg = call_model_fn(model, messages, TOOLS, temperature=temperature)
        except OpenRouterError as e:
            trajectory.append({"step": step, "from_qid": current_qid, "action": "error",
                                "args": {}, "free_text": None, "result": str(e)})
            stop_reason = "error"
            _log(f"[explore step {step}] 呼叫失敗：{e}")
            break

        messages.append(assistant_msg)
        tool_calls = assistant_msg.get("tool_calls") or []
        free_text = assistant_msg.get("content")

        if not tool_calls:
            trajectory.append({"step": step, "from_qid": current_qid, "action": "no_tool_call",
                                "args": {}, "free_text": free_text, "result": ""})
            stop_reason = "no_tool_call"
            _log(f"[explore step {step}] 模型沒有呼叫任何工具，視為結束探索")
            break

        stopped_this_step = False
        reached_pivot_this_step = False
        for tool_call in tool_calls:
            fn = tool_call["function"]
            name = fn["name"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            origin_qid = current_qid
            node = backend.fetch_node(current_qid)

            if name == "list_neighbors":
                result = _format_neighbors(node["neighbors"], visited)
            elif name == "view_current_node":
                result = _format_facts(node["facts"])
            elif name == "move_to":
                neighbor_id = args.get("neighbor_id")
                valid_ids = {nb["qid"] for nb in node["neighbors"]}
                if neighbor_id not in valid_ids:
                    result = f"Error: {neighbor_id} is not a neighbor of the current node. Please choose again."
                else:
                    current_qid = neighbor_id
                    visited.add(neighbor_id)
                    visited_path.append(neighbor_id)
                    # 移動的同時直接把目的地節點的 facts 帶出來，模擬真的點連結進到
                    # 一個頁面會直接看到頁面內容（不用再點一次「查看」）。踩過的坑：
                    # 原本 move_to 只回「已移動到 X」，不會自動帶出 facts，導致模型
                    # 走到 pivot 節點卻從來沒有實際「看到」pivot fact 本身就被判定
                    # hit、直接進 Line B——測到的其實是「模型根本沒被曝光」，不是先驗
                    # 反悔。真的曝光需要模型親眼看過內容，不是單純路徑上經過而已。
                    dest_node = backend.fetch_node(neighbor_id)
                    result = (f"Moved to \"{dest_node['label']}\" ({neighbor_id}). Currently known:\n"
                              f"{_format_facts(dest_node['facts'])}")
                    if target_qid is not None and neighbor_id == target_qid:
                        reached_pivot_this_step = True
            elif name == "stop_exploring":
                result = "Exploration ended."
                stopped_this_step = True
            else:
                result = f"Error: unknown tool {name}."

            trajectory.append({"step": step, "from_qid": origin_qid, "action": name,
                                "args": args, "free_text": free_text, "result": result})
            messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})

            result_preview = result.replace("\n", " ")[:80]
            _log(f"[explore step {step}] {name}({args if args else ''}) -> {result_preview}"
                 f"{'...' if len(result) > 80 else ''}")

        if reached_pivot_this_step:
            stop_reason = "pivot_reached"
            _log(f"[explore] 走到 pivot 節點 {target_qid} 了，停止探索、準備接上 Line B")
            break
        if stopped_this_step:
            stop_reason = "stop_exploring"
            _log("[explore] 模型自己結束了探索")
            break

    hit = target_qid is not None and target_qid in visited
    _log(f"[explore] 結束：hit={hit}, stop_reason={stop_reason}, 走了 {len(trajectory)} 步, "
         f"最終節點={current_qid}")

    return {
        "start_qid": start_qid, "final_qid": current_qid, "hit": hit,
        "stop_reason": stop_reason, "visited_qids": visited_path,
        "trajectory": trajectory, "messages": messages,
    }
