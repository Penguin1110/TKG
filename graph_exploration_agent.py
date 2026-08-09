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

from openrouter_client import call_model_with_tools, OpenRouterError

MAX_NEIGHBORS_SHOWN = 20  # 真實 Wikidata 節點的鄰居可能有上百個，全部列出會爆 token，這裡設實務上限
MAX_FACTS_SHOWN = 30      # 同理，view_current_node() 也設上限

TOOLS = [
    {"type": "function", "function": {
        "name": "list_neighbors",
        "description": "列出目前所在節點可以移動過去的所有相鄰節點。",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "view_current_node",
        "description": "查看目前所在節點的詳細資訊（已知的事實列表）。",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "move_to",
        "description": "移動到指定的相鄰節點。neighbor_id 必須是 list_neighbors() 回傳過的節點 id。",
        "parameters": {
            "type": "object",
            "properties": {"neighbor_id": {"type": "string", "description": "要移動過去的節點 id"}},
            "required": ["neighbor_id"],
        },
    }},
    {"type": "function", "function": {
        "name": "stop_exploring",
        "description": "結束這趟探索，不再繼續移動到其他節點。",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
]


def _format_neighbors(neighbors: list, visited: set) -> str:
    if not neighbors:
        return "（目前節點沒有可移動的相鄰節點）"
    shown = neighbors[:MAX_NEIGHBORS_SHOWN]
    lines = []
    for nb in shown:
        tag = "（已訪問）" if nb["qid"] in visited else ""
        lines.append(f"- id={nb['qid']}｜{nb['label']}｜關係：{nb['property']}{tag}")
    truncated_note = ""
    if len(neighbors) > MAX_NEIGHBORS_SHOWN:
        truncated_note = f"\n（這個節點其實有 {len(neighbors)} 個相鄰節點，這裡只列出前 {MAX_NEIGHBORS_SHOWN} 個）"
    return "\n".join(lines) + truncated_note


def _format_facts(facts: list) -> str:
    if not facts:
        return "（目前節點沒有已知的事實）"
    shown = facts[:MAX_FACTS_SHOWN]
    note = f"\n（總共有 {len(facts)} 條事實，這裡只列出前 {MAX_FACTS_SHOWN} 條）" if len(facts) > MAX_FACTS_SHOWN else ""
    return "\n".join(f"- {f}" for f in shown) + note


def run_free_exploration(model: str, backend, start_qid: str, max_steps: int,
                          task_prompt: str, target_qid: str = None,
                          temperature: float = 0.7,
                          call_model_fn=call_model_with_tools) -> dict:
    """
    跑一趟自由探索。backend 只要有 fetch_node(qid) -> {"qid","label","facts","neighbors"}
    這個介面就行（WikidataGraphBackend 或測試用的 MockGraphBackend 都可以）。

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
        {"role": "user", "content": "開始探索。你可以使用工具查看目前節點、列出鄰居、移動過去，或結束探索。"},
    ]

    stop_reason = "max_steps"
    for step in range(1, max_steps + 1):
        try:
            assistant_msg = call_model_fn(model, messages, TOOLS, temperature=temperature)
        except OpenRouterError as e:
            trajectory.append({"step": step, "from_qid": current_qid, "action": "error",
                                "args": {}, "free_text": None, "result": str(e)})
            stop_reason = "error"
            break

        messages.append(assistant_msg)
        tool_calls = assistant_msg.get("tool_calls") or []
        free_text = assistant_msg.get("content")

        if not tool_calls:
            trajectory.append({"step": step, "from_qid": current_qid, "action": "no_tool_call",
                                "args": {}, "free_text": free_text, "result": ""})
            stop_reason = "no_tool_call"
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
                    result = f"錯誤：{neighbor_id} 不是目前節點的相鄰節點，請重新選擇。"
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
                    result = (f"已移動到「{dest_node['label']}」（{neighbor_id}）。目前已知：\n"
                              f"{_format_facts(dest_node['facts'])}")
                    if target_qid is not None and neighbor_id == target_qid:
                        reached_pivot_this_step = True
            elif name == "stop_exploring":
                result = "探索結束。"
                stopped_this_step = True
            else:
                result = f"錯誤：未知的工具 {name}。"

            trajectory.append({"step": step, "from_qid": origin_qid, "action": name,
                                "args": args, "free_text": free_text, "result": result})
            messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})

        if reached_pivot_this_step:
            stop_reason = "pivot_reached"
            break
        if stopped_this_step:
            stop_reason = "stop_exploring"
            break

    hit = target_qid is not None and target_qid in visited
    return {
        "start_qid": start_qid, "final_qid": current_qid, "hit": hit,
        "stop_reason": stop_reason, "visited_qids": visited_path,
        "trajectory": trajectory, "messages": messages,
    }
