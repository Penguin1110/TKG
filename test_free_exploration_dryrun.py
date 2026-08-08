"""
test_free_exploration_dryrun.py
----------------------------------
選項 B（自由探索）的驗收測試，全部用 mock（見 mock_graph_fixtures.py），
不打真的 Wikidata/OpenRouter API。驗證這幾件事：

  1. bfs_distance() 算對（用 mock 圖，跟 find_nodes_at_distance() 一起測）
  2. 探索終止條件：模型呼叫 stop_exploring()、或走到 max_steps 都會停
  3. 環狀圖不會無窮迴圈
  4. hit/miss 判斷正確，且 miss 的軌跡不會被送進 Line B 排程
  5. 洩漏檢查只檢查 pivot 節點自己的 facts，不會誤判模型探索到的下游節點

執行：
    python3 test_free_exploration_dryrun.py
"""

import os

from mock_graph_fixtures import MockGraphBackend, build_mock_graph, build_cyclic_graph, build_mock_case, PIVOT_QID
from graph_exploration_agent import run_free_exploration
from run_free_exploration_batch import check_pivot_leak, find_pre_seen_ripple_distances


# ---------- 1. bfs_distance / find_nodes_at_distance ----------

def test_bfs_distance_correct():
    backend = MockGraphBackend(build_mock_graph())
    assert backend.bfs_distance(PIVOT_QID, PIVOT_QID) == 0
    assert backend.bfs_distance(PIVOT_QID, "D1_A") == 1
    assert backend.bfs_distance(PIVOT_QID, "D1_B") == 1
    assert backend.bfs_distance(PIVOT_QID, "D2_A") == 2
    assert backend.bfs_distance(PIVOT_QID, "D3_A") == 3
    assert backend.bfs_distance(PIVOT_QID, "D3_B") == 3
    assert backend.bfs_distance(PIVOT_QID, "NOT_IN_GRAPH", max_depth=3) is None

    d1_pool = backend.find_nodes_at_distance(PIVOT_QID, 1)
    assert set(d1_pool) == {"D1_A", "D1_B"}
    d3_pool = backend.find_nodes_at_distance(PIVOT_QID, 3)
    assert set(d3_pool) == {"D3_A", "D3_B"}
    print("[OK] test_bfs_distance_correct")


# ---------- 2. 終止條件 ----------

class _ScriptedModel:
    """依序回放固定的 assistant 訊息，用來精準控制測試情境（不是隨機走）。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def __call__(self, model, messages, tools, temperature=0.7):
        msg = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return msg


def _move_to(qid, call_id="c"):
    return {"role": "assistant", "content": None,
             "tool_calls": [{"id": call_id, "type": "function",
                              "function": {"name": "move_to", "arguments": f'{{"neighbor_id": "{qid}"}}'}}]}


def _stop(call_id="c"):
    return {"role": "assistant", "content": "結束", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "stop_exploring", "arguments": "{}"}}]}


def test_termination_stop_exploring():
    backend = MockGraphBackend(build_mock_graph())
    model = _ScriptedModel([_move_to("D1_A"), _stop()])
    result = run_free_exploration("m", backend, PIVOT_QID, max_steps=10, task_prompt="t",
                                    target_qid=None, call_model_fn=model)
    assert result["stop_reason"] == "stop_exploring"
    assert len(result["trajectory"]) == 2
    print("[OK] test_termination_stop_exploring")


def test_termination_max_steps():
    backend = MockGraphBackend(build_mock_graph())
    # 一直在 pivot 跟 D1_A 之間來回，永遠不呼叫 stop_exploring、也碰不到 target
    model = _ScriptedModel([_move_to("D1_A"), _move_to(PIVOT_QID)])
    result = run_free_exploration("m", backend, PIVOT_QID, max_steps=5, task_prompt="t",
                                    target_qid=None, call_model_fn=model)
    assert result["stop_reason"] == "max_steps"
    assert len(result["trajectory"]) == 5
    print("[OK] test_termination_max_steps")


def test_no_tool_call_terminates():
    backend = MockGraphBackend(build_mock_graph())
    model = _ScriptedModel([{"role": "assistant", "content": "我不知道要做什麼"}])
    result = run_free_exploration("m", backend, PIVOT_QID, max_steps=10, task_prompt="t",
                                    target_qid=None, call_model_fn=model)
    assert result["stop_reason"] == "no_tool_call"
    assert len(result["trajectory"]) == 1
    print("[OK] test_no_tool_call_terminates")


# ---------- 3. 環狀圖不會無窮迴圈 ----------

def test_cyclic_graph_terminates():
    backend = MockGraphBackend(build_cyclic_graph())
    # 一直繞圈：A -> B -> C -> A -> B -> C -> ...，沒有 target、沒有 stop_exploring
    model = _ScriptedModel([_move_to("B"), _move_to("C"), _move_to("A")])
    result = run_free_exploration("m", backend, "A", max_steps=8, task_prompt="t",
                                    target_qid=None, call_model_fn=model)
    assert result["stop_reason"] == "max_steps"
    assert len(result["trajectory"]) == 8  # 真的跑滿 8 步才停，不是卡死也不是提早結束
    print("[OK] test_cyclic_graph_terminates")


# ---------- 4. hit/miss 判斷 + miss 不會被送進 Line B ----------

def test_hit_detection():
    backend = MockGraphBackend(build_mock_graph())
    model = _ScriptedModel([_move_to("D1_A"), _move_to(PIVOT_QID)])
    result = run_free_exploration("m", backend, "D2_A", max_steps=10, task_prompt="t",
                                    target_qid=PIVOT_QID, call_model_fn=model)
    assert result["hit"] is True
    assert result["stop_reason"] == "pivot_reached"
    assert result["final_qid"] == PIVOT_QID
    print("[OK] test_hit_detection")


def test_miss_detection_and_not_sent_to_line_b():
    backend = MockGraphBackend(build_mock_graph())
    # 一路往 D3_A 走，永遠碰不到 pivot，max_steps 到了就停
    model = _ScriptedModel([_move_to("D2_A"), _move_to("D3_A"), _move_to("D2_A")])
    result = run_free_exploration("m", backend, "D1_A", max_steps=3, task_prompt="t",
                                    target_qid=PIVOT_QID, call_model_fn=model)
    assert result["hit"] is False
    assert result["stop_reason"] == "max_steps"

    # 用 run_free_exploration_batch 的批次邏輯驗證：miss 的話 hit_counts 有記錄，
    # 但不會呼叫 run_round_schedule()（用一個會直接 raise 的假 run_round_schedule
    # 偵測「有沒有被呼叫」，miss 的情況下不該被呼叫到）
    import run_free_exploration_batch as batch
    import run_experiment

    original_run_round_schedule = batch.run_round_schedule
    called = {"n": 0}

    def spy_run_round_schedule(*args, **kwargs):
        called["n"] += 1
        return True

    batch.run_round_schedule = spy_run_round_schedule
    try:
        case = build_mock_case()
        log_path = "test_free_exploration_dryrun_tmp.jsonl"
        with open(log_path, "w", encoding="utf-8") as log_fh:
            batch.run_one_exploration("m", backend, "D1_A", PIVOT_QID, max_steps=3,
                                        temperature=0.7, case=case, arm="conflict", distance=3,
                                        repeat_idx=0, log_fh=log_fh, call_model_fn=model)
        assert called["n"] == 0, "miss 的軌跡不應該呼叫 run_round_schedule()"
    finally:
        batch.run_round_schedule = original_run_round_schedule
        if os.path.exists(log_path):
            os.remove(log_path)
    print("[OK] test_miss_detection_and_not_sent_to_line_b")


# ---------- 5. 洩漏檢查只管 pivot 節點 ----------

def test_leak_check_only_pivot_node():
    case = build_mock_case()

    # pivot 節點自己的 facts 很乾淨（不含任何 ripple 關鍵字）-> 不該被判成洩漏
    clean_pivot_facts = ["這是中性的 pivot 現況描述（as of 2026-01-01）"]
    assert check_pivot_leak(clean_pivot_facts, case) == {}

    # pivot 節點自己的 facts 直接寫出 ripple 答案 -> 應該被抓到
    leaking_pivot_facts = ["pivot 現況：答案是 ripple1_new_keyword"]
    leaks = check_pivot_leak(leaking_pivot_facts, case)
    assert "1" in leaks and "ripple1_new_keyword" in leaks["1"]

    # 模型探索到「別的節點」（不是 pivot）看到 ripple 關鍵字 -> 不算洩漏，
    # 是 find_pre_seen_ripple_distances 要記錄的「探索中已看過答案」現象
    backend = MockGraphBackend(build_mock_graph())
    model = _ScriptedModel([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "move_to",
             "arguments": '{"neighbor_id": "D1_A"}'}}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "view_current_node", "arguments": "{}"}}]},
        _move_to(PIVOT_QID, "c3"),
    ])
    result = run_free_exploration("m", backend, PIVOT_QID, max_steps=10, task_prompt="t",
                                    target_qid=PIVOT_QID, call_model_fn=model)
    pre_seen = find_pre_seen_ripple_distances(result["trajectory"], case, PIVOT_QID)
    assert pre_seen == [1], f"應該偵測到 D1_A 節點提前洩漏了 distance 1 的答案，實際: {pre_seen}"

    # 確認 pivot 節點自己被 view_current_node 時不會被誤判成「提前看到」
    # （find_pre_seen_ripple_distances 排除 from_qid == pivot_qid 的 view_current_node）
    pivot_view_step = [{"step": 1, "from_qid": PIVOT_QID, "action": "view_current_node",
                          "result": "答案是 ripple2_new_keyword"}]
    assert find_pre_seen_ripple_distances(pivot_view_step, case, PIVOT_QID) == [], \
        "pivot 節點自己的內容不該被 find_pre_seen_ripple_distances 算進去（那是 check_pivot_leak 的責任）"
    print("[OK] test_leak_check_only_pivot_node")


if __name__ == "__main__":
    test_bfs_distance_correct()
    test_termination_stop_exploring()
    test_termination_max_steps()
    test_no_tool_call_terminates()
    test_cyclic_graph_terminates()
    test_hit_detection()
    test_miss_detection_and_not_sent_to_line_b()
    test_leak_check_only_pivot_node()
    print("\n[OK] 全部通過。")
