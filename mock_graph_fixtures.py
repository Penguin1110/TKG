"""
mock_graph_fixtures.py
------------------------
自由探索用的 mock 圖跟 mock case，給兩個地方共用：
  1. run_free_exploration_batch.py / run_control_exploration_batch.py 的 --dry-run
  2. test_free_exploration_dryrun.py 的單元測試

拆成獨立模組是為了不讓正式程式（run_free_exploration_batch.py）反過來
import 測試檔案——那是反過來的依賴方向，測試檔案改了/搬了，正式程式的
--dry-run 就會跟著壞掉。

MockGraphBackend 刻意只實作 fetch_node()，bfs_distance()/find_nodes_at_distance()
委派給 wikidata_graph_backend.py 的模組級函式（不是另外重寫一份演算法），這樣
測試驗證的才是真正在跑的 BFS 邏輯，不是另一套巧合看起來正確的邏輯。
"""

from wikidata_graph_backend import bfs_distance as _bfs_distance
from wikidata_graph_backend import find_nodes_at_distance as _find_nodes_at_distance

PIVOT_QID = "MOCKPIVOT"


class MockGraphBackend:
    def __init__(self, graph: dict):
        self.graph = graph

    def fetch_node(self, qid: str) -> dict:
        return self.graph[qid]

    def bfs_distance(self, start_qid: str, target_qid: str, max_depth: int = 5):
        return _bfs_distance(self.fetch_node, start_qid, target_qid, max_depth)

    def find_nodes_at_distance(self, start_qid: str, distance: int,
                                max_depth: int = None, max_results: int = 50) -> list:
        return _find_nodes_at_distance(self.fetch_node, start_qid, distance, max_depth, max_results)


def _node(qid, label, facts, neighbor_qids):
    return {"qid": qid, "label": label, "facts": facts,
            "neighbors": [{"qid": nq, "label": nq, "property": "rel"} for nq in neighbor_qids]}


def build_mock_graph() -> dict:
    """
    以 PIVOT_QID 為中心，往外展開 distance 1/2/3，每層 2 個節點（分支因子小，
    讓隨機走的 mock 模型還是有不錯的機率走到 pivot，且距離越遠機率越低——
    這是驗收標準要求「距離1命中率應該高於距離3」的結構性保證，不是碰運氣）。
    每個節點的鄰居都雙向連通（可以走回去），模擬真實圖的樣子。
    """
    graph = {
        PIVOT_QID: _node(PIVOT_QID, "Pivot Node",
                          ["這是中性的 pivot 現況描述（as of 2026-01-01）"],
                          ["D1_A", "D1_B"]),
        # D1_A 的 facts 刻意藏一個 ripple1_new_keyword，用來測試「模型探索到下游
        # 節點、自然看到了 ripple 答案」這個分析維度（find_pre_seen_ripple_distances），
        # 這不算洩漏——洩漏檢查只管 pivot 節點自己的 facts 乾不乾淨
        "D1_A": _node("D1_A", "Distance-1 Node A",
                       ["distance-1 節點 A 的事實，剛好提到 ripple1_new_keyword"],
                       [PIVOT_QID, "D2_A"]),
        "D1_B": _node("D1_B", "Distance-1 Node B", ["distance-1 節點 B 的中性事實"],
                       [PIVOT_QID, "D2_B"]),
        "D2_A": _node("D2_A", "Distance-2 Node A", ["distance-2 節點 A 的中性事實"],
                       ["D1_A", "D3_A"]),
        "D2_B": _node("D2_B", "Distance-2 Node B", ["distance-2 節點 B 的中性事實"],
                       ["D1_B", "D3_B"]),
        "D3_A": _node("D3_A", "Distance-3 Node A", ["distance-3 節點 A 的中性事實"],
                       ["D2_A"]),
        "D3_B": _node("D3_B", "Distance-3 Node B", ["distance-3 節點 B 的中性事實"],
                       ["D2_B"]),
    }
    return graph


def build_cyclic_graph() -> dict:
    """A<->B<->C<->A 三角環，沒有出口，專門測「環狀圖不會無窮迴圈」。"""
    return {
        "A": _node("A", "Node A", ["fact a"], ["B", "C"]),
        "B": _node("B", "Node B", ["fact b"], ["A", "C"]),
        "C": _node("C", "Node C", ["fact c"], ["A", "B"]),
    }


def build_mock_case() -> dict:
    """
    最小可用的合成 case，給 --dry-run 跟測試共用。ripples/control 只有 distance
    1/2/3 各一個候選，關鍵字刻意跟 mock 圖的 facts 內容不重疊（乾淨的 pivot），
    但故意在 D1_A 的 facts 裡藏一個 ripple 關鍵字，用來測試「探索到下游節點
    自然看到答案」跟「pivot 節點洩漏」的差別（見
    find_pre_seen_ripple_distances() / check_pivot_leak()）。
    """
    return {
        "id": "mock_case",
        "category": "test",
        "pivot_qid": PIVOT_QID,
        "pk_question": "mock pk question",
        "pk_threshold": 0.8,
        "old_answer_keywords": ["mock_old"],
        "new_answer_keywords": ["mock_new"],
        "ripples": {
            "1": [{"question": "mock ripple 1？", "paraphrases": [],
                    "old_keywords": ["ripple1_old"], "new_keywords": ["ripple1_new_keyword"]}],
            "2": [{"question": "mock ripple 2？", "paraphrases": [],
                    "old_keywords": ["ripple2_old"], "new_keywords": ["ripple2_new_keyword"]}],
            "3": [{"question": "mock ripple 3？", "paraphrases": [],
                    "old_keywords": ["ripple3_old"], "new_keywords": ["ripple3_new_keyword"]}],
        },
        "control": {
            "1": [{"question": "mock control 1？", "paraphrases": [], "answer_keywords": ["control1_answer"]}],
            "2": [{"question": "mock control 2？", "paraphrases": [], "answer_keywords": ["control2_answer"]}],
            "3": [{"question": "mock control 3？", "paraphrases": [], "answer_keywords": ["control3_answer"]}],
        },
    }
