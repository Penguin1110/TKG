"""
wikidata_graph_backend.py
--------------------------
把 Wikidata 包裝成一個可以被 agent 逐步探索的圖，供 graph_exploration_agent.py
使用。用 wbgetentities 抓單一節點的 claims，轉成人類可讀的 facts（帶
start/end time qualifier）跟 neighbors（僅 wikibase-item 型態的 claim），
本地用 sqlite 快取避免重複打 API。

介面設計成跟 MockGraphBackend（見 test_free_exploration_dryrun.py）duck-type
相容：只要有 fetch_node(qid)，graph_exploration_agent.py 就能運作，測試時
可以完全不打真的 API。

**TKG 的「現在」語意（current_only，預設開啟）**：Wikidata 上一個實體常常
帶著完整的歷史 claims（例如一個國家的元首清單可以回溯上百年）。這個模組
預設只回傳「現在仍然有效」的 claims（沒有 P582 結束時間的、或完全沒有時間
qualifier 的常態性事實），不回傳已經結束的歷史 claim——這是刻意的設計，
把 Wikidata 當成一個「查詢 as of 現在」的時序知識圖（TKG），而不是一份完整
的歷史檔案庫。這樣做同時解決兩件事：(1) 探索時不會被無關的歷史資訊淹沒；
(2) 避免 pivot 節點的歷史紀錄裡剛好包含某個 ripple 事實的答案（例如一個國家
的「現任元首」claim 本身很乾淨，但如果連同幾十年前的歷史元首一起攤開，
可能剛好某個歷史元首正好是某個 ripple 問題要問的人）——見
run_free_exploration_batch.py 的 check_pivot_leak() 使用說明跟
cases.json 裡幾個案例的 _pivot_qid_note。時間戳本身仍然保留在每條 fact 的
字串裡（`_qualifier_time_suffix()`），只是不再把已結束的歷史 claim 也
一起攤開來看。

速率限制：每次真的打 Wikidata API 的請求間隔至少 1 秒，且一律帶 User-Agent
（Wikimedia 的使用規範要求，沒帶容易被降速或封鎖）。
"""

import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import requests

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
DEFAULT_USER_AGENT = (
    "TKG-Pilot-Fetcher/1.0 (research pilot; contact: your-email@example.com)"
)
MIN_REQUEST_INTERVAL = 1.0  # 秒
LABEL_BATCH_SIZE = 50  # wbgetentities 一次最多查 50 個 id


class WikidataError(Exception):
    pass


class WikidataGraphBackend:
    def __init__(self, cache_path: str = "tkg_cache.db",
                 user_agent: str = DEFAULT_USER_AGENT, lang: str = "en",
                 current_only: bool = True, offline_only: bool = False):
        """
        offline_only=True 時，只要 fetch_node() 遇到本地快取沒有的 qid 就直接
        報錯，不會悄悄地即時打 Wikidata API。用來在跑正式實驗前先用
        build_tkg_snapshot.py 把需要的節點都快取好，之後探索過程完全離線、
        不會受即時 API 的速率限制或連線問題影響，也不會因為模型走到快照沒
        涵蓋的節點而在跑到一半時默默改變資料來源（快照 vs 即時）。
        """
        self.user_agent = user_agent
        self.lang = lang
        self.current_only = current_only
        self.offline_only = offline_only
        self._last_request_time = 0.0
        self._conn = sqlite3.connect(cache_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS node_cache "
            "(qid TEXT PRIMARY KEY, data TEXT NOT NULL, fetched_at TEXT NOT NULL)"
        )
        self._conn.commit()

    def close(self):
        self._conn.close()

    # ---------- 底層：帶速率限制、User-Agent 的請求 ----------

    def _rate_limited_get(self, params: dict, max_retries: int = 5) -> dict:
        """
        帶速率限制 + 重試的 GET。實測發現固定 1 秒間隔不代表不會被 Wikidata 節流
        （連續查證階段有踩過 429），這裡才補上重試+指數退避——建 snapshot 時會
        連續打幾百次請求，沒有重試邏輯遇到一次 429 整個流程就中斷太脆弱了。
        """
        last_err = None
        for attempt in range(1, max_retries + 1):
            elapsed = time.time() - self._last_request_time
            if elapsed < MIN_REQUEST_INTERVAL:
                time.sleep(MIN_REQUEST_INTERVAL - elapsed)
            try:
                resp = requests.get(WIKIDATA_API, params=params,
                                     headers={"User-Agent": self.user_agent}, timeout=30)
                self._last_request_time = time.time()
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_err = requests.exceptions.HTTPError(
                        f"{resp.status_code} for {resp.url}", response=resp)
                    wait = 2 ** attempt
                    print(f"[warn] Wikidata API 第 {attempt} 次失敗（HTTP {resp.status_code}），"
                          f"{wait}s 後重試...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                self._last_request_time = time.time()
                last_err = e
                wait = 2 ** attempt
                print(f"[warn] Wikidata API 第 {attempt} 次失敗（{e}），{wait}s 後重試...")
                time.sleep(wait)
        raise WikidataError(f"打 Wikidata API 失敗，已重試 {max_retries} 次：{last_err}")

    def _fetch_labels(self, ids: list) -> dict:
        """
        回傳 {id: label}。查不到 label 時，先退而求其次用 description 當 fallback
        （踩過的坑：apple_ceo 這個案例裡，John Ternus 對應的 Wikidata 項目
        Q106028933 是最近才建的、還沒有人補上英文 label，只有 description
        "Apple senior vice president of hardware engineering"——如果直接顯示
        原始 QID，模型看到的 pivot fact 會是「chief executive officer:
        Q106028933」，等於完全沒曝光到新 CEO 是誰。description 雖然不是真正
        的人名，但至少比一串看起來像亂碼的 QID 有意義，且這是 Wikidata 資料
        本身的缺口，不是我們能補的，description fallback 是務實的次佳選擇），
        description 也沒有才真的用 id 本身當最後手段。
        """
        labels = {}
        ids = sorted(set(ids))
        for i in range(0, len(ids), LABEL_BATCH_SIZE):
            chunk = ids[i:i + LABEL_BATCH_SIZE]
            data = self._rate_limited_get({
                "action": "wbgetentities", "ids": "|".join(chunk),
                "format": "json", "languages": self.lang, "props": "labels|descriptions",
            })
            for eid, entity in data.get("entities", {}).items():
                label_obj = entity.get("labels", {}).get(self.lang)
                if label_obj:
                    labels[eid] = label_obj["value"]
                    continue
                desc_obj = entity.get("descriptions", {}).get(self.lang)
                labels[eid] = f"{eid}（{desc_obj['value']}）" if desc_obj else eid
        return labels

    # ---------- 核心：抓一個節點 ----------

    def fetch_node(self, qid: str) -> dict:
        """
        回傳 {"qid":..., "label":..., "facts": [str,...],
              "neighbors": [{"qid":..., "label":..., "property": str}]}

        facts / neighbors 都來自這個 QID 的 claims：
          - wikibase-item 型態的 value -> 算一條 neighbor 邊，同時也生成一條
            facts 描述句（方便 view_current_node() 用）
          - 其他型態（時間、字串、數量…）-> 只生成 facts 描述句，不當 neighbor
          - 如果 statement 帶 P580（start time）/P582（end time）qualifier，
            一併附加在 facts 描述句裡
        """
        claim_records = self._fetch_claim_records(qid)
        return self._render_node(qid, claim_records, current_only=self.current_only)

    def refresh_node(self, qid: str) -> dict:
        """
        強制即時重打 API 重抓一個 qid（不管快取有沒有），拿新資料跟快取裡
        原本的比對，回傳有沒有變動，並把快取更新成新資料。給
        build_tkg_snapshot.py 的 --verify 模式用：快照建好之後，Wikidata
        本身還是持續在被編輯，這個方法可以回答「快照裡的 pivot 現在還準嗎」。

        回傳：{"qid", "had_cache"（快照裡原本有沒有這個節點）,
               "changed"（新舊 claim 記錄是否不同）,
               "old_record_count", "new_record_count"}
        """
        old_row = self._conn.execute(
            "SELECT data FROM node_cache WHERE qid=?", (qid,)
        ).fetchone()
        old = json.loads(old_row[0]) if old_row else None

        self._conn.execute("DELETE FROM node_cache WHERE qid=?", (qid,))
        self._conn.commit()
        new = self._fetch_claim_records(qid)  # 快取已刪，這裡一定會真的打 API

        changed = old is None or old.get("records") != new.get("records")
        return {
            "qid": qid, "had_cache": old is not None, "changed": changed,
            "old_record_count": len(old["records"]) if old else 0,
            "new_record_count": len(new["records"]),
        }

    def _fetch_claim_records(self, qid: str) -> dict:
        """
        抓一個 QID 的完整、未過濾 claim 記錄（含每條 claim 是不是「現在仍然
        有效」），存進 sqlite 快取。快取存的是完整歷史，過不過濾（current_only）
        是 fetch_node() 讀出來之後才做的事——這樣同一個 qid 不管用
        current_only=True 還是 False 查，都只需要打一次 API，快取共用。
        """
        cached = self._conn.execute(
            "SELECT data FROM node_cache WHERE qid=?", (qid,)
        ).fetchone()
        if cached:
            return json.loads(cached[0])

        if self.offline_only:
            raise WikidataError(
                f"offline_only=True，但快取裡沒有 {qid}。這代表探索走到了 snapshot "
                f"沒有涵蓋的節點——先用 build_tkg_snapshot.py 把需要的範圍（種子節點 + "
                f"更大的 --max-depth/--branch-cap）重新建過，或這次先不要用 offline_only。"
            )

        data = self._rate_limited_get({
            "action": "wbgetentities", "ids": qid, "format": "json",
            "languages": self.lang, "props": "labels|claims",
        })
        entity = data.get("entities", {}).get(qid)
        if entity is None or "missing" in entity:
            raise WikidataError(f"Wikidata 上找不到 {qid}")

        node_label = (entity.get("labels", {}).get(self.lang) or {}).get("value", qid)
        claims = entity.get("claims", {})

        # 先掃一輪，收集所有要查 label 的 property id / item value id
        prop_ids, item_ids = set(), set()
        for prop_id, statements in claims.items():
            prop_ids.add(prop_id)
            for st in statements:
                snak = st.get("mainsnak", {})
                if snak.get("datatype") == "wikibase-item":
                    val = snak.get("datavalue", {}).get("value", {})
                    if "id" in val:
                        item_ids.add(val["id"])
        labels = self._fetch_labels(list(prop_ids | item_ids)) if (prop_ids or item_ids) else {}

        records = []
        for prop_id, statements in claims.items():
            prop_label = labels.get(prop_id, prop_id)
            for st in statements:
                snak = st.get("mainsnak", {})
                if snak.get("snaktype") != "value":
                    continue
                datatype = snak.get("datatype")
                value = snak.get("datavalue", {}).get("value")
                value_label, is_item = _describe_value(datatype, value, labels)
                if value_label is None:
                    continue

                qualifiers = st.get("qualifiers", {})
                records.append({
                    "prop_label": prop_label, "value_label": value_label, "is_item": is_item,
                    "value_qid": value.get("id") if is_item else None,
                    "time_suffix": _qualifier_time_suffix(qualifiers),
                    "is_current": _is_current_claim(qualifiers),
                })

        result = {"qid": qid, "label": node_label, "records": records}
        self._conn.execute(
            "INSERT OR REPLACE INTO node_cache (qid, data, fetched_at) VALUES (?, ?, ?)",
            (qid, json.dumps(result, ensure_ascii=False), datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()
        return result

    @staticmethod
    def _render_node(qid: str, claim_records: dict, current_only: bool) -> dict:
        """
        把快取裡完整、未過濾的 claim 記錄，依 current_only 篩選後組成
        fetch_node() 對外回傳的 {"qid","label","facts","neighbors"} 格式。

        踩過的坑：熱門節點（例如 Apple Inc.）就算做完 current_only 過濾，
        剩下的 claims 還是可能上千條（一堆 EIN、Legal Entity Identifier 之類
        永遠不會有 end time 的靜態識別碼，加上創辦人這種同樣永遠是「現在」的
        claim）。graph_exploration_agent.py 的 view_current_node()/move_to()
        只會顯示前 MAX_FACTS_SHOWN=30 條（token 預算限制），如果照原始 claims
        順序排，CEO 這種真正動態、我們在乎的 claim 完全可能被埋在一千多條
        靜態識別碼後面、連前 30 條都排不到——等於曝光步驟看起來有跑，模型卻
        根本沒機會看到 pivot fact 本身。

        修法：帶時間 qualifier 的 claim（`time_suffix` 非空，代表這是會隨時間
        變動的動態事實，例如「誰現在是 CEO」）優先排在前面，沒有時間資訊的
        靜態事實（識別碼、常態性描述）排後面——用穩定排序，同一組內原本的
        相對順序不變。這剛好也符合 TKG 的核心關注點：動態、有時間性的事實
        本來就該比靜態識別碼優先被看到。
        """
        current_records = [r for r in claim_records["records"] if not current_only or r["is_current"]]
        ordered_records = sorted(current_records, key=lambda r: 0 if r["time_suffix"] else 1)

        facts, neighbors = [], []
        for rec in ordered_records:
            facts.append(f"{rec['prop_label']}: {rec['value_label']}{rec['time_suffix']}")
            if rec["is_item"]:
                neighbors.append({"qid": rec["value_qid"],
                                    "label": rec["value_label"], "property": rec["prop_label"]})
        return {"qid": qid, "label": claim_records["label"], "facts": facts, "neighbors": neighbors}

    # ---------- BFS：距離查詢 / 候選起點池 ----------
    # 實際演算法在下面的模組級函式 bfs_frontier()，這裡只是薄包裝（this.fetch_node
    # 當 fetch_node_fn 用）。拆出來是為了讓測試可以直接對一個 mock 圖（只要有
    # fetch_node(qid) 這個介面）驗證同一套 BFS 邏輯，而不是另外重寫一份不同的
    # 演算法去測──那樣測到的就不是真正在跑的程式碼了。

    def bfs_distance(self, start_qid: str, target_qid: str, max_depth: int = 5,
                      branch_cap: int = None):
        return bfs_distance(self.fetch_node, start_qid, target_qid, max_depth, branch_cap)

    def find_nodes_at_distance(self, start_qid: str, distance: int,
                                max_depth: int = None, max_results: int = 50,
                                branch_cap: int = None) -> list:
        return find_nodes_at_distance(self.fetch_node, start_qid, distance, max_depth,
                                       max_results, branch_cap)


DEFAULT_BRANCH_CAP = 25  # 見 bfs_frontier() 的說明：沒有這個上限，熱門節點（國家/大型組織）
                          # 光是展開一層鄰居就可能是幾百個，兩三層下去會組合爆炸


def bfs_frontier(fetch_node_fn, start_qid: str, max_depth: int, target_qid: str = None,
                  branch_cap: int = None, verbose: bool = True) -> dict:
    """
    從 start_qid 做 BFS，回傳 {distance: [qid,...]}（distance 0 = start_qid 自己）。
    如果有給 target_qid，一找到就提早結束（省 API 呼叫／減少 mock 圖裡的重複走訪）。

    fetch_node_fn 只要是「輸入 qid、回傳 {"neighbors": [{"qid":...}, ...]}」的
    callable 就行，真正的 WikidataGraphBackend.fetch_node 或測試用的
    MockGraphBackend.fetch_node 都適用。

    branch_cap：每個節點展開下一層時最多只走前 N 個鄰居（預設
    DEFAULT_BRANCH_CAP=25，傳 None 也是用這個預設值，傳 0 或負數才是真的不設上限）。
    這不是效能微調，是必要的正確性防護——像秘魯這種國家實體，light 節點的鄰居數
    可以到兩三百個，沒有上限的話，distance=3 的 BFS 光是展開 distance=2 那一層
    可能就要對每個 distance=1 節點都各打上百次 API，實測會直接變成幾千次請求、
    以現在 1 秒一次的 rate limit 來算要跑好幾個小時。設了上限之後找到的候選起點
    池會是「圖上一部分」而不是「全部」，但這本來就跟 --n-starts-per-distance
    只抽樣一部分起點的精神一致，不影響 pilot 的正確性，只是需要在報告裡如實
    交代取樣範圍有這層限制（跟 --restrict-subgraph-k 是同一類型的、需要誠實
    交代的工程取捨，但這裡是預設就開啟，因為完全不設上限在實務上根本跑不完）。

    verbose=True（預設）會印出每個 depth 展開到第幾個節點——這一步是連續對
    Wikidata 打 API（受 1 秒/次限速 + 429 重試影響），沒有快取的情況下光是
    展開到 depth 3 就可能要幾十秒到幾分鐘，完全沒有輸出會讓人以為卡住了
    （實測發生過：使用者以為是探索階段卡住，其實是根本還沒進到那一步，還在
    這裡找候選起點池）。
    """
    def _log(msg):
        if verbose:
            print(msg)

    cap = DEFAULT_BRANCH_CAP if branch_cap is None else branch_cap
    frontier_by_distance = {0: [start_qid]}
    visited = {start_qid}
    current_level = [start_qid]
    for depth in range(1, max_depth + 1):
        next_level = []
        _log(f"[bfs] 展開 depth {depth}（上一層有 {len(current_level)} 個節點要查）...")
        for i, qid in enumerate(current_level, start=1):
            _log(f"[bfs] depth {depth}: 查詢第 {i}/{len(current_level)} 個節點 {qid}...")
            try:
                node = fetch_node_fn(qid)
            except WikidataError as e:
                _log(f"[bfs]   {qid} 查詢失敗，跳過：{e}")
                continue
            neighbors = node["neighbors"] if cap <= 0 else node["neighbors"][:cap]
            for nb in neighbors:
                nb_qid = nb["qid"]
                if nb_qid in visited:
                    continue
                visited.add(nb_qid)
                next_level.append(nb_qid)
                if target_qid is not None and nb_qid == target_qid:
                    frontier_by_distance[depth] = next_level
                    _log(f"[bfs] 在 depth {depth} 找到目標 {target_qid}，提早結束")
                    return frontier_by_distance
        if not next_level:
            _log(f"[bfs] depth {depth} 沒有新節點可以展開，提早結束")
            break
        _log(f"[bfs] depth {depth} 完成，找到 {len(next_level)} 個新節點")
        frontier_by_distance[depth] = next_level
        current_level = next_level
    return frontier_by_distance


def bfs_distance(fetch_node_fn, start_qid: str, target_qid: str, max_depth: int = 5,
                  branch_cap: int = None):
    """回傳 start_qid 到 target_qid 的最短跳數，超過 max_depth 找不到就回傳 None。"""
    if start_qid == target_qid:
        return 0
    frontier = bfs_frontier(fetch_node_fn, start_qid, max_depth, target_qid=target_qid,
                             branch_cap=branch_cap)
    for depth, qids in frontier.items():
        if depth > 0 and target_qid in qids:
            return depth
    return None


def find_nodes_at_distance(fetch_node_fn, start_qid: str, distance: int,
                            max_depth: int = None, max_results: int = 50,
                            branch_cap: int = None) -> list:
    """回傳距離 start_qid 剛好 `distance` 跳的候選節點 qid 清單（做為起點池）。"""
    frontier = bfs_frontier(fetch_node_fn, start_qid, max_depth or distance, branch_cap=branch_cap)
    return frontier.get(distance, [])[:max_results]


def _describe_value(datatype, value, labels):
    """回傳 (value_label:str|None, is_item:bool)。datatype 不支援的回傳 (None, False)。"""
    if datatype == "wikibase-item" and isinstance(value, dict) and "id" in value:
        return labels.get(value["id"], value["id"]), True
    if datatype == "time" and isinstance(value, dict):
        return _format_wikidata_time(value.get("time", "")), False
    if datatype in ("string", "url", "external-id"):
        return str(value), False
    if datatype == "quantity" and isinstance(value, dict):
        return str(value.get("amount", "")).lstrip("+"), False
    if datatype == "monolingualtext" and isinstance(value, dict):
        return value.get("text"), False
    return None, False


def _format_wikidata_time(raw: str) -> str:
    """Wikidata 時間格式是 '+2026-09-01T00:00:00Z'，只取日期部分。"""
    raw = raw.lstrip("+")
    return raw.split("T")[0] if "T" in raw else raw


def _qualifier_time_suffix(qualifiers: dict) -> str:
    start = qualifiers.get("P580", [{}])[0].get("datavalue", {}).get("value", {}).get("time")
    end = qualifiers.get("P582", [{}])[0].get("datavalue", {}).get("value", {}).get("time")
    if not start and not end:
        return ""
    start_s = _format_wikidata_time(start) if start else "?"
    end_s = _format_wikidata_time(end) if end else "至今"
    return f"（{start_s} ~ {end_s}）"


def _is_current_claim(qualifiers: dict) -> bool:
    """
    這條 claim 現在是不是仍然有效：沒有 P582（結束時間）就算現在仍有效——
    不管是完全沒有時間 qualifier 的常態性事實（例如總部所在地），還是有
    P580（開始時間）但沒有 P582（代表「從那時候開始一直到現在」）都算。
    只有明確帶了 P582 結束時間的 claim（代表已經是過去式）才會被排除。
    """
    return not bool(qualifiers.get("P582"))
