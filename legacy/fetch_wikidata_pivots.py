"""
fetch_wikidata_pivots.py
--------------------------
注意給未來讀者：run_control_exploration_batch.py 的設計文件裡提到「沿用
fetch_wikidata_pivots.py 規劃裡已有的 pageviews 過濾邏輯」，但這個檔案在
專案裡原本並不存在（寫這個模組時特地確認過）。這裡是照設計文件裡描述的
邏輯從頭實作，不是真的復用了什麼舊程式碼——取一樣的檔名只是讓文件裡的
交接說明對得起來。

職責：幫 control arm 找「同類型、瀏覽量相近、但近期沒有更替」的穩定實體
當 control pivot。跟 wikidata_graph_backend.py 一樣，所有 Wikimedia API
請求都帶 User-Agent、有速率限制。

目前的限制（誠實寫在這裡，不要在報告裡含糊帶過）：
    find_stable_control_pivot() 是在「呼叫端提供的候選池」裡面篩選/排序，
    不是自動從整個 Wikidata 用 SPARQL 查詢「同一個 class 底下的所有實體」
    ——後者需要針對每個 case 的實體類型寫對應的 SPARQL 查詢，這部分留給
    案例作者在 cases.json 的 control_pivot_candidates 欄位手動整理候選清單
    （比照原本 5 個真實案例研究 ripple 事實時，也是人工查證、不是全自動）。
"""

import time
from datetime import datetime, timedelta

import requests

from legacy.wikidata_graph_backend import DEFAULT_USER_AGENT, WIKIDATA_API

PAGEVIEWS_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "{project}/all-access/user/{article}/daily/{start}/{end}"
)
MIN_REQUEST_INTERVAL = 1.0

_last_request_time = [0.0]


def _rate_limited_get(url: str, headers: dict) -> dict:
    elapsed = time.time() - _last_request_time[0]
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    finally:
        _last_request_time[0] = time.time()


def get_wikipedia_title(qid: str, lang: str = "en", user_agent: str = DEFAULT_USER_AGENT) -> str:
    """回傳這個 QID 對應的 Wikipedia 條目標題（給 pageviews API 用），查不到回傳 None。"""
    data = _rate_limited_get(
        f"{WIKIDATA_API}?action=wbgetentities&ids={qid}&format=json&props=sitelinks",
        headers={"User-Agent": user_agent},
    )
    entity = data.get("entities", {}).get(qid, {})
    sitelink = entity.get("sitelinks", {}).get(f"{lang}wiki")
    return sitelink["title"] if sitelink else None


def fetch_pageviews(article_title: str, project: str = "en.wikipedia.org", days: int = 30,
                     user_agent: str = DEFAULT_USER_AGENT):
    """回傳過去 `days` 天的平均每日瀏覽量，查不到回傳 None。"""
    if not article_title:
        return None
    end = datetime.utcnow().date() - timedelta(days=1)  # pageviews API 通常有 1 天延遲
    start = end - timedelta(days=days)
    url = PAGEVIEWS_API.format(
        project=project, article=article_title.replace(" ", "_"),
        start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"),
    )
    try:
        data = _rate_limited_get(url, headers={"User-Agent": user_agent})
    except requests.HTTPError:
        return None
    items = data.get("items", [])
    if not items:
        return None
    return sum(item["views"] for item in items) / len(items)


def is_pageview_similar(a: float, b: float, tolerance_pct: float = 50.0) -> bool:
    """
    兩個瀏覽量是不是「差不多」，容忍度用百分比表示（相對於較大的那個值）。
    a 或 b 是 None（其中一邊查不到瀏覽量）時，回傳 True——代表「這一關跳過，
    不卡這個候選」，而不是「一定不像」。呼叫端如果想在查不到瀏覽量時整個
    拒絕候選，要自己另外判斷，這裡不用 inf 之類的技巧去湊出「永遠通過」，
    那樣在算 (bigger-smaller)/bigger 時會因為 inf-數字還是 inf、inf/inf 變成
    NaN，NaN 跟任何數字比較都是 False，反而變成「永遠不通過」，是個真的會
    衝突實際行為的陷阱，直接在這裡判斷清楚比較安全。
    """
    if a is None or b is None:
        return True
    bigger, smaller = max(a, b), min(a, b)
    if bigger == 0:
        return smaller == 0
    return (bigger - smaller) / bigger * 100 <= tolerance_pct


def is_claim_stable(backend, qid: str, property_id: str, min_stable_years: float = 3.0) -> bool:
    """
    這個 QID 在 property_id（例如 P169 chief executive officer）上目前的 claim，
    是不是已經維持了至少 min_stable_years 年沒有變動（沒有比它更新的同 property
    claim、且這個 claim 沒有 end time，代表現在還有效）。

    做法：直接看 backend.fetch_node(qid) 生成的 facts 字串裡，這個 property 對應
    的最新一條有沒有帶「~ 至今」且 start time 夠久。這裡故意不重新解析原始
    claims（那是 fetch_node 已經做過的事），只在 facts 字串上做簡單比對，維持
    跟其他模組一樣「少依賴、夠用就好」的風格。
    """
    # fetch_node() 產生的 facts 是 "{property_label}: ..."，所以這裡的 property_id
    # 參數實際上要傳 property 的 label（例如 "chief executive officer"），不是 P-id
    node = backend.fetch_node(qid)
    prop_facts = [f for f in node["facts"] if f.lower().startswith(property_id.lower() + ":")]
    current = [f for f in prop_facts if "至今" in f]
    if not current:
        return False
    # 取第一個「至今」的 claim，看 start time 是不是夠久以前
    import re
    m = re.search(r"（(\d{4}-\d{2}-\d{2})\s*~\s*至今）", current[0])
    if not m:
        return False
    start_date = datetime.strptime(m.group(1), "%Y-%m-%d")
    return (datetime.utcnow() - start_date).days >= min_stable_years * 365


def find_stable_control_pivot(backend, candidate_qids: list, target_pageviews: float,
                               position_property_label: str, tolerance_pct: float = 50.0,
                               min_stable_years: float = 3.0):
    """
    從 candidate_qids 裡挑一個當 control pivot：瀏覽量要跟 target_pageviews
    差不多、position_property_label 對應的 claim 要夠穩定（沒有近期更替）。
    候選不是自動從全 Wikidata 搜出來的，見檔案開頭的限制說明。

    回傳第一個符合條件的 qid，都不符合回傳 None。
    """
    for qid in candidate_qids:
        if not is_claim_stable(backend, qid, position_property_label, min_stable_years):
            continue
        title = get_wikipedia_title(qid)
        pv = fetch_pageviews(title) if title else None
        if is_pageview_similar(pv, target_pageviews, tolerance_pct):
            return qid
    return None
