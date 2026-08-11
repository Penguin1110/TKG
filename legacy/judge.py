"""
legacy.judge
--------
MVP 階段先用「關鍵字比對」做粗略分類，不呼叫額外的 LLM 當裁判
（等 pilot 訊號出來、覺得值得投入，再升級成像 FutureBench / PTC
那樣用獨立模型當 judge，並做 inter-judge agreement 驗證）。

分類邏輯（依序判斷，命中即停）：
    1. stick_new  - 回答裡出現新答案的關鍵字
    2. stick_old  - 回答裡出現舊答案的關鍵字
    3. hedge      - 沒有任何關鍵字命中，但出現「我不確定」「截至我所知」等保留措辭
    4. other      - 都沒命中（可能答非所問，或關鍵字設得不夠好）

注意：關鍵字比對優先於 hedge 判斷，是因為實測發現很多模型（尤其 Claude）
會先講「as of my last update...」這種保留措辭，但緊接著還是給出明確的舊答案
——這種情況我們要算 stick_old，不能因為前面那句保留措辭就整句被判成 hedge，
否則會嚴重低估模型真正的舊先驗強度（實測 kill gate 曾因此誤判成 0%）。
hedge 只在完全沒有具體答案（真的不知道）時才成立。

這是關鍵字比對，不是語意判斷，同義詞、換句話說都可能漏接，MVP 階段先求
「跑得起來、看得出訊號方向」，精確度後續再補強。
"""

HEDGE_PHRASES = [
    "i don't have information",
    "i do not have information",
    "as of my last update",
    "as of my knowledge cutoff",
    "i'm not sure",
    "i am not sure",
    "i don't know",
    "i do not know",
    "cannot confirm",
    "i am unable to confirm",
    "no longer certain",
    "my training data",
]


def _contains_any(text: str, keywords: list) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords if kw)


def classify(response_text: str, old_keywords: list, new_keywords: list) -> str:
    """
    回傳: "stick_new" | "stick_old" | "hedge" | "other"
    """
    if _contains_any(response_text, new_keywords):
        return "stick_new"
    if _contains_any(response_text, old_keywords):
        return "stick_old"
    if _contains_any(response_text, HEDGE_PHRASES):
        return "hedge"
    return "other"


def classify_single(response_text: str, answer_keywords: list) -> str:
    """
    給控制組（無版本衝突、只有一個正確答案）用的判斷。

    回傳: "correct" | "hedge" | "incorrect"
    """
    if _contains_any(response_text, answer_keywords):
        return "correct"
    if _contains_any(response_text, HEDGE_PHRASES):
        return "hedge"
    return "incorrect"
