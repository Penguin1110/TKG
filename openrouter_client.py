"""
openrouter_client.py
---------------------
輕量包裝 OpenRouter 的 chat completions API。

環境需求：
    export OPENROUTER_API_KEY="sk-or-v1-xxxxxxxx"

OpenRouter 相容 OpenAI 的 /chat/completions 介面，所以這裡直接用
requests 打 POST，不依賴 openai 這個 SDK，減少依賴。
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(Exception):
    pass


def call_model(model: str, messages: list, temperature: float = 0.0,
                max_retries: int = 3, timeout: int = 60) -> str:
    """
    呼叫 OpenRouter，回傳模型的文字回應（字串）。

    參數：
        model: OpenRouter 的模型代號，例如 "openai/gpt-4.1-mini"、
               "anthropic/claude-3.7-sonnet"、"meta-llama/llama-4-maverick"
               （完整清單見 https://openrouter.ai/models）
        messages: OpenAI 格式的訊息列表，例如
               [{"role": "user", "content": "你好"}]
        temperature: 設 0 讓輸出盡量穩定、可重現（做研究比較適合）

    回傳：
        模型回應的純文字內容（string）
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterError(
            "找不到 OPENROUTER_API_KEY 環境變數，請先 export OPENROUTER_API_KEY=你的金鑰"
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # 下面兩個是 OpenRouter 建議加、非必填，方便他們排行榜統計來源
        "HTTP-Referer": "https://local-research-pilot.example",
        "X-Title": "TKG-Prior-Reversion-Pilot",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            if not resp.ok:
                # 把 OpenRouter 回傳的錯誤內容（通常會講清楚原因：模型代號錯、缺參數、
                # 額度用完等）一起帶出來，不然只看 raise_for_status() 的通用文字完全
                # 看不出問題在哪
                try:
                    detail = resp.json().get("error", {}).get("message", resp.text)
                except Exception:
                    detail = resp.text
                message = f"HTTP {resp.status_code}：{detail}"
                # 4xx（除了 429 rate limit）是請求本身有問題，不是暫時性的，重試也會
                # 得到一樣的結果，直接放棄比較快；5xx／429／逾時才值得重試（用
                # RuntimeError 交給下面 except Exception 走重試路徑，OpenRouterError
                # 則直接往外拋、跳過重試迴圈）
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise OpenRouterError(f"呼叫 {model} 失敗（client error，不重試）：{message}")
                raise RuntimeError(message)
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except OpenRouterError:
            raise  # fail-fast：直接往外拋，不進入下面的重試邏輯
        except Exception as e:  # noqa: BLE001 - 網路/逾時/429/5xx 等真正暫時性的錯誤才重試
            last_err = e
            wait = 2 ** attempt
            print(f"[warn] call_model 第 {attempt} 次失敗（{e}），{wait}s 後重試...")
            time.sleep(wait)

    raise OpenRouterError(f"呼叫 {model} 失敗，已重試 {max_retries} 次：{last_err}")
