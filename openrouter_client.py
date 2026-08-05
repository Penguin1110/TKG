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
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001 - MVP 階段先統一接住重試
            last_err = e
            wait = 2 ** attempt
            print(f"[warn] call_model 第 {attempt} 次失敗（{e}），{wait}s 後重試...")
            time.sleep(wait)

    raise OpenRouterError(f"呼叫 {model} 失敗，已重試 {max_retries} 次：{last_err}")
