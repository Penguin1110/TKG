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


def _post_chat_completion(payload: dict, max_retries: int, timeout: int) -> dict:
    """
    共用的 POST + 重試邏輯，回傳完整的 message 物件（dict，可能含 "content"
    跟／或 "tool_calls"）。call_model() 跟 call_model_with_tools() 都疊在這上面。
    """
    model = payload.get("model")
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
            return data["choices"][0]["message"]
        except OpenRouterError:
            raise  # fail-fast：直接往外拋，不進入下面的重試邏輯
        except Exception as e:  # noqa: BLE001 - 網路/逾時/429/5xx 等真正暫時性的錯誤才重試
            last_err = e
            wait = 2 ** attempt
            print(f"[warn] call_model 第 {attempt} 次失敗（{e}），{wait}s 後重試...")
            time.sleep(wait)

    raise OpenRouterError(f"呼叫 {model} 失敗，已重試 {max_retries} 次：{last_err}")


def call_model(model: str, messages: list, temperature: float = 0.0,
                max_retries: int = 3, timeout: int = 60) -> str:
    """
    呼叫 OpenRouter，回傳模型的文字回應（字串）。給 PK 探針、劇本式曝光、
    round schedule 追問這種不需要 tool-calling 的純文字場景用。

    參數：
        model: OpenRouter 的模型代號，例如 "openai/gpt-4.1-mini"、
               "anthropic/claude-sonnet-4.5"、"meta-llama/llama-4-maverick"
               （完整清單見 https://openrouter.ai/models）
        messages: OpenAI 格式的訊息列表，例如
               [{"role": "user", "content": "你好"}]
        temperature: 設 0 讓輸出盡量穩定、可重現（做研究比較適合）

    回傳：
        模型回應的純文字內容（string）
    """
    payload = {"model": model, "messages": messages, "temperature": temperature}
    message = _post_chat_completion(payload, max_retries, timeout)
    return message.get("content") or ""


def call_model_with_tools(model: str, messages: list, tools: list, temperature: float = 0.7,
                           max_retries: int = 3, timeout: int = 60) -> dict:
    """
    呼叫 OpenRouter 的 tool-calling 介面，給 graph_exploration_agent.py 的自由
    探索用。跟 call_model() 的差別：多帶 tools 參數、回傳完整的 message 物件
    （dict），不是只回傳文字——呼叫端需要看 message.get("tool_calls") 才知道
    模型選了哪個工具、帶了什麼參數。

    temperature 預設改成 0.7（不是 call_model() 的 0）：自由探索需要同一個
    起點能跑出不同路徑（--repeats-per-start 才有意義），跟 PK 探針/round
    schedule 追問要求穩定可重現的目的不一樣。

    回傳的 message dict 形如：
        {"role": "assistant", "content": "...(可能是 None 或空字串)...",
         "tool_calls": [{"id": "...", "type": "function",
                          "function": {"name": "...", "arguments": "...(JSON字串)..."}}]}
    如果模型這輪沒有呼叫任何工具，"tool_calls" 會不存在或是空列表。
    """
    payload = {
        "model": model, "messages": messages, "temperature": temperature,
        "tools": tools,
    }
    return _post_chat_completion(payload, max_retries, timeout)
