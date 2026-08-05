# TKG Prior-Reversion Pilot

測試 LLM 在「先驗知識 vs 眼前更新的事實」衝突下，多輪推理過程中會不會反悔回舊答案。

流程分兩段：**Line A**（曝光）讓模型走一段 5 步的劇本式知識圖探索，順路經過
帶時間戳的 pivot fact 節點；**Line B**（多輪追問）曝光結束後，用交錯距離的
round schedule 連續追問下游 ripple 事實，觀察正確率隨輪數/距離怎麼變化。

## 檔案結構

```
cases.json                        案例資料庫：pk_question、graph_walk（conflict arm
                                   的 5 步劇本式圖探索）、control_graph_walk（control
                                   arm 對應版本），以及依「ripple 距離」分組的
                                   ripples（衝突事實）跟 control（無衝突對照事實）
openrouter_client.py              呼叫 OpenRouter API 的輕量包裝（自動載入 .env）
judge.py                          classify()（新/舊/hedge，給 conflict arm 用）跟
                                   classify_single()（對/錯/hedge，給 control arm 用）
run_experiment.py                 主流程：PK探針 → 圖探索曝光（Line A）→ 交錯距離的
                                   多輪追問（Line B，conflict arm + control arm）→
                                   存成 results.jsonl
analyze.py                        彙總成「occurrence x 距離」正確率、bootstrap CI、
                                   半衰輪數、有效傳播半徑、conflict vs control 顯著性檢定
test_dryrun.py                    不用真的打 API，用模擬回應驗證流程邏輯對不對
relabel.py                        judge.py/cases.json 關鍵字修正後，用來重新
                                   分類 results.jsonl 裡已經打過 API 的原始回應，
                                   不必重花額度重整組實驗（用法見下方）
pyproject.toml / uv.lock          uv 管理的依賴
```

## 安裝（用 uv）

```bash
# 沒裝 uv 的話先裝（Windows PowerShell）：
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

uv sync                 # 建立 .venv、依 pyproject.toml/uv.lock 裝依賴
cp .env.example .env
# 編輯 .env，把 OPENROUTER_API_KEY 換成你自己的金鑰
```

`.env` 已加進 `.gitignore`，不會被 commit；`openrouter_client.py` 會在啟動時自動載入它。
之後每個指令前面加 `uv run`，例如 `uv run python test_dryrun.py`。

## 先跑一次模擬測試（不花 API 額度）

```bash
uv run python test_dryrun.py
```

會看到 conflict arm 的正確率隨 occurrence 從 100% 掉到接近 0%，control arm 全程維持
接近 100%——這正是 kill gate 要看的對比形狀，代表整條流程（含 control arm、
distance interleave、bootstrap 分析）邏輯是通的。

## 正式跑實驗

```bash
uv run python run_experiment.py --models "openai/gpt-4.1-mini,anthropic/claude-sonnet-4.5"
uv run python analyze.py
```

常用參數：
- `--models`：逗號分隔的 OpenRouter 模型代號，完整清單見 https://openrouter.ai/models
- `--case-ids portugal_first_lady,apple_ceo`：只想先跑一兩個案例時用
- `--arms conflict` / `--arms control`：只跑其中一個 arm（預設兩個都跑）
- `--rounds 8`：每個 arm 跑幾輪追問（含 distractor），design doc 建議 5-8 輪
- `--distractor-every 3`：每隔幾輪插一個無關干擾題
- `--repeats 5`：每個 case x model 重跑幾次（每次用不同候選/paraphrase 抽樣），
  增加每格樣本數，bootstrap CI 才有意義——design doc 建議 pilot 每格 5-10 次
- 結果會累加寫進 `results.jsonl`（append 模式，重跑不會覆蓋舊資料，想重新開始
  就手動刪掉這個檔案）

跑完後 `run_experiment.py` 會印出每個 case 的 PK 探針 kill gate 結果（先驗夠不夠強、
有沒有過門檻），`analyze.py` 會輸出兩份 CSV：

- `summary.csv`：每個 (model, arm, distance, occurrence) 格子的正確率 + bootstrap CI，
  篩 `arm==conflict` 就是報告要畫的「輪數 x ripple距離」反悔熱力圖
- `summary_conflict_vs_control.csv`：conflict vs control 在每個格子的差異顯著性檢定，
  這是實際的 kill gate 判準——只有 `significant_reversion=True`（conflict 顯著低於
  control）的格子，才能宣稱是先驗反悔，而不是通用的 lost-in-the-middle

## 改過 judge.py 或 cases.json 關鍵字之後，不用重打 API

```bash
uv run python relabel.py --input results.jsonl --output results_relabeled.jsonl
uv run python analyze.py --input results_relabeled.jsonl
```

`relabel.py` 只是拿 `results.jsonl` 裡已經存好的 `response` 文字，套用目前版本的
`judge.classify()`/`classify_single()` 跟 `cases.json` 關鍵字重新打標籤，不會
再呼叫 OpenRouter。實測踩過的坑：judge.py 原本 hedge 判斷排在關鍵字比對前面，
導致 Claude 那種「先講 as of my last update...、後面還是給出明確舊答案」的回答
被整句誤判成 hedge，PK 探針 kill gate 因此假性掉到 0%；改成關鍵字優先判斷、
hedge 只在完全沒有具體答案時才成立之後，用 `relabel.py` 重跑一次就修正了，
不必重花錢重整組實驗。

## 實驗設計對應關係

| 程式裡的東西 | 對應報告裡的概念 |
|---|---|
| `pk_probe`（PK探針）+ `pk_threshold` | 確認模型的舊先驗夠強（預設門檻 80%），才有「反悔」的意義 |
| `graph_walk` / `control_graph_walk`（Line A：曝光） | 5 步劇本式圖探索，模型在探索過程中「順路」經過 pivot fact 節點，不是被明講「這是編輯」；每步都帶 `as_of` 時間戳 |
| `run_graph_exploration()` 用純文字模擬、不用 tool-calling | 讓不同模型的 tool-use 能力差異不會混進先驗反悔訊號；路徑是劇本式（模型的自由文字回覆只被記錄，不影響下一步顯示哪個節點），刻意排除「模型導航能力」這個變數，只留「曝光後會不會反悔」——這點跟 Think-on-Graph／KG-Agent 那種讓模型真的自主選路的做法不同，是有意識的方法論分歧 |
| `ripples["1"]` / `["2"]` / `["3"]`（Line B：多輪追問） | 對應 RippleEdits 系的「距離 1／2／3」下游依賴事實，每個距離可放多個候選 |
| `control["1"]` / `["2"]` / `["3"]` | 同樣距離位置、但無版本衝突的一般事實，kill gate 對照組 |
| `build_round_schedule()` | 讓「第幾輪」跟「問哪個距離」解耦（交錯排程），避免順序本身造成偏誤 |
| `occurrence` | 這個距離被問第幾次（比原始輪數更準的 x 軸，因為排程會交錯） |
| `--repeats` + paraphrases | 同一格用不同候選/換句話說多問幾次，估變異、算 bootstrap CI |
| `analyze.py` 的 conflict vs control 顯著性檢定 | 排除純粹 lost-in-the-middle 的 kill gate 判準 |

## 目前 MVP 的簡化之處（之後要擴大規模時要處理）

1. **judge 是關鍵字比對，不是語意判斷**——換句話說法容易漏接，正式規模化
   時建議換成獨立 LLM 當 judge（比照 PTC／FutureBench 的作法），並抽樣做
   inter-judge agreement 驗證。
2. **bestbuy_ceo 的 distance 3 還是空陣列**——Jason Bonfig 空出的 Chief
   Customer, Product and Fulfillment Officer 一職，截至查證當下（見對話記錄的
   WebSearch 結果）Best Buy 官方還沒公布繼任者，故意沒有編造。`run_experiment.py`
   的排程只會抽「非空」的距離，所以留空陣列可以安全先跑，等官方公布後再補。
3. **每個距離大多只有 1 個已查證候選**（外加 paraphrases；apple_ceo 的
   distance 1、portugal_first_lady 的 distance 2 已經有 2 個）——design doc
   建議每個距離備 2-3 個候選下游事實，之後比照現有格式在 `ripples`/`control`
   底下新增元素即可，不用改程式。
4. **conflict vs control 的顯著性檢定用 bootstrap 取代 McNemar**——因為兩組
   問的不是同一題（不是配對資料），是兩組獨立比例的差異檢定，效果類似但
   統計上不是嚴格的 McNemar；資料量夠大後可以考慮換更嚴謹的檢定方法。
5. **圖探索是劇本式、不是模型自主選路**——跟 Think-on-Graph／KG-Agent 那種
   讓模型真的自己決定走哪條路的做法不同（那些論文測的是「模型能不能找到
   正確路徑」，我們測的是「曝光後會不會反悔」，兩個問題不同，所以刻意不讓
   路徑受模型影響）。如果之後想追加「導航方式會不會影響反悔程度」這個問題，
   可以另外做一個 ReAct 風格、路徑真的由模型決定的 arm 來比較，但不是這個
   pilot 的範圍。
6. **舊版結果檔案**：`results_v2_snippet_exposure.jsonl`（曝光只有單輪新聞摘要，
   已用 `relabel.py` 修過 judge 標籤）、`results_legacy_v1.jsonl`（更早、schema
   完全不同的版本）、`results_v3_leaky_graphwalk.jsonl`（第一版 graph_walk，見下一點）
   都是舊版曝光機制留下的資料，跟現在的 `graph_walk` 版本不能混在同一份
   `results.jsonl` 裡分析（對話長度、曝光形式都不同，會混淆「先驗反悔」和
   「曝光形式改變」兩個效應），純粹保留備查。
7. **踩過的坑：graph_walk 曾經直接洩漏 ripple 答案**——第一版 `graph_walk`
   為了讓每個節點「看起來有內容」，把 ripple 事實的答案直接寫進了節點的
   facts 裡（例如 apple_ceo 的 CEO職位節點直接寫「前任 Tim Cook 轉任
   Executive Chairman」，那正是 distance-2 問題的標準答案）。這樣一來，後面
   的多輪追問測到的其實是「模型還記不記得自己兩三輪前讀過的東西」（短期
   記憶），不是「pivot fact 的更新有沒有正確傳播到下游事實」（RippleEdits
   要測的東西）——這也是為什麼 `results_v3_leaky_graphwalk.jsonl` 那輪
   conflict arm 正確率異常高、跟 control arm 幾乎沒有顯著差異。現在的版本
   已經修正：pivot 節點只講清楚「現在是誰、什麼時候生效」，其他節點只放
   跟任何 ripple 問題都無關的中性背景資訊（出生地、任職年資、公司背景等），
   不會直接或用近似說法講出 ripple 答案。改資料內容時要注意同一條紀律：
   曝光步驟只能講清楚 pivot fact 本身，ripple 事實必須留給模型自己用「更新後
   的認知」去推論/回答，不能先劇透。
