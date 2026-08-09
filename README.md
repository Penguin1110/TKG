# TKG Prior-Reversion Pilot

測試 LLM 在「先驗知識 vs 眼前更新的事實」衝突下，多輪推理過程中會不會反悔回舊答案。

流程分兩段：**Line A**（曝光）讓模型接觸一個帶時間戳的 pivot fact 更新；
**Line B**（多輪追問）曝光結束後，用交錯距離的 round schedule 連續追問下游
ripple 事實，觀察正確率隨輪數/距離怎麼變化。Line A 有兩種互斥的實作方式：

- **選項 A（劇本式固定路徑）**：模型走一段 5 步寫死的知識圖探索，順路經過
  pivot fact 節點，路徑不受模型影響——刻意排除「模型導航能力」這個變數
- **選項 B（自由探索，真實 Wikidata 圖）**：模型在真的從 Wikidata 抓下來的
  圖上用 tool-calling 自由探索，自己決定路徑，事後判斷有沒有「碰巧」撞見
  pivot 節點

兩者回答的是不同問題，不是一個取代另一個（詳見下方「選項 B」章節的取捨說明）。
Line B（追問排程、判斷、統計分析）兩個選項完全共用同一套程式碼。

## 檔案結構

```
--- 共用核心 ---
cases.json                        案例資料庫，見下方「cases.json 欄位一覽」
openrouter_client.py              呼叫 OpenRouter API 的輕量包裝（自動載入 .env，
                                   同時提供純文字 call_model() 跟 tool-calling
                                   call_model_with_tools()）
judge.py                          classify()（新/舊/hedge，給 conflict arm 用）跟
                                   classify_single()（對/錯/hedge，給 control arm 用）
run_experiment.py                 選項 A 主流程 + run_round_schedule()（Line B，
                                   兩個選項共用）→ 存成 results.jsonl
analyze.py                        彙總成「occurrence x 距離」正確率、bootstrap CI、
                                   半衰輪數、有效傳播半徑、conflict vs control 顯著性
                                   檢定、occ1→occ2 配對轉移 McNemar 檢定
relabel.py                        judge.py/cases.json 關鍵字修正後，重新分類已收集
                                   的原始回應，不必重花額度重整組實驗
backfill_checkpoints.py           一次性工具：幫沒有 checkpoint 紀錄的舊資料補寫
test_dryrun.py                    選項 A 的驗收測試，不打真的 API
pyproject.toml / uv.lock          uv 管理的依賴

--- 選項 B 專用（自由探索） ---
wikidata_graph_backend.py         真的打 Wikidata API 抓節點（facts + neighbors，
                                   帶時間戳），sqlite 本地快取，bfs_distance()/
                                   find_nodes_at_distance() 找候選起點池
fetch_wikidata_pivots.py          幫 control arm 找「同類型、瀏覽量相近、但近期
                                   沒有更替」的穩定實體（pageviews API）
graph_exploration_agent.py        自由探索的 tool-calling 迴圈（list_neighbors/
                                   view_current_node/move_to/stop_exploring）
mock_graph_fixtures.py            自由探索用的 mock 圖，給 --dry-run 跟測試共用
run_free_exploration_batch.py     選項 B 的 conflict arm 主流程
run_control_exploration_batch.py  選項 B 的 control arm（結構配對，不是路徑配對）
test_free_exploration_dryrun.py   選項 B 的驗收測試，全部用 mock，不打真的 API
```

## 安裝（用 uv）

```bash
# 沒裝 uv 的話先裝（Windows PowerShell）：
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

uv sync                 # 建立 .venv、依 pyproject.toml/uv.lock 裝依賴
cp .env.example .env
# 編輯 .env，把 OPENROUTER_API_KEY 換成你自己的金鑰
```

`.env` 已加進 `.gitignore`，不會被 commit；`openrouter_client.py` 會在啟動時自動
載入它。之後每個指令前面加 `uv run`。

---

# 選項 A：劇本式固定路徑

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
有沒有過門檻），`analyze.py` 會輸出三份 CSV：

- `summary.csv`：每個 (model, arm, distance, occurrence) 格子的正確率 + bootstrap CI，
  篩 `arm==conflict` 就是報告要畫的「輪數 x ripple距離」反悔熱力圖
- `summary_conflict_vs_control.csv`：conflict vs control 在每個格子的差異顯著性檢定，
  這是實際的 kill gate 判準——只有 `significant_reversion=True`（conflict 顯著低於
  control）的格子，才能宣稱是先驗反悔，而不是通用的 lost-in-the-middle
- `summary_paired_transitions.csv`：同一段對話裡 occurrence 相鄰兩次的軌跡轉移
  （stable_correct/reversion/recovery/stable_incorrect），對 reversion vs recovery
  做真正配對的 McNemar exact test——比 bootstrap 獨立樣本檢定更嚴謹，因為這裡
  是同一條軌跡的前後比較，不是兩組獨立樣本

## Checkpoint／續跑：想多測幾個模型時不用重花已經花過的錢

`run_experiment.py` 每完成一個 (case, model, repeat, arm) 就會寫一筆 `slot:"checkpoint"`
紀錄。下次執行時會先讀 `results.jsonl` 裡已經有的 checkpoint，已經完成的組合直接跳過，
只跑缺的部分——最常見的用法是「想多加測幾個模型」：

```bash
uv run python run_experiment.py --models "openai/gpt-4.1-mini,anthropic/claude-sonnet-4.5,google/gemini-2.5-pro" --repeats 5
```

前兩個模型如果已經跑過，會印 `[checkpoint] 跳過 ...`，只有新加的 `google/gemini-2.5-pro`
真的會打 API。也適用於「跑到一半斷線/手動中斷」的情況——重新執行同一條指令會自動接續，
只有真正跑失敗、沒完整跑完的那個 arm 會重跑（那一個 arm 可能會有少量重複資料，這是
刻意的簡化：checkpoint 是「整個 arm」這個粒度，不是每一輪都存檔，換取實作簡單）。

**如果你手上已經有「加 checkpoint 機制之前」跑出來的 `results.jsonl`**（沒有任何
`slot:"checkpoint"` 紀錄），直接重跑會被誤判成「什麼都沒跑過」而整組重來一次。
先跑一次性的補檔工具，把舊資料反推出 checkpoint 再繼續：

```bash
uv run python backfill_checkpoints.py --input results.jsonl
```

它會重算當初的 round_schedule、比對實際列數是否剛好對得上，抓不完整的 arm 會印
`[skip]` 訊息並保留原樣（下次執行 `run_experiment.py` 會整個 arm 重跑一次），完整的
才補寫 checkpoint。這個工具只需要對每份舊資料跑一次。

## 改過 judge.py 或 cases.json 關鍵字之後，不用重打 API

```bash
uv run python relabel.py --input results.jsonl --output results_relabeled.jsonl
uv run python analyze.py --input results_relabeled.jsonl
```

`relabel.py` 只是拿 `results.jsonl` 裡已經存好的 `response` 文字，套用目前版本的
`judge.classify()`/`classify_single()` 跟 `cases.json` 關鍵字重新打標籤，不會
再呼叫 OpenRouter。

## 實驗設計對應關係

| 程式裡的東西 | 對應報告裡的概念 |
|---|---|
| `pk_probe`（PK探針）+ `pk_threshold` | 確認模型的舊先驗夠強（預設門檻 80%），才有「反悔」的意義 |
| `graph_walk` / `control_graph_walk`（Line A：曝光） | 5 步劇本式圖探索，模型在探索過程中「順路」經過 pivot fact 節點，不是被明講「這是編輯」；每步都帶 `as_of` 時間戳 |
| `run_graph_exploration()` 用純文字模擬、不用 tool-calling | 讓不同模型的 tool-use 能力差異不會混進先驗反悔訊號；路徑是劇本式（模型的自由文字回覆只被記錄，不影響下一步顯示哪個節點），刻意排除「模型導航能力」這個變數，只留「曝光後會不會反悔」——這點跟 Think-on-Graph／KG-Agent 那種讓模型真的自主選路的做法不同，是有意識的方法論分歧（選項 B 就是刻意換回這個做法，見下方） |
| `ripples["1"]` / `["2"]` / `["3"]`（Line B：多輪追問） | 對應 RippleEdits 系的「距離 1／2／3」下游依賴事實，每個距離可放多個候選 |
| `control["1"]` / `["2"]` / `["3"]` | 同樣距離位置、但無版本衝突的一般事實，kill gate 對照組 |
| `build_round_schedule()` | 讓「第幾輪」跟「問哪個距離」解耦（交錯排程），避免順序本身造成偏誤 |
| `occurrence` | 這個距離被問第幾次（比原始輪數更準的 x 軸，因為排程會交錯） |
| `--repeats` + paraphrases | 同一格用不同候選/換句話說多問幾次，估變異、算 bootstrap CI |
| `analyze.py` 的 conflict vs control 顯著性檢定 | 排除純粹 lost-in-the-middle 的 kill gate 判準 |
| `analyze.py` 的 occ1→occ2 配對轉移 McNemar 檢定 | 同一條軌跡內「持有過又反悔」vs「修正」是否對稱，比獨立樣本檢定更嚴謹的第二道判準 |

---

# 選項 B：自由探索（真實 Wikidata 圖）

## 為什麼要有選項 B（跟選項 A 的取捨）

選項 A 排除了「模型導航能力」這個變數，但代價是曝光本身不是模型自己找到的，
沒辦法回答「agent 在 TKG 上真的自主探索時，有多大機率會撞見這個更新、撞見後
又會不會反悔」這個更貼近真實 agent 使用情境的問題。選項 B 換回這個真實性，
代價是要處理「自然命中率通常 <100%」——不是每趟探索都會碰到 pivot 節點，
需要 oversample（多起點、多重複）並事後篩選，而且比選項 A 貴很多（多輪
tool-calling + 篩選後只有部分軌跡能用）。兩個選項回答的是不同的問題，不是
一個取代另一個；如果你已經有選項 A 的結果，選項 B 是互補的驗證，不是重跑。

用 OpenRouter 的 tool-calling 介面（不是純文字劇本），給模型四個工具：
`list_neighbors()`、`view_current_node()`、`move_to(neighbor_id)`、
`stop_exploring()`。跟 Think-on-Graph／KG-Agent 那類論文的做法比較接近（模型的
選擇真的會影響路徑），但我們的任務指令刻意寫成中性、探索導向，不能暗示要往哪裡
走——一旦暗示，就不再是「碰巧路過」，整個自然命中率的量測就沒有意義了。

## 曝光機制怎麼運作

1. 對每個 case 的 `pivot_qid`，用 `find_nodes_at_distance()` 從 Wikidata 圖上
   反推「距離 pivot 剛好 1/2/3 步」的候選起點池
2. 每個距離抽 `--n-starts-per-distance` 個起點，每個起點跑
   `--repeats-per-start` 趟自由探索（`temperature > 0`，同一起點才可能走出
   不同路徑）
3. 每趟結束後判斷 hit（有沒有走到 `pivot_qid`）／miss：
   - **hit** → 從走到 pivot 那一刻的完整對話記錄開始，接上 `run_experiment.py`
     的 `run_round_schedule()`（Line B 的邏輯完全不用重寫，兩種曝光機制共用
     同一份追問排程程式碼）
   - **miss** → 存檔但不進入 Line B，只用來算自然命中率
4. 命中率本身是要在報告裡呈現的描述性統計（`hit_rate_by_distance.csv`），
   不是可以丟掉的副產品——一個案例如果距離 1 都很難自然命中，那本身就是一個
   值得討論的發現

## control arm：結構配對，不是路徑配對（重要的方法論轉變）

選項 A 可以讓 conflict/control 走一模一樣的 5 步、逐字對應。**自由探索下做
不到**——兩邊的起點、圖結構本來就不同，沒辦法強迫走相同路徑。改成配對雙方
的「探索預算與起點距離結構」：一樣的 `--n-starts-per-distance`、
`--repeats-per-start`、`--max-steps`，讓 conflict/control 的距離分佈跟步數
分佈對稱，而不是要求逐字相同的路徑。

**為什麼這樣配對仍然合理**：kill gate 要排除的是「距離／步數造成的通用衰退」
（lost-in-the-middle），不是要證明兩邊問了逐字相同的問題。只要兩邊「曝光要走
幾步」「追問距離分佈」對稱，這個排除目的就達到了——如果 control arm 在同樣的
步數/距離結構下也衰退，代表看到的是通用衰退；如果只有 conflict 衰退、control
沒有，才能宣稱是先驗反悔專屬現象。

control pivot 用 `fetch_wikidata_pivots.py` 挑：找「同類型、瀏覽量相近、但
近期沒有更替」的穩定實體。每個 case 的 `control_position_property_label`
（例如 `"chief executive officer"`、`"head coach"`、`"chairperson"`、
`"head of state"`）決定要檢查哪個 property 的 claim 穩不穩定——不同 case 的
職位類型不一樣，這個欄位如果沒標，`--position-property-label` 只是全域
fallback，不建議多 case 批次跑時只靠這個。

**誠實的限制**：`control_pivot_candidates` 目前是人工整理的候選 QID 清單，
不是自動查詢全 Wikidata 找同一個 class 底下的所有實體——那需要針對每個 case
的實體類型寫對應的 SPARQL 查詢，這部分留給案例作者比照原本查證 ripple 事實
時的做法（人工查證），或用 `--control-pivot-qid` 直接指定已經驗證過的候選。

## 洩漏檢查的語意重新定義

選項 A 的「洩漏」指劇本寫死的文字意外包含了 ripple 答案。選項 B 下，**模型
自己走到某個節點、正好看到下游事實，是正常的多跳推理行為，不是洩漏**——
新規則只檢查 **pivot 節點自己的 facts**（`check_pivot_leak()`），不檢查
其他節點。如果模型探索到別的節點時自然看到了 ripple 答案，會被
`find_pre_seen_ripple_distances()` 記錄成 `pre_seen_ripple_distances` 欄位
（存在 `free_explore_summary` 這筆資料列裡），`analyze.py` 的
`summarize_pre_seen()` 會統計每個 (model, case) 有多少 hit 軌跡「提早看過
答案」——這是一個分析維度，不是要排除的雜訊：如果某個模型的低反悔率主要是
因為它探索時提早看過答案（等於變相被劇透），跟「這個模型先驗真的比較穩固」
是兩件不同的事，寫報告時要對照著看。

## `--restrict-subgraph-k`：命中率太低時的折衷選項（預設關閉）

如果自然命中率太低，樣本數不夠，可以用 `--restrict-subgraph-k <k>` 把模型
能探索的地圖範圍縮小到 pivot 的 k-hop 鄰域內，其餘節點在 `list_neighbors()`
裡不會出現。模型還是完全自由選路，只是地圖變小了——跟「強迫走某條路徑」
性質不同，但**這會人為墊高自然命中率，用了這個選項就必須在報告的方法論
小節誠實交代**，不能含糊帶過說「命中率是 X%」而不提有沒有開這個選項。

## TKG 的「現在」語意：current_only（重要設計）

`wikidata_graph_backend.py` 預設只回傳一個實體「現在仍然有效」的 claims
（沒有 P582 結束時間的、或完全沒有時間 qualifier 的常態性事實），不回傳
已經結束的歷史 claim。這不只是效能/token 考量，是刻意把 Wikidata 當成
「查詢 as of 現在」的**時序**知識圖（這才是 TKG 的價值所在），不是把整份
歷史檔案攤開給模型看。

這個設計參考了 MQuAKE-T（[Zhong et al., 2023](https://arxiv.org/abs/2305.14795)）
的做法：MQuAKE-T 比較兩個時間點的 Wikidata 快照，只挑出真的代表事實更新的
diff（例如 head of government 換人），再從那個「當下」的事實往外取樣多跳
鏈，而不是把一個實體的完整歷史都當成資料源。我們原本的實作（用組織/國家
實體當 pivot、把 `fetch_node()` 抓到的所有 claims 都當成 facts）踩到的
洩漏問題（見下方「已知限制」）本質上就是沒有做這個「只看現在」的過濾，
才會讓早就結束的歷史 claim（例如某人多年前也當過同一個職位）混進 pivot
節點的內容裡，恰好撞上某個 ripple 事實的答案。改成 `current_only=True`
之後直接解決了。

## 本地 TKG 快照：`build_tkg_snapshot.py`

正式跑實驗前，建議先把會用到的圖區域抓下來存成本地快照，理由有兩個：

1. **可重現**：Wikidata 隨時有人在編輯，如果每個模型都各自即時打 API 探索，
   不同模型實際上可能看到不一樣版本的圖（例如模型 A 探索的當下某個 claim
   還沒被編輯，模型 B 探索時已經被改了）。先建好快照、固定住，才能說「這次
   pilot 裡所有模型面對的是同一份圖」。
2. **離線、不受即時 API 狀況影響**：探索是多輪 tool-calling，一個案例走幾十步
   很正常，中途因為 Wikidata 那邊速率限制或連線問題失敗會很煩人。

```bash
# 建快照：種子 = cases.json 裡所有 pivot_qid + control_pivot_candidates，
# 每個種子往外展開到 distance 3（對應 ripple distance 1-3 的需求）
uv run python build_tkg_snapshot.py --max-depth 3 --branch-cap 25

# 之後正式跑實驗時加 --offline，只用快照、快取沒有的節點直接報錯
uv run python run_free_exploration_batch.py --models "..." --offline
uv run python run_control_exploration_batch.py --models "..." --offline --control-pivot-qid Q2283
```

**`--branch-cap`（預設 25）是必要的正確性防護，不是效能微調**：像 `peru_president`
的 pivot（Q419，Peru）這種國家實體，過濾掉歷史 claim 之後仍然有 266 個
current 鄰居——沒有上限的話，distance=3 的 BFS 光是展開 distance=2 那一層
就要對每個 distance=1 節點各打上百次 API，會直接變成幾千次請求。實測抓一個
種子的 distance=2（`branch_cap=25`、`max_results=50`）花了超過 5 分鐘（含幾次
Wikidata 429 限流的重試退避）——這是即時查詢完全不可行的規模，也是為什麼
要先建快照、之後探索才能用快照的原因。設了 `branch_cap` 之後找到的候選起點
池是「圖上一部分」而不是「全部」，這點會直接反映在 `find_nodes_at_distance()`
回傳的候選數量上，寫報告時如實交代取樣範圍即可，不影響 pilot 本身的正確性。

**快照不會自動保鮮**：建好之後 Wikidata 還是持續在被編輯，快照跟「現在的
Wikidata」的落差只會越來越大。用 `--verify` 檢查種子節點（`pivot_qid`，
不含衍生鄰居）現在有沒有變動：

```bash
uv run python build_tkg_snapshot.py --verify
```

會對 manifest 裡的每個種子重新即時查一次、跟快照裡的版本比對，印出有沒有
差異——**不會自動幫你決定「差異不大所以沒關係」**，多大算「不大」是研究
判斷（例如 pivot 本身的 claim 變了 vs 只是某個不相關的旁支 claim 變了，
嚴重程度不一樣），工具只負責把差異攤開來，判斷交給人。

## 目前的資料現況（真實 Wikidata 查證結果，2026-08 查證）

7 個案例，`pivot_qid` 查證結果分兩類：

**1. 可以跑（pivot 乾淨，選項 B 現在就能用）**——4 個
- `apple_ceo`（Q312，Apple Inc.）——CEO claims 完整帶時間戳，一路到
  John Ternus 生效日
- `manutd_coach`（Q18656，Manchester United F.C.）——head coach claims
  完整對上 old/new_answer_keywords，一路到 Michael Carrick 生效日
- `fed_chair`（Q53536，Federal Reserve System）——chairperson claims
  完整對上，一路到 Kevin Warsh 生效日
- `peru_president`（Q419，Peru）——head of state claims 完整對上，一路到
  Keiko Fujimori 生效日

  這四個都已經用 `check_pivot_leak()` 對真實 Wikidata 資料驗證過乾淨。
  後三個原本各有一個 ripple 因為「同一機構的另一號人物」（Darren Fletcher
  當過過渡教練、Alberto Fujimori 也曾是秘魯總統）而被判成洩漏，改成
  `current_only` 過濾歷史 claim 之後就修好了（見上一節）；`fed_chair` 還
  額外修了一個關鍵字問題——原本 distance-2 的 `new_keywords` 有一個太
  籠統的單字 `"governor"`，會誤判到不相關的常態性事實（機構名稱裡剛好
  帶了這個字），已經換成更精確的片語。

**2. 查過，Wikidata 上沒有可用的結構，沒辦法標 pivot_qid**——3 個
- `bestbuy_ceo` —— Q533415 的 CEO claim 只有 `Hubert Joly`（沒有時間戳，
  且早就卸任），Wikidata 條目沒跟上真實世界異動
- `portugal_first_lady` / `honduras_first_lady` —— 兩國的「First Lady」在
  Wikidata 上都只是一個 `instance of: position` 的條目，沒有可查詢的
  officeholder claim（First Lady 通常不是正式憲政職位，Wikidata 沒有用
  「組織持有職位」的方式建模它）

`control_pivot_candidates` 幫這 4 個 case 都準備了候選池（同類型、長年沒換
人的穩定實體，例如 Atlético Madrid 的 Diego Simeone 自 2011 年、瑞典國王
Carl XVI Gustaf 自 1973 年），已用 `is_claim_stable()` 邏輯抽查驗證過候選
確實穩定，但 `find_stable_control_pivot()` 的自動篩選（含 pageviews 比對）
還沒有實跑過。

## 跑通的驗證方式

```bash
# 1. 全部用 mock，不打真的 API，確認機制本身是通的
uv run python test_free_exploration_dryrun.py
uv run python run_free_exploration_batch.py --dry-run --models "mock/model-A" \
    --n-starts-per-distance 2 --repeats-per-start 2

# 2. 小規模跑一次真的（4 個 case 都已驗證乾淨，先挑一個試）
uv run python run_free_exploration_batch.py --models "openai/gpt-4.1-mini" \
    --case-ids apple_ceo --n-starts-per-distance 2 --repeats-per-start 2

# 3. control arm（--control-pivot-qid 可以跳過自動篩選，先手動指定一個
#    驗證過的候選，例如 Q2283 微軟）
uv run python run_control_exploration_batch.py --models "openai/gpt-4.1-mini" \
    --case-ids apple_ceo --control-pivot-qid Q2283 \
    --n-starts-per-distance 2 --repeats-per-start 2
```

輸出寫進 `results_free_exploration.jsonl`（跟舊版 `results.jsonl` 分開存，
理由見下方「已踩過的坑」：曝光機制完全不同，混在一起分析會搞不清楚衰退是
先驗反悔還是曝光方式改變造成的）。

---

# cases.json 欄位一覽

```
pk_question / old_answer_keywords / new_answer_keywords / pk_threshold   -- PK 探針
exploration_task / graph_walk / control_graph_walk       -- 選項 A 專用
pivot_qid / control_pivot_candidates /
  control_position_property_label                        -- 選項 B 專用（見上）
ripples["1"/"2"/"3"]   -- 各距離的下游事實候選（question/paraphrases/old_keywords/new_keywords）
control["1"/"2"/"3"]   -- 各距離對應的無衝突對照事實（question/paraphrases/answer_keywords）
exposure_snippet       -- 已不使用，保留做人類查證來源
```

7 個案例涵蓋政治(3)/企業(2)/金融(1)/體育(1)：`portugal_first_lady`、
`honduras_first_lady`、`apple_ceo`、`bestbuy_ceo`、`peru_president`、
`fed_chair`、`manutd_coach`。每個案例都有 `_xxx_note` 開頭的欄位記錄查證
來源跟已知限制，讀資料時可以順便看。

---

# 已知限制 / 踩過的坑

1. **judge 是關鍵字比對，不是語意判斷**——換句話說法容易漏接，正式規模化
   時建議換成獨立 LLM 當 judge（比照 PTC／FutureBench 的作法），並抽樣做
   inter-judge agreement 驗證。
2. **bestbuy_ceo 的 distance 3 還是空陣列**——Jason Bonfig 空出的職位，
   查證當下 Best Buy 官方還沒公布繼任者，故意沒有編造。`run_experiment.py`
   的排程只會抽「非空」的距離，留空陣列可以安全先跑，等官方公布後再補。
3. **每個距離大多只有 1 個已查證候選**（外加 paraphrases）——design doc
   建議每個距離備 2-3 個候選下游事實，之後比照現有格式在 `ripples`/`control`
   底下新增元素即可，不用改程式。
4. **conflict vs control 的 bootstrap 顯著性檢定不是嚴格的 McNemar**——因為
   `summarize_cells()` 那組是兩組獨立比例的差異檢定，不是配對資料；真正配對
   的 McNemar 檢定在 `summary_paired_transitions.csv`（occ1→occ2 同軌跡轉移），
   兩份要對照著看。
5. **選項 A 的圖探索是劇本式、不是模型自主選路**——這是刻意的方法論分歧，
   不是不知道 Think-on-Graph／KG-Agent 那種做法（選項 B 就是換回那個做法）。
6. **踩過的坑：graph_walk 曾經直接洩漏 ripple 答案**——第一版 `graph_walk`
   為了讓每個節點「看起來有內容」，把 ripple 事實的答案直接寫進了節點的
   facts 裡（例如 apple_ceo 的 CEO職位節點直接寫「前任 Tim Cook 轉任
   Executive Chairman」，那正是 distance-2 問題的標準答案）。這樣一來後面
   的多輪追問測到的其實是「模型還記不記得自己兩三輪前讀過的東西」（短期
   記憶），不是「pivot fact 的更新有沒有正確傳播到下游事實」——這也是為什麼
   `results_v3_leaky_graphwalk.jsonl` 那輪 conflict arm 正確率異常高、跟
   control arm 幾乎沒有顯著差異。現在的版本已經修正：pivot 節點只講清楚
   「現在是誰、什麼時候生效」，其他節點只放跟任何 ripple 問題都無關的中性
   背景資訊，不會直接或用近似說法講出 ripple 答案。改資料內容時要注意同一條
   紀律：曝光步驟只能講清楚 pivot fact 本身，ripple 事實留給模型自己推論。
7. **踩過的坑：選項 B 的 pivot 洩漏是不同性質的問題，已修好**——不是文字
   寫死洩題，是「用組織實體當 pivot、`fetch_node()` 預設回傳完整歷史」的
   結構性限制：某個 ripple 事實剛好也是「同一機構的另一號人物」時（例如
   Alberto Fujimori 自己也曾是秘魯總統）就會撞上，被 `check_pivot_leak()`
   正確擋下。修法是把 `wikidata_graph_backend.py` 改成預設只回傳「現在仍然
   有效」的 claims（`current_only=True`），這也是更正確的 TKG 語意（查詢
   as of 現在，不是攤開整份歷史檔案）——參考 MQuAKE-T 從 Wikidata 快照 diff
   取樣事實鏈的做法，見上方「TKG 的『現在』語意」章節。4 個有 `pivot_qid`
   的案例現在都已驗證乾淨。
8. **舊版結果檔案歸檔慣例**：每次曝光機制或 schema 有重大變更，就把舊結果
   歸檔成 `results_vN_描述.jsonl`（例如 `results_v2_snippet_exposure.jsonl`
   單輪新聞摘要曝光、`results_v3_leaky_graphwalk.jsonl` 洩題版
   graph_walk、`results_legacy_v1.jsonl` 更早的 schema），不會跟目前版本的
   `results.jsonl` 混在一起分析——原因都一樣：對話長度/曝光形式不同，混著看
   會搞不清楚變化是先驗反悔還是曝光機制改變造成的。選項 B 的
   `results_free_exploration.jsonl` 也遵守同一個慣例，是全新的一份，不會
   跟選項 A 的 `results.jsonl` 混在一起。
