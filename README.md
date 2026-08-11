# Temporal Wikipedia Explorer

這個專案測量 LLM 能否在 Wikipedia 的「頁面 × 時間」圖中自主探索，找到不同
revision 的事實並回答一個時間敏感問題。

```text
node = (Wikipedia page title, revision id)
hyperlink edge = 該 revision 正文中實際存在的連結
temporal edge = 同一 page 在兩個 revision 之間的模型切換紀錄
```

主流程只有一個 cutoff-relative 多跳問題，先做 per-model PK admission，再做 graph
navigation；不使用 ripple、control、distractor 或多輪 reversion protocol。
舊版 prior-reversion 程式與 artifacts 只留作歷史稽核，不屬於 `tkg-run`。

## 完整流程

```text
指定被測 model，從 pinned cutoff registry 取得候選 cutoff 日期
  ↓
提供一條 2–6 hop 的語意關係 seed
  ↓
逐跳抓指定日期 Wikipedia revision
  ↓
逐跳驗證 verbatim evidence + 真實 hyperlink + chain connectivity
  ↓
把關係反向巢狀組合成相對時間問題，隱藏所有中間 entity 與答案
  ↓
以 target pivot revision 建 bounded reverse BFS，確認 declared chain 是最短路徑
  ↓
獨立 LLM multi-hop semantic judge
  ↓
每個被測模型做 fresh-context PK probes
  ↓
只保留不知道 target answer 的題目 × 模型配對
  ↓
固定從關係鏈 anchor page 開始，pivot title 不提供給模型
  ↓
反覆 switch_snapshot / follow_link
  ↓
submit_answer
  ↓
獨立 LLM answer judge
  ↓
JSONL、score CSV、動態 trajectory viewer
```

## 安裝

```bash
uv sync
cp .env.example .env
# 在 .env 設定 OPENROUTER_API_KEY
```

## Multi-hop Case 格式

```json
{
  "id": "example_ceo_chain",
  "category": "corporate",
  "wikipedia_title": "Former spouse",
  "wikipedia_before": "2024-06-01",
  "wikipedia_as_of": "2026-01-01",
  "start_title": "Example Corp",
  "hide_pivot_title": true,
  "temporal_question": "At the target snapshot, who is the former spouse of the spouse of the second successor as chief executive officer to the person who was chief executive officer of Example Corp at the tested model's registered knowledge cutoff?",
  "new_answer_keywords": ["Former spouse"],
  "reasoning_hop_count": 4,
  "expected_navigation_distance": 5,
  "knowledge_cutoff": {"cutoff_date": "2024-06-01", "model_ids": ["openai/gpt-4.1-mini"]},
  "reasoning_chain": ["full revision/link/evidence records omitted here"]
}
```

新版 CLI 預設讀取 generator 寫出的 `generated_cases.json`。舊 `cases.json` 的
`pk_question` 只供明確啟用的相容模式讀取。新版 PK 直接使用
`temporal_question` 與相同 target date，不需要另存一份問題。

## 多跳出題

Cutoff registry 參考 `HaoooWang/llm-knowledge-cutoff-dates`，並在程式內保存來源 URL
與 pinned commit。它只用來篩選 cutoff 之後的候選事件，不代表模型一定不知道該事實；
PK admission 才是正式 gate。未知 model ID 會 fail closed，不猜日期。

先依 `examples/multihop_seeds.example.json` 建 seed。每個 hop 必須提供：

- `source_title`、`target_title`、`as_of`。
- 語意關係 `relation`。
- 含一個 `{source}` 的 `relative_clause`，供程式組合問題。
- source revision 的逐字 `evidence` 與 `target_aliases`。

程式不把「有 hyperlink」自行解讀成 spouse／successor；relation seed 提出語意，Wikipedia
revision 與獨立 judge 負責驗證：

```bash
uv run tkg-generate-multihop \
  --seed-file examples/multihop_seeds.example.json \
  --judge-model "INDEPENDENT_JUDGE_MODEL" \
  --output multihop_question_packets.jsonl \
  --cases-output generated_multihop_cases.json
```

只跑 Wikipedia deterministic gates、不花 LLM judge 費用：

```bash
uv run tkg-generate-multihop \
  --seed-file MY_SEEDS.json \
  --validate-only \
  --output validation_packets.jsonl
```

多跳 deterministic gates 會拒絕：少於 2 hops、cutoff 後沒有新 hop、revision 原文不符、
缺少 source→target hyperlink、鏈不連續、鏈自己繞圈、問題洩漏中間 entity/答案。runner
建完 arena 後還會拒絕不是 anchor→pivot 最短路徑的鏈。

### 單頁變更出題（保留）

出題器比較同一 Wikipedia page 的 before/after revision，只產生一個問題：

```bash
uv run tkg-generate-cases \
  --title "Prime Minister of Canada" \
  --before 2025-03-13 \
  --after 2025-03-15 \
  --generator-model "GENERATOR_MODEL" \
  --judge-model "INDEPENDENT_JUDGE_MODEL"
```

輸出：

- `generated_question_packets.jsonl`：所有候選、原始模型輸出與退件理由。
- `generated_cases.json`：通過機器檢查、等待人工 review 的 cases。

正式執行前可用 `uv run tkg-validate-cases` 檢查 strict temporal schema；舊資料只有在
明確使用 `--cases cases.json --allow-legacy` 時才會通過相容驗證。

單頁模式的 deterministic checks 不使用模型，會確認：

- before/after evidence 是對應 revision 的逐字 substring。
- old/new aliases 確實出現在各自 evidence。
- old/new 答案不重疊。
- 問題不洩漏答案。
- 問題格式與必要欄位完整。

之後由另一個 LLM judge 確認兩段證據描述同一個 current-valued property、變更已生效，
不是公告、推測或歷史敘述。

## PK Admission

Wikipedia revision 只能證明事實有變，不能證明某個模型不知道新值。因此 runner
會在建圖前，對每個「題目 × 被測模型」執行 fresh-context PK probes：

- 問與正式任務相同的 `temporal_question` 與 target date。
- 不給 Wikipedia、tools、arena 或先前對話。
- 每次 probe 都是新對話，且不會把 PK response 塞回後續 navigation context。
- 獨立 judge 標記 `stick_new` / `stick_old` / `hedge` / `unsupported` /
  `unjudgeable`。
- 預設做 3 次，`stick_new` 比例必須為 0，且不得有 `unjudgeable`，才能進入建圖與導航。

這是 per-model admission：同一題可能對模型 A 是未知、對模型 B 已知，不會把 case
本身宣稱為所有模型都不知道。Cutoff-relative case 另外綁定 exact model ID；不同 cutoff
的模型不能共用同一題而偷偷改變 anchor 的意思。

## 建立最短路徑 Arena

以 `(pivot page, target revision)` 為距離 0，沿 verified historical backlinks 做 reverse
BFS；同 page 的不同 revision 之間加入 temporal edges：

```bash
uv run tkg-snapshot \
  --case-ids example_ceo \
  --snapshot-dates 2024-01-01,2025-01-01 \
  --max-depth 3 \
  --branch-cap 25
```

Wikipedia 原圖可以有環；正式 reasoning chain 本身不能有環。候選 nodes 蒐集完後，程式會在 bounded induced graph 上重新計算
一次完整 reverse BFS，因此每個 node 的 `distance_to_pivot` 是這張 arena 內的精確最小值；
pivot 永遠是 0，也不會因環再次出現在更深層。同一 page 的不同 revision 不會被合併。

歷史 backlinks 的候選來自 MediaWiki 目前的 backlink index，再逐一用指定日期 revision
驗證。因此已回傳的 edge 是真的，但早已刪除且目前 backlink index 找不到的歷史 edge
可能漏掉；「最短」是對本次 manifest 保存的 bounded arena 而言，不宣稱全 Wikipedia。

## 模型可用 Tools

- `switch_snapshot(as_of, brief_reason)`：切換目前 page 的時間，可重複呼叫。
- `view_current_page()`：閱讀目前 page revision。
- `list_links()`：列出目前 revision 真正存在的 outgoing hyperlinks。
- `follow_link(target)`：沿目前 revision 的 hyperlink 移動，時間保持不變。
- `search_within_page(query)`：搜尋目前 revision 的可見正文。
- `submit_answer(answer)`：提交最終答案並結束。

模型第一個有效動作必須先選一個 snapshot，但之後可以不限次數切換。多跳題會告訴模型
其 registered cutoff snapshot 與 target snapshot，但不告訴 pivot title。切換時間時保留
page title，並重新載入該日期的 revision 與 links。題目的 `wikipedia_as_of` 會明確告訴
模型那個日期是最終答案的 target；before、target 與其他 allowed dates 都仍可自由切換。
模型只可沿 arena 內仍真實存在的 hyperlinks 移動，外部連結會保留在頁面文字中但不能作為
本次 graph transition。

## 執行

```bash
uv run tkg-run \
  --models "TESTED_MODEL" \
  --judge-model "INDEPENDENT_JUDGE_MODEL" \
  --case-ids example_ceo \
  --snapshot-dates 2024-01-01,2025-01-01 \
  --start-distance 3 \
  --backlink-branch-cap 25 \
  --pk-repeats 3 \
  --pk-max-known-rate 0 \
  --repeats 3 \
  --max-steps 16 \
  --offline
```

若 case 是新版 generator 產生的，`--snapshot-dates` 可省略，runner 會使用保存的所有
required hop dates。多跳 case 固定使用 seed 的 `start_title`；`--start-distance` 只作為
arena 至少要建多深的下限。

輸出：

- `temporal_results.jsonl`
- `temporal_scores.csv`

JSONL 會保存每次 tool call 的 page title、revision、來源/目的日期、tool output、完整
visited versions、最終答案與 judge 結果。另保存：

- `pk_probe`：每次 fresh-context 問題、原始回答與 judge label。
- `pk_gate`：題目 × 模型的新/舊/其他回答率與 admission 結果。
- `navigation_arena`：本次 bounded graph 的 nodes、edges、距離與 graph hash id。
- `distance_to_pivot`：每次 navigation 後的最短剩餘距離。
- `shortest_navigation_steps`：包含第一次選時間的理論最少步數。
- `actual_steps_to_first_pivot` 與 `detour_steps`。
- `revisit_count`、`cycle_detected`、`shortest_arrival`。

`view_current_page`、`list_links`、`search_within_page` 不算移動；成功的
`switch_snapshot` 與 `follow_link` 各算一步。API error 不會寫 complete checkpoint。

## Answer Judge

Judge 和被測模型必須不同，labels 為：

- `correct_after`
- `old_snapshot_answer`
- `supported_other_time`
- `unsupported`
- `no_answer`
- `unjudgeable`

Judge 只會看到模型實際讀過的 revision evidence。正式實驗前可用人工 gold 驗證：

```bash
uv run tkg-validate-judge \
  --gold judge_gold.jsonl \
  --judge-model "JUDGE_MODEL" \
  --min-agreement 0.90
```

## Visualization

```bash
uv run tkg-visualize \
  --cache wikipedia_snapshot.db \
  --results temporal_results.jsonl
```

Viewer 初始只顯示 pivot 的 target revision。播放 trajectory 時：

- 模型切到另一時間：產生同 page 另一 revision node 與紫色 temporal edge。
- 模型進入某 revision：只展開該 revision 的一層 outgoing hyperlink nodes。
- 沒有實際抵達的 page 不會展開下一層。
- 未下載的 hyperlink target 顯示為灰色 stub。
- 黃色 edge 是模型實際走過的 hyperlink trajectory。
- 時間軸倒退時，尚未發現的 nodes/edges 會消失。
- hover node 可查看 revision id、timestamp、`distance_to_pivot` 與摘要；點擊 node 後可
  開啟 exact revision URL。
- Run badges 顯示 PK admission、pivot hit/miss、shortest/detour、cycle/revisit。

只建立靜態網站：

```bash
uv run tkg-visualize --build-only --output-dir .tkg_visualization
```

## 測試

```bash
uv sync
uv run pytest
uv run mypy
uv run pyflakes src tests
uv build
```

離線測試涵蓋相對多跳組合、逐跳 evidence/link gates、隱藏 pivot、最短鏈拒絕、fresh-context PK admission、含環 graph 的精確最短距離、外圍 page 到 pivot 的
最短 trajectory、重複時間切換、independent judge，以及 page-revision 動態圖不提前洩漏
下一層。

## 舊資料

根目錄既有 `results*.jsonl`、`summary*.csv`、`report.*` 與舊 graph manifest 皆為
prior-reversion 實驗 artifacts。程式禁止新版 runner 覆寫它們；舊 Python 實作保存在
`legacy/` 或僅供 regression test 使用。
