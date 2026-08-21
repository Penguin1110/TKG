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

老師報告版的研究動機、方法設計與 pilot 解讀，見
[Temporal Wikipedia Graph QA：研究動機、設計與小樣本結果](docs/advisor_report_motivation_design_and_pilot.md)；
完整工程 gates、action traces 與 artifacts 則見
[TKG 目前出題流程與小樣本結果](docs/current_question_pipeline_and_smoke_results.md)。

## 完整流程

```text
指定被測 model，從 pinned cutoff registry 取得候選 cutoff 日期
  ↓
以 Wikipedia revision/link change、通過 profile 的 Wikidata 時間 qualifiers，
或人工/LLM seed 提出 2–6 hop 候選（保存 candidate-source provenance）
  ↓
逐跳抓指定日期 Wikipedia revision
  ↓
逐跳驗證 verbatim evidence + 真實 hyperlink + 舊版尚未出現下一 target
  ↓
把每個 relation 寫成一個短步驟，隱藏所有中間 entity 與答案；另存 nested canonical form 稽核
  ↓
建立 product-graph semantic waypoints；另算 raw hyperlink shortest 作診斷
  ↓
獨立 LLM multi-hop semantic judge
  ↓
每個被測模型做 factorized fresh-context PK probes
  ↓
只保留不知道關鍵 post-cutoff bridge fact 的題目 × 模型配對；
tail 與完整 composed answer 分開報告
  ↓
只對 PK-admitted cases 做 hash-bound human approval
（無人力的 machine screening 不冒充 approval，也不跑 A/B/C/D）
  ↓
從關係鏈 anchor page 的 cutoff revision 開始，pivot title 不提供給模型
  ↓
按需 list_revisions / switch_snapshot / follow_link
  ↓
submit_answer
  ↓
accepted-alias deterministic judge；只有模糊回答才呼叫獨立 LLM judge
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
  "temporal_question": "Start with Example Corp. First, identify the person who was its chief executive officer at the registered cutoff. Next, identify that person's second successor. Next, identify that person's spouse. At the target snapshot, who is that person's former spouse?",
  "new_answer_keywords": ["Former spouse"],
  "reasoning_hop_count": 4,
  "expected_navigation_distance": 7,
  "required_temporal_switches": 3,
  "semantic_shortest_distance": 7,
  "temporal_waypoints": ["ordered (entity, revision, as_of) records omitted here"],
  "knowledge_cutoff": {"cutoff_date": "2024-06-01", "model_ids": ["openai/gpt-4.1-mini"]},
  "reasoning_chain": ["full revision/link/evidence records omitted here"]
}
```

新版 CLI 預設讀取 generator 寫出的 `generated_cases.json`。每個 case 另存
`factorized-prior-knowledge-v2` contract，分別定義 anchor、所有 designated acquisition
bridges、attribute tail 與 composed probes；每條 acquisition bridge 都必須通過 PK gate。
Accepted case 同時內嵌 exact rendered pages、links 與 content-addressed hash manifest。
沒有這份 contract 的舊 case 只走相容模式。

## 多跳出題

Cutoff registry 參考 `HaoooWang/llm-knowledge-cutoff-dates`，並在程式內保存來源 URL
與 pinned commit。它只用來篩選 cutoff 之後的候選事件，不代表模型一定不知道該事實；
PK admission 才是正式 gate。未知 model ID 會 fail closed，不猜日期。

先依 `examples/multihop_seeds.example.json` 建 seed。每個 hop 必須提供：

- `source_title`、`target_title`、`as_of`。
- 語意關係 `relation`。
- 含一個 `{source}` 的 `relative_clause`，供程式組合問題。
- source revision 的逐字 `evidence` 與 `target_aliases`。

任何 relation clause 若聲稱 `first` / `next`，還必須附
`wikidata-event-order-v1` certificate：固定 world-event boundary、coverage end、完整候選
events、selected date/QID 與 source hash。選中事件不是 boundary 後唯一最早事件、SPARQL
結果碰到 LIMIT，或 certificate/hash 被改動時一律 fail closed。

temporal hop 必須換到更晚的日期；最後可接一個有結構化 statement 的同 snapshot
attribute tail。因此正式 temporal spine 固定交錯：

```text
H -> T -> H -> T -> H ...
```

先用 Wikidata `P286` 的 `start time` / `end time` 找教練跨隊候選，再以 exact Wikipedia
revision 做 deterministic 驗證：

```bash
uv run tkg-discover-temporal-candidates \
  --model-id openai/gpt-4.1-mini \
  --popularity-month 2026-07 \
  --popular-prefilter 100 \
  --tails-per-entity 2 \
  --max-per-relation-family 4 \
  --max-per-property 1 \
  --output discovered_multihop_seeds.json \
  --packets-output candidate_discovery_packets.jsonl
```

候選事件日期會測試 `+0/+1/+3/+7` 日，處理 Wikidata 生效日與 Wikipedia 更新日的落差。
只有 cutoff 頁有第一個人、換時間後才出現下一 entity、再次換時間後才出現答案的鏈會輸出。
不同日期若解析到同一個 immutable revision，只會下載並 parse 一次；其後以 revision ID
重用正文與 hyperlinks。attribute-tail 的 property/family quota 填滿後 discovery 會提早停止，
而且每完成一個候選就立即 flush discovery packet，長時間執行時可以直接追蹤進度。

### Renewable temporal relation registry

Temporal relations 不再只存在於 Python 分支。版本化 bootstrap registry 目前包含 leadership、
politics、career、sports、affiliation、education、family、geography、ownership、organization、
awards 與 participation 等 24 個 entity-valued Wikidata properties。先用 profiler 對指定時間窗
取樣，再以少量 Wikipedia revisions 測量實際 hyperlink yield：

```bash
uv run tkg-profile-temporal-relations \
  --since 2024-06-01 \
  --until 2026-08-12 \
  --properties P286,P169,P488,P6,P35,P108,P54,P26 \
  --kg-limit 10 \
  --time-buckets 3 \
  --wiki-samples 2 \
  --judge-model openai/gpt-4.1 \
  --judge-workers 4 \
  --mine-properties 50 \
  --output temporal_relation_profile.json \
  --packets-output temporal_relation_profile_packets.jsonl
```

輸出逐 relation 保存 KG 候選數、日期 qualifier 完整度、Wikipedia hyperlink 支援率、
前一日 novelty 與 promotion recommendation。新 property 只能成為 `discovered`；link gate
達到門檻也只會成為 `wikipedia_link_candidate_semantic_review_required`，必須再通過 relation
語意審核。指定 `--judge-model` 時，link-supported samples 會用內容雜湊快取及 bounded
`--judge-workers` 並行審核；API failure 與 semantic reject 分開統計。通過後仍只會成為
`semantic_validated_candidate_human_review_required`，profiler 本身不會改寫 registry 或放寬正式題目 gates。新 property discovery
使用 bounded Wikidata RecentChanges sample，不以昂貴的全域 WDQS aggregate 阻塞既有 relation profiling。
長時間窗若只用 `--time-buckets 1`，`ORDER BY` 會偏向窗口末端；正式 renewable run 應固定
2–4 個時間桶，使早／中／晚事件都有 bounded 配額。每個桶的日期範圍、limit 與 query hash
都併入 profile query manifest。

### Registry renewal

Profiler 的 `property_mining.candidates` 不再是死路。renewal command 會把新 property 以
`discovered` 狀態寫入一個**新的**版本化 registry；之後可在下一個 profiling window 對它
取樣。通過 Wikipedia link 與 semantic judge 的 relation 先成為 `validated`，只有明確列入
`--activate-properties` 才成為 `active`：

```bash
uv run tkg-renew-relation-registry \
  --profile temporal_relation_profile.json \
  --registry-version renewal-2026-08-12 \
  --activate-properties P286,P6,P35 \
  --output temporal_relations.renewed.json
```

若 preregistration 明確採用全自動 semantic-threshold promotion，可改傳
`--activate-validated`；輸出 provenance 會記錄這個 policy，不會把自動 promotion 當成人工
review。未經 semantic validation 的 `discovered` relation 無法啟用。每次輸出保存 base
registry version、所有 profile SHA-256、promotion decisions 與 deprecated properties。

### Renewable multi-hop question engine

正式 renewable engine 直接讀取一個或多個 immutable relation profiles，而不是固定 P286
分支。預設 contract 是 4–5 relation hops、至少 2 個 relation families、每換一個 entity 後
snapshot 必須嚴格往後、至少 2 次 temporal switches、同一 temporal property 最多使用 3 次、不可重複
entity：

```bash
uv run tkg-renew-questions \
  --model-id openai/gpt-4.1-mini \
  --registry temporal_relations.renewed.json \
  --profile temporal_relation_profile.json \
  --until 2026-08-12 \
  --popularity-month 2026-07 \
  --judge-model openai/gpt-4.1 \
  --judge-workers 4 \
  --ledger-path renewable_question_ledger.db \
  --seeds-output renewable_multihop_seeds.json \
  --packets-output renewable_question_packets.jsonl \
  --cases-output renewable_generated_cases.json
```

CLI 只接受 registry 中 `active` 且 profile recommendation 已通過 semantic validation 的交集；
只有 profile pass、但尚未經 renewal activation 的 relation 不會進入正式出題搜尋。

搜尋同時支援兩個有明確語意的方向：

```text
forward: team --P286--> coach
inverse: coach --inverse(P286)--> the next team that appointed that coach
```

inverse 不是把箭頭偷偷倒過來；registry 必須提供 `inverse_relative_clause` 與 answer kind，
Wikidata statement 仍保存原始 subject/object/direction。每條後續 edge 只考慮目前 snapshot
之後最早的 `start time` 或 `point in time`。同日多個 target、日期缺少日精度、無 enwiki
sitelink 或新版沒有真實 hyperlink 都會 fail closed。若 target hyperlink 在舊 revision 已因
其他歷史角色出現，預設仍拒絕；formal mode 可由 cached edge-contrast judge 證明「新版明確
表達指定 relation、舊版沒有表達同一 relation」後放行。這個 override 保存 before/after
revision IDs、逐字 evidence、judge model/confidence/reason 與 raw-response hash，正式 validator
會重新核對，不能只在 seed 裡寫一個 `pass`。
`selection_policy` 明確區分 `single_active`、`latest_start` 與 `latest_point`，避免把多值
property 任選成唯一答案。

預設允許在至少 3 個嚴格遞增的 temporal hops 後接一個同 snapshot attribute tail；tail 仍需
Wikidata cardinality selector、真實 revision hyperlink、逐字 evidence、無 entity cycle，且
必須讓全題達到 relation-family diversity。可用 `--no-terminal-attribute-tails` 要求所有 hops
都必須是 temporal；這個較嚴格設定可能在短 cutoff window 得到零題，零 yield 會原樣回報。
預設也排除 `easy` tails（citizenship、birthplace、native language），優先 education、career、
team、party、award、membership 等較難由人物先驗猜中的答案；只有顯式傳
`--allow-easy-tails` 才會重新納入。

候選通過 beam search 後仍須依序通過：

```text
MultiHopSeed contract
  → exact revision/evidence/hyperlink gates
  → bounded arena semantic-shortest audit
  → independent whole-chain LLM judge
  → machine_pass_human_review_required
  → runner 的 per-model fresh-context PK admission
```

生成資料的三層 LLM gate（relation profile、before/after edge contrast、whole-chain）都要求固定名稱的
boolean `checks` 全為 true；只回 `decision=pass`、缺欄位或回自由格式陣列都會 fail closed。
prompt version 會進 cache key，修改 judge contract 不會沿用舊判定。

`renewable_question_ledger.db` 以 model cutoff、QID path、property＋direction sequence、snapshot
dates 與 answer QID 建 content fingerprint。改寫題目文字不會製造假新題；deterministic/judge
reject 與已接受題目預設不再生成，API/網路 failure 則可重試。`--retry-rejected` 是明確的
再審政策。seed artifact 保存 registry/profile hashes、搜尋 contract、beam score、完整 QID path
與 qualifiers；JSONL 保存所有 anchor/expansion rejection reason。只想稽核 discovery 而不花
judge 費用可傳 `--seeds-only`。

這個 engine 產生的是等待 review 的 case，不宣稱模型必然不知道答案；正式未知性仍由下游
PK admission 對每個「題目 × 被測模型」獨立判定。

預設出題不是停在最後找到的人物，而是在 target snapshot 接一個經驗證的 attribute tail。
白名單包含 birthplace/citizenship、education、employer/position、sports team、political
party、spouse、award 與 organization membership；其中前述 `easy` 類別預設不取樣。每個 tail 必須同時具備：

- Wikidata statement 與明確的 `single` / `current` / `latest` selector；多答案無法唯一化就拒絕。
- target Wikipedia revision 內真的可見的 target hyperlink 與逐字 evidence。
- target entity 自己也有 Wikipedia page，並且不造成 entity cycle。

候選 spine 先以 Wikimedia Analytics API 的固定月份 top-pageviews 排序；輸出保存月份與各
entity views。最後的 sampler 同時對 property ID、relation family、property sequence 與 anchor/source entity 設硬上限，
避免熱門題全部集中在 birthplace、spouse 或同一人物。若要重播舊 P286-only 行為，可傳
`--no-attribute-tails`；若
刻意不要熱門度排序，必須顯式傳 `--no-popularity-ranking`。

目前的體育 v4 實例在 `examples/popular_diverse_v4/`。實際非體育 artifacts 在
`examples/non_sports_v4/`：兩題分別覆蓋 politics/career/family 與
politics/career/geography，另保存一條 Wikidata-first deterministic rejection。所有通過案例
都仍是 `machine_pass_human_review_required`，不是人工核准題目。

### Candidate source：Wikipedia-first 與 Wikidata-first

Wikidata 不是正式證據，也不是必要入口。統一流程是：

```text
Wikipedia revision/link changes ─┐
Wikidata temporal statements ────┼→ candidate edge + provenance
manual/LLM hypotheses ───────────┘
                                  ↓
                    exact Wikipedia revision proof gate
                                  ↓
              whole-chain judge → per-model PK → admitted-only human review
```

Wikipedia-first 能補到 Wikidata 漏掉的歷史角色；Wikidata-first 適合大量提名與日期/QID
cross-check，但每個候選仍可能因 Wikipedia 沒有可見 hyperlink、關係已提前出現或動態 template
展開造成未來資訊洩漏而被拒絕。`action=parse&oldid=...` 的 template 是用今天的 template 狀態
展開；因此 gate 以「模型實際會看到的 rendered page」為準，若舊 snapshot 已顯示後來 target，
不會因 Wikidata 日期正確就放行。

比較 discovery source 時使用相同 proof gates：

```bash
uv run tkg-compare-candidate-sources \
  --packets wikipedia_packets.jsonl wikidata_packets.jsonl \
  --output candidate_source_comparison.json
```

報告分開統計 deterministic pass、whole-chain judge pass、semantic-contrast 與 verified redirect；
任一來源少於 20 題時會標示不可排名，避免用 smoke 樣本宣稱哪個來源較好。

`examples/popular_diverse/` 是舊 v3 provenance，保留稽核但不再作為目前可執行案例。

### 先大量產生 KG 題目候選

若要先累積題目、之後再分批跑昂貴的 Wikipedia 與 LLM gates，可使用 candidate-only CLI：

```bash
uv run tkg-generate-candidate-batch \
  --model-id openai/gpt-4.1-mini \
  --registry temporal_relations.renewed.json \
  --profile temporal_relation_profile.json \
  --until 2026-08-14 \
  --max-questions 100 \
  --output questions.json \
  --packets-output query_packets.jsonl \
  --markdown-output QUESTIONS.md
```

新批次的 relation 不再由 Python 內固定的 property 清單決定。正式模式會取 renewed registry
中 `active` 與 immutable profile 中 semantic pass 的交集，再依
`candidate_topologies.v1.json` 的版本化 topology contract 自動選出可執行 relations；手動傳
`--properties` 或 `--staged-properties` 只能縮小這個集合，不能繞過 admission。staged adapter
也會檢查相依 relation，例如 `P39` office succession 必須同時核准 `P39` 與 `P1308`。
18 個 attribute tails 同樣記錄在 topology registry，不再藏在 Python tuple。

如果只是要延續早期的 cheap discovery、尚未完成 profile/renewal，必須明確加上
`--allow-bootstrap-candidates`。這個模式仍只產生 provisional candidates，輸出 metadata 會標成
`explicit_provisional_bootstrap`，不能冒充正式 admission。`--reuse-batch` 與 `--merge-batches`
不要求重跑 admission，因為它們只重寫/合併既有 artifact，不重新發現 relation。

它以 Wikidata qualified statements 建立固定的 4-hop private chain，題面不公開中繼日期。
除了傳統的 `entity → person → entity → person` leadership topology，也支援分段抓 event、
本地 join 的 `specific office → holder → later office → next officeholder`，其 relations 為
`P39 → P39 → P1308`。尾端可輪替 geography、identity、career、language、education、family、
awards、name、works 等 relation。輸出保留答案、QID、完整 chain、查詢與 provenance，但所有題目
一律標成 `pending_wikipedia_validation`；若實際產量少於 `--max-questions`，CLI 會寫出已找到的
候選並以非零狀態結束。這些候選不得直接送進正式 runner，仍需依序通過 event-order、exact
Wikipedia revision/hyperlink、whole-chain judge、per-model PK 與 admitted-only human review gates。

`--max-sports-share`、`--max-topology-share` 與 `--max-per-anchor` 是 hard caps；供給不足時寧可
少於指定題數，也不會突破配額補滿。`P39` discovery 只接受有 `P1308 officeholder` statements
且有 English Wikipedia sitelink 的 office item；這是 candidate heuristic，仍不能取代正式的
singleton-office 與完整事件順序稽核。

目前較平衡的 100 題候選與檢查摘要在 `examples/question_batch_diverse_v5/`；它與其他
`examples/question_batch_*` 都保留為歷史 artifact，不因新 admission contract 被刪除或覆寫。
舊的 `examples/question_batch_100_v1/` 保留作為 80% 運動偏差的歷史 artifact，不再是推薦批次。

程式不把「有 hyperlink」自行解讀成 spouse／successor；relation seed 提出語意，Wikipedia
revision 與獨立 judge 負責驗證：

```bash
uv run tkg-generate-multihop \
  --seed-file examples/multihop_seeds.example.json \
  --generator-model openai/gpt-5.4-mini \
  --judge-model openai/gpt-5-mini \
  --judge-workers 4 \
  --output multihop_question_packets.jsonl \
  --cases-output generated_multihop_cases.json
```

Judge 預設最多四個並行 request，結果依 model、prompt contract、問題與 verified chain 的
內容雜湊保存在 `<cache-path>.judge.db`；重跑相同題目不會再次付費。可用
`--judge-workers` 與 `--judge-cache-path` 調整，但輸出仍維持 seed 原始順序。

Question writer 與 judge 必須使用不同模型。Writer 會看到已驗證的完整 private entity
chain，以便正確理解 person／position／place 與指涉，但不會看到中繼 oracle 日期；被測模型
只看到通過 gate 的題目。任何 hidden canonical title 或 alias 出現在題目中都會被 deterministic
gate 退回重寫。每次換時間的步驟必須以世界事件為邊界，例如「前面找出的人開始擔任該職位
之後，下一位開始擔任者」，而不能以模型查看的 previous snapshot 為邊界。`P39` 另要求
`first/next` 與 tenure-start 語意；獨立 judge 再檢查證據、唯一性、時間清楚度與自然語句。

只跑 Wikipedia deterministic gates、不花 LLM judge 費用：

```bash
uv run tkg-generate-multihop \
  --seed-file MY_SEEDS.json \
  --validate-only \
  --output validation_packets.jsonl
```

多跳 deterministic gates 會拒絕：少於 2 hops、相鄰 hop 沒有使用更晚時間、revision
原文不符、缺少 source→target hyperlink、舊 revision 已提前出現下一 target、鏈不連續、
鏈自己繞圈、問題洩漏中間 entity/答案。

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

Machine pass 不能直接進正式 runner。先建立 hash-bound draft，人工逐題檢查問題、日期、
答案 aliases 與 evidence chain，再把 `pending` 明確改成 `approved` 或 `rejected`：

```bash
uv run tkg-create-review-template \
  --cases generated_cases.json \
  --output human_review.json
```

reviewer 必須填姓名／代號與 ISO-8601 `reviewed_at`；case 任何受審內容改動都會使 SHA-256
失效。若只是工程 smoke 而明確不做人工審查，runner 只接受
`--waive-human-review`，且 artifact 會記成 `waived_by_user`，絕不記成 approved。

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

- `critical_bridge` 逐一直接問所有 designated cutoff 後 acquisition edges；每一條都控制
  admission，不只檢查最後一條 bridge。
- `tail` 單獨問既有 attribute，例如配偶或出生地，用來確認 composition 所需舊知識。
- `composed` 再問完整多跳題，但只作診斷，不控制 admission。
- 不給 Wikipedia、tools、arena 或先前對話。
- Tested-model prompt 不含 probe role/objective、cutoff、target date、admission policy 或
  `must_be_unknown`；這些 metadata 只存在 private logs 與 judge context。
- 每次 probe 都是新對話，使用不同的簡短 elicitation wording 與 sampling temperature，且不會把 PK response 塞回後續 navigation context。
- factorized labels 為 `known` / `unknown` / `wrong` / `unjudgeable`。Composed probe 另帶
  intermediate aliases；judge 若把 David Lammy 之類的中繼答案當成 final answer，會被
  deterministic contract guard 改成 `unjudgeable`，不會污染 critical-bridge gate。
- 預設每個 probe 做 5 次，temperatures 為 `0,0.2,0.5,0.7,1.0`；`known` 比例必須逐
  critical bridge 不超過 `--pk-max-known-rate`，且不得有 `unjudgeable`，才能導航。

只做 PK 時使用 `--pk-only`；此模式不開 Wikipedia backend、不建 graph、不跑 trajectory。
完整 run 也先做 PK，只有 admitted model 才建 navigation arena。

這是 per-model admission：同一題可能對模型 A 是未知、對模型 B 已知，不會把 case
本身宣稱為所有模型都不知道。Cutoff-relative case 另外綁定 exact model ID；不同 cutoff
的模型不能共用同一題而偷偷改變 anchor 的意思。

## Raw Arena 與 Semantic Route

以 `(pivot page, target revision)` 為距離 0，沿 verified historical backlinks 做 reverse
BFS；同 page 的不同 revision 之間加入 temporal edges：

```bash
uv run tkg-snapshot \
  --case-ids example_ceo \
  --snapshot-dates 2024-01-01,2025-01-01 \
  --max-depth 3 \
  --branch-cap 25 \
  --max-nodes 500
```

Wikipedia 原圖可以有環；正式 reasoning chain 本身不能有環。候選 nodes 蒐集完後，程式會在 bounded induced graph 上重新計算
一次完整 reverse BFS，因此每個 node 的 `distance_to_pivot` 是這張 arena 內的精確最小值；
pivot 永遠是 0，也不會因環再次出現在更深層。同一 page 的不同 revision 不會被合併。

歷史 backlinks 的候選來自 MediaWiki 目前的 backlink index，再逐一用指定日期 revision
驗證。因此已回傳的 edge 是真的，但早已刪除且目前 backlink index 找不到的歷史 edge
可能漏掉；「最短」是對本次 manifest 保存的 bounded arena 而言，不宣稱全 Wikipedia。
`branch_cap` 限制每個節點的 backlink 候選，`max_nodes`／`--arena-node-cap` 限制整張
reference arena；必要 reasoning route 會先放入，不會被候選排序擠掉。artifact 會保存
`arena_truncated` 與 `discovery_mode`。offline run 只對已凍結在本機 cache 的 arena 精確，
不等同 live MediaWiki coverage。

Wikipedia 後期 revision 常保留舊事件的 hyperlink，因此 raw graph 可能出現語意捷徑。
新版不再用這種捷徑否決題目，而是同時保存。當模型可任選日期時，raw arena 只是在
generator 已知日期上建立的 reference graph，不宣稱涵蓋日期區間內的所有 revision：

- `raw_shortest_navigation_steps`：page/revision hyperlink arena 的最短距離。
- `reference_route_match` / `reference_route_coverage`：模型是否碰巧走過 generator 保存的一條 proof route，只作診斷。
- `semantic_route_complete` 與 `semantic_waypoints_completed`：舊欄位保留相容性，語意等同上面的 reference-route 診斷，不是成功 gate。

任何 Wikipedia hyperlink 路徑都可合法作答，不要求 generator 的路徑唯一。答案正確、target-date
證據是否可見、時間探索、hyperlink 導航與 reference route 分開計分，避免把「猜對答案」寫成「完成時間探索」。
空白 final answer 會在呼叫 LLM judge 前直接標成 `no_answer`；任何 positive judge label 的
`answer_extracted` 也必須實際出現在被測模型輸出中。derived scorer 會覆核舊 artifact，並以
`blank_answer_judge_overrides` 揭露歷史 judge 假陽性，不改寫 raw JSONL。

主要 knowledge-acquisition 指標另要求模型實際開過每一個 PK-gated bridge 的 exact source
revision，且 frozen relation evidence 在可見文字中。`correct_after` 但缺少這些 bridge
evidence 不算 `acquisition_success`。

## 模型可用 Tools

- `switch_snapshot(as_of, brief_reason)`：切換目前 page 的時間，可重複呼叫；`as_of` 可由
  模型在 cutoff–target 閉區間內任選 `YYYY-MM-DD`。
- `list_revisions(from, to, limit)`：只回傳目前 page 的 revision 日期 JSON array；不回傳 diff、內容、edit comment、使用者或重要日期提示。
- `view_current_page()`：閱讀目前 page revision。
- `list_links()`：列出目前 revision 真正存在的 outgoing hyperlinks。
- `follow_link(target)`：沿目前 revision 的 hyperlink 移動，時間保持不變。
- `search_within_page(query)`：搜尋目前 revision 的可見正文。
- `submit_answer(answer)`：提交最終答案並結束。

### Temporal graph-constrained beam prototype

另有一條不取代既有 external-tool agent 的 inference-only 原型。每條 beam 明確保存
`(page title, revision ID)`、可見 evidence、時間限制、entity state、完整 action trace 與累積分數；
候選 `FOLLOW_LINK` 與 `SWITCH_SNAPSHOT` 會在 controller 端依目前 rendered revision 驗證。
API 模型使用的是明確標記的 utility-ranker fallback，不宣稱為 logits/decoding integration；
open-weight conditional log-probability adapter、A/B/C/D arms、PK 接線與 engineering-smoke 限制詳見
[設計文件](docs/temporal_graph_constrained_beam.md)。

`SUBMIT_ANSWER` 不再是和有限 graph actions 一起憑空填答案的空 action。每個 state 會先獨立
產生一個短 answer candidate；controller 只在答案逐字出現在它引用的可見 evidence ID 時建立
parameterized submit action，再與 graph actions 一起排名。這是 evidence-support gate，不使用 gold
answer 或 private chain。兩題 non-scoring forced-state 診斷可用：

```bash
uv run tkg-run-forced-diagnostics \
  --cases CASE_A.json CASE_B.json \
  --model openai/gpt-4.1-mini \
  --ranker-cache forced_ranker.db \
  --output forced_diagnostics.jsonl
```

結果與 candidate funnel 見
[forced-state 診斷](docs/FORCED_STATE_DIAGNOSTICS.md)。API utility ranker 仍不是 integrated decoding。
API dense fallback 只接受最多 30 個 compacted actions，並要求回傳 ID 集合與輸入完全相等；
missing、unexpected、duplicate 或非法分數會 corrective retry 一次，仍失敗即記為
`ranker_infrastructure_error`，不補 floor、不產生假排名。

起始 page 已在 registered cutoff revision 載入並直接顯示，不耗用 tool step，也不強迫第一個
動作切時間；之後可以不限次數切換。預設只告訴模型
registered cutoff 與 target 形成的日期範圍，不提供 generator 用來驗證題目的中繼日期。
模型必須根據每次頁面輸出與 hyperlink 自己決定下一個日期；每次 `as_of`、`brief_reason`、
解析到的 revision timestamp 都寫進 trajectory。切換時間時保留 page title，並即時載入該
日期最後一個 revision 及其 links。兩個日期若解析到同一 revision，graph/viewer 會合併成
同一 node。模型可沿該 revision 真正存在的任一 article hyperlink 移動，不再受 hidden
reference arena 的 title 白名單限制。

評分時，generator 保存的中繼日期與 revisions 仍是 hidden reference evidence，只用來計算
reference-route coverage。模型可以走其他合法 page/revision 路徑；成功與否由答案以及實際看見
的 target-date evidence 判定，不要求匹配 generator route。

## 執行

```bash
uv run tkg-run \
  --models "TESTED_MODEL" \
  --judge-model "INDEPENDENT_JUDGE_MODEL" \
  --human-review-file human_review.json \
  --case-ids example_ceo \
  --start-distance 3 \
  --backlink-branch-cap 25 \
  --pk-repeats 5 \
  --pk-max-known-rate 0 \
  --repeats 3 \
  --workers 3 \
  --max-steps 16
```

只做獨立 PK admission；此階段不要求 human review，只有 PK-admitted cases 才進後續 review：

```bash
uv run tkg-run \
  --models "TESTED_MODEL" \
  --judge-model "INDEPENDENT_JUDGE_MODEL" \
  --cases generated_cases.json \
  --pk-only --pk-repeats 5 \
  --output pk_results.jsonl --score-output pk_scores.csv
```

`--snapshot-dates` 預設省略，此時模型可在 cutoff–target 間任選日期。只有重播舊實驗或
debug 固定時間清單時才顯式傳入它；這會恢復 enum allowlist 與 bounded arena 限制。
多跳 case 固定使用 seed 的 `start_title`；`--start-distance` 只作為 reference arena 至少
要建多深的下限。

`--workers` 只並行已通過 PK 與 graph gate 的 trajectory；PK 與建圖仍序列執行。每個 worker
有自己的 SQLite connection，JSONL write 有 lock，所有 worker 共用一個 aggregate Wikimedia
request throttle。每次 OpenRouter call 另外寫入預設的 `<output>.usage.jsonl`，保存 call role、
cache hit、prompt/completion/total tokens 與 provider 回傳的 cost；可用 `--usage-output` 改路徑。
生成器也有相同 ledger，方便在移除任何 judge layer 前先量實際 call/cache/cost。

自由選時通常不能搭配 `--offline`，因為模型可能挑到 cache 尚未出現的日期；offline
replay 應使用原 trajectory 已選過的 cache，或顯式傳 `--snapshot-dates` 固定清單。

輸出：

- `temporal_results.jsonl`
- `temporal_scores.csv`

JSONL 會保存每次 tool call 的 page title、revision、來源/目的日期、tool output、完整
visited versions、最終答案與 judge 結果；`temporal_summary` 也保存實際送入模型的兩則
初始 messages 與 tool JSON schema，方便確認沒有洩漏 hidden dates。另保存：

- `pk_probe`：每次 fresh-context 問題、原始回答與 judge label。
- `pk_gate`：題目 × 模型的 critical-bridge、tail、composed known rates、各 probe
  labels、deterministic/LLM judge 次數與 admission 結果。
- `navigation_arena`：本次 bounded graph 的 nodes、edges、距離與 graph hash id。
- `snapshot_mode` / `snapshot_range`：模型任選日期或 legacy 固定清單，以及公開日期範圍。
- `reference_snapshot_dates`：只供事後稽核的 hidden oracle dates；不會送進模型 prompt。
- `distance_to_pivot`：每次 navigation 後的最短剩餘距離。
- `shortest_navigation_steps` / `raw_shortest_navigation_steps`：raw arena 距離。
- `semantic_waypoints_completed`、`semantic_route_complete`、
  `reference_route_match`、`reference_route_coverage`：reference proof route 診斷。
- `revision_discovery_*`、`temporal_switch_*`、`hyperlink_follow_*`、
  `target_snapshot_evidence_seen`、`answer_submitted`：分開的能力指標。
- `failure_mode`：區分 revision discovery、時間探索、link navigation、找不到 target、看見證據後答錯與未作答。
- `judge_mode`：明示本次是 `deterministic_alias` 或真正呼叫 `llm_fallback`。
- `actual_steps_to_first_pivot` 與 `detour_steps`。
- `revisit_count`、`cycle_detected`、`shortest_arrival`。

`view_current_page`、`list_links`、`search_within_page` 不算移動；成功的
`switch_snapshot` 與 `follow_link` 各算一步；cutoff initial state 與 `list_revisions` 不算移動。API error 不會寫 complete checkpoint。

## Answer Judge

accepted alias 的明確回答先走 deterministic gate；模糊回答才使用與被測模型不同的 LLM
judge。labels 為：

- `correct_after`
- `correct_without_visible_support`（答案字面正確，但 trajectory 沒顯示該答案的 target-date evidence）
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

Temporal viewer 初始只顯示模型實際收到的 start page cutoff revision。播放 trajectory 時：

- 模型切到另一時間：產生同 page 另一 revision node 與紫色 temporal edge。
- 模型進入某 revision：只展開該 revision 的一層 outgoing hyperlink nodes。
- 沒有實際抵達的 page 不會展開下一層。
- 未下載的 hyperlink target 顯示為灰色 stub。
- 黃色 edge 是模型實際走過的 hyperlink trajectory。
- 時間軸倒退時，尚未發現的 nodes/edges 會消失。
- hover node 可查看 revision id、timestamp、`distance_to_pivot` 與摘要；點擊 node 後可
  開啟 exact revision URL。
- Run badges 顯示 PK admission、pivot/alternate legal proof path、shortest/detour、cycle/revisit；
  若模型未碰到 generator pivot，但從另一條合法路徑看見 target-date evidence 並答對，不會被
  顯示成醒目的 pivot failure。

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
