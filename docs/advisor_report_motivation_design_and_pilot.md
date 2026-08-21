# Temporal Wikipedia Graph QA：動機、設計與目前稽核結果

> 老師報告版｜2026-08-14  
> 階段：benchmark validity 與工程可行性驗證  
> 目前沒有可解讀的 accuracy 結果

## 一、研究動機

本研究不只是測模型能不能背出靜態答案，而是問：

> 當一條關鍵關係在模型 knowledge cutoff 後才成立時，模型能否從歷史 Wikipedia
> revision graph 取得新 bridge fact，再和既有 attribute 組合成答案？

例如 Ed Balls 和 Inverness 都是舊知識；真正的新事實是某人在 2025 年開始擔任
Foreign Secretary。完整能力應拆成：

```text
post-cutoff bridge acquisition
              +
known attribute composition
              ↓
         final answer
```

只看 final answer 會混在一起：模型可能真的取得新事實、只知道舊 tail、猜中，或根本沒
理解多跳題。

## 二、資料與工具設計

一個 graph node 是 `(Wikipedia title, revision id)`；正文中實際存在的 hyperlink 是
跨 entity edge，同頁不同 revision 是 temporal edge。模型可自行：

- 列出 revisions、切換任意允許日期；
- 查看目前頁面、搜尋內文、列出或跟隨 hyperlink；
- 最後提交答案。

Reference route 只用來證明題目可解、計算 semantic shortest path 和 detour，不會提供給
tested model。模型走過一個 node 後，viewer 才展開該層，並動畫顯示 trajectory。

Wikidata 負責候選 QID、property、方向與時間 qualifiers；Wikipedia exact revision 才是模型
實際可見的正式 evidence。Hyperlink 本身不等於 relation，仍需正文 evidence 與語意 gate。

## 三、出題流程

```text
熱門 Wikipedia/Wikidata entity 提名
  → temporal relation registry 選可再生關係
  → Wikidata qualifiers 建候選時間邊
  → first/next edge 的 exhaustive event-order certificate
  → exact Wikipedia revision 的 evidence + hyperlink 驗證
  → 凍結 rendered pages、links、revision metadata 與 hashes
  → LLM writer 自然改寫 private chain
  → deterministic leakage、時間、文法與唯一性 gates
  → independent LLM whole-chain judge
  → hash-bound human review
  → per-model factorized PK admission
  → admitted model 才執行 temporal navigation
```

LLM 不負責發明答案或 relation，只負責改寫和第二層語意審查。可再生性來自 relation
registry、Wikidata candidate search 與 content-addressed ledger；新資料到來後可重新提名，但
每一題仍必須通過同一組 frozen gates。

## 四、本次 validity audit 發現的問題

### 1. Solver-selected snapshot bug（舊 v5）

舊題寫「比 previous step 使用的 snapshot 更晚」，但 snapshot 是模型選的，因此正解會隨
瀏覽行為改變。現行 contract 只允許固定世界事件邊界，例如比較兩段任期開始時間。

### 2. PK prompt 洩漏（v6）

v6 的 tested-model prompt 曾直接包含 `critical_bridge`、`must_be_unknown`、cutoff 和 target
metadata。這會暗示模型哪一題應回答不知道，所以該一次 PK smoke 全部作廢。

現行 prompt 只含直接事實問題與中性的 memory-only instruction；所有 admission metadata
只留在 private logs / judge context。

### 3. `first` relation 的 factual bug（v6）

v6 問 David Lammy 在 2024-06-01 後第一個開始擔任的 P39 government position，預期
Foreign Secretary（2024-07-05）。Live Wikidata audit 顯示另有一個國會 P39 從
2024-07-04 開始，因此「第一個任意 P39」不成立。

這說明只驗證「選中的事件存在」不夠；還要驗證 boundary 與 selected event 之間沒有別的
合資格事件。

## 五、修正後的 deterministic contracts

### Event-order certificate

任何題目聲稱 `first/next` 時，必須保存 boundary、coverage end、完整候選事件集合、selected
date/QID 與 source hash。Gate 驗證 selected event 是 boundary 後唯一最早事件；SPARQL
結果若碰到 LIMIT，直接 fail closed。

Corrected provisional spine 不再把 Foreign Secretary 稱為第一個任意 P39。它用語意範圍問
與 Shadow Foreign Secretary 對應、之後開始擔任的政府職位；只有下一任 Foreign Secretary
仍使用 `next`，並以 certificate 證明 2025-09-05 的事件。

### Factorized PK v2

Anchor、所有 post-cutoff acquisition bridges、tail 與 composed 分別測試。Admission 要求
每一條 designated bridge 都是 unknown（依預先固定 threshold）；tail/composed 只做診斷。
`--pk-only` 不開 Wikipedia backend、不建圖，完整實驗也改成先 PK、後建圖。

### Frozen evidence

Accepted case 內含 exact rendered page 全文、links、revision ID、timestamp、各層 SHA-256 和
整體 manifest hash。這使日後 Wikipedia 或 renderer 改變時可以檢測 drift，並從 case 本身
重播當時 judge/trajectory 所依據的文字。

## 六、評估指標

Final correctness 與 knowledge acquisition 分開報告：

```text
correct_after
  = final answer judged correct

acquisition_success
  = correct_after
    AND 每一個 PK-gated bridge 的 exact source revision 都實際被模型看過
    AND frozen relation evidence 在可見文字中
```

另報 revision discovery、temporal switches、hyperlink follows、semantic route coverage、raw
detour、cycles、tool errors 與 failure mode。找到 page 卻答錯、猜中但沒看到 bridge，都不會
被合併成成功。

## 七、目前結果與可說／不可說的結論

可說：

- runner、PK v2、event-order、frozen evidence、acquisition metric 與 viewer 已有離線回歸；
- full offline suite 為 126 passed；
- 稽核成功攔下 prompt leakage 和一個真實的 intervening-event factual error。

不可說：

- 不可引用 v6 的 PK pass 或兩條 no-answer trajectory；
- 不可說 GPT-4.1-mini 不知道 bridge；洩漏 prompt 使該結果無效；
- 不可估 benchmark accuracy 或 tool-use success rate。

目前 v7 live rerun 受到 Wikimedia HTTP 429：第一題四 hops 已驗證但 shortest arena 未完成，
第二題最後 page fetch 未完成。它們被保存為 `infrastructure_error`，沒有補值、沒有視為 pass，
也尚未執行新 PK 或 navigation。

## 八、下一個正式 pilot 的進場條件

1. Wikimedia rate limit 恢復後完成 v7 deterministic evidence 與 shortest-arena validation；
2. 獨立 whole-chain judge 通過，之後完成 hash-bound human review；
3. 用 metadata-free `--pk-only` 做預先固定 repeats / threshold；
4. 只對 admitted cases 跑 online navigation；
5. 擴充跨領域、獨立 temporal spines，避免兩題共用同一核心 bridge；
6. 原始 logs、judge outputs 與 invalid artifacts 保持不可變，derived analysis 另存。

## 九、Artifacts

- [技術流程與狀態](current_question_pipeline_and_smoke_results.md)
- [v6 作廢說明](../examples/non_sports_v6/README.md)
- [v7 provisional seeds](../examples/non_sports_v7/seeds.json)
- [v7 rate-limited validation packets](../examples/non_sports_v7/validation_only_packets.jsonl)
- [v5 作廢說明](../examples/non_sports_v5/README.md)
