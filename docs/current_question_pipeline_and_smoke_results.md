# TKG 現行出題、PK 與結果狀態

> 更新：2026-08-14  
> Contract：generation v6、renewable engine v3、runner v11

## 目前結論

舊 `non_sports_v6` 的兩題與 PK smoke 已作廢，現在沒有可報告的模型成功率。
作廢不是因為模型答錯，而是 benchmark contract 有兩個問題：

- PK prompt 把 `critical_bridge`、`must_be_unknown`、cutoff 與 target metadata 告訴了
  tested model，會誘導「不知道」的回答；
- 題目把 Foreign Secretary 稱為 David Lammy 在 cutoff 後「第一個」開始擔任的 P39
  職位，但 Wikidata 還有一個 2024-07-04 開始的國會職位，早於 2024-07-05 的
  Foreign Secretary 任期。

因此 v6 的 admission pass、tail/composed rates 與 offline trajectories 都只能保留為
invalid audit history，不能解讀。

## 修正後的標準流程

```text
候選 temporal relation
  → Wikidata 提供 QID、方向、qualifier 與候選事件集合
  → 若題目聲稱 first/next，建立「邊界後所有候選事件」certificate
  → deterministic gate 證明選中事件是唯一最早事件
  → exact Wikipedia revision 驗證正文 evidence 與 hyperlink
  → 凍結完整 rendered page、links 與 SHA-256 manifest
  → LLM writer 只自然改寫已驗證的 private chain
  → deterministic wording/leakage gates
  → independent whole-chain judge
  → hash-bound human review
  → PK-only admission（完全不開 Wikipedia、不建 graph）
  → 通過 PK 才建 temporal graph 並執行 navigation
  → 分開計算 answer correctness 與 acquisition success
```

## Fixed-answer temporal semantics

題目答案只能依賴世界事件，不能依賴模型查看哪個 snapshot。`first/next` edge 的
`wikidata-event-order-v1` certificate 至少包含：

- 固定 boundary event date；
- 選中的 event date / QID；
- boundary 到 coverage end 的完整候選集合；
- 唯一最早事件檢查；
- content-addressed source hash；
- SPARQL 路徑另存 query hash，LIMIT 飽和則 fail closed。

新版 provisional spine 不再聲稱 Foreign Secretary 是第一個任意 P39 職位。它改問
「與 Shadow Foreign Secretary 對應、該人物在 cutoff 後開始擔任的政府職位」；只有
後續「下一位 Foreign Secretary」保留 `next`，且其 ordering certificate 指向
2025-09-05 的 Yvette Cooper。

## Factorized PK v2

Tested model 的每個 prompt 現在只有：

```text
Question: <direct factual question>
Instruction: <neutral memory-only instruction; no tools; final answer only>
```

它看不到 probe role、objective、cutoff、target date、admission policy 或 expected unknown
標籤；這些只存在 private log 與 judge context。每個 probe 仍是 fresh one-turn context。

`factorized-prior-knowledge-v2` 將所有 designated post-cutoff acquisition edges 都列入
`primary_admission_probe_ids`。Admission 要求每一條的 known rate 都不超過 threshold，且
不能有 unjudgeable；不再只測最後一條 bridge。Tail 與 composed probes 只做診斷。

CLI 新增 `--pk-only`。此模式不 instantiate Wikipedia backend、不做 reverse BFS、也不跑
trajectory；完整 run 也改成先 PK，只有 admitted model 才支付 graph construction 成本。

## Answer correctness 不等於 acquisition

`correct_after` 只表示 final answer 正確。主要 acquisition 指標為：

```text
acquisition_success = final label == correct_after
                      and every PK-gated bridge's exact source revision was viewed
                      and its frozen relation evidence was visible
```

所以模型即使猜中 Ed Balls / Inverness，只要沒有看到 post-cutoff bridge evidence，就不算
acquisition success。Score CSV 與 viewer 分別顯示 critical bridge coverage 和 acquisition
success。

## Reproducibility

Generation v6 在 case 內保存 `frozen-wikipedia-evidence-v1`：每個用到的 rendered page
全文、links、revision ID、timestamp、content/link hash、snapshot hash，以及整體 manifest
hash。Case validator 重新計算全部 hashes；任一正文、link 或 manifest 被改動都會 fail。

## 驗證狀態

- Offline suite：126 passed。
- `non_sports_v6`：明確 invalidated；原始 artifacts 保留且未覆寫。
- `non_sports_v7/seeds.json`：已修正 relation semantics，並加入 remaining `next` edge
  certificate。
- v7 live deterministic rerun：第一題四個 Wikipedia hops 已驗證，但 shortest-arena API
  遇到 HTTP 429；第二題在最後一個 page fetch 遇到 HTTP 429。因此目前只有
  `infrastructure_error` packets，沒有 accepted v7 cases、沒有新 PK 或 navigation 結果。

## Artifacts

- [v6 invalidation note](../examples/non_sports_v6/README.md)
- [v7 provisional seeds](../examples/non_sports_v7/seeds.json)
- [v7 rate-limited validation packets](../examples/non_sports_v7/validation_only_packets.jsonl)
- [老師報告版](advisor_report_motivation_design_and_pilot.md)
