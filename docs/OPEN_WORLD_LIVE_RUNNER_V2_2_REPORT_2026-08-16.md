# Temporal Wikipedia Graph QA：live v2.2 runner 工程報告

## 結論

新版環境、candidate retrieval、30-action compaction、dense ranker、global beam、structured submission 已經接成一個可從公開起點實際執行的 runner。合成多路徑 smoke 是由 runner 從 `Start` 逐步產生 action，沒有輸入預製 trajectory，也沒有在 inference 載入 private route。

這只證明工程原型成立，不是 benchmark 跑成功，也不構成模型能力、accuracy 或方法優越性結論。

## 實際解題流程

每條 beam 保存 page/revision、snapshot time、模型摘要、抽取 entity、可見 evidence、pagination 狀態、visited `(page, revision)`、action trace、累積分數與 structured submission。

每輪對每個未完成 state：

1. structured proposer 僅用問題與目前可見 evidence 產生一次 answer candidate；證據不足時回傳空 submission。
2. runner 產生已取回的 `FOLLOW_LINK`／`SWITCH_SNAPSHOT`、必要的 `LIST_LINKS`／`LIST_REVISIONS`，以及通過公開 evidence gate 的 `SUBMIT_ANSWER`。
3. 依 retrieval/document order 壓到最多 30 個，保留 pagination 與 submit control；不使用 reference route、gold distance、答案正確性或 Wikidata。
4. ranker 必須為全部 action 回傳唯一、有限數值的分數。missing、unexpected、duplicate ID 最多重試一次，仍不完整就 fail closed；不再用 `-100` 補 omitted action。
5. 展開每個 parent 的最高分 actions，驗證 hyperlink 確實存在於當前 revision，或 revision switch 沒有換頁。
6. 對等價 observable state 去重，跨所有 parent 保留 global top `beam_width`。
7. `SUBMIT_ANSWER` 必須引用 trajectory 自己取得的 evidence，答案須是最多八詞的 noun phrase，且在 tail evidence 有 literal support。
8. 搜尋完成後，private witnesses、aliases 與 reference route 才能進入 evaluator；它們不會回流影響搜尋。

## 合成 live smoke

固定 policy 從 `Start@revision 1` 走出：

```text
LIST_LINKS
→ FOLLOW_LINK(Route B)
→ LIST_REVISIONS
→ SWITCH_SNAPSHOT(revision 20)
→ LIST_LINKS
→ FOLLOW_LINK(Person X)
→ SUBMIT_ANSWER(Answer City)
```

共 7 expansions。私有可行性 witness 是 Route A；runner 沒有完成該 reference route，但事後 evaluator 以 Route B 的另一條 evidence chain 驗證成功。這驗證的是 open-world 多路徑 runner 與 evaluator 能接起來，不是模型表現，因為此 smoke 使用 deterministic fixture ranker/proposer。

完整可稽核產物在 `examples/temporal_eval_v2/live_synthetic_multiroute_v22.json`，包含 state、visible evidence、完整環境 manifest、retrieved/compacted actions、dense scores、expanded/retained actions、pruning reason、transition validity、resulting state 與事後評分。

## 驗證

- 全套測試：200 passed。
- 新 runner 專屬測試：9 passed。
- mypy：50 source files，0 issues。
- pyflakes：0 findings。
- 相同輸入與 seed 產生相同 beam。
- max expansions 嚴格停止，未提交不記成功。
- API ranker 的 missing、unexpected、duplicate IDs 都會在一次 retry 後 fail closed。
- cumulative score 可由 action trace 重算。

凍結契約與 hashes 在 `docs/OPEN_WORLD_LIVE_RUNNER_V2_2_FREEZE_2026-08-16.json`。Germany／Canada 沒有重跑，也沒有用來調整 v2.2。

## 凍結後 API-model smoke

凍結 runner 後，另以 `openai/gpt-5.4-mini` 的 API dense ranker 與 structured proposer 從同一個 synthetic `Start` 執行一次；完整產物是 `examples/temporal_eval_v2/live_synthetic_api_gpt54mini_v22.json`。

這次模型實際走到 `Route B@revision 20`，取得 post-cutoff bridge，再走到 `Person X@revision 30` 取得 birthplace tail。11 次 ranker calls 全部回傳完整、唯一的 dense action IDs，沒有 omission floor 或 contract failure。

但 structured proposer 在已看見兩段 evidence 時仍兩次回傳空 submission。runner 隨後用完該 state 的 link/revision queries，以 `exhausted_no_legal_progress` 停止；沒有 submit，不能算成功。ranker 的摘要雖然明確辨認 Person X 與 Answer City，也不能替代 structured submission。

因此這個 post-freeze API smoke 的定位是：

```text
LIVE NAVIGATION AND TEMPORAL SWITCH PASSED
DENSE RANKER CONTRACT PASSED
BRIDGE AND TAIL PAGES ACQUIRED
STRUCTURED SUBMISSION CANDIDATE FAILED
END-TO-END VALIDATED SUCCESS = FALSE
```

這是 failure localization，不是 benchmark accuracy。runner 與 prompt 沒有因這個單一 synthetic development result 回頭調整。

## 還沒有完成的研究工作

- 尚無 fresh post-freeze development cases。
- 尚無 fresh development cases 上的 API model A/B/C/D runs；目前只有上述 post-freeze synthetic API smoke。
- 尚無 benchmark accuracy。
- 尚無人工 evidence review；machine pass 不能寫成人工核准。
- API utility ranker 仍是外部控制器，不是 graph-integrated decoding。
- open-weight action conditional log-probability adapter 有介面，但尚未接 Qwen 等模型 logits 做 live run。

因此目前可進入的下一階段，是凍結此 runner 後生成 fresh development cases、先做 PK admission，再用同一份 cases/model/evidence/action budget 跑 A/B/C/D。只有這一步成立後，才應接真正的 open-weight logits condition。
