"""
relabel.py
----------
重新用目前版本的 judge.py + cases.json 關鍵字，把 results.jsonl 裡已經打過
API、存好的原始回應（response 欄位）重新分類一次，不用再花 API 額度。

用途：judge.py 的分類邏輯或 cases.json 的關鍵字修正後（例如修掉 hedge 判斷
搶在關鍵字比對前面的 bug、修掉某個 case 的 old_answer_keywords 設錯），可以
用這個腳本重新產生正確的 label，而不必整組實驗重跑。

執行：
    python3 relabel.py --input results.jsonl --output results_relabeled.jsonl
"""

import argparse
import json

from judge import classify, classify_single


def build_lookup(cases_data):
    """case_id -> {"pk": (old_kw, new_kw), "questions": {question_text: item}}"""
    lookup = {}
    for case in cases_data["cases"]:
        questions = {}
        for pool_name in ("ripples", "control"):
            for distance, items in case[pool_name].items():
                for item in items:
                    variants = [item["question"]] + item.get("paraphrases", [])
                    for q in variants:
                        questions[q] = (pool_name, item)
        lookup[case["id"]] = {
            "pk": (case["old_answer_keywords"], case["new_answer_keywords"]),
            "questions": questions,
        }
    return lookup


def relabel_row(row, lookup):
    if row["slot"] in ("exposure", "distractor", "explore"):
        return row  # 這幾種本來就不判斷關鍵字，維持原樣

    case_lookup = lookup.get(row["case_id"])
    if case_lookup is None:
        return row  # cases.json 裡找不到這個 case，保留原樣，不亂猜

    if row["slot"] == "pk_probe":
        old_kw, new_kw = case_lookup["pk"]
        row["label"] = classify(row["response"], old_kw, new_kw)
        return row

    match = case_lookup["questions"].get(row["question"])
    if match is None:
        return row  # 舊資料的問題文字在目前 cases.json 裡找不到對應項目，保留原樣
    pool_name, item = match
    if pool_name == "ripples":
        row["label"] = classify(row["response"], item.get("old_keywords", []), item.get("new_keywords", []))
    else:
        row["label"] = classify_single(row["response"], item.get("answer_keywords", []))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results.jsonl")
    parser.add_argument("--output", type=str, default="results_relabeled.jsonl")
    parser.add_argument("--cases", type=str, default="cases.json")
    args = parser.parse_args()

    with open(args.cases, "r", encoding="utf-8") as f:
        cases_data = json.load(f)
    lookup = build_lookup(cases_data)

    with open(args.input, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    changed = 0
    for row in rows:
        old_label = row["label"]
        row = relabel_row(row, lookup)
        if row["label"] != old_label:
            changed += 1

    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"重新分類完成：{len(rows)} 筆資料，{changed} 筆 label 有變動，已寫入 {args.output}")


if __name__ == "__main__":
    main()
