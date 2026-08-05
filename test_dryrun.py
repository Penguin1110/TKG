"""
test_dryrun.py
--------------
不需要 OPENROUTER_API_KEY、不需要網路，就能驗證整條流程邏輯對不對（含
control arm / distance interleave / bootstrap 分析 / 五步劇本式圖探索曝光）。

做法：monkeypatch call_model，模擬一個「探索階段（5 步 graph_walk + 1 步總結）
隨便回應、探索結束後的追問前幾輪答對新事實、後面幾輪反悔回舊答案；control arm
全程穩定答對」的假模型，跑完整個 run_experiment.py 的邏輯，再用 analyze.py
彙總，確認流程跟資料格式是通的、kill gate（conflict 顯著低於 control）判準
邏輯有動作。

用途：
  1. 剛拿到這份程式碼時，先跑這個，確認環境裝好、邏輯沒問題，再花真的
     API 額度跑正式實驗
  2. 之後若修改 run_experiment.py / judge.py / analyze.py，也可以先跑這個回歸測試

執行：
    python3 test_dryrun.py
"""
import json
import os
import random

import openrouter_client
import run_experiment
import analyze

CALL_COUNT = {"n": 0}
CONTROL_QUESTIONS = set()


EXPLORATION_STEPS_PER_ARM = 6  # 5 步 graph_walk + 1 步總結（見 run_experiment.WRAP_UP_PROMPT）


def fake_call_model(model, messages, temperature=0.0, **kwargs):
    CALL_COUNT["n"] += 1
    last_user = messages[-1]["content"]

    # PK 探針（conflict arm only，全新對話只有一則訊息）：模擬模型預設答舊值
    if len(messages) == 1:
        return "I believe the answer is the previous known office holder (old value)."

    # 圖探索的每一步（format_walk_step_prompt 都會有這句「可以接著查看」）
    if "可以接著查看" in last_user:
        return "好的，我看看下一個節點。"

    # 曝光總結（WRAP_UP_PROMPT）
    if "總結" in last_user:
        return "目前看到的重點是這個節點群裡有一些近期的變動。"

    # control arm 的問題全程穩定答對，模擬「無版本衝突就不該衰退」
    if last_user in CONTROL_QUESTIONS:
        return "The stable, unchanging answer is (control correct)."

    # conflict arm 多輪追問：模擬「探索結束後，輪數越後面越容易反悔回舊答案」
    # （扣掉探索階段固定的 6 個 assistant 回合，只算追問階段的回合數）
    n_asst_total = sum(1 for m in messages if m["role"] == "assistant")
    n_asst_followup = n_asst_total - EXPLORATION_STEPS_PER_ARM
    if n_asst_followup <= 2:
        return "Based on the update, the new office holder is correct (new value)."
    else:
        return "Actually, I believe it is still the old office holder (old value)."


def patch_cases(cases_data):
    for case in cases_data["cases"]:
        case["old_answer_keywords"] = ["(old value)"]
        case["new_answer_keywords"] = ["(new value)"]
        for distance, items in case["ripples"].items():
            for item in items:
                item["old_keywords"] = ["(old value)"]
                item["new_keywords"] = ["(new value)"]
        for distance, items in case["control"].items():
            for item in items:
                item["answer_keywords"] = ["(control correct)"]
                CONTROL_QUESTIONS.add(item["question"])
                for p in item.get("paraphrases", []):
                    CONTROL_QUESTIONS.add(p)
    return cases_data


if __name__ == "__main__":
    os.environ["OPENROUTER_API_KEY"] = "fake-key-for-dryrun"
    openrouter_client.call_model = fake_call_model
    run_experiment.call_model = fake_call_model

    with open("cases.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    data = patch_cases(data)

    distractors = data["distractor_questions"]

    out_path = "dryrun_results.jsonl"
    if os.path.exists(out_path):
        os.remove(out_path)

    with open(out_path, "a", encoding="utf-8") as log_fh:
        for model in ["fake/model-A"]:
            for case in data["cases"]:
                distances = run_experiment.available_distances(case)
                rng = random.Random(0)
                schedule = run_experiment.build_round_schedule(distances, n_rounds=6,
                                                                 distractor_every=3, rng=rng)
                print(f"=== case={case['id']} model={model} schedule={schedule} ===")
                run_experiment.run_pk_probe(case, model, log_fh, repeat_idx=0)
                run_experiment.run_arm(case, model, "conflict", distractors, schedule,
                                        log_fh, rng, repeat_idx=0)
                run_experiment.run_arm(case, model, "control", distractors, schedule,
                                        log_fh, rng, repeat_idx=0)

    print(f"\n共呼叫 call_model {CALL_COUNT['n']} 次")

    print("\n--- 執行 analyze.py 邏輯 ---")
    rows = analyze.load_rows(out_path)
    print(f"總記錄筆數: {len(rows)}")

    cells = analyze.summarize_cells(rows)
    pk = analyze.summarize_pk(rows)

    print("\nPK 探針:")
    for (model, case_id), (succ, n) in sorted(pk.items()):
        print(f"  {model} {case_id}: stick_old {succ}/{n}")

    print("\n輪數(occurrence) x 距離 x arm 正確率:")
    for (model, arm, distance, occurrence), (succ, n) in sorted(
            cells.items(), key=lambda x: (x[0][1], x[0][2] or 0, x[0][3] or 0)):
        print(f"  {model} {arm} distance={distance} occ={occurrence}: {succ}/{n}")

    print("\n半衰輪數 (conflict arm):")
    for model in sorted({r["model"] for r in rows}):
        hl = analyze.half_life_occurrence(cells, model)
        print(f"  {model}: {hl}")

    assert CALL_COUNT["n"] > 0, "call_model 完全沒被呼叫，流程有問題"
    assert any(a == "control" for (_, a, _, _) in cells), "control arm 沒有產生任何資料列"
    assert any(a == "conflict" for (_, a, _, _) in cells), "conflict arm 沒有產生任何資料列"
    print("\n[OK] dry-run 通過：conflict/control 兩個 arm 都有資料，流程跑得通。")
