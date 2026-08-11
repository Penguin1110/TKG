"""
run_experiment.py
------------------
流程（對每個 case × 每個 model × 兩個 arm: conflict / control）：

  conflict arm（先驗反悔組）：
    1. PK 探針：全新對話、只問 pk_question，測模型「不給任何曝光」時的預設答案，
       用來確認舊先驗夠強（門檻見 case["pk_threshold"]，預設 0.8）
    2. 曝光（Line A）：讓模型走一段 5 步的劇本式圖探索（case["graph_walk"]），
       每一步用純文字呈現「目前節點的現況事實（帶 as_of 時間戳）+ 可查看的鄰居
       節點」，模擬 agent 在 TKG 上瀏覽時「順路」經過 pivot fact 節點——刻意不用
       OpenRouter 的 tool-calling API，模型每步用自由文字回覆想查哪個鄰居，這個
       回覆會被記錄，但下一步顯示哪個節點是照劇本走、不受影響。這樣做是為了排除
       「模型導航能力」這個干擾變數，只留下「曝光後會不會反悔」這一個變數（詳見
       README；跟 Think-on-Graph/KG-Agent 那種讓模型真的自主選路的作法不同，是
       刻意的方法論分歧，不是不知道有那種做法）
    3. 多輪追問（Line B）：問 case["ripples"][distance] 底下的問題（新舊事實會衝突）

  control arm（無衝突對照組 / kill gate）：
    走同一個 case 的 control_graph_walk（一樣 5 步、一樣格式，但內容全是穩定不變
    的事實），維持跟 conflict arm 相同的對話長度／結構，只有內容不衝突。多輪追問
    改問 case["control"][distance] 底下的問題——這些事實從頭到尾只有一個正確答案，
    沒有版本更新可反悔。如果 control arm 的正確率也隨輪數下滑，代表看到的是通用的
    lost-in-the-middle，不是先驗反悔專屬現象。

兩個 arm 共用同一套「輪數 x 距離」交錯排程（build_round_schedule），刻意讓
「第幾輪」跟「ripple 距離」解耦（例如輪1問距離1、輪2問距離2、輪3又回來問距離1），
這樣才能把「輪數效應」和「距離效應」分開估計，而不是兩者完全綁死。

每一輪的回答都會被 judge.classify()/classify_single() 貼標籤，記錄進
results.jsonl，供 analyze.py 算「輪數 x 距離」正確率、bootstrap CI、半衰輪數、
conflict vs control 顯著性檢定。
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone

# 見 run_free_exploration_batch.py 的同一行說明：強制 stdout 一行一行 flush，
# 避免 uv run 之類的包裝工具把 stdout 變成整塊緩衝，導致進度訊息卡在緩衝區
# 不會即時顯示、看起來像卡住了。
sys.stdout.reconfigure(line_buffering=True)

from legacy.judge import classify, classify_single
from tkg.api.openrouter import OpenRouterError, call_model


def build_round_schedule(distances: list, n_rounds: int, distractor_every: int, rng: random.Random) -> list:
    """
    產生長度 n_rounds 的排程，每個元素是 distance（int）或 "distractor"。

    做法：把 distances 排成一輪（每個 lap 開始前重新洗牌一次，避免距離出現的
    順序被記死），逐輪填入排程；每隔 distractor_every 輪插一個無關干擾題，
    藉此讓「第幾輪」跟「這輪問的是哪個距離」解耦。
    """
    if not distances:
        return ["distractor"] * n_rounds

    schedule = []
    lap = []
    for i in range(1, n_rounds + 1):
        if distractor_every and i % distractor_every == 0:
            schedule.append("distractor")
            continue
        if not lap:
            lap = list(distances)
            rng.shuffle(lap)
        schedule.append(lap.pop())
    return schedule


WRAP_UP_PROMPT = "你已經瀏覽完這幾個節點。請用一兩句話總結你目前看到的重點。"


def format_walk_step_prompt(step: dict, is_first: bool, task_prompt: str) -> str:
    lines = []
    if is_first:
        lines.append(task_prompt)
        lines.append("")
        lines.append(f"你現在在「{step['node']}」節點。")
    else:
        lines.append(f"你查看了「{step['node']}」節點。")
    if step.get("as_of"):
        lines[-1] += f"（現況 as of {step['as_of']}）"
    lines.append("目前已知：")
    for fact in step["facts"]:
        lines.append(f"- {fact}")
    lines.append("")
    lines.append(f"可以接著查看：{'、'.join(step['neighbors'])}。你想接著查哪一個？")
    return "\n".join(lines)


def run_graph_exploration(case: dict, model: str, arm: str, history: list,
                           log_fh, case_id: str, repeat_idx: int) -> bool:
    """
    劇本式圖探索（Line A 的曝光步驟）：依序把 graph_walk（或 control_graph_walk）
    的每一步餵給模型，模型的自由文字回覆只被記錄、不影響下一步顯示哪個節點。
    回傳 False 代表中途呼叫失敗，呼叫端應該放棄這個 arm。
    """
    walk = case["graph_walk"] if arm == "conflict" else case["control_graph_walk"]
    task_prompt = case["exploration_task"]

    for step_idx, step in enumerate(walk):
        prompt = format_walk_step_prompt(step, is_first=(step_idx == 0), task_prompt=task_prompt)
        history.append({"role": "user", "content": prompt})
        try:
            response = call_model(model, history)
        except OpenRouterError as e:
            print(f"[error] 圖探索第 {step_idx + 1} 步失敗 case={case_id} model={model} arm={arm}: {e}")
            return False
        history.append({"role": "assistant", "content": response})
        _write_row(log_fh, case_id, model, arm=arm, round_idx=0, slot="explore",
                   distance=None, occurrence=step_idx + 1, question=prompt,
                   response=response, label="n/a", repeat_idx=repeat_idx)

    history.append({"role": "user", "content": WRAP_UP_PROMPT})
    try:
        response = call_model(model, history)
    except OpenRouterError as e:
        print(f"[error] 曝光總結失敗 case={case_id} model={model} arm={arm}: {e}")
        return False
    history.append({"role": "assistant", "content": response})
    _write_row(log_fh, case_id, model, arm=arm, round_idx=0, slot="exposure",
               distance=None, occurrence=None, question=WRAP_UP_PROMPT,
               response=response, label="n/a", repeat_idx=repeat_idx)
    return True


def _pick_item_and_question(candidates: list, rng: random.Random):
    item = rng.choice(candidates)
    variants = [item["question"]] + item.get("paraphrases", [])
    question = rng.choice(variants)
    return item, question


def run_pk_probe(case: dict, model: str, log_fh, repeat_idx: int) -> str:
    case_id = case["id"]
    pk_messages = [{"role": "user", "content": case["pk_question"]}]
    try:
        pk_response = call_model(model, pk_messages)
    except OpenRouterError as e:
        print(f"[error] PK probe 失敗 case={case_id} model={model}: {e}")
        return "error"
    pk_label = classify(pk_response, case["old_answer_keywords"], case["new_answer_keywords"])
    _write_row(log_fh, case_id, model, arm="conflict", round_idx=0, slot="pk_probe",
               distance=None, occurrence=None, question=case["pk_question"],
               response=pk_response, label=pk_label, repeat_idx=repeat_idx)
    print(f"[pk] {case_id} / {model}: {pk_label}")
    return pk_label


def run_round_schedule(case: dict, model: str, arm: str, distractors: list, round_schedule: list,
                        log_fh, rng: random.Random, repeat_idx, history: list) -> bool:
    """
    Line B：接在曝光步驟後面的交錯距離多輪追問。故意獨立出來（不含曝光步驟），
    是因為曝光機制現在有兩種來源會共用這段邏輯：
      1. 舊版劇本式路徑（run_arm() 下面接著呼叫，history 由 run_graph_exploration() 產生）
      2. 新版自由探索（run_free_exploration_batch.py 直接呼叫，history 是
         graph_exploration_agent.run_free_exploration() 回傳的、走到 pivot 那一刻
         為止的完整對話記錄）
    這樣「接上 Line B」的程式碼只有一份，不用維護兩套。

    repeat_idx 允許是 int（舊版排程）或 str（自由探索批次用複合 id，例如
    "d1_Q123_r0"），這裡跟後面的 analyze.py 都只把它當成不透明的分組 key，
    不要求一定是整數。

    回傳 True 代表整段追問都成功跑完，呼叫端應該在這之後才寫 checkpoint。
    """
    case_id = case["id"]
    fact_pool = case["ripples"] if arm == "conflict" else case["control"]

    occurrence_counter = {}
    for round_idx, slot in enumerate(round_schedule, start=1):
        if slot == "distractor":
            question = rng.choice(distractors)
            history.append({"role": "user", "content": question})
            try:
                response = call_model(model, history)
            except OpenRouterError as e:
                print(f"[error] round {round_idx} 失敗 case={case_id} model={model} arm={arm}: {e}")
                return False
            history.append({"role": "assistant", "content": response})
            _write_row(log_fh, case_id, model, arm=arm, round_idx=round_idx, slot="distractor",
                       distance=None, occurrence=None, question=question, response=response,
                       label="distractor_na", repeat_idx=repeat_idx)
            print(f"[r{round_idx}:distractor] {case_id} / {model} / {arm}")
            continue

        distance = slot
        candidates = fact_pool.get(str(distance), [])
        if not candidates:
            # 這個距離還沒有查證過的候選事實（例如 distance 3），跳過
            continue

        occurrence_counter[distance] = occurrence_counter.get(distance, 0) + 1
        item, question = _pick_item_and_question(candidates, rng)

        history.append({"role": "user", "content": question})
        try:
            response = call_model(model, history)
        except OpenRouterError as e:
            print(f"[error] round {round_idx} 失敗 case={case_id} model={model} arm={arm}: {e}")
            return False
        history.append({"role": "assistant", "content": response})

        if arm == "conflict":
            label = classify(response, item.get("old_keywords", []), item.get("new_keywords", []))
        else:
            label = classify_single(response, item.get("answer_keywords", []))

        slot_name = "ripple" if arm == "conflict" else "control"
        _write_row(log_fh, case_id, model, arm=arm, round_idx=round_idx, slot=slot_name,
                   distance=distance, occurrence=occurrence_counter[distance],
                   question=question, response=response, label=label, repeat_idx=repeat_idx)
        print(f"[r{round_idx}:{slot_name}_d{distance}] {case_id} / {model} / {arm}: {label}")

    return True


def run_arm(case: dict, model: str, arm: str, distractors: list, round_schedule: list,
            log_fh, rng: random.Random, repeat_idx: int) -> bool:
    """
    舊版劇本式路徑專用的薄包裝：跑 5 步 graph_walk 曝光，成功的話再接上
    run_round_schedule()。新版自由探索不走這個函式，是直接呼叫
    run_round_schedule()（見該函式的說明）。
    回傳 True 代表這個 arm 的曝光+追問全部成功跑完，checkpoint 才會記這個 arm 完成。
    """
    case_id = case["id"]
    history = []
    if not run_graph_exploration(case, model, arm, history, log_fh, case_id, repeat_idx):
        return False
    return run_round_schedule(case, model, arm, distractors, round_schedule,
                               log_fh, rng, repeat_idx, history)


def _write_row(fh, case_id, model, arm, round_idx, slot, distance, occurrence,
                question, response, label, repeat_idx):
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "model": model,
        "arm": arm,
        "repeat": repeat_idx,
        "round": round_idx,
        "slot": slot,
        "distance": distance,
        "occurrence": occurrence,
        "question": question,
        "response": response,
        "label": label,
    }
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()


def _write_checkpoint(fh, case_id, model, arm, repeat_idx):
    """
    一個 arm（conflict 包含 PK 探針、control 不含）全部成功跑完後寫一筆 checkpoint
    紀錄。之後重跑（例如想多測幾個模型）時，load_completed_arms() 會讀這些紀錄，
    已經完成的 (case, model, repeat, arm) 組合就跳過，不用重花錢重問一次。

    只在整個 arm 都成功跑完才寫，中途失敗（run_arm 回傳 False）不會寫，這樣下次
    重跑會整個 arm 重來一次（可能造成那個 arm 少量重複資料，但換來實作簡單、
    checkpoint 語意清楚：有 checkpoint = 這個 arm 完整且乾淨）。
    """
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id, "model": model, "arm": arm, "repeat": repeat_idx,
        "round": 0, "slot": "checkpoint", "distance": None, "occurrence": None,
        "question": None, "response": None, "label": "done",
    }
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()


def load_completed_arms(path: str) -> set:
    """回傳已經完成的 (case_id, model, repeat, arm) 組合集合，讀自舊的 checkpoint 紀錄。"""
    completed = set()
    if not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("slot") == "checkpoint":
                completed.add((row["case_id"], row["model"], row["repeat"], row["arm"]))
    return completed


def available_distances(case: dict) -> list:
    dists = set()
    for pool in (case["ripples"], case["control"]):
        for d, candidates in pool.items():
            if candidates:
                dists.add(int(d))
    return sorted(dists)


def main():
    parser = argparse.ArgumentParser(description="TKG prior-reversion pilot runner")
    parser.add_argument("--models", type=str, required=True,
                         help="逗號分隔的 OpenRouter 模型代號，例如："
                              "openai/gpt-4.1-mini,anthropic/claude-sonnet-4.5")
    parser.add_argument("--cases", type=str, default="cases.json")
    parser.add_argument("--output", type=str, default="results.jsonl")
    parser.add_argument("--case-ids", type=str, default=None,
                         help="只跑指定的 case id（逗號分隔），不填就跑全部")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=6,
                         help="每個 arm 跑幾輪追問（含 distractor），design doc 建議 5-8 輪")
    parser.add_argument("--distractor-every", type=int, default=3,
                         help="每隔幾輪插一個無關干擾題，0 表示不插")
    parser.add_argument("--repeats", type=int, default=1,
                         help="每個 case x model 重跑幾次（用不同 seed 抽不同候選/"
                              "paraphrase），增加每個格子的樣本數以利 bootstrap CI；"
                              "design doc 建議 pilot 每格 5-10 次")
    parser.add_argument("--arms", type=str, default="conflict,control",
                         help="要跑哪些 arm，逗號分隔，預設兩個都跑（control 是 kill gate 對照組）")
    args = parser.parse_args()

    with open(args.cases, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data["cases"]
    if args.case_ids:
        wanted = set(args.case_ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    distractors = data["distractor_questions"]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[error] 請先在 .env 或環境變數設定 OPENROUTER_API_KEY", file=sys.stderr)
        sys.exit(1)

    completed = load_completed_arms(args.output)
    if completed:
        print(f"[checkpoint] 從 {args.output} 讀到 {len(completed)} 個已完成的 (case,model,repeat,arm)，會跳過")

    with open(args.output, "a", encoding="utf-8") as log_fh:
        for model in models:
            for case in cases:
                case_id = case["id"]
                distances = available_distances(case)
                pk_labels = []
                for repeat_idx in range(args.repeats):
                    needed_arms = [a for a in arms if (case_id, model, repeat_idx, a) not in completed]
                    if not needed_arms:
                        print(f"[checkpoint] 跳過 case={case_id} model={model} repeat={repeat_idx}（已完成）")
                        continue

                    rng = random.Random(args.seed + repeat_idx * 1000 + hash(case_id) % 997)
                    round_schedule = build_round_schedule(distances, args.rounds, args.distractor_every, rng)

                    print(f"=== case={case_id} model={model} repeat={repeat_idx} "
                          f"needed_arms={needed_arms} schedule={round_schedule} ===")

                    if "conflict" in needed_arms:
                        pk_label = run_pk_probe(case, model, log_fh, repeat_idx)
                        pk_labels.append(pk_label)
                        if pk_label != "error" and run_arm(case, model, "conflict", distractors,
                                                            round_schedule, log_fh, rng, repeat_idx):
                            _write_checkpoint(log_fh, case_id, model, "conflict", repeat_idx)
                    if "control" in needed_arms:
                        if run_arm(case, model, "control", distractors, round_schedule,
                                   log_fh, rng, repeat_idx):
                            _write_checkpoint(log_fh, case_id, model, "control", repeat_idx)

                if pk_labels:
                    strong_prior_rate = pk_labels.count("stick_old") / len(pk_labels)
                    threshold = case.get("pk_threshold", 0.8)
                    verdict = "OK" if strong_prior_rate >= threshold else "WARN 先驗不夠強，建議從分析中剔除"
                    print(f"[kill-gate:pk] case={case_id} model={model} "
                          f"stick_old_rate={strong_prior_rate:.2f} (threshold={threshold}, 只算這次新跑的) -> {verdict}")

    print(f"完成，結果寫入 {args.output}")


if __name__ == "__main__":
    main()
