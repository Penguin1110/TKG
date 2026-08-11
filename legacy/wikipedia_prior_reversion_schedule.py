"""Archived round scheduling for the prior-reversion protocol."""

from __future__ import annotations

import random


def build_round_schedule(
    distances: list[int], n_rounds: int, distractor_every: int, rng: random.Random
) -> list[int | str]:
    if not distances:
        return ["distractor"] * n_rounds
    schedule: list[int | str] = []
    lap: list[int] = []
    for index in range(1, n_rounds + 1):
        if distractor_every and index % distractor_every == 0:
            schedule.append("distractor")
            continue
        if not lap:
            lap = list(distances)
            rng.shuffle(lap)
        schedule.append(lap.pop())
    return schedule
