"""Popularity snapshots from the official Wikimedia Analytics API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable
from urllib.parse import unquote

import requests


ANALYTICS_BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews"
PAGEVIEW_USER_AGENT = "TKG-Wikipedia-Browser/1.0 (research; contact: local-user)"


def normalize_article_title(value: str) -> str:
    return " ".join(unquote(value).replace("_", " ").split()).casefold()


def last_complete_month(today: date | None = None) -> str:
    current = today or date.today()
    previous_month_day = current.replace(day=1) - timedelta(days=1)
    return previous_month_day.strftime("%Y-%m")


@dataclass(frozen=True)
class PopularityRecord:
    title: str
    views: int
    rank: int


@dataclass(frozen=True)
class PopularitySnapshot:
    project: str
    month: str
    records: dict[str, PopularityRecord]

    def record(self, title: str) -> PopularityRecord | None:
        return self.records.get(normalize_article_title(title))

    def views(self, title: str) -> int:
        value = self.record(title)
        return value.views if value else 0

    def score(self, titles: list[str] | tuple[str, ...]) -> int:
        return sum(self.views(title) for title in dict.fromkeys(titles))


def fetch_top_pageviews(
    *,
    lang: str = "en",
    month: str | None = None,
    request_get: Callable = requests.get,
) -> PopularitySnapshot:
    """Fetch one immutable top-article month for deterministic popularity ranking."""
    selected_month = month or last_complete_month()
    try:
        year, month_number = selected_month.split("-", 1)
        if len(year) != 4 or len(month_number) != 2:
            raise ValueError
        parsed = date(int(year), int(month_number), 1)
    except (TypeError, ValueError) as exc:
        raise ValueError("popularity month must use YYYY-MM") from exc
    if parsed.strftime("%Y-%m") != selected_month:
        raise ValueError("popularity month must use YYYY-MM")
    project = f"{lang}.wikipedia.org"
    url = (
        f"{ANALYTICS_BASE_URL}/top/{project}/all-access/"
        f"{parsed.year:04d}/{parsed.month:02d}/all-days"
    )
    response = request_get(
        url,
        headers={"Accept": "application/json", "User-Agent": PAGEVIEW_USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items", [])
    articles = items[0].get("articles", []) if items else []
    records: dict[str, PopularityRecord] = {}
    for raw in articles:
        title = str(raw.get("article", "")).strip()
        if not title or title in {"Main_Page", "-"} or title.startswith("Special:"):
            continue
        try:
            views = int(raw.get("views", 0))
            rank = int(raw.get("rank", 0))
        except (TypeError, ValueError):
            continue
        normalized = normalize_article_title(title)
        if normalized and normalized not in records:
            records[normalized] = PopularityRecord(
                title=" ".join(unquote(title).replace("_", " ").split()),
                views=max(0, views), rank=max(0, rank),
            )
    return PopularitySnapshot(project=project, month=selected_month, records=records)
