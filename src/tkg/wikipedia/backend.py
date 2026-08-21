"""Wikipedia revision and hyperlink backend.

Nodes are rendered Wikipedia pages and directed edges are hyperlinks that are
actually present in the rendered revision.  A page can be fetched as it exists
now or at the last revision on/before an ``as_of`` timestamp.  Snapshots are
stored in SQLite so a formal run can be repeated without silently mixing live
and cached pages.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import unquote, urlparse

import requests

from tkg.experiment.contracts import PageLink, PageSnapshot


DEFAULT_USER_AGENT = "TKG-Wikipedia-Browser/1.0 (research; contact: local-user)"


class WikipediaError(RuntimeError):
    pass


class WikimediaRequestThrottle:
    """Process-local aggregate throttle shared by concurrent API workers."""

    def __init__(self, min_interval: float = 0.1):
        if min_interval < 0:
            raise ValueError("min_interval must be >= 0")
        self.min_interval = min_interval
        self._last_request = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_request = time.monotonic()


def normalize_title(title: str) -> str:
    return " ".join(unquote(title).replace("_", " ").split()).strip()


def _as_of_key(as_of: str | None) -> str:
    return as_of or "__CURRENT_SNAPSHOT__"


def _timestamp_for_api(value: str) -> str:
    """Normalize a date/datetime to MediaWiki's UTC timestamp form."""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value + "T23:59:59Z"
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _VisiblePageParser(HTMLParser):
    """Extract readable blocks while retaining visible anchor/target pairs."""

    # Infoboxes are HTML tables and frequently contain the very office-holder/
    # date facts this experiment needs.  Keep th/td text instead of dropping
    # tables wholesale; navigation/reference noise remains excluded.
    BLOCK_TAGS = {
        "p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt", "caption", "th", "td"
    }
    SKIP_TAGS = {"script", "style", "sup", "figure", "nav", "noscript"}

    def __init__(self, allowed_titles: set[str]):
        super().__init__(convert_charrefs=True)
        self.allowed = {normalize_title(t).casefold(): normalize_title(t) for t in allowed_titles}
        self.skip_depth = 0
        self.block_depth = 0
        self.parts: list[str] = []
        self.blocks: list[str] = []
        self.links: list[PageLink] = []
        self._active_target: str | None = None
        self._anchor_parts: list[str] = []

    @staticmethod
    def _href_title(href: str) -> str | None:
        parsed = urlparse(href)
        path = parsed.path
        if path.startswith("/wiki/"):
            return normalize_title(path[len("/wiki/"):])
        if path.startswith("./"):
            return normalize_title(path[2:])
        return None

    def handle_starttag(self, tag: str, attrs):
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in self.BLOCK_TAGS:
            if self.block_depth == 0:
                self.parts = []
            self.block_depth += 1
        if tag == "a" and self.block_depth:
            href = dict(attrs).get("href", "")
            raw_target = self._href_title(href)
            canonical = self.allowed.get((raw_target or "").casefold())
            if canonical:
                self._active_target = canonical
                self._anchor_parts = []

    def handle_data(self, data: str):
        if self.skip_depth or not self.block_depth:
            return
        if self._active_target:
            self._anchor_parts.append(data)
        else:
            self.parts.append(data)

    def handle_endtag(self, tag: str):
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "a" and self._active_target:
            anchor = " ".join("".join(self._anchor_parts).split()) or self._active_target
            self.parts.append(f"[{anchor} -> {self._active_target}]")
            self.links.append(PageLink(target=self._active_target, anchor=anchor))
            self._active_target = None
            self._anchor_parts = []
        if tag in self.BLOCK_TAGS and self.block_depth:
            self.block_depth -= 1
            if self.block_depth == 0:
                block = " ".join("".join(self.parts).split())
                if block:
                    self.blocks.append(block)
                self.parts = []

    def result(self) -> tuple[str, list[PageLink]]:
        seen: set[tuple[str, str]] = set()
        links = []
        for link in self.links:
            key = (link.target.casefold(), link.anchor.casefold())
            if key not in seen:
                seen.add(key)
                links.append(link)
        return "\n\n".join(self.blocks), links


class WikipediaPageBackend:
    def __init__(
        self,
        cache_path: str = "wikipedia_snapshot.db",
        lang: str = "en",
        offline_only: bool = False,
        user_agent: str = DEFAULT_USER_AGENT,
        min_request_interval: float = 0.1,
        request_get: Callable | None = None,
        request_throttle: WikimediaRequestThrottle | None = None,
        max_api_calls: int | None = None,
        api_max_retries: int = 6,
        backoff_base_seconds: float = 2.0,
        backoff_max_seconds: float = 60.0,
        backlink_verify_workers: int = 1,
    ):
        self.lang = lang
        self.offline_only = offline_only
        self.user_agent = user_agent
        self.api_url = f"https://{lang}.wikipedia.org/w/api.php"
        self.wikidata_url = "https://www.wikidata.org/w/api.php"
        self.min_request_interval = min_request_interval
        self._request_throttle = request_throttle or WikimediaRequestThrottle(
            min_request_interval
        )
        self._request_get = request_get or requests.get
        if max_api_calls is not None and max_api_calls <= 0:
            raise ValueError("max_api_calls must be > 0 when provided")
        if api_max_retries <= 0:
            raise ValueError("api_max_retries must be > 0")
        if backoff_base_seconds < 0 or backoff_max_seconds < 0:
            raise ValueError("API backoff values must be >= 0")
        if backlink_verify_workers <= 0:
            raise ValueError("backlink_verify_workers must be > 0")
        self.max_api_calls = max_api_calls
        self.api_max_retries = api_max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.backlink_verify_workers = backlink_verify_workers
        self._api_calls_used = 0
        self._api_counter_lock = threading.Lock()
        self._conn = sqlite3.connect(cache_path, timeout=30)
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS page_cache ("
            "cache_key TEXT PRIMARY KEY, title TEXT NOT NULL, as_of TEXT NOT NULL, "
            "data TEXT NOT NULL, fetched_at TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS page_alias ("
            "alias_key TEXT PRIMARY KEY, cache_key TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS page_links ("
            "source_key TEXT NOT NULL, target_folded TEXT NOT NULL, target_title TEXT NOT NULL, "
            "PRIMARY KEY(source_key, target_folded, target_title))"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS revision_cache ("
            "lang TEXT NOT NULL, revision_id INTEGER NOT NULL, title TEXT NOT NULL, "
            "content TEXT NOT NULL, links_json TEXT NOT NULL, fetched_at TEXT NOT NULL, "
            "PRIMARY KEY(lang, revision_id))"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS revision_date_cache ("
            "lang TEXT NOT NULL, title_folded TEXT NOT NULL, from_date TEXT NOT NULL, "
            "to_date TEXT NOT NULL, dates_json TEXT NOT NULL, fetched_at TEXT NOT NULL, "
            "PRIMARY KEY(lang,title_folded,from_date,to_date))"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS revision_metadata_page_cache ("
            "cache_key TEXT PRIMARY KEY, data TEXT NOT NULL, fetched_at TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS backlink_candidate_cache ("
            "lang TEXT NOT NULL, target_folded TEXT NOT NULL, max_results INTEGER NOT NULL, "
            "candidates_json TEXT NOT NULL, fetched_at TEXT NOT NULL, "
            "PRIMARY KEY(lang,target_folded,max_results))"
        )
        self._conn.commit()

    def close(self):
        self._conn.close()

    def request_stats(self) -> dict[str, int | None]:
        return {
            "api_calls_used": self._api_calls_used,
            "api_call_budget": self.max_api_calls,
            "api_calls_remaining": (
                None if self.max_api_calls is None
                else max(0, self.max_api_calls - self._api_calls_used)
            ),
        }

    def _reserve_api_call(self) -> None:
        with self._api_counter_lock:
            if (
                self.max_api_calls is not None
                and self._api_calls_used >= self.max_api_calls
            ):
                raise WikipediaError(
                    "Wikimedia API call budget exhausted: "
                    f"used={self._api_calls_used}:budget={self.max_api_calls}"
                )
            self._api_calls_used += 1

    def _api_get(self, url: str, params: dict, max_retries: int | None = None) -> dict:
        if self.offline_only:
            raise WikipediaError("offline_only=True，禁止即時呼叫 Wikimedia API")
        retries = self.api_max_retries if max_retries is None else max_retries
        last_error = None
        for attempt in range(1, retries + 1):
            self._request_throttle.wait()
            try:
                self._reserve_api_call()
                response = self._request_get(
                    url, params=params, headers={"User-Agent": self.user_agent}, timeout=30
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                if "error" in payload:
                    raise WikipediaError(str(payload["error"]))
                return payload
            except WikipediaError:
                raise
            except Exception as exc:  # network/rate-limit only
                last_error = exc
                if attempt < retries:
                    retry_after = 0.0
                    headers = getattr(locals().get("response"), "headers", {})
                    try:
                        retry_after = float(headers.get("Retry-After", 0))
                    except (TypeError, ValueError, AttributeError):
                        retry_after = 0.0
                    exponential = self.backoff_base_seconds * (2 ** (attempt - 1))
                    time.sleep(min(
                        self.backoff_max_seconds,
                        max(retry_after, exponential),
                    ))
        raise WikipediaError(f"Wikimedia API 呼叫失敗：{last_error}")

    def _cache_key(self, title: str, as_of: str | None) -> str:
        return f"{self.lang}|{normalize_title(title).casefold()}|{_as_of_key(as_of)}"

    def _load_cached(self, title: str, as_of: str | None) -> PageSnapshot | None:
        alias_key = self._cache_key(title, as_of)
        row = self._conn.execute(
            "SELECT p.data FROM page_alias a JOIN page_cache p ON p.cache_key=a.cache_key "
            "WHERE a.alias_key=?", (alias_key,)
        ).fetchone()
        if not row:
            row = self._conn.execute(
                "SELECT data FROM page_cache WHERE cache_key=?", (alias_key,)
            ).fetchone()
        return PageSnapshot.from_dict(json.loads(row[0])) if row else None

    def _store(self, requested_title: str, page: PageSnapshot):
        key = self._cache_key(page.title, page.as_of)
        data = json.dumps(page.to_dict(), ensure_ascii=False)
        self._conn.execute(
            "INSERT OR REPLACE INTO page_cache(cache_key,title,as_of,data,fetched_at) VALUES(?,?,?,?,?)",
            (key, page.title, _as_of_key(page.as_of), data, datetime.now(timezone.utc).isoformat()),
        )
        for alias in {requested_title, page.title}:
            self._conn.execute(
                "INSERT OR REPLACE INTO page_alias(alias_key,cache_key) VALUES(?,?)",
                (self._cache_key(alias, page.as_of), key),
            )
        self._conn.execute("DELETE FROM page_links WHERE source_key=?", (key,))
        self._conn.executemany(
            "INSERT OR IGNORE INTO page_links(source_key,target_folded,target_title) VALUES(?,?,?)",
            [(key, link.target.casefold(), link.target) for link in page.links],
        )
        self._conn.commit()

    def _load_parsed_revision(
        self, revision_id: int
    ) -> tuple[str, str, list[PageLink]] | None:
        """Load date-independent rendered content for one immutable revision."""
        row = self._conn.execute(
            "SELECT title,content,links_json FROM revision_cache "
            "WHERE lang=? AND revision_id=?",
            (self.lang, revision_id),
        ).fetchone()
        if not row:
            return None
        links = [PageLink(**value) for value in json.loads(row[2])]
        return str(row[0]), str(row[1]), links

    def _store_parsed_revision(
        self, revision_id: int, title: str, content: str, links: list[PageLink]
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO revision_cache("
            "lang,revision_id,title,content,links_json,fetched_at) VALUES(?,?,?,?,?,?)",
            (
                self.lang,
                revision_id,
                title,
                content,
                json.dumps(
                    [{"target": link.target, "anchor": link.anchor} for link in links],
                    ensure_ascii=False,
                ),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def _resolve_revision(self, title: str, as_of: str | None) -> tuple[str, int, int, str]:
        params = {
            "action": "query", "format": "json", "formatversion": "2", "redirects": "1",
            "titles": normalize_title(title), "prop": "revisions", "rvprop": "ids|timestamp",
            "rvlimit": "1",
        }
        if as_of:
            params.update({"rvstart": _timestamp_for_api(as_of), "rvdir": "older"})
        data = self._api_get(self.api_url, params)
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            raise WikipediaError(f"Wikipedia 找不到頁面：{title}")
        page = pages[0]
        revisions = page.get("revisions", [])
        if not revisions:
            when = f" at/before {as_of}" if as_of else ""
            raise WikipediaError(f"頁面 {title}{when} 沒有可用 revision")
        rev = revisions[0]
        return page["title"], int(page["pageid"]), int(rev["revid"]), rev["timestamp"]

    def _parse_revision(self, revision_id: int) -> tuple[str, str, set[str]]:
        data = self._api_get(self.api_url, {
            "action": "parse", "format": "json", "formatversion": "2", "oldid": revision_id,
            "prop": "text|links|displaytitle",
        })
        parsed = data["parse"]
        allowed = {
            link["title"] for link in parsed.get("links", [])
            if link.get("ns") == 0 and not link.get("missing")
        }
        return parsed.get("title", ""), parsed.get("text", ""), allowed

    def fetch_page(self, title: str, as_of: str | None = None, refresh: bool = False) -> PageSnapshot:
        if not refresh:
            cached = self._load_cached(title, as_of)
            if cached:
                return cached
        if self.offline_only:
            raise WikipediaError(
                f"offline_only=True，但 snapshot 沒有 {title!r}（as_of={as_of!r}）"
            )
        resolved, page_id, revision_id, timestamp = self._resolve_revision(title, as_of)
        parsed_revision = None if refresh else self._load_parsed_revision(revision_id)
        if parsed_revision is None:
            parsed_title, html, allowed = self._parse_revision(revision_id)
            parser = _VisiblePageParser(allowed)
            parser.feed(html)
            content, links = parser.result()
            canonical = normalize_title(parsed_title or resolved)
            self._store_parsed_revision(revision_id, canonical, content, links)
        else:
            canonical, content, links = parsed_revision
        page = PageSnapshot(
            title=canonical, page_id=page_id, revision_id=revision_id, timestamp=timestamp,
            as_of=as_of, content=content, links=links,
            source_url=f"https://{self.lang}.wikipedia.org/?oldid={revision_id}",
        )
        self._store(title, page)
        return page

    def _fetch_page_network_only(
        self, title: str, as_of: str | None,
    ) -> PageSnapshot | None:
        """Fetch without SQLite access so independent candidates can run in threads."""
        try:
            resolved, page_id, revision_id, timestamp = self._resolve_revision(
                title, as_of,
            )
            parsed_title, html, allowed = self._parse_revision(revision_id)
            parser = _VisiblePageParser(allowed)
            parser.feed(html)
            content, links = parser.result()
            canonical = normalize_title(parsed_title or resolved)
            return PageSnapshot(
                title=canonical, page_id=page_id, revision_id=revision_id,
                timestamp=timestamp, as_of=as_of, content=content, links=links,
                source_url=f"https://{self.lang}.wikipedia.org/?oldid={revision_id}",
            )
        except WikipediaError:
            return None

    def fetch_revision(self, revision_id: int, refresh: bool = False) -> PageSnapshot:
        """Fetch one exact immutable revision for the v2 graph environment."""
        if isinstance(revision_id, bool) or int(revision_id) <= 0:
            raise ValueError("revision_id must be a positive integer")
        revision_id = int(revision_id)
        metadata_key = f"{self.lang}|exact-revision|{revision_id}"
        cached_metadata = self._conn.execute(
            "SELECT data FROM revision_metadata_page_cache WHERE cache_key=?",
            (metadata_key,),
        ).fetchone()
        metadata = None if refresh or not cached_metadata else json.loads(
            cached_metadata[0]
        )
        if metadata is None:
            data = self._api_get(self.api_url, {
                "action": "query", "format": "json", "formatversion": "2",
                "revids": str(revision_id), "prop": "revisions",
                "rvprop": "ids|timestamp",
            })
            pages = data.get("query", {}).get("pages", [])
            revisions = pages[0].get("revisions", []) if pages else []
            if not pages or pages[0].get("missing") or not revisions:
                raise WikipediaError(f"Wikipedia 找不到 revision：{revision_id}")
            metadata = {
                "title": str(pages[0]["title"]),
                "page_id": int(pages[0]["pageid"]),
                "revision_id": int(revisions[0]["revid"]),
                "timestamp": str(revisions[0]["timestamp"]),
            }
            self._conn.execute(
                "INSERT OR REPLACE INTO revision_metadata_page_cache("
                "cache_key,data,fetched_at) VALUES(?,?,?)",
                (
                    metadata_key, json.dumps(metadata, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()
        parsed = None if refresh else self._load_parsed_revision(revision_id)
        if parsed is None:
            parsed_title, html, allowed = self._parse_revision(revision_id)
            parser = _VisiblePageParser(allowed)
            parser.feed(html)
            content, links = parser.result()
            title = normalize_title(parsed_title or str(metadata["title"]))
            self._store_parsed_revision(revision_id, title, content, links)
        else:
            title, content, links = parsed
        return PageSnapshot(
            title=title,
            page_id=int(metadata["page_id"]),
            revision_id=revision_id,
            timestamp=str(metadata["timestamp"]),
            as_of=str(metadata["timestamp"]),
            content=content,
            links=links,
            source_url=f"https://{self.lang}.wikipedia.org/?oldid={revision_id}",
        )

    def list_revision_metadata_page(
        self,
        title: str,
        from_date: str,
        to_date: str,
        *,
        cursor: str | None = None,
        page_size: int = 50,
    ) -> dict:
        """Return one reproducible MediaWiki revision page without sampling."""
        start = self._strict_date(from_date, "from")
        end = self._strict_date(to_date, "to")
        if start > end:
            raise ValueError("from must not be after to")
        if isinstance(page_size, bool) or not 1 <= int(page_size) <= 500:
            raise ValueError("page_size must be an integer from 1 through 500")
        page_size = int(page_size)
        key_payload = {
            "lang": self.lang,
            "title": normalize_title(title).casefold(),
            "from": start,
            "to": end,
            "cursor": cursor,
            "page_size": page_size,
        }
        cache_key = "revision-page|" + hashlib.sha256(json.dumps(
            key_payload, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        cached = self._conn.execute(
            "SELECT data FROM revision_metadata_page_cache WHERE cache_key=?",
            (cache_key,),
        ).fetchone()
        if cached:
            return json.loads(cached[0])
        params = {
            "action": "query", "format": "json", "formatversion": "2",
            "redirects": "1", "titles": normalize_title(title),
            "prop": "revisions", "rvprop": "ids|timestamp",
            "rvlimit": str(page_size), "rvdir": "newer",
            "rvstart": f"{start}T00:00:00Z", "rvend": f"{end}T23:59:59Z",
        }
        if cursor:
            params["rvcontinue"] = cursor
        data = self._api_get(self.api_url, params)
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            raise WikipediaError(f"Wikipedia 找不到頁面：{title}")
        result = {
            "title": str(pages[0]["title"]),
            "from_date": start,
            "to_date": end,
            "cursor": cursor,
            "next_cursor": data.get("continue", {}).get("rvcontinue"),
            "revisions": [
                {
                    "revision_id": int(row["revid"]),
                    "parent_id": int(row.get("parentid") or 0),
                    "timestamp": str(row["timestamp"]),
                }
                for row in pages[0].get("revisions", [])
            ],
        }
        self._conn.execute(
            "INSERT OR REPLACE INTO revision_metadata_page_cache("
            "cache_key,data,fetched_at) VALUES(?,?,?)",
            (
                cache_key, json.dumps(result, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return result

    @staticmethod
    def _strict_date(value: str, field: str) -> str:
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d").date()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must use strict YYYY-MM-DD form") from exc
        if parsed.isoformat() != value:
            raise ValueError(f"{field} must use strict YYYY-MM-DD form")
        return value

    @staticmethod
    def _sample_revision_dates(values: list[str], limit: int) -> list[str]:
        """Evenly retain dates across the interval instead of only early edits."""
        unique = sorted(set(values))
        if len(unique) <= limit:
            return unique
        if limit == 1:
            return [unique[-1]]
        indices = {
            round(index * (len(unique) - 1) / (limit - 1))
            for index in range(limit)
        }
        return [unique[index] for index in sorted(indices)]

    def list_revision_dates(
        self, title: str, from_date: str, to_date: str, limit: int = 10
    ) -> list[str]:
        """List sampled revision dates, deliberately omitting content and comments."""
        start = self._strict_date(from_date, "from")
        end = self._strict_date(to_date, "to")
        if start > end:
            raise ValueError("from must not be after to")
        if isinstance(limit, bool) or not 1 <= int(limit) <= 20:
            raise ValueError("limit must be an integer from 1 through 20")
        limit = int(limit)
        key = (self.lang, normalize_title(title).casefold(), start, end)
        cached = self._conn.execute(
            "SELECT dates_json FROM revision_date_cache WHERE "
            "lang=? AND title_folded=? AND from_date=? AND to_date=?", key,
        ).fetchone()
        if cached:
            return self._sample_revision_dates(list(json.loads(cached[0])), limit)
        if self.offline_only:
            raise WikipediaError(
                f"offline_only=True，但沒有 {title!r} 在 {start}..{end} 的 revision 日期快取"
            )

        dates: list[str] = []
        continuation: str | None = None
        # A hard bound prevents an extremely active article from causing an
        # unbounded API scan.  Sampling is deterministic and spans all fetched
        # dates, so the result is still useful for choosing temporal probes.
        while len(dates) < 5000:
            params = {
                "action": "query", "format": "json", "formatversion": "2",
                "redirects": "1", "titles": normalize_title(title),
                "prop": "revisions", "rvprop": "timestamp", "rvlimit": "500",
                "rvdir": "newer", "rvstart": f"{start}T00:00:00Z",
                "rvend": f"{end}T23:59:59Z",
            }
            if continuation:
                params["rvcontinue"] = continuation
            data = self._api_get(self.api_url, params)
            pages = data.get("query", {}).get("pages", [])
            if not pages or pages[0].get("missing"):
                raise WikipediaError(f"Wikipedia 找不到頁面：{title}")
            dates.extend(
                str(item.get("timestamp", ""))[:10]
                for item in pages[0].get("revisions", [])
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}T.*", str(item.get("timestamp", "")))
            )
            continuation = data.get("continue", {}).get("rvcontinue")
            if not continuation:
                break
        if continuation:
            # Extremely active pages can exceed the 5,000-revision scan cap.
            # In that case, discard the early-date-biased scan and probe fixed
            # calendar quantiles, retaining only real revision dates returned
            # by MediaWiki.  This keeps the tool bounded without presenting a
            # misleading cluster from the beginning of the interval.
            start_day = date.fromisoformat(start)
            end_day = date.fromisoformat(end)
            span = (end_day - start_day).days
            probed_dates = []
            probe_count = 20 if span else 1
            for index in range(probe_count):
                probe = start_day + timedelta(
                    days=round(span * index / (probe_count - 1)) if span else 0
                )
                data = self._api_get(self.api_url, {
                    "action": "query", "format": "json", "formatversion": "2",
                    "redirects": "1", "titles": normalize_title(title),
                    "prop": "revisions", "rvprop": "timestamp", "rvlimit": "1",
                    "rvdir": "older", "rvstart": f"{probe.isoformat()}T23:59:59Z",
                })
                pages = data.get("query", {}).get("pages", [])
                revisions = pages[0].get("revisions", []) if pages else []
                if revisions:
                    value = str(revisions[0].get("timestamp", ""))[:10]
                    if start <= value <= end:
                        probed_dates.append(value)
            all_dates = sorted(set(probed_dates))
        else:
            all_dates = sorted(set(dates))
        self._conn.execute(
            "INSERT OR REPLACE INTO revision_date_cache("
            "lang,title_folded,from_date,to_date,dates_json,fetched_at) "
            "VALUES(?,?,?,?,?,?)",
            (*key, json.dumps(all_dates), datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()
        return self._sample_revision_dates(all_dates, limit)

    def resolve_qid_title(self, qid: str) -> str:
        data = self._api_get(self.wikidata_url, {
            "action": "wbgetentities", "format": "json", "ids": qid, "props": "sitelinks",
        })
        entity = data.get("entities", {}).get(qid, {})
        link = entity.get("sitelinks", {}).get(f"{self.lang}wiki")
        if not link:
            raise WikipediaError(f"{qid} 沒有 {self.lang} Wikipedia sitelink")
        return normalize_title(link["title"])

    def resolve_title_qid(self, title: str) -> str:
        """Resolve a Wikipedia article title to its Wikidata item ID."""
        data = self._api_get(self.api_url, {
            "action": "query", "format": "json", "formatversion": "2",
            "redirects": "1", "titles": normalize_title(title),
            "prop": "pageprops", "ppprop": "wikibase_item",
        })
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            raise WikipediaError(f"Wikipedia 找不到頁面：{title}")
        qid = pages[0].get("pageprops", {}).get("wikibase_item")
        if not isinstance(qid, str) or not re.fullmatch(r"Q\d+", qid):
            raise WikipediaError(f"Wikipedia 頁面 {title!r} 沒有 Wikidata item")
        return qid

    def get_wikidata_entities(
        self, qids: list[str], *, props: str = "claims|labels|sitelinks"
    ) -> dict[str, dict]:
        """Fetch a bounded Wikidata entity batch through the official Action API."""
        unique = list(dict.fromkeys(qid for qid in qids if re.fullmatch(r"Q\d+", qid)))
        if not unique:
            return {}
        if len(unique) > 50:
            raise ValueError("get_wikidata_entities accepts at most 50 QIDs per call")
        data = self._api_get(self.wikidata_url, {
            "action": "wbgetentities", "format": "json", "ids": "|".join(unique),
            "props": props, "languages": "en", "sitefilter": f"{self.lang}wiki",
        })
        entities = data.get("entities", {})
        return {
            qid: value for qid, value in entities.items()
            if isinstance(qid, str) and isinstance(value, dict) and not value.get("missing")
        }

    def _cached_backlinks(self, title: str, as_of: str | None) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT p.title FROM page_links l "
            "JOIN page_cache p ON p.cache_key=l.source_key "
            "WHERE l.target_folded=? AND p.as_of=? ORDER BY p.title",
            (normalize_title(title).casefold(), _as_of_key(as_of)),
        ).fetchall()
        return [row[0] for row in rows]

    def find_backlinks(
        self, title: str, as_of: str | None = None, max_results: int = 50
    ) -> list[str]:
        """Return pages whose selected revision really links to ``title``.

        For historical snapshots MediaWiki cannot directly query historical
        backlinks.  We use current backlinks as candidates and verify each
        candidate's selected historical revision.  This is sound for returned
        pages, but cannot recover links that existed historically and have since
        disappeared; snapshot manifests must disclose this coverage limitation.
        """
        cached = self._cached_backlinks(title, as_of)
        if self.offline_only:
            return cached[:max_results]

        target_folded = normalize_title(title).casefold()
        candidate_row = self._conn.execute(
            "SELECT candidates_json FROM backlink_candidate_cache "
            "WHERE lang=? AND target_folded=? AND max_results=?",
            (self.lang, target_folded, max_results),
        ).fetchone()

        if candidate_row:
            candidates = list(json.loads(candidate_row[0]))
        else:
            candidates = []
            continuation = None
            while len(candidates) < max_results:
                params = {
                    "action": "query", "format": "json", "formatversion": "2",
                    "list": "backlinks", "bltitle": normalize_title(title),
                    "blnamespace": "0", "bllimit": str(min(500, max_results * 3)),
                }
                if continuation:
                    params["blcontinue"] = continuation
                data = self._api_get(self.api_url, params)
                candidates.extend(
                    item["title"]
                    for item in data.get("query", {}).get("backlinks", [])
                )
                continuation = data.get("continue", {}).get("blcontinue")
                if not continuation:
                    break
            candidates = list(dict.fromkeys(candidates))
            self._conn.execute(
                "INSERT OR REPLACE INTO backlink_candidate_cache("
                "lang,target_folded,max_results,candidates_json,fetched_at) "
                "VALUES(?,?,?,?,?)",
                (
                    self.lang, target_folded, max_results,
                    json.dumps(candidates, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

        verified = list(cached)
        seen = {item.casefold() for item in verified}
        uncached: list[str] = []
        pages_by_candidate: dict[str, PageSnapshot] = {}
        network_attempted: set[str] = set()
        for candidate in candidates:
            if candidate.casefold() in seen:
                continue
            page = self._load_cached(candidate, as_of)
            if page is None:
                uncached.append(candidate)
            else:
                pages_by_candidate[candidate] = page
        if uncached and self.backlink_verify_workers > 1:
            network_attempted.update(uncached)
            with ThreadPoolExecutor(
                max_workers=self.backlink_verify_workers,
            ) as executor:
                fetched = executor.map(
                    lambda candidate: self._fetch_page_network_only(candidate, as_of),
                    uncached,
                )
                for candidate, page in zip(uncached, fetched, strict=True):
                    if page is not None:
                        pages_by_candidate[candidate] = page
                        self._store_parsed_revision(
                            page.revision_id, page.title, page.content, page.links,
                        )
                        self._store(candidate, page)
        for candidate in candidates:
            if candidate.casefold() in seen:
                continue
            page = pages_by_candidate.get(candidate)
            if page is None:
                if candidate in network_attempted:
                    continue
                try:
                    page = self.fetch_page(candidate, as_of=as_of)
                except WikipediaError:
                    continue
            if any(link.target.casefold() == target_folded for link in page.links):
                verified.append(page.title)
                seen.add(page.title.casefold())
                if len(verified) >= max_results:
                    break
        return verified[:max_results]


def reverse_bfs_frontier(
    backend: WikipediaPageBackend,
    target_title: str,
    max_depth: int,
    as_of: str | None = None,
    branch_cap: int = 25,
) -> dict[int, list[str]]:
    """Find start pages with a directed hyperlink path to the target page."""
    target = backend.fetch_page(target_title, as_of=as_of).title
    frontiers = {0: [target]}
    visited = {target.casefold()}
    current = [target]
    for depth in range(1, max_depth + 1):
        next_level: list[str] = []
        for page_title in current:
            for source in backend.find_backlinks(page_title, as_of=as_of, max_results=branch_cap):
                folded = source.casefold()
                if folded not in visited:
                    visited.add(folded)
                    next_level.append(source)
        if not next_level:
            break
        frontiers[depth] = next_level
        current = next_level
    return frontiers
