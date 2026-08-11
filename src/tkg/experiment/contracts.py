"""Shared contracts for the Wikipedia browsing experiment.

The old implementation passed loosely shaped dictionaries between the graph
backend, agent, gates and runner.  These small dataclasses make the boundaries
explicit while keeping JSONL output straightforward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class PageLink:
    target: str
    anchor: str


@dataclass
class PageSnapshot:
    title: str
    page_id: int
    revision_id: int
    timestamp: str
    as_of: str | None
    content: str
    links: list[PageLink] = field(default_factory=list)
    source_url: str | None = None

    @property
    def node_id(self) -> str:
        return self.title

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PageSnapshot":
        data = dict(value)
        data["links"] = [PageLink(**link) for link in data.get("links", [])]
        return cls(**data)


class PageBackend(Protocol):
    def fetch_page(self, title: str, as_of: str | None = None) -> PageSnapshot:
        ...

    def find_backlinks(
        self, title: str, as_of: str | None = None, max_results: int = 50
    ) -> list[str]:
        ...


@dataclass
class JudgeResult:
    decision: str
    confidence: float
    reason: str
    evidence: str = ""
    answer_extracted: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
