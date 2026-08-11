"""Tool-calling agent that browses rendered Wikipedia pages via hyperlinks."""

from __future__ import annotations

import json
import re
import uuid

from tkg.api.openrouter import OpenRouterError, call_model_with_tools
from tkg.wikipedia.backend import WikipediaError, normalize_title


MAX_PAGE_CHARS = 16_000
MAX_LINKS_SHOWN = 80
SEARCH_CONTEXT_CHARS = 280

SNAPSHOT_INTENTS = [
    "latest_available",
    "seek_post_cutoff_update",
    "historical_comparison",
    "follow_assigned_time",
    "uncertain",
    "other",
]

TOOLS = [
    {"type": "function", "function": {
        "name": "view_current_page",
        "description": "Read the visible main content of the current Wikipedia page.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "list_links",
        "description": "List hyperlinks actually present in the current page. Use offset to page "
                       "through a long list, or query to filter by anchor/target text.",
        "parameters": {"type": "object", "properties": {
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "query": {"type": "string", "default": ""},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "follow_link",
        "description": "Open one hyperlink from the current page.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "Exact target title returned by list_links"},
        }, "required": ["target"]},
    }},
    {"type": "function", "function": {
        "name": "search_within_page",
        "description": "Find text snippets containing a phrase in the current page.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "stop_browsing",
        "description": "End this browsing session.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
]


def _snapshot_token(as_of: str | None) -> str:
    return as_of or "CURRENT"


def _snapshot_tool(allowed_as_of: list[str | None]) -> list[dict]:
    tokens = [_snapshot_token(value) for value in allowed_as_of]
    return [{"type": "function", "function": {
        "name": "select_snapshot",
        "description": (
            "Bind this browsing session to one allowed Wikipedia snapshot. This must be the "
            "first action. The selected time cannot be changed later. Give only a brief action "
            "justification, not private chain-of-thought."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "as_of": {"type": "string", "enum": tokens},
                "intent_code": {"type": "string", "enum": SNAPSHOT_INTENTS},
                "brief_reason": {
                    "type": "string",
                    "description": "One short sentence explaining this observable action.",
                },
            },
            "required": ["as_of", "intent_code", "brief_reason"],
        },
    }}]


def _temporal_tools(allowed_as_of: list[str | None]) -> list[dict]:
    tokens = [_snapshot_token(value) for value in allowed_as_of]
    switch_tool = {"type": "function", "function": {
        "name": "switch_snapshot",
        "description": (
            "Load the same current Wikipedia page at another allowed time. You may call this "
            "repeatedly. The page revision and its outgoing hyperlinks both change with time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "as_of": {"type": "string", "enum": tokens},
                "brief_reason": {
                    "type": "string",
                    "description": "One short observable reason for changing time.",
                },
            },
            "required": ["as_of", "brief_reason"],
        },
    }}
    submit_tool = {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Finish exploration and submit the final answer to the question.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    }}
    return [switch_tool, *TOOLS[:-1], submit_tool]


def run_snapshot_selection(
    model: str,
    allowed_as_of: list[str | None],
    *,
    snapshot_mode: str,
    task_prompt: str,
    call_model_fn=call_model_with_tools,
    max_attempts: int = 3,
) -> dict:
    """Require an explicit, auditable snapshot-binding tool call before browsing."""
    if snapshot_mode not in {"assigned", "agent_selected"}:
        raise ValueError(f"unknown snapshot_mode: {snapshot_mode}")
    unique: list[str | None] = []
    for value in allowed_as_of:
        if value not in unique:
            unique.append(value)
    if not unique:
        raise ValueError("allowed_as_of must not be empty")
    if snapshot_mode == "assigned" and len(unique) != 1:
        raise ValueError("assigned snapshot mode requires exactly one allowed snapshot")

    token_to_value = {_snapshot_token(value): value for value in unique}
    tokens = list(token_to_value)
    if snapshot_mode == "assigned":
        instruction = (
            f'The experiment assigned snapshot "{tokens[0]}". Your first action must be '
            "select_snapshot with that exact value."
        )
    else:
        instruction = (
            "Choose the Wikipedia snapshot you want to browse from the allowed values. Your "
            "choice is part of the measured behavior; after selection it is immutable."
        )
    messages = [
        {"role": "system", "content": task_prompt},
        {"role": "user", "content": (
            f"{instruction}\nAllowed snapshots: {json.dumps(tokens)}\n"
            "Call select_snapshot now. Do not browse or answer in plain text yet."
        )},
    ]
    attempts = []
    selection_id = uuid.uuid4().hex
    for attempt_index in range(1, max_attempts + 1):
        assistant = call_model_fn(
            model, messages, _snapshot_tool(unique), temperature=0.0
        )
        messages.append(assistant)
        calls = assistant.get("tool_calls") or []
        if not calls:
            attempts.append({
                "attempt": attempt_index, "status": "no_tool_call",
                "free_text": assistant.get("content"),
            })
            messages.append({
                "role": "user",
                "content": "You must call select_snapshot with one allowed value.",
            })
            continue
        call = calls[0]
        fn = call.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        token = str(args.get("as_of", ""))
        intent_code = str(args.get("intent_code", "other"))
        if intent_code not in SNAPSHOT_INTENTS:
            intent_code = "other"
        brief_reason = " ".join(str(args.get("brief_reason", "")).split())[:300]
        error = None
        if fn.get("name") != "select_snapshot":
            error = f"unknown tool {fn.get('name')!r}; select_snapshot is required"
        elif token not in token_to_value:
            error = f"snapshot {token!r} is not allowed"
        elif not brief_reason:
            error = "brief_reason must be one short non-empty sentence"
        if error:
            result = f"Error: {error}"
            attempts.append({
                "attempt": attempt_index, "status": "invalid", "args": args,
                "result": result,
            })
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            continue
        selected_as_of = token_to_value[token]
        payload = {
            "snapshot_id": f"wikipedia@{token}",
            "selected_as_of": selected_as_of,
            "selection_token": token,
            "revision_policy": "latest_revision_at_or_before",
            "frozen": True,
        }
        result = json.dumps(payload, ensure_ascii=False)
        attempts.append({
            "attempt": attempt_index, "status": "selected", "args": args,
            "result": payload,
        })
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        return {
            "selection_id": selection_id,
            "status": "selected",
            "snapshot_mode": snapshot_mode,
            "allowed_as_of": unique,
            "selected_as_of": selected_as_of,
            "selection_token": token,
            "intent_code": intent_code,
            "brief_reason": brief_reason,
            "attempts": attempts,
            "messages": messages,
        }
    return {
        "selection_id": selection_id,
        "status": "failed",
        "snapshot_mode": snapshot_mode,
        "allowed_as_of": unique,
        "selected_as_of": None,
        "selection_token": None,
        "intent_code": None,
        "brief_reason": "",
        "attempts": attempts,
        "messages": messages,
    }


def _visible_page_content(page) -> str:
    content = page.content
    if len(content) > MAX_PAGE_CHARS:
        content = content[:MAX_PAGE_CHARS] + "\n\n[Page truncated by the experiment UI.]"
    return content


def _render_page(page) -> str:
    content = _visible_page_content(page)
    return (
        f'Wikipedia page: "{page.title}"\n'
        f"Revision: {page.revision_id} at {page.timestamp}\n\n{content}"
    )


def _evidence_page(page, visible_content: str) -> dict:
    value = page.to_dict()
    # The judge must see exactly what the tested model saw, never hidden text
    # from below the UI truncation boundary.
    value["content"] = visible_content
    return value


def _render_links(
    page,
    visited: set[str],
    offset: int = 0,
    query: str = "",
    allowed_targets: set[str] | None = None,
) -> str:
    if not page.links:
        return "This page has no eligible article hyperlinks."
    links = page.links
    if allowed_targets is not None:
        links = [link for link in links if link.target.casefold() in allowed_targets]
    if query.strip():
        needle = query.casefold()
        links = [link for link in links
                 if needle in link.target.casefold() or needle in link.anchor.casefold()]
    offset = max(0, offset)
    lines = []
    for link in links[offset:offset + MAX_LINKS_SHOWN]:
        tag = " (visited)" if link.target.casefold() in visited else ""
        lines.append(f'- target="{link.target}" | anchor="{link.anchor}"{tag}')
    if not lines:
        return f"No hyperlinks matched query={query!r} at offset={offset}."
    next_offset = offset + len(lines)
    if next_offset < len(links):
        lines.append(
            f"[Showing {offset + 1}-{next_offset} of {len(links)} links; "
            f"call list_links with offset={next_offset} for the next page.]"
        )
    return "\n".join(lines)


def _search_page(content: str, query: str) -> str:
    if not query.strip():
        return "Error: query must not be empty."
    matches = list(re.finditer(re.escape(query), content, flags=re.IGNORECASE))[:8]
    if not matches:
        return f'No matches for "{query}" on this page.'
    snippets = []
    for match in matches:
        start = max(0, match.start() - SEARCH_CONTEXT_CHARS)
        end = min(len(content), match.end() + SEARCH_CONTEXT_CHARS)
        snippets.append(" ".join(content[start:end].split()))
    return "\n---\n".join(snippets)


def run_wikipedia_browsing(
    model: str,
    backend,
    start_title: str,
    max_steps: int,
    task_prompt: str,
    target_title: str | None = None,
    as_of: str | None = None,
    temperature: float = 0.7,
    call_model_fn=call_model_with_tools,
    verbose: bool = True,
    initial_messages: list[dict] | None = None,
) -> dict:
    """Run one autonomous browsing trajectory.

    Reaching the target is only a ``page_hit``.  It intentionally does *not*
    mean that the pivot was visible or understood; those are separate gates.
    """
    current = backend.fetch_page(start_title, as_of=as_of)
    target_folded = normalize_title(target_title).casefold() if target_title else None
    visited_path = [current.title]
    visited = {current.title.casefold()}
    evidence_pages: list[dict] = []
    trajectory: list[dict] = []
    snapshot_context = (
        f'This browsing session is frozen at or before "{as_of}". Every page tool is '
        "restricted to that same snapshot date; you cannot change the date."
        if as_of else
        "This browsing session uses the experiment's current frozen snapshot. Every page tool "
        "is restricted to that same snapshot; you cannot change it."
    )
    if initial_messages is None:
        messages = [
            {"role": "system", "content": task_prompt},
            {"role": "user", "content": (
                f"{snapshot_context}\n\n"
                f'You are starting on the Wikipedia page "{current.title}". Use the tools to '
                "read it, inspect its hyperlinks, follow links, search within the current page, "
                "or stop."
            )},
        ]
    else:
        messages = list(initial_messages)
        messages.append({"role": "user", "content": (
            f'Snapshot binding is complete. You are starting on the Wikipedia page "{current.title}". '
            "Use the page tools to read it, inspect its hyperlinks, follow links, search within "
            "the current page, or stop."
        )})

    def log(message):
        if verbose:
            print(message)

    stop_reason = "max_steps"
    page_hit = current.title.casefold() == target_folded if target_folded else False
    if page_hit:
        # Starting at the pivot still requires actually displaying the page.
        rendered = _render_page(current)
        messages.append({"role": "user", "content": rendered})
        evidence_pages.append(_evidence_page(current, _visible_page_content(current)))
        stop_reason = "target_page_opened"
        return {
            "start_title": start_title, "final_title": current.title, "page_hit": True,
            "stop_reason": stop_reason, "visited_titles": visited_path,
            "trajectory": trajectory, "messages": messages, "evidence_pages": evidence_pages,
        }

    for step in range(1, max_steps + 1):
        try:
            assistant = call_model_fn(model, messages, TOOLS, temperature=temperature)
        except OpenRouterError as exc:
            trajectory.append({"step": step, "from_title": current.title, "action": "error",
                               "args": {}, "result": str(exc)})
            stop_reason = "error"
            break
        messages.append(assistant)
        calls = assistant.get("tool_calls") or []
        if not calls:
            trajectory.append({"step": step, "from_title": current.title,
                               "action": "no_tool_call", "args": {},
                               "free_text": assistant.get("content"), "result": ""})
            stop_reason = "no_tool_call"
            break

        should_stop = False
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            origin = current.title
            try:
                if name == "view_current_page":
                    result = _render_page(current)
                    evidence_pages.append(_evidence_page(current, _visible_page_content(current)))
                elif name == "list_links":
                    try:
                        offset = int(args.get("offset", 0))
                    except (TypeError, ValueError):
                        offset = 0
                    result = _render_links(
                        current, visited, offset=offset, query=str(args.get("query", ""))
                    )
                    evidence_pages.append(_evidence_page(current, "[VISIBLE LINK LIST]\n" + result))
                elif name == "search_within_page":
                    result = _search_page(current.content, str(args.get("query", "")))
                    # Search reveals page text, so the page belongs in the evidence set.
                    evidence_pages.append(_evidence_page(current, "[SEARCH RESULT]\n" + result))
                elif name == "follow_link":
                    requested = normalize_title(str(args.get("target", "")))
                    valid = {link.target.casefold(): link.target for link in current.links}
                    canonical = valid.get(requested.casefold())
                    if not canonical:
                        result = f'Error: "{requested}" is not a hyperlink on the current page.'
                    else:
                        current = backend.fetch_page(canonical, as_of=as_of)
                        visited.add(current.title.casefold())
                        visited_path.append(current.title)
                        result = _render_page(current)
                        evidence_pages.append(_evidence_page(
                            current, _visible_page_content(current)
                        ))
                        if target_folded and current.title.casefold() == target_folded:
                            page_hit = True
                            stop_reason = "target_page_opened"
                            should_stop = True
                elif name == "stop_browsing":
                    result = "Browsing ended."
                    stop_reason = "stop_browsing"
                    should_stop = True
                else:
                    result = f"Error: unknown tool {name}."
            except WikipediaError as exc:
                result = f"Error: {exc}"

            record = {"step": step, "from_title": origin, "action": name, "args": args,
                      "free_text": assistant.get("content"), "result": result}
            trajectory.append(record)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            log(f"[browse {step}] {origin} --{name}--> {current.title}")
            if should_stop:
                break
        if should_stop:
            break

    # Deduplicate exact displays, while retaining later search snippets/link
    # lists from the same revision because those are distinct visible evidence.
    unique_evidence = []
    seen_revisions = set()
    for page in evidence_pages:
        key = (page["title"].casefold(), page["revision_id"], page.get("content", ""))
        if key not in seen_revisions:
            seen_revisions.add(key)
            unique_evidence.append(page)
    return {
        "start_title": start_title, "final_title": current.title, "page_hit": page_hit,
        "stop_reason": stop_reason, "visited_titles": visited_path,
        "trajectory": trajectory, "messages": messages, "evidence_pages": unique_evidence,
    }


def run_temporal_browsing(
    model: str,
    backend,
    start_title: str,
    question: str,
    allowed_as_of: list[str | None],
    max_steps: int,
    *,
    target_title: str,
    target_as_of: str | None,
    allowed_version_keys: set[str] | None = None,
    allowed_titles_by_snapshot: dict[str, list[str]] | None = None,
    reveal_target_title: bool = True,
    cutoff_reference: str | None = None,
    temperature: float = 0.7,
    call_model_fn=call_model_with_tools,
    verbose: bool = True,
) -> dict:
    """Browse a page-revision graph with repeatable time switching."""
    unique_dates: list[str | None] = []
    for value in allowed_as_of:
        if value not in unique_dates:
            unique_dates.append(value)
    if len(unique_dates) < 2:
        raise ValueError("temporal browsing requires at least two allowed snapshots")
    token_to_value = {_snapshot_token(value): value for value in unique_dates}
    tokens = list(token_to_value)
    target_token = _snapshot_token(target_as_of)
    if target_token not in token_to_value:
        raise ValueError(f"target snapshot {target_token!r} is not in allowed snapshots")
    arena_versions = set(allowed_version_keys or [])
    arena_titles = {
        token: {title.casefold() for title in titles}
        for token, titles in (allowed_titles_by_snapshot or {}).items()
    }

    def version_key(page) -> str:
        return f"{int(page.revision_id)}:{page.title.casefold()}"
    target_instruction = (
        f'Target pivot page title: "{target_title}"\n'
        "Navigate through real page hyperlinks to the target pivot, then answer the "
        "question as of the target snapshot."
        if reveal_target_title else
        "The target page title is deliberately hidden. Resolve the relative relation "
        "chain from the starting page through real hyperlinks, then answer from the "
        "page you identify at the target snapshot."
    )
    cutoff_instruction = (
        f"\nRegistered knowledge-cutoff snapshot for this tested model: {cutoff_reference}"
        if cutoff_reference else ""
    )
    messages = [
        {"role": "system", "content": (
            "Answer the user's temporal question by exploring Wikipedia page revisions. "
            "You may switch time repeatedly and follow only hyperlinks exposed by the current "
            "revision. Use submit_answer when finished. Give brief tool reasons, not private "
            "chain-of-thought. Only page-version nodes inside the experiment's bounded "
            "navigation arena are traversable."
        )},
        {"role": "user", "content": (
            f"Question: {question}\n"
            f'Starting page title: "{start_title}"\n'
            f"{target_instruction}\n"
            f"Allowed snapshots: {json.dumps(tokens)}\n"
            f"Target snapshot to answer: {target_token}{cutoff_instruction}\n"
            "Other snapshots are available for comparison.\n"
            "Your first action must be switch_snapshot. You may switch_snapshot again at any "
            "later step."
        )},
    ]
    tools = _temporal_tools(unique_dates)
    current = None
    current_as_of: str | None = None
    current_token: str | None = None
    current_title = normalize_title(start_title)
    visited_titles: set[str] = set()
    visited_versions: list[dict] = []
    evidence_pages: list[dict] = []
    trajectory: list[dict] = []
    final_answer = ""
    stop_reason = "max_steps"

    def log(message: str) -> None:
        if verbose:
            print(message)

    def remember_page(page, token: str) -> None:
        record = {
            "title": page.title,
            "revision_id": page.revision_id,
            "timestamp": page.timestamp,
            "as_of": page.as_of,
            "snapshot_token": token,
        }
        if record not in visited_versions:
            visited_versions.append(record)

    for _turn in range(1, max_steps + 1):
        try:
            assistant = call_model_fn(model, messages, tools, temperature=temperature)
        except OpenRouterError as exc:
            trajectory.append({
                "step": len(trajectory) + 1,
                "from_title": current_title, "to_title": current_title,
                "action": "error", "args": {}, "result": str(exc),
                "snapshot_token": current_token, "revision_id": (
                    current.revision_id if current else None
                ),
            })
            stop_reason = "error"
            break
        messages.append(assistant)
        calls = assistant.get("tool_calls") or []
        if not calls:
            free_text = " ".join(str(assistant.get("content") or "").split())
            trajectory.append({
                "step": len(trajectory) + 1,
                "from_title": current_title, "to_title": current_title,
                "action": "no_tool_call", "args": {}, "result": "",
                "free_text": free_text, "snapshot_token": current_token,
                "revision_id": current.revision_id if current else None,
            })
            if free_text:
                final_answer = free_text
                stop_reason = "free_text_answer"
            else:
                stop_reason = "no_tool_call"
            break

        should_stop = False
        for call in calls:
            if len(trajectory) >= max_steps:
                stop_reason = "max_steps"
                should_stop = True
                break
            fn = call.get("function", {})
            name = str(fn.get("name", ""))
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            origin_title = current_title
            origin_token = current_token
            origin_revision = current.revision_id if current else None
            result: str
            try:
                if name == "switch_snapshot":
                    token = str(args.get("as_of", ""))
                    reason = " ".join(str(args.get("brief_reason", "")).split())[:300]
                    if token not in token_to_value:
                        result = f"Error: snapshot {token!r} is not allowed."
                    elif not reason:
                        result = "Error: brief_reason must not be empty."
                    else:
                        next_as_of = token_to_value[token]
                        switched = backend.fetch_page(current_title, as_of=next_as_of)
                        if arena_versions and version_key(switched) not in arena_versions:
                            result = (
                                "Error: that page revision is outside the bounded "
                                "shortest-path arena."
                            )
                        else:
                            current_as_of = next_as_of
                            current_token = token
                            current = switched
                            current_title = current.title
                            visited_titles.add(current.title.casefold())
                            remember_page(current, token)
                            result = _render_page(current)
                            evidence_pages.append(_evidence_page(
                                current, _visible_page_content(current)
                            ))
                elif current is None:
                    result = "Error: call switch_snapshot before using page tools."
                elif name == "view_current_page":
                    result = _render_page(current)
                    evidence_pages.append(_evidence_page(
                        current, _visible_page_content(current)
                    ))
                elif name == "list_links":
                    try:
                        offset = int(args.get("offset", 0))
                    except (TypeError, ValueError):
                        offset = 0
                    result = _render_links(
                        current, visited_titles, offset=offset,
                        query=str(args.get("query", "")),
                        allowed_targets=arena_titles.get(current_token or "CURRENT"),
                    )
                    evidence_pages.append(_evidence_page(
                        current, "[VISIBLE LINK LIST]\n" + result
                    ))
                elif name == "search_within_page":
                    result = _search_page(current.content, str(args.get("query", "")))
                    evidence_pages.append(_evidence_page(
                        current, "[SEARCH RESULT]\n" + result
                    ))
                elif name == "follow_link":
                    requested = normalize_title(str(args.get("target", "")))
                    eligible = arena_titles.get(current_token or "CURRENT")
                    valid = {
                        link.target.casefold(): link.target for link in current.links
                        if eligible is None or link.target.casefold() in eligible
                    }
                    canonical = valid.get(requested.casefold())
                    if not canonical:
                        result = (
                            f'Error: "{requested}" is not an eligible hyperlink on the '
                            "current page in this arena."
                        )
                    else:
                        destination = backend.fetch_page(canonical, as_of=current_as_of)
                        if arena_versions and version_key(destination) not in arena_versions:
                            result = "Error: hyperlink destination is outside the bounded arena."
                        else:
                            current = destination
                            current_title = current.title
                            visited_titles.add(current.title.casefold())
                            remember_page(current, current_token or "CURRENT")
                            result = _render_page(current)
                            evidence_pages.append(_evidence_page(
                                current, _visible_page_content(current)
                            ))
                elif name == "submit_answer":
                    final_answer = " ".join(str(args.get("answer", "")).split())
                    if not final_answer:
                        result = "Error: answer must not be empty."
                    else:
                        result = "Final answer submitted."
                        stop_reason = "submit_answer"
                        should_stop = True
                else:
                    result = f"Error: unknown tool {name}."
            except WikipediaError as exc:
                result = f"Error: {exc}"

            record = {
                "step": len(trajectory) + 1,
                "from_title": origin_title,
                "to_title": current_title,
                "action": name,
                "args": args,
                "free_text": assistant.get("content"),
                "result": result,
                "from_snapshot_token": origin_token,
                "snapshot_token": current_token,
                "from_revision_id": origin_revision,
                "revision_id": current.revision_id if current else None,
            }
            trajectory.append(record)
            messages.append({
                "role": "tool", "tool_call_id": call["id"], "content": result
            })
            log(
                f"[temporal {record['step']}] {origin_title}@{origin_token} "
                f"--{name}--> {current_title}@{current_token}"
            )
            if should_stop:
                break
        if should_stop:
            break

    unique_evidence = []
    seen_evidence = set()
    for page in evidence_pages:
        key = (page["title"].casefold(), page["revision_id"], page.get("content", ""))
        if key not in seen_evidence:
            seen_evidence.add(key)
            unique_evidence.append(page)
    return {
        "start_title": start_title,
        "final_title": current_title,
        "final_snapshot_token": current_token,
        "final_answer": final_answer,
        "stop_reason": stop_reason,
        "trajectory": trajectory,
        "messages": messages,
        "visited_versions": visited_versions,
        "evidence_pages": unique_evidence,
        "target_title_revealed": reveal_target_title,
    }
