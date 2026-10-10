"""
G.I.D.E.O.N HackerNews Firebase API Tool

Fetches top, new, best, ask, show, or job stories from official HackerNews Firebase REST API.
Does not require any API keys.
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from gideon.core.logs import get_logger
from gideon.tools.base import Param, tool

log = get_logger("gideon.tools.news")

VALID_CATEGORIES = {"top", "new", "best", "ask", "show", "job", "jobs"}


def _fetch_item(item_id: int) -> dict[str, Any] | None:
    url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
    req = urllib.request.Request(url, headers={"User-Agent": "GIDEON-Assistant/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


@tool(
    "get_hackernews",
    "Fetch top tech news stories, Ask HN, Show HN, or recent posts from HackerNews. Call when user asks for tech news or HackerNews stories.",
    Param(
        "category",
        "HackerNews category.",
        enum=("top", "new", "best", "ask", "show", "jobs"),
        aliases=("cat",),
        default="top",
    ),
    Param("limit", "Number of stories (1-10, default 5).", type="integer", default=5),
    untrusted_output=True,
    triggers={"hackernews", "hn", "headlines", "story", "stories", "article", "news"},
)
def get_hackernews(category: str = "top", limit: int = 5) -> str:
    """Fetch stories from HackerNews (categories: top, new, best, ask, show, jobs)."""
    cat = (category or "top").strip().lower().rstrip("s")
    if cat == "job":
        cat_key = "job"
    elif cat in ("top", "new", "best", "ask", "show"):
        cat_key = cat
    else:
        cat_key = "top"

    count = max(1, min(int(limit), 10))
    stories_url = f"https://hacker-news.firebaseio.com/v0/{cat_key}stories.json"

    req = urllib.request.Request(stories_url, headers={"User-Agent": "GIDEON-Assistant/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            story_ids = json.loads(resp.read().decode("utf-8"))
    except Exception as err:
        log.error("Failed to fetch HN story list: %s", err)
        return f"Error fetching HackerNews {cat_key} stories: {err}"

    if not story_ids:
        return f"No stories found for HackerNews category '{cat_key}'."

    target_ids = story_ids[:count]
    items: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=min(5, count)) as executor:
        futures = {executor.submit(_fetch_item, sid): sid for sid in target_ids}
        for future in as_completed(futures):
            res = future.result()
            if res:
                items.append(res)

    # Sort items by original order in target_ids
    id_order = {sid: idx for idx, sid in enumerate(target_ids)}
    items.sort(key=lambda item: id_order.get(item.get("id", 0), 99))

    lines = [f"HackerNews ({cat_key.upper()} Stories):"]
    for idx, item in enumerate(items, 1):
        title = item.get("title", "Untitled")
        score = item.get("score", 0)
        by = item.get("by", "anonymous")
        comments = item.get("descendants", 0)
        url = item.get("url") or f"https://news.ycombinator.com/item?id={item.get('id')}"

        lines.append(f"{idx}. {title}")
        lines.append(f"   Points: {score} | By: {by} | Comments: {comments}")
        lines.append(f"   URL: {url}")

    return "\n".join(lines)
