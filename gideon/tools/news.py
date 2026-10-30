"""
G.I.D.E.O.N News Tools

get_news — headlines by section from Google News RSS (the edition set by
GIDEON_REGION), and football stories from ESPN. No keys, no accounts.

get_hackernews — top, new, best, ask, show, or job stories from the official
HackerNews Firebase REST API.
"""

from __future__ import annotations

import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from gideon.core.config import REGION
from gideon.core.logs import get_logger
from gideon.tools.base import Param, tool
from gideon.tools.football import football_stories

log = get_logger("gideon.tools.news")

_GOOGLE_NEWS = "https://news.google.com/rss"
# get_news topic → Google News section (None: the edition's front page).
_SECTIONS = {
    "top": None,
    "world": "WORLD",
    "national": "NATION",
    "business": "BUSINESS",
    "technology": "TECHNOLOGY",
    "science": "SCIENCE",
    "health": "HEALTH",
    "sports": "SPORTS",
    "entertainment": "ENTERTAINMENT",
    "football": None,  # ESPN, see football.py
}
# What small models write instead of the enum value.
_TOPIC_SYNONYMS = {
    "tech": "technology", "sport": "sports", "nation": "national", "india": "national",
    "soccer": "football", "general": "top", "headlines": "top", "top stories": "top",
    "economy": "business", "finance": "business", "space": "science",
}
_REGION_WORDS = {
    "IN": ("india", "indian"),
    "US": ("america", "american", "usa"),
    "GB": ("uk", "britain", "british"),
    "AU": ("australia", "australian"),
    "CA": ("canada", "canadian"),
}
# Never-assume: a section the user did not ask for is dropped (top stories).
_TOPIC_WORDS = {
    "top": ("top", "headline", "headlines", "news", "today", "latest", "happening", "breaking"),
    "world": ("world", "international", "global", "worldwide", "foreign", "abroad"),
    "national": ("national", "domestic", "country", *_REGION_WORDS.get(REGION, ())),
    "business": (
        "business", "economy", "economic", "market", "markets", "finance", "financial",
        "companies", "corporate",
    ),
    "technology": (
        "tech", "technology", "gadget", "gadgets", "ai", "software", "computing", "startup",
        "startups",
    ),
    "science": ("science", "scientific", "space", "research", "climate"),
    "health": ("health", "medical", "medicine", "disease", "fitness"),
    "sports": ("sport", "sports", "cricket", "tennis", "f1", "formula", "olympics", "nba", "nfl"),
    "entertainment": (
        "entertainment", "movie", "movies", "film", "films", "celebrity", "bollywood", "hollywood",
        "tv", "shows",
    ),
    "football": (
        "football", "soccer", "transfer", "transfers", "premier", "liga", "isl", "ucl", "champions",
    ),
}
_TOPIC_EVIDENCE = {
    **_TOPIC_WORDS,
    **{alias: _TOPIC_WORDS[topic] for alias, topic in _TOPIC_SYNONYMS.items()},
}
_HEADLINES = 8
# Daily puzzle-answer pages flood the technology section ("Today's NYT
# Strands Hints, Spangram and Answers…").
_PUZZLE_SPAM = re.compile(
    r"\b(wordle|strands|connections|crossword|quordle)\b.*\b(hints?|answers?)\b", re.IGNORECASE
)


def _topic(value: Any) -> str:
    text = " ".join(str(value or "").lower().replace("_", " ").split())
    text = _TOPIC_SYNONYMS.get(text, text)
    return text if text in _SECTIONS else "top"


def _ago(moment: datetime) -> str:
    minutes = max(0, int((datetime.now(UTC) - moment).total_seconds() // 60))
    if minutes < 60:
        return f"{minutes} min ago"
    if minutes < 48 * 60:
        return f"{minutes // 60} h ago"
    return f"{minutes // 1440} days ago"


def _fetch_text(url: str) -> str | None:
    """One GET; None on any failure (tests replace this)."""
    try:
        response = httpx.get(
            url,
            timeout=8.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 GIDEON-Assistant/1.0"},
        )
        response.raise_for_status()
        return response.text
    except Exception as exc:
        log.warning("News request failed for %s: %s", url, exc)
        return None


def google_news_stories(feed: str) -> list[tuple[datetime, str, str]]:
    """(published, headline, source) from a Google News RSS document."""
    try:
        root = ET.fromstring(feed)
    except ET.ParseError as exc:
        log.warning("Unreadable news feed: %s", exc)
        return []
    stories, seen = [], set()
    for item in root.iter("item"):
        title = " ".join((item.findtext("title") or "").split())
        source = (item.findtext("source") or "").strip()
        # Titles end with " - Source"; the source is listed separately.
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].rstrip()
        key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
        if not title or key in seen or _PUZZLE_SPAM.search(title):
            continue
        seen.add(key)
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            continue
        stories.append((published, title, source))
    return stories


@tool(
    "get_news",
    "Latest news headlines, top stories or one section. Call for news, headlines or what's happening.",
    Param(
        "topic",
        "Section; omit for top stories.",
        enum=tuple(_SECTIONS),
        aliases=("section", "category"),
        default="top",
        evidence=_TOPIC_EVIDENCE,
        normalize=_topic,
    ),
    untrusted_output=True,
    triggers={
        "news", "headlines", "headline", "happening", "breaking", "world", "football", "soccer",
        "transfer",
    },
)
def get_news(topic: str = "top") -> str:
    """Headlines for one news section, with each story's age."""
    stamp = datetime.now(UTC).astimezone().strftime("%a %d %b, %H:%M")
    if topic == "football":
        stories = football_stories()
        if not stories:
            return "Couldn't reach the football news service right now. Try again shortly."
        lines = [f"FOOTBALL NEWS (ESPN, fetched {stamp}):"]
        for i, (published, headline, summary) in enumerate(stories[:_HEADLINES], 1):
            lines.append(f"{i}. {headline} ({_ago(published)})")
            if summary and summary != headline:
                lines.append(f"   {summary[:180]}")
        return "\n".join(lines)

    region = REGION
    edition = f"hl=en-{region}&gl={region}&ceid={region}:en"
    section = _SECTIONS.get(topic)
    url = (
        f"{_GOOGLE_NEWS}/headlines/section/topic/{section}?{edition}"
        if section
        else f"{_GOOGLE_NEWS}?{edition}"
    )
    feed = _fetch_text(url)
    stories = google_news_stories(feed) if feed else []
    if not stories:
        return "Couldn't reach the news service right now. Try again shortly."
    # Google ranks stories by importance, not time: keep its order and show
    # each story's age.
    label = "TOP STORIES" if topic == "top" else f"{topic.upper()} NEWS"
    lines = [f"{label} (Google News, {region} edition, fetched {stamp}):"]
    for i, (published, headline, source) in enumerate(stories[:_HEADLINES], 1):
        lines.append(f"{i}. {headline} ({source + ', ' if source else ''}{_ago(published)})")
    return "\n".join(lines)


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
    "Hacker News stories: top, new, best, Ask HN, Show HN, jobs. Call when the user asks about Hacker News.",
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
