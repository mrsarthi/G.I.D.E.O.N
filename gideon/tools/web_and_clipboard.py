"""
G.I.D.E.O.N Web Search & Clipboard Tools

Search goes through the duckduckgo_search library rather than a regex over
DuckDuckGo's HTML — the old `result__snippet` scrape broke every time they
touched their markup. The Instant Answer / Wikipedia paths are kept as
fallbacks for when the library is rate-limited or unavailable.
"""

from __future__ import annotations

import html as html_parser
import json
import re
import urllib.parse
import urllib.request

import pyperclip

from gideon.core.logs import get_logger
from gideon.tools.base import Param, tool

log = get_logger("gideon.tools.web")

MAX_RESULTS = 5
CLIPBOARD_LIMIT = 2500

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# The package is renaming itself from duckduckgo_search to ddgs; support both.
try:  # pragma: no cover - import shim
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    try:
        from duckduckgo_search import DDGS  # type: ignore[assignment]
    except ImportError:
        DDGS = None  # type: ignore[assignment,misc]


def _search_via_library(query: str) -> list[str]:
    """Primary path: structured results from the maintained client."""
    if DDGS is None:
        return []
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=MAX_RESULTS))
    except Exception as exc:
        log.warning("DDGS library search failed: %s", exc)
        return []

    lines = []
    for hit in hits:
        title = (hit.get("title") or "").strip()
        body = (hit.get("body") or "").strip()
        if body:
            lines.append(f"- {title}: {body}" if title else f"- {body}")
    return lines


def _search_instant_answer(query: str) -> list[str]:
    """Fallback: DuckDuckGo's Instant Answer JSON API."""
    lines = []
    try:
        url = (
            f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}"
            "&format=json&no_html=1&no_redirect=1"
        )
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))

        abstract = (data.get("AbstractText") or "").strip()
        heading = (data.get("Heading") or "").strip()
        if abstract:
            lines.append(f"Summary ({heading}): {abstract}" if heading else abstract)

        topics = [
            t.get("Text", "").strip()
            for t in data.get("RelatedTopics", [])
            if t.get("Text")
        ]
        if topics:
            lines.append("Related: " + " | ".join(topics[:2]))
    except Exception as exc:
        log.debug("Instant Answer fallback failed: %s", exc)
    return lines


def _search_wikipedia(query: str) -> list[str]:
    """Last resort: Wikipedia REST summary."""
    try:
        slug = urllib.parse.quote(query.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        extract = (data.get("extract") or "").strip()
        if extract:
            return [f"Wikipedia: {extract}"]
    except Exception as exc:
        log.debug("Wikipedia fallback failed: %s", exc)
    return []


@tool(
    "search_web",
    "Perform an invisible background internet search for online news, live facts, documentation, or answers. Call ONLY when the user explicitly asks to search the web, check online information, or asks about events after your training cut-off. NEVER open a browser window; report the findings verbally.",
    Param(
        "query",
        "The search terms to look up.",
        required=True,
        aliases=("search_query", "q"),
    ),
    triggers={
        "search", "google", "look", "lookup", "browse", "internet", "web",
        "online", "news", "latest", "current",
    },
)
def search_web(query: str) -> str:
    """Headless background web search. Never opens a browser window."""
    clean_query = (query or "").strip()
    if not clean_query:
        return "Search query was empty."

    for strategy in (_search_via_library, _search_instant_answer, _search_wikipedia):
        results = strategy(clean_query)
        if results:
            log.info("Web search '%s' answered by %s", clean_query, strategy.__name__)
            body = "\n".join(results)
            # Strip any stray markup the sources leak through.
            body = html_parser.unescape(re.sub(r"<[^>]+>", "", body))
            return f"WEB SEARCH RESULTS FOR '{clean_query}':\n{body}"

    log.info("Web search '%s' returned nothing", clean_query)
    return (
        f"Could not retrieve web results for '{clean_query}'. "
        "No quick answers were found online."
    )


@tool(
    "get_clipboard",
    "Read whatever text or code is currently copied to the user's Windows clipboard. Call ONLY when the user asks you to read, analyze, check, or review their clipboard.",
    triggers={"clipboard", "paste", "copied"},
)
def get_clipboard() -> str:
    """Read current text content from the Windows clipboard."""
    try:
        content = pyperclip.paste().strip()
    except Exception as exc:
        log.error("Clipboard read failed: %s", exc)
        return f"Failed to read from the Windows clipboard: {exc}"

    if not content:
        return "The Windows clipboard is currently empty."
    if len(content) > CLIPBOARD_LIMIT:
        content = content[:CLIPBOARD_LIMIT] + "\n[...truncated...]"
    return f"[CURRENT WINDOWS CLIPBOARD CONTENTS]:\n{content}"


@tool(
    "set_clipboard",
    "Copy text, code, or answers into the user's Windows clipboard so they can paste with Ctrl+V. Call ONLY when the user explicitly instructs you to copy something.",
    Param(
        "text",
        "The content to copy into the clipboard.",
        required=True,
        aliases=("content",),
    ),
    triggers={"clipboard", "copy"},
)
def set_clipboard(text: str) -> str:
    """Write content to the Windows clipboard."""
    if not text:
        return "Nothing was provided to copy."
    try:
        pyperclip.copy(text)
        return "Copied to your Windows clipboard — paste anywhere with Ctrl+V."
    except Exception as exc:
        log.error("Clipboard write failed: %s", exc)
        return f"Failed to copy to the Windows clipboard: {exc}"
