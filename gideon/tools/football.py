"""
G.I.D.E.O.N Football Tool

Scores, results, fixtures and league tables from ESPN's public soccer
endpoints — no key, no account. They are unofficial, so every failure comes
back as a plain "couldn't reach" line instead of an exception.

Football *news* is served through get_news(topic="football") — see
football_stories() — so the model has one place to look for news.
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from typing import Any

import httpx

from gideon.core.logs import get_logger
from gideon.tools.base import Param, tool

log = get_logger("gideon.tools.football")

_SITE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
_STANDINGS = "https://site.api.espn.com/apis/v2/sports/soccer"
_TIMEOUT = 8.0

# The competition names the model sees → (ESPN code, words that show the user
# meant it). Never-assume: a league the user did not name is dropped.
LEAGUES: dict[str, tuple[str, tuple[str, ...]]] = {
    "premier league": ("eng.1", ("premier", "epl", "english", "england")),
    "la liga": ("esp.1", ("liga", "laliga", "spanish", "spain")),
    "serie a": ("ita.1", ("serie", "italian", "italy")),
    "bundesliga": ("ger.1", ("bundesliga", "german", "germany")),
    "ligue 1": ("fra.1", ("ligue", "french", "france")),
    "champions league": ("uefa.champions", ("champions", "ucl")),
    "europa league": ("uefa.europa", ("europa", "uel")),
    "indian super league": ("ind.1", ("isl", "indian", "india")),
    "mls": ("usa.1", ("mls", "usa", "american")),
}
# Searched for a team's home league; cup competitions only hold these clubs.
_DOMESTIC = (
    "premier league", "la liga", "serie a", "bundesliga", "ligue 1", "indian super league", "mls",
)
# "Any football today?" covers these.
_OVERVIEW = (
    "premier league", "la liga", "serie a", "bundesliga", "ligue 1", "champions league",
    "indian super league",
)
# News sources for get_news(topic="football"), widest coverage first.
_NEWS_LEAGUES = ("eng.1", "esp.1", "uefa.champions", "ita.1", "ger.1", "ind.1")

# Nicknames ESPN does not carry.
_TEAM_ALIASES = {
    "barca": "barcelona",
    "man u": "manchester united",
    "man utd": "manchester united",
    "inter milan": "internazionale",
    "inter": "internazionale",
    "psg": "paris saint germain",
    "atleti": "atletico madrid",
    "juve": "juventus",
    "bayern": "bayern munich",
    # Fixture nicknames: a club's results include the match (2026-10-30: "the
    # latest el classico scores" named no club).
    "el clasico": "barcelona",
    "clasico": "barcelona",
    "el classico": "barcelona",
    "classico": "barcelona",
}
_NOISE_WORDS = frozenset(["fc", "cf", "afc", "sc", "the", "club"])

_TABLE_WORDS = (
    "table", "standings", "standing", "position", "rank", "ranked", "ranking", "place", "top",
    "bottom", "points",
)
_MATCH_WORDS = (
    "match", "matches", "game", "games", "score", "scores", "result", "results", "fixture",
    "fixtures", "play", "playing", "played", "next", "last", "won", "lost", "beat", "draw", "drew",
    "do", "did", "doing",
)
# Keyed by what the model may send, not only the enum value: evidence is
# checked before `normalize` runs ("standings", "EPL").
_VIEW_EVIDENCE = {"table": _TABLE_WORDS, "standings": _TABLE_WORDS, "matches": _MATCH_WORDS}
_LEAGUE_EVIDENCE = {
    key: words for name, (code, words) in LEAGUES.items() for key in (name, code, *words)
}


def _get_json(url: str) -> dict[str, Any] | None:
    """One GET; None on any failure (tests replace this)."""
    try:
        response = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        log.warning("ESPN request failed for %s: %s", url, exc)
        return None


def _get_many(urls: list[str]) -> list[dict[str, Any] | None]:
    if len(urls) <= 1:
        return [_get_json(url) for url in urls]
    with ThreadPoolExecutor(max_workers=min(8, len(urls))) as pool:
        return list(pool.map(_get_json, urls))


def _normalize(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    words = re.findall(r"[a-z0-9]+", plain.lower().replace("&", " and "))
    return " ".join(w for w in words if w not in _NOISE_WORDS)


def _display(league: str) -> str:
    return "MLS" if league == "mls" else league.title()


# ---------------------------------------------------------------------------
#  Teams
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Team:
    id: str
    name: str
    league: str  # a LEAGUES key
    aliases: frozenset[str]


_TEAMS_TTL_SECONDS = 24 * 3600
_teams_lock = threading.Lock()
_teams_cache: tuple[float, list[Team]] | None = None


def _load_teams() -> list[Team]:
    """Every club in the domestic leagues, fetched once a day."""
    global _teams_cache
    with _teams_lock:
        if _teams_cache and time.monotonic() - _teams_cache[0] < _TEAMS_TTL_SECONDS:
            return _teams_cache[1]
        pages = _get_many([f"{_SITE}/{LEAGUES[name][0]}/teams" for name in _DOMESTIC])
        teams: list[Team] = []
        for league, page in zip(_DOMESTIC, pages, strict=True):
            try:
                entries = page["sports"][0]["leagues"][0]["teams"] if page else []
            except (KeyError, IndexError, TypeError):
                entries = []
            for entry in entries:
                info = entry.get("team", {})
                names = {
                    info.get(k)
                    for k in ("displayName", "shortDisplayName", "name", "nickname", "location")
                }
                aliases = {_normalize(n) for n in names if n}
                if info.get("abbreviation"):
                    aliases.add(info["abbreviation"].lower())
                teams.append(
                    Team(
                        str(info.get("id")),
                        info.get("displayName", "?"),
                        league,
                        frozenset(aliases - {""}),
                    )
                )
        if teams:
            _teams_cache = (time.monotonic(), teams)
        return teams


def find_team(query: str, teams: list[Team]) -> Team | list[Team] | None:
    """The one club `query` names; a list when it could be several; None."""
    wanted = _normalize(query)
    wanted = _TEAM_ALIASES.get(wanted, wanted)
    if not wanted:
        return None
    exact = [t for t in teams if wanted in t.aliases]
    if exact:
        return exact[0]
    tokens = wanted.split()
    containing = [
        t for t in teams if any(all(tok in a.split() for tok in tokens) for a in t.aliases)
    ]
    if len(containing) == 1:
        return containing[0]
    if containing:
        return containing
    # A misspelling: "Barcelonna", "Tottenam".
    scored = sorted(
        (
            (max(SequenceMatcher(None, wanted, a).ratio() for a in t.aliases), t)
            for t in teams
            if t.aliases
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if scored and scored[0][0] >= 0.8 and (len(scored) == 1 or scored[1][0] < scored[0][0]):
        return scored[0][1]
    return None


# ---------------------------------------------------------------------------
#  Matches
# ---------------------------------------------------------------------------
def _when(event: dict[str, Any]) -> datetime | None:
    try:
        return datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
    except (KeyError, ValueError, AttributeError):
        return None


def _local(moment: datetime, with_time: bool = True) -> str:
    local = moment.astimezone()
    day = f"{local:%a} {local.day} {local:%b}"
    return f"{day}, {local:%H:%M}" if with_time else day


def _score(competitor: dict[str, Any]) -> tuple[str, float | None]:
    """(goals, shootout goals) — scoreboards give a string, schedules a dict."""
    score = competitor.get("score")
    shootout = competitor.get("shootoutScore")
    if isinstance(score, dict):
        shootout = score.get("shootoutScore", shootout)
        score = score.get("displayValue")
    return (str(score) if score not in (None, "") else "?"), shootout


def event_line(event: dict[str, Any], competition: str = "") -> str:
    """One match as a line: result, live score, or kick-off time."""
    comp = (event.get("competitions") or [{}])[0]
    status = (comp.get("status") or event.get("status") or {}).get("type", {})
    sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
    home, away = sides.get("home", {}), sides.get("away", {})
    home_name = home.get("team", {}).get("displayName", "?")
    away_name = away.get("team", {}).get("displayName", "?")
    moment = _when(event)
    label = f", {competition}" if competition else ""
    state = status.get("state")

    if state == "pre":
        when = _local(moment) if moment else "Time to be confirmed"
        return f"{when}{label}: {home_name} vs {away_name}"
    if state == "post" and not status.get("completed"):
        # Postponed, cancelled, abandoned: a 0-0 here would be invented.
        day = _local(moment, with_time=False) if moment else ""
        return (
            f"{day}{label}: {home_name} vs {away_name} — {status.get('description', 'not played')}"
        )

    (home_goals, home_pens), (away_goals, away_pens) = _score(home), _score(away)
    line = f"{home_name} {home_goals}-{away_goals} {away_name}"
    if state == "in":
        return f"LIVE ({status.get('shortDetail') or status.get('detail') or 'in play'}){label}: {line}"
    if home_pens is not None and away_pens is not None:
        winner = home_name if home_pens > away_pens else away_name
        high, low = sorted((int(home_pens), int(away_pens)), reverse=True)
        line += f" — {winner} won {high}-{low} on penalties"
    day = _local(moment, with_time=False) if moment else ""
    return f"{day}{label}: {line}"


def _month_keys(since: datetime, until: datetime) -> list[str]:
    keys, cursor = [], since.replace(day=1)
    while cursor <= until:
        keys.append(f"{cursor:%Y%m}")
        cursor = (cursor + timedelta(days=32)).replace(day=1)
    return keys


def _league_events(
    leagues: list[str], since: datetime, until: datetime
) -> dict[str, list[dict[str, Any]]]:
    """Each league's matches between `since` and `until`, in date order."""
    months = _month_keys(since, until)
    urls = [
        f"{_SITE}/{LEAGUES[name][0]}/scoreboard?dates={month}"
        for name in leagues
        for month in months
    ]
    pages = iter(_get_many(urls))
    found: dict[str, list[dict[str, Any]]] = {}
    for name in leagues:
        events: dict[str, dict[str, Any]] = {}
        for _month in months:
            page = next(pages)
            for event in (page or {}).get("events", []):
                moment = _when(event)
                if moment and since <= moment <= until:
                    events[str(event.get("id"))] = event
        found[name] = sorted(events.values(), key=lambda e: _when(e) or since)
    return found


def _state(event: dict[str, Any]) -> str:
    comp = (event.get("competitions") or [{}])[0]
    return (comp.get("status") or event.get("status") or {}).get("type", {}).get("state", "")


def _stamp() -> str:
    return datetime.now(UTC).astimezone().strftime("%a %d %b, %H:%M")


def _team_matches(team: Team) -> str:
    code = LEAGUES[team.league][0]
    played, upcoming, summary = _get_many(
        [
            f"{_SITE}/all/teams/{team.id}/schedule",
            f"{_SITE}/all/teams/{team.id}/schedule?fixture=true",
            f"{_SITE}/{code}/teams/{team.id}",
        ]
    )
    if played is None and upcoming is None:
        return f"Couldn't reach the football scores service for {team.name}. Try again shortly."

    standing = ((summary or {}).get("team") or {}).get("standingSummary")
    lines = [f"{team.name}" + (f" — {standing}" if standing else "") + f" (as of {_stamp()}):"]

    def competition(event: dict[str, Any]) -> str:
        league = event.get("league")
        return league.get("name", "") if isinstance(league, dict) else ""

    results = [e for e in (played or {}).get("events", []) if _state(e) == "post"]
    results.sort(key=lambda e: _when(e) or datetime.min.replace(tzinfo=UTC), reverse=True)
    live = [
        e
        for e in (played or {}).get("events", []) + (upcoming or {}).get("events", [])
        if _state(e) == "in"
    ]
    fixtures = [e for e in (upcoming or {}).get("events", []) if _state(e) == "pre"]
    fixtures.sort(key=lambda e: _when(e) or datetime.max.replace(tzinfo=UTC))

    if live:
        lines.append("Playing now:")
        lines += [f"- {event_line(e, competition(e))}" for e in live[:1]]
    lines.append("Last results:" if results else "Last results: none this season yet.")
    lines += [f"- {event_line(e, competition(e))}" for e in results[:5]]
    lines.append("Next matches:" if fixtures else "Next matches: none scheduled yet.")
    lines += [f"- {event_line(e, competition(e))}" for e in fixtures[:3]]
    return "\n".join(lines)


def _league_matches(league: str) -> str:
    now = datetime.now(UTC)
    events = _league_events([league], now - timedelta(days=7), now + timedelta(days=7))[league]
    if not events:
        return f"No {_display(league)} matches in the last or next 7 days (as of {_stamp()})."
    results = [e for e in events if _state(e) == "post"][-10:]
    live = [e for e in events if _state(e) == "in"]
    fixtures = [e for e in events if _state(e) == "pre"][:10]
    lines = [f"{_display(league)} (as of {_stamp()}):"]
    for title, group in (
        ("Playing now", live),
        ("Results, last 7 days", results),
        ("Fixtures, next 7 days", fixtures),
    ):
        if group:
            lines.append(f"{title}:")
            lines += [f"- {event_line(e)}" for e in group]
    return "\n".join(lines)


def _overview() -> str:
    now = datetime.now(UTC)
    by_league = _league_events(
        list(_OVERVIEW), now - timedelta(hours=24), now + timedelta(hours=24)
    )
    lines = [f"Football in the last and next 24 hours (as of {_stamp()}):"]
    for league in _OVERVIEW:
        events = by_league.get(league) or []
        if events:
            lines.append(f"{_display(league)}:")
            lines += [f"- {event_line(e)}" for e in events[:8]]
    if len(lines) == 1:
        covered = ", ".join(_display(name) for name in _OVERVIEW)
        return f"No matches in the last or next 24 hours in: {covered} (as of {_stamp()})."
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Tables
# ---------------------------------------------------------------------------
def _table(league: str, highlight: Team | None = None) -> str:
    page = _get_json(f"{_STANDINGS}/{LEAGUES[league][0]}/standings")
    groups = (page or {}).get("children") or []
    if not groups:
        return f"Couldn't get the {_display(league)} table right now. Try again shortly."
    lines = [f"{_display(league)} table (as of {_stamp()}):"]
    for group in groups:
        entries = group.get("standings", {}).get("entries", [])
        if len(groups) > 1:
            lines.append(f"{group.get('name', '')}:")
        rows: list[tuple[int, str]] = []
        for entry in entries:
            stats = {s.get("name"): s.get("displayValue") for s in entry.get("stats", [])}
            team = entry.get("team", {})
            rank = int(stats.get("rank") or 0) or len(rows) + 1
            mark = "  <-" if highlight and str(team.get("id")) == highlight.id else ""
            rows.append(
                (
                    rank,
                    f"{rank}. {team.get('displayName', '?')} — {stats.get('points', '?')} pts "
                    f"(P{stats.get('gamesPlayed', '?')} W{stats.get('wins', '?')} D{stats.get('ties', '?')} "
                    f"L{stats.get('losses', '?')}, GD {stats.get('pointDifferential', '?')}){mark}",
                )
            )
        lines += [text for _rank, text in sorted(rows)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  News (served by get_news)
# ---------------------------------------------------------------------------
def football_stories() -> list[tuple[datetime, str, str]]:
    """Recent football stories as (published, headline, summary), newest first."""
    pages = _get_many([f"{_SITE}/{code}/news?limit=10" for code in _NEWS_LEAGUES])
    seen: set[str] = set()
    stories: list[tuple[datetime, str, str]] = []
    for page in pages:
        for article in (page or {}).get("articles", []):
            headline = " ".join((article.get("headline") or "").split())
            key = _normalize(headline)
            if not headline or key in seen:
                continue
            seen.add(key)
            try:
                published = datetime.fromisoformat(
                    article.get("published", "").replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                continue
            stories.append(
                (published, headline, " ".join((article.get("description") or "").split()))
            )
    return sorted(stories, key=lambda story: story[0], reverse=True)


# ---------------------------------------------------------------------------
#  Tool
# ---------------------------------------------------------------------------
def _league_name(value: Any) -> str | None:
    text = _normalize(str(value or ""))
    if not text:
        return None
    for name, (code, words) in LEAGUES.items():
        if text in (_normalize(name), code) or text in words:
            return name
    return None


@tool(
    "get_football",
    "Football (soccer) only: live scores, results, when a club plays next, and league tables. "
    "Call for any football match or score question. Not for other sports; for football news use get_news.",
    Param(
        "team",
        "Club the user named, e.g. 'Arsenal'. Omit if none.",
        aliases=("club",),
        default=None,
        grounded=True,
    ),
    Param(
        "league",
        "Only if the user named one.",
        enum=tuple(LEAGUES),
        aliases=("competition",),
        default=None,
        evidence=_LEAGUE_EVIDENCE,
        normalize=_league_name,
    ),
    Param(
        "view",
        "'table' for standings, else 'matches'.",
        enum=("matches", "table"),
        default="matches",
        evidence=_VIEW_EVIDENCE,
        normalize=lambda v: (
            "table" if str(v or "").strip().lower() in ("table", "standings") else "matches"
        ),
    ),
    untrusted_output=True,
    triggers={
        "football", "soccer", "match", "matches", "fixture", "fixtures", "score", "scores",
        "scoreline", "result", "results", "standings", "goal", "goals", "kickoff", "league", "epl",
        "isl", "ucl", "laliga", "bundesliga", "premier", "champions",
        # Clubs, so "how did Arsenal do?" reaches the tools at all.
        "arsenal", "chelsea", "liverpool", "tottenham", "spurs", "everton", "newcastle",
        "barcelona", "barca", "madrid", "atletico", "sevilla", "bayern", "dortmund", "leverkusen",
        "juventus", "juve", "napoli", "inter", "milan", "roma", "psg", "marseille", "bengaluru",
        "bagan", "blasters", "clasico", "classico", "derby",
    },
)
def get_football(team: str | None = None, league: str | None = None, view: str = "matches") -> str:
    """Scores, fixtures or a table for a club, a league, or today's big leagues."""
    found: Team | None = None
    if team:
        teams = _load_teams()
        if not teams:
            return "Couldn't reach the football scores service right now. Try again shortly."
        match = find_team(team, teams)
        if isinstance(match, list):
            names = ", ".join(t.name for t in match[:6])
            return f"'{team}' could be several clubs: {names}. Ask the user which one."
        if match is None:
            covered = ", ".join(_display(name) for name in _DOMESTIC)
            return f"No club called '{team}' in the leagues covered ({covered})."
        found = match

    if view == "table":
        target = league or (found.league if found else None)
        if not target:
            return "Which league's table? Ask the user."
        return _table(target, highlight=found)
    if found:
        return _team_matches(found)
    if league:
        return _league_matches(league)
    return _overview()
