"""get_news (Google News RSS + ESPN football stories) and get_football (ESPN
scores, fixtures, tables). Network calls are replaced; nothing leaves the
machine."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from gideon.brain.intent import should_offer_tools
from gideon.tools import football, news
from gideon.tools.football import Team, event_line, find_team
from gideon.tools.registry import dispatch_tool_call, get_tool
from gideon.tools.validation import check_call


def _rss_date(hours_ago: float) -> str:
    moment = datetime.now(UTC) - timedelta(hours=hours_ago)
    return moment.strftime("%a, %d %b %Y %H:%M:%S GMT")


FEED = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Top stories</title>
<item><title>Rover finds water ice - Space Daily</title><source url="x">Space Daily</source>
<pubDate>{_rss_date(2)}</pubDate></item>
<item><title>Rover finds water ice - Other Paper</title><source url="y">Other Paper</source>
<pubDate>{_rss_date(3)}</pubDate></item>
<item><title>Today's NYT Strands Hints, Spangram and Answers for Friday - CNET</title><source>CNET</source>
<pubDate>{_rss_date(1)}</pubDate></item>
<item><title>Markets close higher - Daily Ledger</title><source>Daily Ledger</source>
<pubDate>{_rss_date(30)}</pubDate></item>
</channel></rss>"""


# ---------------------------------------------------------------------------
#  get_news
# ---------------------------------------------------------------------------
def test_feed_parsing_strips_sources_dedupes_and_drops_puzzle_pages():
    stories = news.google_news_stories(FEED)
    assert [(title, source) for _when, title, source in stories] == [
        ("Rover finds water ice", "Space Daily"),
        ("Markets close higher", "Daily Ledger"),
    ]


def test_unreadable_feed_is_no_stories():
    assert news.google_news_stories("<html>not a feed") == []


def test_top_stories_name_the_edition_source_and_age():
    with patch.object(news, "_fetch_text", return_value=FEED) as fetch:
        out = news.get_news()
    url = fetch.call_args.args[0]
    assert "/headlines/" not in url and f"gl={news.REGION}" in url
    assert out.startswith("TOP STORIES (Google News")
    assert "1. Rover finds water ice (Space Daily, 2 h ago)" in out
    assert "2. Markets close higher (Daily Ledger, 30 h ago)" in out


def test_a_section_reads_its_own_feed():
    with patch.object(news, "_fetch_text", return_value=FEED) as fetch:
        out = news.get_news("technology")
    assert "/headlines/section/topic/TECHNOLOGY?" in fetch.call_args.args[0]
    assert out.startswith("TECHNOLOGY NEWS")


def test_news_service_down_is_said_plainly():
    with patch.object(news, "_fetch_text", return_value=None):
        assert news.get_news("world").startswith("Couldn't reach the news service")


def test_football_news_comes_from_espn_stories():
    stories = [
        (
            datetime.now(UTC) - timedelta(minutes=30),
            "Striker signs new deal",
            "He stays until 2030.",
        ),
        (datetime.now(UTC) - timedelta(hours=5), "Coach sacked", ""),
    ]
    with patch.object(news, "football_stories", return_value=stories):
        out = news.get_news("football")
    assert out.startswith("FOOTBALL NEWS (ESPN")
    assert "1. Striker signs new deal (30 min ago)\n   He stays until 2030." in out
    assert "2. Coach sacked (5 h ago)" in out


def test_topic_synonyms_normalize():
    bind = get_tool("get_news").bind
    assert bind({"topic": "tech"})["topic"] == "technology"
    assert bind({"category": "Soccer"})["topic"] == "football"
    assert bind({"topic": "celebrity gossip"})["topic"] == "top"
    assert bind({})["topic"] == "top"


def test_the_section_the_user_named_wins_over_the_models():
    """Eval 2026-10-30: "what's the latest football news?" → topic="sports"."""
    spec = get_tool("get_news")
    check = check_call(spec, {"topic": "sports"}, "what's the latest football news?")
    assert check.ok and check.arguments == {"topic": "football"} and check.corrected == ["topic"]
    # Two sections fit "tech and business news": nothing to choose between.
    check = check_call(spec, {"topic": "world"}, "any tech and business news?")
    assert check.arguments == {} and check.dropped == ["topic"]


def test_a_section_the_user_did_not_ask_for_is_dropped():
    spec = get_tool("get_news")
    assert check_call(spec, {"topic": "technology"}, "any tech news?").arguments == {
        "topic": "technology"
    }
    assert check_call(spec, {"topic": "tech"}, "what's new in tech").arguments == {"topic": "tech"}
    assert check_call(spec, {"topic": "football"}, "latest transfer gossip").arguments == {
        "topic": "football"
    }
    dropped = check_call(spec, {"topic": "business"}, "what's in the news today?")
    assert dropped.ok and dropped.arguments == {} and dropped.dropped == ["topic"]


# ---------------------------------------------------------------------------
#  get_football — teams
# ---------------------------------------------------------------------------
def _team(id_: str, name: str, league: str, *extra: str) -> Team:
    aliases = {football._normalize(name), *(football._normalize(e) for e in extra)}
    return Team(id_, name, league, frozenset(aliases))


TEAMS = [
    _team("359", "Arsenal", "premier league", "ARS"),
    _team("367", "Tottenham Hotspur", "premier league", "Spurs", "TOT"),
    _team("382", "Manchester City", "premier league", "Man City"),
    _team("360", "Manchester United", "premier league", "Man United"),
    _team("86", "Real Madrid", "la liga"),
    _team("1068", "Atlético Madrid", "la liga", "Atlético"),
    _team("83", "Barcelona", "la liga", "Barça"),
]


def test_team_names_resolve_by_name_nickname_alias_and_typo():
    assert find_team("arsenal fc", TEAMS).id == "359"
    assert find_team("Spurs", TEAMS).id == "367"
    assert find_team("man utd", TEAMS).id == "360"
    assert find_team("Barca", TEAMS).id == "83"
    assert find_team("atletico", TEAMS).id == "1068"
    assert find_team("Barcelonna", TEAMS).id == "83"
    # "the latest el classico scores": Barcelona's results include it.
    assert find_team("El Classico", TEAMS).id == "83"
    assert find_team("el clásico", TEAMS).id == "83"


def test_a_name_that_fits_several_clubs_is_a_question():
    match = find_team("madrid", TEAMS)
    assert sorted(t.name for t in match) == ["Atlético Madrid", "Real Madrid"]
    with patch.object(football, "_load_teams", return_value=TEAMS):
        out = football.get_football(team="Manchester")
    assert "could be several clubs" in out and "Ask the user which one" in out


def test_an_unknown_club_is_said_not_guessed():
    assert find_team("Gotham Rovers", TEAMS) is None
    with patch.object(football, "_load_teams", return_value=TEAMS):
        assert football.get_football(team="Gotham Rovers").startswith(
            "No club called 'Gotham Rovers'"
        )


# ---------------------------------------------------------------------------
#  get_football — matches and tables
# ---------------------------------------------------------------------------
def _event(
    home,
    away,
    *,
    state,
    days,
    home_score="0",
    away_score="0",
    completed=True,
    desc="",
    pens=None,
    league="",
):
    when = (datetime.now(UTC) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%MZ")
    sides = []
    for side, name, score, pen in (
        ("home", home, home_score, (pens or (None, None))[0]),
        ("away", away, away_score, (pens or (None, None))[1]),
    ):
        sides.append(
            {
                "homeAway": side,
                "team": {"displayName": name},
                "score": {"displayValue": score, "shootoutScore": pen},
            }
        )
    status = {
        "type": {"state": state, "completed": completed, "description": desc, "shortDetail": desc}
    }
    event = {
        "id": f"{home}-{away}-{days}",
        "date": when,
        "competitions": [{"competitors": sides, "status": status}],
    }
    if league:
        event["league"] = {"name": league}
    return event


def test_event_lines_cover_results_live_fixtures_penalties_and_postponements():
    assert event_line(
        _event("A", "B", state="post", days=-1, home_score="2", away_score="1")
    ).endswith(": A 2-1 B")
    assert (
        event_line(_event("A", "B", state="in", days=0, home_score="1", desc="63'"))
        == "LIVE (63'): A 1-0 B"
    )
    assert ": A vs B" in event_line(_event("A", "B", state="pre", days=2))
    pens = event_line(
        _event("A", "B", state="post", days=-1, home_score="1", away_score="1", pens=(3, 5))
    )
    assert pens.endswith("A 1-1 B — B won 5-3 on penalties")
    postponed = event_line(
        _event("A", "B", state="post", days=-1, completed=False, desc="Postponed")
    )
    assert postponed.endswith("A vs B — Postponed") and "0-0" not in postponed


def test_team_view_lists_recent_results_then_next_matches():
    played = {
        "events": [
            _event(
                "Arsenal", "Everton", state="post", days=-6, home_score="2", league="Premier League"
            ),
            _event(
                "Fleetwood",
                "Arsenal",
                state="post",
                days=-3,
                home_score="1",
                away_score="1",
                pens=(5, 3),
                league="League Cup",
            ),
        ]
    }
    upcoming = {
        "events": [
            _event("Arsenal", "Hull", state="pre", days=8, league="Premier League"),
            _event("Liverpool", "Arsenal", state="pre", days=2, league="Premier League"),
        ]
    }
    summary = {"team": {"standingSummary": "2nd in Premier League"}}
    with (
        patch.object(football, "_load_teams", return_value=TEAMS),
        patch.object(football, "_get_many", return_value=[played, upcoming, summary]),
    ):
        out = football.get_football(team="Arsenal")
    lines = out.splitlines()
    assert lines[0].startswith("Arsenal — 2nd in Premier League")
    assert lines[1] == "Last results:"
    assert "League Cup: Fleetwood 1-1 Arsenal — Fleetwood won 5-3 on penalties" in lines[2]
    assert "Premier League: Arsenal 2-0 Everton" in lines[3]
    assert lines[4] == "Next matches:"
    assert "Liverpool vs Arsenal" in lines[5] and "Arsenal vs Hull" in lines[6]


def test_table_marks_the_named_club_and_orders_by_rank():
    def row(id_, name, rank, pts):
        stats = [
            {"name": "rank", "displayValue": str(rank)},
            {"name": "points", "displayValue": str(pts)},
        ]
        return {"team": {"id": id_, "displayName": name}, "stats": stats}

    page = {
        "children": [
            {
                "standings": {
                    "entries": [
                        row("367", "Tottenham Hotspur", 2, 9),
                        row("382", "Manchester City", 1, 21),
                    ]
                }
            }
        ]
    }
    with (
        patch.object(football, "_load_teams", return_value=TEAMS),
        patch.object(football, "_get_json", return_value=page),
    ):
        out = football.get_football(team="Spurs", view="table")
    lines = out.splitlines()
    assert lines[0].startswith("Premier League table")
    assert lines[1].startswith("1. Manchester City — 21 pts")
    assert lines[2].startswith("2. Tottenham Hotspur — 9 pts") and lines[2].endswith("<-")


def test_a_table_needs_a_league_or_a_club():
    assert football.get_football(view="table") == "Which league's table? Ask the user."


def test_league_view_and_overview_ask_espn_by_month():
    page = {
        "events": [_event("Kerala Blasters", "Mumbai City", state="post", days=-2, home_score="0")]
    }
    with patch.object(football, "_get_many", side_effect=lambda urls: [page] * len(urls)) as many:
        out = football.get_football(league="indian super league")
    assert all("/ind.1/scoreboard?dates=" in url for url in many.call_args.args[0])
    assert out.startswith("Indian Super League (as of")
    assert "Kerala Blasters 0-0 Mumbai City" in out


def test_mls_is_spelled_as_initials():
    with patch.object(football, "_get_many", side_effect=lambda urls: [{"events": []}] * len(urls)):
        assert football.get_football(league="mls").startswith("No MLS matches")


def test_view_and_league_synonyms_normalize_through_dispatch():
    bind = get_tool("get_football").bind
    assert bind({"view": "standings"})["view"] == "table"
    assert bind({"view": "fixtures"})["view"] == "matches"
    assert bind({"competition": "EPL"})["league"] == "premier league"
    assert bind({"league": "ucl"})["league"] == "champions league"


def test_a_league_or_view_the_user_did_not_ask_for_is_dropped():
    spec = get_tool("get_football")
    check = check_call(
        spec,
        {"team": "Arsenal", "league": "premier league", "view": "table"},
        "how did arsenal do?",
    )
    assert check.ok and check.arguments == {"team": "Arsenal"}
    kept = check_call(spec, {"league": "EPL", "view": "standings"}, "show me the epl standings")
    assert kept.arguments == {"league": "EPL", "view": "standings"}


def test_an_invented_club_is_dropped():
    check = check_call(get_tool("get_football"), {"team": "Arsenal"}, "any football scores today?")
    assert check.ok and check.dropped == ["team"]


def test_football_and_news_questions_reach_the_tools():
    for prompt in (
        "how did arsenal do last night?",
        "when do spurs play next",
        "premier league table",
        "what's the latest in football",
        "any news today?",
        "what's happening in the world",
    ):
        assert should_offer_tools(prompt), prompt


def test_registered_and_observe_only():
    for name in ("get_news", "get_football"):
        spec = get_tool(name)
        assert spec is not None and spec.untrusted_output
    with patch.object(football, "_load_teams", return_value=[]):
        assert dispatch_tool_call("get_football", {"team": "Arsenal"}).startswith("Couldn't reach")
