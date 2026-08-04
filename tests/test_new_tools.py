"""
Unit tests for new G.I.D.E.O.N tools:
- Weather (Open-Meteo)
- Financial Markets (yfinance)
- HackerNews (Firebase API)
- Geolocation (IP-API)
- Wolfram Alpha (Short Answers API)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.finance import get_financial_data
from tools.geolocation import get_ip_info
from tools.knowledge import query_wolfram_alpha
from tools.news import get_hackernews
from tools.registry import dispatch_tool_call
from tools.weather import get_weather


# ---------------------------------------------------------------------------
#  Geolocation (IP-API) Tests
# ---------------------------------------------------------------------------
def test_get_ip_info_success():
    fake_response = {
        "status": "success",
        "query": "8.8.8.8",
        "city": "Mountain View",
        "regionName": "California",
        "country": "United States",
        "lat": 37.4056,
        "lon": -122.0775,
        "isp": "Google LLC",
        "timezone": "America/Los_Angeles",
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(fake_response).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = get_ip_info("8.8.8.8")
        assert "Mountain View" in res
        assert "37.4056" in res
        assert "Google LLC" in res


def test_get_ip_info_failure():
    fake_response = {"status": "fail", "message": "invalid query"}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(fake_response).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = get_ip_info("invalid_ip")
        assert "IP lookup failed: invalid query" in res


# ---------------------------------------------------------------------------
#  Weather (Open-Meteo) Tests
# ---------------------------------------------------------------------------
def test_get_weather_with_location():
    geo_response = {
        "results": [
            {
                "latitude": 35.6895,
                "longitude": 139.6917,
                "name": "Tokyo",
                "country": "Japan",
            }
        ]
    }
    weather_response = {
        "current": {
            "temperature_2m": 22.5,
            "relative_humidity_2m": 60,
            "wind_speed_10m": 12.3,
            "weather_code": 0,
        },
        "daily": {
            "temperature_2m_max": [26.0],
            "temperature_2m_min": [18.0],
        },
    }

    def mock_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        m = MagicMock()
        if "geocoding-api" in url:
            m.read.return_value = json.dumps(geo_response).encode("utf-8")
        else:
            m.read.return_value = json.dumps(weather_response).encode("utf-8")
        m.__enter__.return_value = m
        return m

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        res = get_weather("Tokyo")
        assert "Weather for Tokyo, Japan" in res
        assert "22.5°C" in res
        assert "Clear sky" in res
        assert "High 26.0°C" in res


# ---------------------------------------------------------------------------
#  Finance (yfinance) Tests
# ---------------------------------------------------------------------------
def test_get_financial_data_summary():
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "shortName": "Apple Inc.",
        "currentPrice": 225.50,
        "currency": "USD",
        "regularMarketChangePercent": 1.45,
        "marketCap": 3400000000000,
        "trailingPE": 33.2,
        "fiftyTwoWeekHigh": 237.23,
        "fiftyTwoWeekLow": 164.08,
    }

    with patch("yfinance.Ticker", return_value=mock_ticker):
        res = get_financial_data("AAPL", data_type="summary")
        assert "Apple Inc. (AAPL)" in res
        assert "225.5 USD" in res
        assert "1.45%" in res


def test_get_financial_data_news():
    mock_ticker = MagicMock()
    mock_ticker.news = [
        {"title": "Apple releases new update", "publisher": "TechCrunch", "link": "https://example.com/1"}
    ]

    with patch("yfinance.Ticker", return_value=mock_ticker):
        res = get_financial_data("AAPL", data_type="news")
        assert "Recent News for AAPL" in res
        assert "TechCrunch" in res
        assert "Apple releases new update" in res


# ---------------------------------------------------------------------------
#  HackerNews Tests
# ---------------------------------------------------------------------------
def test_get_hackernews_success():
    story_ids = [101, 102]
    item_101 = {
        "id": 101,
        "title": "Python 3.13 Released",
        "score": 450,
        "by": "guido",
        "descendants": 85,
        "url": "https://python.org",
    }

    def mock_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        m = MagicMock()
        if "topstories.json" in url:
            m.read.return_value = json.dumps(story_ids).encode("utf-8")
        else:
            m.read.return_value = json.dumps(item_101).encode("utf-8")
        m.__enter__.return_value = m
        return m

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        res = get_hackernews("top", limit=1)
        assert "HackerNews (TOP Stories)" in res
        assert "Python 3.13 Released" in res
        assert "Points: 450" in res


# ---------------------------------------------------------------------------
#  Wolfram Alpha Tests
# ---------------------------------------------------------------------------
def test_query_wolfram_alpha_unconfigured():
    with patch("tools.knowledge.WOLFRAM_ALPHA_APP_ID", ""):
        res = query_wolfram_alpha("distance to mars")
        assert "integration is not configured" in res


def test_query_wolfram_alpha_success():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"225 million km"
    mock_resp.__enter__.return_value = mock_resp

    with patch("tools.knowledge.WOLFRAM_ALPHA_APP_ID", "TEST_APP_ID"), patch(
        "urllib.request.urlopen", return_value=mock_resp
    ):
        res = query_wolfram_alpha("distance to mars")
        assert "Wolfram Alpha Answer: 225 million km" in res


# ---------------------------------------------------------------------------
#  Registry Dispatch Tests
# ---------------------------------------------------------------------------
def test_dispatch_new_tools():
    with patch("tools.registry.get_weather", return_value="Weather OK"):
        assert dispatch_tool_call("get_weather", {"location": "London"}) == "Weather OK"

    with patch("tools.registry.get_financial_data", return_value="Finance OK"):
        assert dispatch_tool_call("get_financial_data", {"symbol": "TSLA"}) == "Finance OK"

    with patch("tools.registry.get_hackernews", return_value="HN OK"):
        assert dispatch_tool_call("get_hackernews", {"category": "top"}) == "HN OK"

    with patch("tools.registry.get_ip_info", return_value="IP OK"):
        assert dispatch_tool_call("get_ip_info", {}) == "IP OK"

    with patch("tools.registry.query_wolfram_alpha", return_value="Wolfram OK"):
        assert dispatch_tool_call("query_wolfram_alpha", {"query": "5+5"}) == "Wolfram OK"


# ---------------------------------------------------------------------------
#  Context budget must track the real schema size
# ---------------------------------------------------------------------------
def test_prompt_budget_cannot_overflow_the_context_window():
    """Adding tools grew the schemas past the hardcoded reserve, letting the
    assembled prompt exceed num_ctx (Ollama then truncates the system prompt)."""
    import config
    import orchestrator

    worst_case = (
        orchestrator.EFFECTIVE_PROMPT_BUDGET
        + orchestrator.TOOL_SCHEMA_TOKENS
        + config.RESPONSE_RESERVE
        + config.MEMORY_RESERVE
    )
    assert worst_case <= config.NUM_CTX


def test_measured_schema_cost_is_used_not_the_stale_constant():
    import orchestrator
    from tools.registry import OLLAMA_TOOLS

    assert orchestrator.TOOL_SCHEMA_TOKENS > 0
    assert len(OLLAMA_TOOLS) >= 16


# ---------------------------------------------------------------------------
#  Wolfram intermittent 501 retry
# ---------------------------------------------------------------------------
def test_wolfram_retries_transient_501(monkeypatch):
    """Live-computation queries ("distance to mars") intermittently return
    HTTP 501 even though Wolfram can answer them. Measured ~25% failure."""
    import urllib.error
    from unittest.mock import MagicMock

    import tools.knowledge as k

    monkeypatch.setattr(k, "WOLFRAM_ALPHA_APP_ID", "TESTID")
    monkeypatch.setattr(k.time, "sleep", lambda _: None)

    calls = {"n": 0}

    def flaky(req, timeout=8):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 501, "No short answer", {}, None)
        resp = MagicMock()
        resp.read.return_value = b"1.984 au"
        resp.__enter__ = lambda s: resp
        resp.__exit__ = lambda *a: False
        return resp

    monkeypatch.setattr(k.urllib.request, "urlopen", flaky)
    result = k.query_wolfram_alpha("distance to mars")

    assert "1.984 au" in result
    assert calls["n"] == 2, "should have retried exactly once"


def test_wolfram_gives_up_after_max_attempts(monkeypatch):
    """A genuinely unanswerable query must not retry forever."""
    import urllib.error

    import tools.knowledge as k

    monkeypatch.setattr(k, "WOLFRAM_ALPHA_APP_ID", "TESTID")
    monkeypatch.setattr(k.time, "sleep", lambda _: None)

    calls = {"n": 0}

    def always_501(req, timeout=8):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 501, "No short answer", {}, None)

    monkeypatch.setattr(k.urllib.request, "urlopen", always_501)
    result = k.query_wolfram_alpha("qwertyuiop asdfgh")

    assert "could not interpret" in result
    assert calls["n"] == k.MAX_ATTEMPTS


def test_wolfram_invalid_appid_does_not_retry(monkeypatch):
    """403 is permanent; retrying wastes the user's monthly quota."""
    import urllib.error

    import tools.knowledge as k

    monkeypatch.setattr(k, "WOLFRAM_ALPHA_APP_ID", "BAD")
    calls = {"n": 0}

    def forbidden(req, timeout=8):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(k.urllib.request, "urlopen", forbidden)
    result = k.query_wolfram_alpha("anything")

    assert "403" in result
    assert calls["n"] == 1
