"""get_financial_data's market overview and the names people use for markets
("nifty", "gold", "dollar to rupee"). Yahoo is never called."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from gideon.brain.intent import should_offer_tools
from gideon.tools import finance
from gideon.tools.registry import dispatch_tool_call, get_tool
from gideon.tools.validation import check_call


def _symbol(value: str) -> str:
    return get_tool("get_financial_data").bind({"symbol": value})["symbol"]


@pytest.mark.parametrize(
    ("said", "ticker"),
    [
        ("market", finance.OVERVIEW),
        ("stock markets", finance.OVERVIEW),
        ("global market overview", finance.OVERVIEW),
        ("Nifty", "^NSEI"),
        ("SENSEX", "^BSESN"),
        ("bank nifty", "^NSEBANK"),
        ("S&P 500", "^GSPC"),
        ("gold price", "GC=F"),
        ("crude oil", "BZ=F"),
        ("btc", "BTC-USD"),
        ("ETH", "ETH-USD"),
        ("usd to inr", "USDINR=X"),
        ("dollar to rupee", "USDINR=X"),
        ("AAPL", "AAPL"),
        ("BTC-USD", "BTC-USD"),
        ("^GSPC", "^GSPC"),
    ],
)
def test_market_names_become_tickers(said, ticker):
    assert _symbol(said) == ticker


def test_one_currency_is_priced_in_the_users_own():
    with patch.object(finance, "REGION", "IN"):
        assert [_symbol(c) for c in ("dollar", "rupee", "euro")] == ["USDINR=X", "USDINR=X", "EURINR=X"]
    with patch.object(finance, "REGION", "US"):
        assert [_symbol(c) for c in ("dollar", "euro")] == ["DX-Y.NYB", "EURUSD=X"]


QUOTES = {
    "^NSEI": (23053.10, 22936.75),
    "^GSPC": (7624.24, 7720.39),
    "USDINR=X": (94.94, 95.21),
    "GC=F": (3926.90, 4031.40),
    "BTC-USD": (76886.12, 77775.11),
}


def _fake_quotes(stale=()):
    now = datetime.now()
    return {t: (last, prev, now - timedelta(days=3) if t in stale else now) for t, (last, prev) in QUOTES.items()}


def test_overview_leads_with_the_users_market_and_skips_what_yahoo_lacks():
    with patch.object(finance, "REGION", "IN"), patch.object(finance, "_quotes", return_value=_fake_quotes()) as q:
        out = dispatch_tool_call("get_financial_data", {"symbol": "markets"})
    asked = q.call_args.args[0]
    assert asked[:3] == ["^NSEI", "^BSESN", "^NSEBANK"] and "USDINR=X" in asked and "EURINR=X" in asked
    lines = out.splitlines()
    assert lines[0].startswith("MARKET OVERVIEW (Yahoo Finance")
    assert lines[1] == "India: Nifty 50 23,053.10 (+0.51%)"
    assert lines[2].startswith("World: S&P 500 7,624.24 (-1.25%)")
    assert "Sensex" not in out  # no quote came back for it
    assert lines[3] == "Currencies: USD/INR 94.94 (-0.28%)"
    assert "Gold $3,926.90/oz (-2.59%), about ₹119,864 per 10 g" in lines[4]
    assert lines[5] == "Crypto: Bitcoin $76,886.12 (-1.14%)"


def test_a_closed_market_says_when_its_price_is_from():
    with patch.object(finance, "REGION", "IN"), patch.object(finance, "_quotes", return_value=_fake_quotes(stale={"^GSPC"})):
        out = finance._overview()
    day = (datetime.now() - timedelta(days=3)).strftime("%a %d %b")
    assert f"S&P 500 7,624.24 (-1.25%) as of {day}" in out
    assert "Nifty 50 23,053.10 (+0.51%);" in out or "Nifty 50 23,053.10 (+0.51%)\n" in out


def test_us_overview_has_no_rupee_gold_line_and_dollar_crosses():
    with patch.object(finance, "REGION", "US"), patch.object(finance, "_quotes", return_value=_fake_quotes()) as q:
        out = finance._overview()
    asked = q.call_args.args[0]
    assert asked[:3] == ["^GSPC", "^IXIC", "^DJI"] and asked.count("^GSPC") == 1
    assert "EURUSD=X" in asked and "per 10 g" not in out
    assert out.splitlines()[1].startswith("US: S&P 500")


def test_overview_failures_are_said_plainly():
    with patch.object(finance, "_quotes", return_value={}):
        assert finance._overview().startswith("Couldn't reach the market data service")
    with patch.object(finance, "_quotes", side_effect=RuntimeError("rate limited")):
        assert finance._overview() == "Error retrieving the market overview: rate limited"


def test_a_stock_price_with_no_company_still_asks():
    """ask-stock-which: "what's the stock price?" names nothing to look up."""
    spec = get_tool("get_financial_data")
    check = check_call(spec, {}, "what's the stock price?")
    assert [p.name for p in check.missing] == ["symbol"]
    # Eval 2026-10-30: the model filled the gap with the overview.
    check = check_call(spec, {"symbol": "market"}, "what's the stock price?")
    assert [p.name for p in check.ungrounded] == ["symbol"]


@pytest.mark.parametrize(
    ("said", "symbol"),
    [
        ("how are the markets doing today?", "market"),
        ("give me a stock market overview", "stock market"),
        ("where's the sensex at?", "market"),
        ("what's apple's stock price?", "AAPL"),
        ("how is bitcoin doing", "BTC-USD"),
    ],
)
def test_the_overview_needs_a_market_question_but_tickers_do_not(said, symbol):
    check = check_call(get_tool("get_financial_data"), {"symbol": symbol}, said)
    assert check.ok and check.arguments == {"symbol": symbol}


def test_market_questions_reach_the_tools():
    for prompt in ("how's the market doing today?", "where's the sensex at", "gold rate today", "dollar to rupee rate"):
        assert should_offer_tools(prompt), prompt
