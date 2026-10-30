"""
G.I.D.E.O.N yfinance Market & Financial Data Tool

Retrieves live stock quotes, market indices, crypto prices, and company financial stats,
or an overview of the markets the user cares about (GIDEON_REGION).
Powered by yfinance.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from gideon.core.config import REGION
from gideon.core.logs import get_logger
from gideon.tools.base import Param, tool

log = get_logger("gideon.tools.finance")

# The symbol that asks for the overview instead of one ticker.
OVERVIEW = "MARKET"
_OVERVIEW_NAMES = {"market", "markets", "overview", "indices"}
# Words that may surround one of those: "stock markets", "global market overview".
_OVERVIEW_WORDS = _OVERVIEW_NAMES | {"stock", "stocks", "share", "shares", "all", "global", "world", "summary"}
# Never-assume: the overview needs the user to have asked about the market.
# "what's the stock price?" was answered with symbol="market" (2026-10-30);
# it names no company, so it must be asked.
_OVERVIEW_EVIDENCE: dict[str, tuple[str, ...]] = {
    name: ("market", "markets", "indices", "index", "overview", "sensex", "nifty")
    for name in (
        *_OVERVIEW_NAMES, "the market", "the markets", "stock market", "stock markets",
        "market overview", "global markets", "world markets",
    )
}

# What people call markets → Yahoo Finance tickers. Small models pass the
# name ("nifty", "gold") as often as the ticker, and a bare "ETH" or "BTC" is
# a different listing on Yahoo.
_SYMBOL_NAMES = {
    "nifty": "^NSEI", "nifty 50": "^NSEI", "nifty50": "^NSEI", "sensex": "^BSESN",
    "bank nifty": "^NSEBANK", "nifty bank": "^NSEBANK", "banknifty": "^NSEBANK",
    "s&p": "^GSPC", "s&p 500": "^GSPC", "sp500": "^GSPC", "s&p500": "^GSPC",
    "nasdaq": "^IXIC", "dow": "^DJI", "dow jones": "^DJI", "ftse": "^FTSE", "ftse 100": "^FTSE",
    "nikkei": "^N225", "nikkei 225": "^N225", "dax": "^GDAXI",
    "gold": "GC=F", "silver": "SI=F", "oil": "BZ=F", "crude": "BZ=F", "crude oil": "BZ=F",
    "brent": "BZ=F", "wti": "CL=F",
    "bitcoin": "BTC-USD", "btc": "BTC-USD", "ethereum": "ETH-USD", "eth": "ETH-USD",
}
_CURRENCY_NAMES = {
    "dollar": "USD", "dollars": "USD", "usd": "USD", "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "pound": "GBP", "pounds": "GBP", "gbp": "GBP", "yen": "JPY", "jpy": "JPY",
    "rupee": "INR", "rupees": "INR", "inr": "INR",
}
_SYMBOL_FILLER = {"price", "prices", "rate", "rates", "today", "index", "the", "of", "to", "vs", "and", "exchange"}

# Region → (label, home indices, home currency). Unknown regions use US.
_HOME = {
    "IN": ("India", [("Nifty 50", "^NSEI"), ("Sensex", "^BSESN"), ("Nifty Bank", "^NSEBANK")], "INR"),
    "US": ("US", [("S&P 500", "^GSPC"), ("Nasdaq", "^IXIC"), ("Dow Jones", "^DJI")], "USD"),
    "GB": ("UK", [("FTSE 100", "^FTSE"), ("FTSE 250", "^FTMC")], "GBP"),
    "AU": ("Australia", [("ASX 200", "^AXJO")], "AUD"),
    "CA": ("Canada", [("S&P/TSX", "^GSPTSE")], "CAD"),
    "JP": ("Japan", [("Nikkei 225", "^N225")], "JPY"),
}
_WORLD = [("S&P 500", "^GSPC"), ("Nasdaq", "^IXIC"), ("Dow Jones", "^DJI"), ("FTSE 100", "^FTSE"), ("Nikkei 225", "^N225")]
_COMMODITIES = [("Gold", "GC=F", "/oz"), ("Silver", "SI=F", "/oz"), ("Brent crude", "BZ=F", "/bbl")]
_CRYPTO = [("Bitcoin", "BTC-USD"), ("Ethereum", "ETH-USD")]
_TROY_OUNCE_GRAMS = 31.1035


def _currency_pair(base: str, quote: str) -> str:
    if base == quote:
        # A currency in itself: the dollar index for the dollar, else its
        # price in dollars ("the rupee" is USD/INR).
        return "DX-Y.NYB" if base == "USD" else f"USD{base}=X"
    return f"{base}{quote}=X"


def _symbol(value: Any) -> str:
    """What the model wrote → a Yahoo ticker, or OVERVIEW."""
    raw = " ".join(str(value or "").split())
    text = raw.lower()
    if text in _SYMBOL_NAMES:
        return _SYMBOL_NAMES[text]
    words = [w for w in re.findall(r"[a-z0-9&]+", text) if w not in _SYMBOL_FILLER]
    if words and set(words) <= _OVERVIEW_WORDS and set(words) & _OVERVIEW_NAMES:
        return OVERVIEW
    trimmed = " ".join(words)
    if trimmed in _SYMBOL_NAMES:
        return _SYMBOL_NAMES[trimmed]
    currencies = [_CURRENCY_NAMES[w] for w in words if w in _CURRENCY_NAMES]
    if currencies and len(currencies) == len(words) <= 2:
        if len(currencies) == 2:
            return _currency_pair(currencies[0], currencies[1])
        # One currency is priced in the user's own: "the euro" to an Indian
        # user is EUR/INR, and "the rupee" is USD/INR.
        home = _HOME.get(REGION, _HOME["US"])[2]
        return _currency_pair(currencies[0], home)
    return raw.upper()


def _quotes(tickers: list[str]) -> dict[str, tuple[float, float, datetime]]:
    """ticker → (last price, previous close, date of the last price), in one
    batched request. Tickers Yahoo returns nothing for are left out."""
    import yfinance as yf

    frame = yf.download(
        tickers, period="5d", interval="1d", group_by="ticker", progress=False, threads=True, auto_adjust=False
    )
    found: dict[str, tuple[float, float, datetime]] = {}
    for ticker in tickers:
        try:
            closes = frame[ticker]["Close"].dropna()
        except KeyError:
            continue
        if len(closes) >= 2:
            found[ticker] = (float(closes.iloc[-1]), float(closes.iloc[-2]), closes.index[-1].to_pydatetime())
    return found


def _number(value: float) -> str:
    return f"{value:,.4f}" if value < 10 else f"{value:,.2f}"


def _overview() -> str:
    label, home_indices, home_currency = _HOME.get(REGION, _HOME["US"])
    home_tickers = {ticker for _name, ticker in home_indices}
    world = [(name, ticker) for name, ticker in _WORLD if ticker not in home_tickers]
    if home_currency == "USD":
        currencies = [("EUR/USD", "EURUSD=X"), ("GBP/USD", "GBPUSD=X"), ("USD/JPY", "JPY=X")]
    else:
        currencies = [(f"USD/{home_currency}", f"USD{home_currency}=X"), (f"EUR/{home_currency}", f"EUR{home_currency}=X")]
    sections = [
        (label, [(n, t, "", "") for n, t in home_indices]),
        ("World", [(n, t, "", "") for n, t in world]),
        ("Currencies", [(n, t, "", "") for n, t in currencies]),
        ("Commodities", [(n, t, "$", unit) for n, t, unit in _COMMODITIES]),
        ("Crypto", [(n, t, "$", "") for n, t in _CRYPTO]),
    ]
    tickers = [ticker for _label, rows in sections for _n, ticker, _p, _u in rows]
    try:
        quotes = _quotes(tickers)
    except Exception as err:
        log.error("Market overview failed: %s", err)
        return f"Error retrieving the market overview: {err}"
    if not quotes:
        return "Couldn't reach the market data service right now. Try again shortly."

    today = datetime.now().date()
    stamp = datetime.now().strftime("%a %d %b, %H:%M")
    lines = [f"MARKET OVERVIEW (Yahoo Finance, fetched {stamp}; change is against the previous close):"]
    for title, rows in sections:
        items = []
        for name, ticker, prefix, unit in rows:
            if ticker not in quotes:
                continue
            last, previous, day = quotes[ticker]
            item = f"{name} {prefix}{_number(last)}{unit} ({(last / previous - 1) * 100:+.2f}%)"
            if day.date() != today:
                item += f" as of {day:%a %d %b}"
            if ticker == "GC=F" and home_currency == "INR" and f"USD{home_currency}=X" in quotes:
                per_10g = last * quotes[f"USD{home_currency}=X"][0] / _TROY_OUNCE_GRAMS * 10
                item += f", about ₹{per_10g:,.0f} per 10 g at today's rate before Indian import duty and GST"
            items.append(item)
        if items:
            lines.append(f"{title}: " + "; ".join(items))
    return "\n".join(lines)


@tool(
    "get_financial_data",
    "Prices of stocks, market indices, currencies, gold, oil and crypto, company news or financials, "
    "or a market overview. Call for market quotes, stock prices, exchange rates or financial news.",
    Param(
        "symbol",
        "Ticker or name, e.g. 'AAPL', 'BTC-USD', 'nifty', 'gold', 'usd inr'; 'market' for an overview "
        "of indices, currencies, commodities and crypto.",
        required=True,
        aliases=("ticker", "stock"),
        normalize=_symbol,
        evidence=_OVERVIEW_EVIDENCE,
    ),
    Param(
        "data_type",
        "Type of market data to fetch.",
        enum=("summary", "news", "financials"),
        aliases=("type",),
        default="summary",
    ),
    # The news data_type returns third-party headlines.
    untrusted_output=True,
    triggers={
        "stock", "stocks", "share", "shares", "ticker", "quote", "market",
        "markets", "nasdaq", "dow", "crypto", "bitcoin", "btc", "eth",
        "ethereum", "price", "trading", "portfolio", "earnings", "dividend",
        "finance", "financial", "sensex", "nifty", "index", "indices", "gold", "silver",
        "oil", "crude", "commodities", "rupee", "dollar", "euro", "forex", "currency",
    },
)
def get_financial_data(symbol: str, data_type: str = "summary") -> str:
    """Retrieve financial data, stock quotes, market metrics, or news for a ticker symbol.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL', 'TSLA', 'BTC-USD', 'NVDA', '^GSPC'), or MARKET.
        data_type: 'summary' (price & metrics), 'news' (recent headlines), or 'financials'.
    """
    clean_symbol = (symbol or "").strip().upper()
    if not clean_symbol:
        return "Error: A ticker symbol is required (e.g., AAPL, TSLA, BTC-USD)."

    try:
        import yfinance as yf
    except ImportError:
        return "Error: 'yfinance' package is not installed."

    if clean_symbol == OVERVIEW:
        return _overview()

    try:
        ticker = yf.Ticker(clean_symbol)

        if data_type.lower() == "news":
            news_items = getattr(ticker, "news", []) or []
            if not news_items:
                return f"No recent news found for symbol '{clean_symbol}'."
            lines = [f"Recent News for {clean_symbol}:"]
            for item in news_items[:5]:
                title = item.get("title") or item.get("headline", "No title")
                publisher = item.get("publisher") or item.get("source", "Unknown")
                link = item.get("link") or item.get("url", "")
                lines.append(f"- [{publisher}] {title} ({link})")
            return "\n".join(lines)

        elif data_type.lower() == "financials":
            info = getattr(ticker, "info", {}) or {}
            rev = info.get("totalRevenue")
            ebitda = info.get("ebitda")
            net_income = info.get("netIncomeToCommon")
            profit_margin = info.get("profitMargins")

            lines = [
                f"Financial Overview for {clean_symbol}:",
                f"- Total Revenue: ${rev:,}" if isinstance(rev, (int, float)) else f"- Total Revenue: {rev}",
                f"- EBITDA: ${ebitda:,}" if isinstance(ebitda, (int, float)) else f"- EBITDA: {ebitda}",
                f"- Net Income: ${net_income:,}" if isinstance(net_income, (int, float)) else f"- Net Income: {net_income}",
                f"- Profit Margin: {profit_margin * 100:.2f}%" if isinstance(profit_margin, (int, float)) else f"- Profit Margin: {profit_margin}",
            ]
            return "\n".join(lines)

        else:
            # Default: summary / price quote
            info = getattr(ticker, "info", {}) or {}
            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )
            name = info.get("shortName") or info.get("longName") or clean_symbol
            currency = info.get("currency", "USD")
            change_pct = info.get("regularMarketChangePercent")
            mcap = info.get("marketCap")
            pe = info.get("trailingPE")
            fifty_two_high = info.get("fiftyTwoWeekHigh")
            fifty_two_low = info.get("fiftyTwoWeekLow")

            lines = [
                f"Financial Quote for {name} ({clean_symbol}):",
                f"- Price: {price} {currency}" if price is not None else "- Price: N/A",
            ]
            if change_pct is not None:
                lines.append(f"- 24h/Day Change: {change_pct:.2f}%")
            if mcap is not None and isinstance(mcap, (int, float)):
                lines.append(f"- Market Cap: {mcap:,} {currency}")
            if pe is not None:
                lines.append(f"- Trailing P/E: {pe}")
            if fifty_two_high is not None and fifty_two_low is not None:
                lines.append(f"- 52-Week Range: {fifty_two_low} - {fifty_two_high} {currency}")

            if len(lines) == 2 and price is None:
                # Fallback to fast_info or history if info dict is sparse
                fast_info = getattr(ticker, "fast_info", {})
                last_price = getattr(fast_info, "last_price", None)
                if last_price is not None:
                    return f"Financial Quote for {clean_symbol}:\n- Current Price: {last_price:.2f}"

            return "\n".join(lines)

    except Exception as err:
        log.error("yfinance query failed for %s: %s", clean_symbol, err)
        return f"Error retrieving market data for '{clean_symbol}': {err}"
