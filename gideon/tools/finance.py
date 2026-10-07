"""
G.I.D.E.O.N yfinance Market & Financial Data Tool

Retrieves live stock quotes, market indices, crypto prices, and company financial stats.
Powered by yfinance.
"""

from __future__ import annotations

from gideon.core.logs import get_logger
from gideon.tools.base import Param, tool

log = get_logger("gideon.tools.finance")


@tool(
    "get_financial_data",
    "Fetch stock quotes, market indices, crypto prices, financial metrics, or company news via yfinance. Call when user asks about market quotes, stock prices, financial news, or crypto.",
    Param(
        "symbol",
        "Ticker symbol (e.g. 'AAPL', 'TSLA', 'BTC-USD', 'NVDA', '^GSPC').",
        required=True,
        aliases=("ticker", "stock"),
    ),
    Param(
        "data_type",
        "Type of market data to fetch.",
        enum=("summary", "news", "financials"),
        aliases=("type",),
        default="summary",
    ),
    triggers={
        "stock", "stocks", "share", "shares", "ticker", "quote", "market",
        "markets", "nasdaq", "dow", "crypto", "bitcoin", "btc", "eth",
        "ethereum", "price", "trading", "portfolio", "earnings", "dividend",
        "finance", "financial",
    },
)
def get_financial_data(symbol: str, data_type: str = "summary") -> str:
    """Retrieve financial data, stock quotes, market metrics, or news for a ticker symbol.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL', 'TSLA', 'BTC-USD', 'NVDA', '^GSPC').
        data_type: 'summary' (price & metrics), 'news' (recent headlines), or 'financials'.
    """
    clean_symbol = (symbol or "").strip().upper()
    if not clean_symbol:
        return "Error: A ticker symbol is required (e.g., AAPL, TSLA, BTC-USD)."

    try:
        import yfinance as yf
    except ImportError:
        return "Error: 'yfinance' package is not installed."

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
                lines.append(f"- Market Cap: ${mcap:,}")
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
