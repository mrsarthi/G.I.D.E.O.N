"""
G.I.D.E.O.N Wolfram Alpha Short Answers Tool

Queries Wolfram Alpha's Short Answers REST API for factual, computational, unit conversion,
or math queries.
Requires GIDEON_WOLFRAM_APP_ID to be set in environment.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request

from gideon.core.config import WOLFRAM_ALPHA_APP_ID
from gideon.core.logs import get_logger
from gideon.tools.base import Param, tool

log = get_logger("gideon.tools.knowledge")

# Total attempts, including the first. Bounded so a genuinely unanswerable
# query costs at most one extra round trip.
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 0.4


@tool(
    "query_wolfram_alpha",
    "Query Wolfram Alpha for math calculations, scientific facts, conversions, or factual questions. Call when user asks for math/computational answers or Wolfram queries.",
    Param(
        "query",
        "The math equation, scientific query, or factual question.",
        required=True,
        aliases=("q", "input"),
    ),
    untrusted_output=True,
    triggers={
        "calculate", "compute", "convert", "solve", "equation", "distance",
        "population", "wolfram",
    },
)
def query_wolfram_alpha(query: str) -> str:
    """Query Wolfram Alpha Short Answers API for computations, conversions, science, or factual data."""
    q = (query or "").strip()
    if not q:
        return "Error: A query string is required for Wolfram Alpha."

    if not WOLFRAM_ALPHA_APP_ID:
        return (
            "Wolfram Alpha integration is not configured. "
            "Set the 'GIDEON_WOLFRAM_APP_ID' environment variable with your Wolfram Alpha AppID."
        )

    encoded = urllib.parse.quote(q)
    url = f"http://api.wolframalpha.com/v1/result?appid={WOLFRAM_ALPHA_APP_ID}&i={encoded}"

    # HTTP 501 ("No short answer available") is not always a real answer to the
    # question — for queries needing live computation, such as planetary
    # distances, Wolfram returns it intermittently. Measured: "distance to
    # mars" and "distance to the moon" each failed ~25% of the time, while
    # static facts and arithmetic succeeded 8/8. One retry cuts that to ~6%.
    last_code = None
    for attempt in range(MAX_ATTEMPTS):
        req = urllib.request.Request(url, headers={"User-Agent": "GIDEON-Assistant/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                answer = resp.read().decode("utf-8").strip()
                if attempt:
                    log.info("Wolfram answered %r on attempt %d", q, attempt + 1)
                return f"Wolfram Alpha Answer: {answer}"
        except urllib.error.HTTPError as http_err:
            last_code = http_err.code
            if http_err.code == 403:
                return "Wolfram Alpha returned HTTP 403: Invalid AppID."
            if http_err.code == 501 and attempt < MAX_ATTEMPTS - 1:
                log.debug("Wolfram 501 for %r; retrying (%d)", q, attempt + 1)
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            break
        except Exception as err:
            log.error("Wolfram Alpha query failed for %r: %s", q, err)
            return f"Error contacting Wolfram Alpha: {err}"

    if last_code == 501:
        return f"Wolfram Alpha could not interpret or find a short answer for: '{q}'."
    return f"Wolfram Alpha query failed (HTTP {last_code})."
