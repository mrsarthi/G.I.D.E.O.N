"""
G.I.D.E.O.N Security Primitives

The API token, and the irreversible-command denylist the terminal tool checks.
Per-tool policy (which channels may use a tool, which need confirmation) is
declared on the tool itself — see tools/base.py.
"""

from __future__ import annotations

import os
import re
import secrets

from gideon.core.config import BASE_DIR

# ---------------------------------------------------------------------------
#  API authentication
# ---------------------------------------------------------------------------
# Shared secret for state-changing endpoints. Generated on first use and
# written to .env so the frontend and server agree across restarts.
API_TOKEN_HEADER = "X-Gideon-Token"


def _load_or_create_api_token() -> str:
    token = os.environ.get("GIDEON_API_TOKEN", "")
    if token:
        return token

    token = secrets.token_urlsafe(32)
    env_file = BASE_DIR / ".env"
    try:
        with env_file.open("a", encoding="utf-8") as handle:
            handle.write(f"\nGIDEON_API_TOKEN={token}\n")
    except OSError:
        # Non-fatal: the token still works for this process.
        pass
    os.environ["GIDEON_API_TOKEN"] = token
    return token


API_TOKEN = _load_or_create_api_token()


def token_matches(supplied: str | None) -> bool:
    """Constant-time comparison against the API token."""
    return secrets.compare_digest(supplied or "", API_TOKEN)


# ---------------------------------------------------------------------------
#  Terminal execution denylist
# ---------------------------------------------------------------------------
# Commands whose damage cannot be undone. Matched case-insensitively against
# every segment of a chained command line, so `dir && format c:` is caught.
# This is a backstop against a wrong tool call or injected instruction, not a
# sandbox — it blocks the catastrophic tail, nothing more.
DESTRUCTIVE_COMMAND_PATTERNS = [
    r"\bformat\s+[a-z]:",                     # format c:
    r"\bdiskpart\b",
    r"\bmkfs(\.\w+)?\b",
    r"\brm\s+(-\w*\s+)*-\w*[rf]",             # rm -rf / rm -fr
    r"\bdel\s+.*[/-][sq]\b",                  # del /s /q
    r"\brmdir\s+.*[/-]s\b",
    r"\bremove-item\b.*-recurse.*-force",
    r"\bformat-volume\b",
    r"\bclear-disk\b",
    r"\breg\s+delete\b",
    r"\bremove-itemproperty\b",
    r"\bcipher\s+/w",                          # secure wipe
    r"\bshutdown\b",
    r"\brestart-computer\b",
    r"\bstop-computer\b",
    r"\bbcdedit\b",
    r"\bvssadmin\s+delete\b",                  # shadow copy deletion
    r"\btakeown\b.*[/-]r",
    r"\bicacls\b.*[/-]reset",
    r"\bnet\s+user\b.*[/-]delete",
    r":\(\)\s*\{.*\};\s*:",                    # fork bomb
    r"\b(curl|wget|iwr|invoke-webrequest)\b.*\|\s*(sh|bash|iex|powershell)",
    r"\biex\b.*\bdownloadstring\b",            # download-and-execute
]

# Shell operators that chain a second command onto the first.
COMMAND_CHAIN_SPLIT = r"&&|\|\||[;&|\n]"

DENY_RE = [re.compile(p, re.IGNORECASE) for p in DESTRUCTIVE_COMMAND_PATTERNS]
CHAIN_RE = re.compile(COMMAND_CHAIN_SPLIT)


def find_destructive_pattern(command: str) -> str | None:
    """Return the first denylist pattern the command matches, if any.

    Each chained segment is checked separately: an allowed prefix must not be
    able to smuggle a destructive suffix past the filter.
    """
    segments = [s.strip() for s in CHAIN_RE.split(command) if s.strip()]
    # Also test the whole line, so patterns spanning an operator still match.
    for candidate in [command, *segments]:
        for pattern in DENY_RE:
            if pattern.search(candidate):
                return pattern.pattern
    return None
