"""
G.I.D.E.O.N Tool Registry & Schema Definitions

OpenAI-compatible JSON schemas for Ollama native function calling, plus the
name -> handler dispatch table.

Every schema here is prepended to each prompt by Ollama, so it costs context
on every turn — see config.TOOL_SCHEMA_RESERVE. Keep descriptions tight.
"""

from __future__ import annotations

from typing import Any, Callable

from config import CHANNEL_BLOCKED_TOOLS, get_logger
from tools.desktop import get_system_status, launch_app, search_files
from tools.finance import get_financial_data
from tools.geolocation import get_ip_info
from tools.knowledge import query_wolfram_alpha
from tools.media_and_docs import control_media, control_volume, read_document
from tools.news import get_hackernews
from tools.reminders import list_reminders, set_reminder
from tools.samay import execute as samay_execute
from tools.terminal import run_terminal_command
from tools.weather import get_weather
from tools.web_and_clipboard import get_clipboard, search_web, set_clipboard
from tools.yt_music import play_youtube_music, search_youtube_music

log = get_logger("gideon.tools.registry")


# ---------------------------------------------------------------------------
#  Tool schemas exposed to the model
# ---------------------------------------------------------------------------
OLLAMA_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_system_status",
            "description": "Retrieve computer hardware diagnostic telemetry (CPU, GPU, RAM, disk, battery usage). MUST be called whenever the user asks to check system status, hardware, diagnostics, or PC performance.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "ONLY call this when the user explicitly instructs you to schedule or save a new future task reminder or alarm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The task or note to be reminded about.",
                    },
                    "time_str": {
                        "type": "string",
                        "description": "Relative time or date expression (e.g. 'in 5 days', 'tomorrow at 9am', 'in 2 hours').",
                    },
                },
                "required": ["text", "time_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "ONLY call this when the user explicitly asks to check, read, or display their currently active saved reminders.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "launch_app",
            "description": "ONLY call this when the user explicitly gives a direct command to open or run a desktop application (e.g. 'open notepad', 'launch calc'). NEVER call this to check the time, date, or answer general questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Friendly name of the application to start, e.g. 'notepad', 'chrome', 'spotify'.",
                    }
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "ONLY call this when the user explicitly commands you to search their local disk for files or folders by filename. NEVER use this for general technical discussion, conceptual research, or summarizing conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name_or_extension": {
                        "type": "string",
                        "description": "Filename or extension pattern (e.g. 'main.py', 'report.pdf', '.txt').",
                    }
                },
                "required": ["file_name_or_extension"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Perform an invisible background internet search for online news, live facts, documentation, or answers. Call ONLY when the user explicitly asks to search the web, check online information, or asks about events after your training cut-off. NEVER open a browser window; report the findings verbally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search terms to look up.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_clipboard",
            "description": "Read whatever text or code is currently copied to the user's Windows clipboard. Call ONLY when the user asks you to read, analyze, check, or review their clipboard.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_clipboard",
            "description": "Copy text, code, or answers into the user's Windows clipboard so they can paste with Ctrl+V. Call ONLY when the user explicitly instructs you to copy something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The content to copy into the clipboard.",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_volume",
            "description": "Adjust the computer master sound volume (mute, unmute, or set a percentage from 0 to 100). Call ONLY when the user instructs you to change volume, mute audio, or check the speaker level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "mute", "unmute", "status"],
                        "description": "'set' to change level, 'mute'/'unmute' to toggle audio, 'status' to report the current level.",
                    },
                    "level": {
                        "type": "integer",
                        "description": "Target volume percentage 0-100 (required when action is 'set').",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_youtube_music",
            "description": "Search and launch playback of a specific song, artist, album, or playlist by name on YouTube Music (e.g. 'play Loser by Tame Impala' or 'play Starboy'). ALWAYS use this tool whenever the user mentions a song name, track title, or artist to play.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The track title, artist name, or playlist to play (e.g., 'Loser Tame Impala' or 'Starboy The Weeknd').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_media",
            "description": "Control hardware media playback buttons (pause active music, resume, next track, previous track). Call ONLY for general button commands like 'pause', 'resume', 'next track'. NEVER use this when the user names a specific song or artist to play.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["play_pause", "next", "previous", "stop"],
                        "description": "The transport button action to perform.",
                    }
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Open and read the text of a local file or document (.pdf, .py, .txt, .md, .json, etc.). Call ONLY when the user explicitly instructs you to read, open, summarize, or analyze a specific local file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path or name of the document, e.g. 'C:\\Users\\me\\Desktop\\report.pdf' or 'main.py'.",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_youtube_music",
            "description": "Search YouTube Music for songs, albums, artists, or playlists. Call ONLY when the user explicitly asks to search for music or find tracks on YouTube Music.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The song title, artist, album name, or playlist search terms.",
                    },
                    "search_filter": {
                        "type": "string",
                        "enum": ["songs", "videos", "albums", "artists", "playlists"],
                        "description": "Optional search filter type.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetch live weather conditions and daily temperature forecast. Call when user asks about weather, temperature, rain, or forecast for a city or local area.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or location name (e.g. 'Tokyo', 'London', 'New York'). Omit for current location.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_data",
            "description": "Fetch stock quotes, market indices, crypto prices, financial metrics, or company news via yfinance. Call when user asks about market quotes, stock prices, financial news, or crypto.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol (e.g. 'AAPL', 'TSLA', 'BTC-USD', 'NVDA', '^GSPC').",
                    },
                    "data_type": {
                        "type": "string",
                        "enum": ["summary", "news", "financials"],
                        "description": "Type of market data to fetch.",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hackernews",
            "description": "Fetch top tech news stories, Ask HN, Show HN, or recent posts from HackerNews. Call when user asks for tech news or HackerNews stories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["top", "new", "best", "ask", "show", "jobs"],
                        "description": "HackerNews category.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of stories (1-10, default 5).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ip_info",
            "description": "Lookup IP address geolocation, ISP, and network details. Call when user asks for their public IP, location info, or IP lookup.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ip_address": {
                        "type": "string",
                        "description": "Optional IP address to look up. Omit for current device public IP.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_wolfram_alpha",
            "description": "Query Wolfram Alpha for math calculations, scientific facts, conversions, or factual questions. Call when user asks for math/computational answers or Wolfram queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The math equation, scientific query, or factual question.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_terminal_command",
            "description": "Execute a shell command line (PowerShell/CMD) on the host system and capture stdout/stderr output. Call when user explicitly asks to run a CLI command, install a package, check git status, or execute terminal commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact CLI command string to execute (e.g., 'git status', 'pip list', 'dir').",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Optional working directory path.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

# Derived from the schemas so the two can never drift apart.
VALID_TOOL_NAMES: set[str] = {t["function"]["name"] for t in OLLAMA_TOOLS}


# ---------------------------------------------------------------------------
#  Dispatch
# ---------------------------------------------------------------------------
def _first_arg(arguments: dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Return the first supplied key. Small models rename parameters freely."""
    for key in keys:
        if key in arguments and arguments[key] not in (None, ""):
            return arguments[key]
    return default


def _handle_system_status(_: dict[str, Any]) -> str:
    return get_system_status()


def _handle_set_reminder(args: dict[str, Any]) -> str:
    return set_reminder(
        _first_arg(args, "text", "task", "reminder"),
        _first_arg(args, "time_str", "time", "when", default="in 1 day"),
    )


def _handle_list_reminders(_: dict[str, Any]) -> str:
    return list_reminders()


def _handle_launch_app(args: dict[str, Any]) -> str:
    return launch_app(_first_arg(args, "app_name", "app", "name"))


def _handle_search_files(args: dict[str, Any]) -> str:
    return search_files(
        _first_arg(args, "file_name_or_extension", "query", "filename", "pattern")
    )


def _handle_search_web(args: dict[str, Any]) -> str:
    return search_web(_first_arg(args, "query", "search_query", "q"))


def _handle_get_clipboard(_: dict[str, Any]) -> str:
    return get_clipboard()


def _handle_set_clipboard(args: dict[str, Any]) -> str:
    return set_clipboard(_first_arg(args, "text", "content"))


def _handle_control_volume(args: dict[str, Any]) -> str:
    level = args.get("level")
    try:
        level = int(level) if level is not None else None
    except (TypeError, ValueError):
        level = None
    return control_volume(_first_arg(args, "action", default="status"), level)


def _handle_control_media(args: dict[str, Any]) -> str:
    return control_media(_first_arg(args, "action", "command", default="play_pause"))


def _handle_read_document(args: dict[str, Any]) -> str:
    return read_document(_first_arg(args, "file_path", "path", "file"))


def _handle_search_youtube_music(args: dict[str, Any]) -> str:
    return search_youtube_music(
        _first_arg(args, "query", "search_query", "q"),
        _first_arg(args, "search_filter", "filter", default=None),
    )


def _handle_play_youtube_music(args: dict[str, Any]) -> str:
    return play_youtube_music(_first_arg(args, "query", "song", "track", "music"))


def _handle_time_date(args: dict[str, Any]) -> str:
    """Retained for CLI/programmatic use; the model gets time via LIVE CONTEXT."""
    return samay_execute(_first_arg(args, "subcommand", default="datetime"))


def _handle_get_weather(args: dict[str, Any]) -> str:
    return get_weather(_first_arg(args, "location", "city", "place", default=None))


def _handle_get_financial_data(args: dict[str, Any]) -> str:
    return get_financial_data(
        _first_arg(args, "symbol", "ticker", "stock"),
        _first_arg(args, "data_type", "type", default="summary"),
    )


def _handle_get_hackernews(args: dict[str, Any]) -> str:
    limit = args.get("limit")
    try:
        limit = int(limit) if limit is not None else 5
    except (TypeError, ValueError):
        limit = 5
    return get_hackernews(_first_arg(args, "category", "cat", default="top"), limit)


def _handle_get_ip_info(args: dict[str, Any]) -> str:
    return get_ip_info(_first_arg(args, "ip_address", "ip", default=None))


def _handle_query_wolfram_alpha(args: dict[str, Any]) -> str:
    return query_wolfram_alpha(_first_arg(args, "query", "q", "input"))


def _handle_run_terminal_command(args: dict[str, Any]) -> str:
    return run_terminal_command(
        _first_arg(args, "command", "cmd", "line"),
        cwd=_first_arg(args, "cwd", "dir", default=None),
    )


_HANDLERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "get_system_status": _handle_system_status,
    "set_reminder": _handle_set_reminder,
    "list_reminders": _handle_list_reminders,
    "launch_app": _handle_launch_app,
    "search_files": _handle_search_files,
    "search_web": _handle_search_web,
    "get_clipboard": _handle_get_clipboard,
    "set_clipboard": _handle_set_clipboard,
    "control_volume": _handle_control_volume,
    "control_media": _handle_control_media,
    "read_document": _handle_read_document,
    "search_youtube_music": _handle_search_youtube_music,
    "play_youtube_music": _handle_play_youtube_music,
    "get_time_date": _handle_time_date,
    "get_weather": _handle_get_weather,
    "get_financial_data": _handle_get_financial_data,
    "get_hackernews": _handle_get_hackernews,
    "get_ip_info": _handle_get_ip_info,
    "query_wolfram_alpha": _handle_query_wolfram_alpha,
    "run_terminal_command": _handle_run_terminal_command,
}


def tools_for_channel(channel: str) -> list[dict[str, Any]]:
    """Schemas this channel is permitted to see.

    Withholding the schema is stronger than refusing the call: the model
    cannot pick a tool it was never shown.
    """
    blocked = CHANNEL_BLOCKED_TOOLS.get(channel, set())
    if not blocked:
        return OLLAMA_TOOLS
    return [t for t in OLLAMA_TOOLS if t["function"]["name"] not in blocked]


def dispatch_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    chroma_collection: Any = None,
    channel: str = "web",
) -> str:
    """Route a tool call to its handler and return a string result."""
    name = (tool_name or "").strip().lower()

    # Second line of defence: even if a schema leaked into the prompt, a
    # blocked channel cannot execute the tool.
    if name in CHANNEL_BLOCKED_TOOLS.get(channel, set()):
        log.warning("Tool %r blocked for channel %r", name, channel)
        return (
            f"The '{name}' tool is not available over {channel}. "
            f"Ask me from the desktop interface instead."
        )

    if name == "run_terminal_command":
        if not isinstance(arguments, dict):
            arguments = {}
        return run_terminal_command(
            _first_arg(arguments, "command", "cmd", "line"),
            cwd=_first_arg(arguments, "cwd", "dir", default=None),
            channel=channel,
        )

    handler = _HANDLERS.get(name)

    if handler is None:
        log.warning("Model requested unknown tool %r", tool_name)
        return (
            f"There is no tool called '{tool_name}'. "
            f"Available tools: {', '.join(sorted(VALID_TOOL_NAMES))}."
        )

    if not isinstance(arguments, dict):
        arguments = {}

    return handler(arguments)
