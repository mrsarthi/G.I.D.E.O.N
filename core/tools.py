"""
G.I.D.E.O.N Tool Definitions

Each function here is a "tool" that the LLM can choose to call.
Ollama reads the function name, type hints, and docstring to understand
what each tool does and when to use it.
"""

from core.actions import samay, din, browser, play_video


def get_current_time() -> str:
    """Get the current time. Use this when the user asks what time it is."""
    return f"The current time is {samay()}"


def get_current_date() -> str:
    """Get today's date. Use this when the user asks what day or date it is."""
    return f"Today is {din()}"


def web_search(query: str, platform: str) -> str:
    """Search the web and show search results. Use this ONLY when the user wants to browse or look up information. Do NOT use this to play music or videos — use play_music instead.

    Args:
        query: The search terms to look up.
        platform: The platform to search on. Must be one of: 'google', 'youtube', 'bing', 'music'.
    """
    return browser(query, platform)


def play_music(query: str) -> str:
    """Play a specific song or video directly on YouTube. Use this whenever the user says 'play', 'put on', 'listen to', or wants to watch a specific video or hear a specific song. This opens the video directly instead of showing search results.

    Args:
        query: The name of the song, artist, or video to play.
    """
    return play_video(query)


# Registry: maps function names to actual functions
# This makes it easy to look up which function to call when the LLM picks one
TOOL_REGISTRY = {
    "get_current_time": get_current_time,
    "get_current_date": get_current_date,
    "web_search": web_search,
    "play_music": play_music,
}

# List of tool functions to pass to ollama.chat()
TOOLS = [get_current_time, get_current_date, web_search, play_music]
