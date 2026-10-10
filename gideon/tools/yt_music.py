"""
G.I.D.E.O.N YouTube Music Integration Tools

Provides searching and track playback for YouTube Music via ytmusicapi.
"""

from __future__ import annotations

import os
import webbrowser
from typing import Any

from gideon.core.logs import get_logger
from gideon.tools.base import Param, tool

log = get_logger("gideon.tools.yt_music")

try:
    from ytmusicapi import YTMusic
except ImportError:
    YTMusic = None  # type: ignore[assignment,misc]


def _get_ytmusic_client() -> Any | None:
    """Instantiate YTMusic client (uses browser.json headers if provided by user, else unauthenticated)."""
    if YTMusic is None:
        return None

    # Check if user provided an auth header file
    for auth_path in ["browser.json", "headers_auth.json", "model/browser.json"]:
        if os.path.exists(auth_path):
            try:
                return YTMusic(auth_path)
            except Exception as e:
                log.warning("YTMusic header auth failed for %s: %s", auth_path, e)

    # Default unauthenticated client
    try:
        return YTMusic()
    except Exception as e:
        log.warning("YTMusic client initialisation failed: %s", e)
        return None


@tool(
    "search_youtube_music",
    "Search YouTube Music for songs, albums, artists, or playlists. Call ONLY when the user explicitly asks to search for music or find tracks on YouTube Music.",
    Param(
        "query",
        "The song title, artist, album name, or playlist search terms.",
        required=True,
        aliases=("search_query", "q"),
        grounded=True,
        ground_ignore=("music", "song", "search", "youtube", "yt"),
    ),
    Param(
        "search_filter",
        "Optional search filter type.",
        enum=("songs", "videos", "albums", "artists", "playlists"),
        aliases=("filter",),
        default=None,
    ),
    untrusted_output=True,
    triggers={"youtube", "yt", "album", "playlist", "artist"},
)
def search_youtube_music(query: str, search_filter: str | None = None) -> str:
    """Search YouTube Music for songs, artists, albums, or playlists."""
    clean_query = query.strip()
    if not clean_query:
        return "Search query was empty."

    client = _get_ytmusic_client()
    if not client:
        return "YouTube Music capabilities are unavailable (ytmusicapi library missing)."

    try:
        valid_filter = search_filter.strip().lower() if search_filter else None
        if valid_filter not in ("songs", "videos", "albums", "artists", "playlists"):
            valid_filter = None

        results = client.search(clean_query, filter=valid_filter)
        if not results:
            return f"No results found on YouTube Music for '{clean_query}'."

        formatted_items = []
        for item in results[:5]:
            title = item.get("title", "Unknown Title")
            result_type = item.get("resultType", "item")
            artists_list = [a.get("name", "") for a in item.get("artists", []) if a.get("name")]
            artists_str = ", ".join(artists_list) if artists_list else "Unknown Artist"
            duration = item.get("duration", "")

            link = ""
            if "videoId" in item and item["videoId"]:
                link = f"https://music.youtube.com/watch?v={item['videoId']}"
            elif "browseId" in item and item["browseId"]:
                link = f"https://music.youtube.com/browse/{item['browseId']}"
            elif "playlistId" in item and item["playlistId"]:
                link = f"https://music.youtube.com/playlist?list={item['playlistId']}"

            dur_str = f" ({duration})" if duration else ""
            formatted_items.append(f"- **{title}** by *{artists_str}*{dur_str} [{result_type.capitalize()}]\n  Link: {link}")

        return f"YOUTUBE MUSIC SEARCH RESULTS FOR '{clean_query}':\n" + "\n\n".join(formatted_items)

    except Exception as e:
        return f"Failed to search YouTube Music: {e}"


@tool(
    "play_youtube_music",
    "Search and launch playback of a specific song, artist, album, or playlist by name on YouTube Music (e.g. 'play Loser by Tame Impala' or 'play Starboy'). ALWAYS use this tool whenever the user mentions a song name, track title, or artist to play.",
    Param(
        "query",
        "The track title, artist name, or playlist to play (e.g., 'Loser Tame Impala' or 'Starboy The Weeknd').",
        required=True,
        aliases=("song", "track", "music"),
        grounded=True,
        ground_ignore=("song", "music", "track", "play", "some", "playlist"),
    ),
    triggers={
        "play", "song", "music", "spotify", "listen", "track", "tune",
        "queue", "blast", "youtube", "yt", "album", "playlist", "artist",
    },
)
def play_youtube_music(query: str) -> str:
    """Search for a track or playlist on YouTube Music and open it directly in default browser/player."""
    clean_query = query.strip()
    if not clean_query:
        return "Please specify a song, artist, or playlist to play on YouTube Music."

    client = _get_ytmusic_client()
    if not client:
        return "YouTube Music playback unavailable (ytmusicapi missing)."

    try:
        # Search songs first for specific queries
        results = client.search(clean_query, filter="songs")
        if not results:
            # Fallback to general search (playlists / albums)
            results = client.search(clean_query)

        if not results:
            return f"Could not find any tracks matching '{clean_query}' on YouTube Music."

        top_match = results[0]
        title = top_match.get("title", clean_query)
        artists_list = [a.get("name", "") for a in top_match.get("artists", []) if a.get("name")]
        artist_str = ", ".join(artists_list) if artists_list else ""

        target_url = None
        if "videoId" in top_match and top_match["videoId"]:
            target_url = f"https://music.youtube.com/watch?v={top_match['videoId']}"
        elif "playlistId" in top_match and top_match["playlistId"]:
            target_url = f"https://music.youtube.com/playlist?list={top_match['playlistId']}"

        if not target_url:
            return f"Found '{title}', but could not resolve a playable YouTube Music URL."

        # Launch playback in user's browser via native Windows Shell startfile
        if hasattr(os, "startfile"):
            os.startfile(target_url)
        else:
            webbrowser.open(target_url)

        by_str = f" by {artist_str}" if artist_str else ""
        return f"Playing '{title}'{by_str} on YouTube Music! Link: {target_url}"

    except Exception as e:
        return f"Failed to play track on YouTube Music: {e}"
