from datetime import *
import webbrowser
import re
import requests


def samay():
    return datetime.now().strftime("%H:%M")

def din():
    return datetime.now().strftime("%A, %B %d, %Y")

def browser(query, platform):
    if platform == "youtube":
        url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        webbrowser.open(url)

    elif platform == "bing":
        url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(url)

    elif platform == "google":
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(url)

    elif platform == "music":
        url = f"https://music.youtube.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(url)

    else:
        url = None

    if url:
        return f"Opening {platform} and searching for '{query}'"
    else:
        return "Unknown platform."


def play_video(query):
    """Search YouTube and play the first matching video directly."""
    try:
        search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(search_url, headers=headers, timeout=10)

        # Extract the first video ID from the page
        video_ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', response.text)

        if video_ids:
            video_id = video_ids[0]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            webbrowser.open(video_url)
            return f"Now playing: {video_url}"
        else:
            return f"No videos found for '{query}'."
    except Exception as e:
        return f"Failed to search YouTube: {e}"