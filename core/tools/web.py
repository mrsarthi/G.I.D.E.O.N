import webbrowser

def web_search(query: str, platform: str) -> str:
    platforms = {
        "youtube": f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}",
        "bing": f"https://www.bing.com/search?q={query.replace(' ', '+')}",
        "google": f"https://www.google.com/search?q={query.replace(' ', '+')}",
        "music": f"https://music.youtube.com/search?q={query.replace(' ', '+')}"
    }
    
    url = platforms.get(platform.lower())
    if url:
        webbrowser.open(url)
        return f"Opening {platform} and searching for '{query}'"
    return "Unknown platform."
