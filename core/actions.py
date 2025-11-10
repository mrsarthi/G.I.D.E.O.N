from datetime import *
import webbrowser

def samay():
    return datetime.now().strftime("%H:%M:%S")

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

    else:
        url = None

    if url:
        return f"G.I.D.E.O.N: Opening {platform} and searching for '{query}'"        
    else:
        return "G.I.D.E.O.N: Unknown platform."