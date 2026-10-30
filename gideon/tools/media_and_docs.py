"""
G.I.D.E.O.N Audio, Media Transport & Document Reader Tools

Volume goes through pycaw (Windows Core Audio). Transport controls
(play/pause/next/previous) are synthesised media keys, so they work with
whatever app currently owns playback — Spotify, a browser tab, VLC.

COM must be initialised on whichever thread touches pycaw. Tools run in
FastAPI's worker pool, not the main thread, so CoInitialize is called on
every volume operation.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from gideon.core.logs import get_logger
from gideon.tools.base import Param, tool

log = get_logger("gideon.tools.media")

try:
    import comtypes
    from pycaw.pycaw import AudioUtilities
except ImportError:  # pragma: no cover
    AudioUtilities = None
    comtypes = None

try:
    import pypdf
except ImportError:  # pragma: no cover
    pypdf = None  # type: ignore[assignment]

# Virtual key codes for the media transport keys.
_VK_MEDIA = {
    "next": 0xB0,
    "previous": 0xB1,
    "stop": 0xB2,
    "play_pause": 0xB3,
}
_KEYEVENTF_KEYUP = 0x0002

_MEDIA_ALIASES = {
    "play": "play_pause",
    "pause": "play_pause",
    "resume": "play_pause",
    "unpause": "play_pause",
    "toggle": "play_pause",
    "next_track": "next",
    "skip": "next",
    "forward": "next",
    "prev": "previous",
    "prev_track": "previous",
    "back": "previous",
}

def normalize_media_action(action: object) -> str:
    """Map whatever the model wrote onto a transport key name.

    Models emit synonyms ("pause", "resume") and malformed variants
    ("pause_pause", "media_next_track"); match on the recognisable parts
    rather than rejecting outright. Unrecognised input is returned as-is.
    """
    key = str(action or "").strip().lower().replace("-", "_").replace(" ", "_")
    key = _MEDIA_ALIASES.get(key, key)
    if key in _VK_MEDIA:
        return key
    for token in key.split("_"):
        resolved = _MEDIA_ALIASES.get(token, token)
        if resolved in _VK_MEDIA:
            log.info("Normalised media action %r -> %r", action, resolved)
            return resolved
    return key


_MEDIA_LABELS = {
    "play_pause": "Toggled media playback.",
    "next": "Skipped to the next track.",
    "previous": "Went back to the previous track.",
    "stop": "Stopped playback.",
}

_TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".csv", ".log", ".html", ".css", ".xml", ".sh", ".bat",
}
_MAX_DOC_CHARS = 20_000


# ---------------------------------------------------------------------------
#  Volume
# ---------------------------------------------------------------------------
@tool(
    "control_volume",
    "Adjust the computer master sound volume (mute, unmute, or set a percentage from 0 to 100). Call ONLY when the user instructs you to change volume, mute audio, or check the speaker level.",
    Param(
        "action",
        "'set' to change level, 'mute'/'unmute' to toggle audio, 'status' to report the current level.",
        required=True,
        enum=("set", "mute", "unmute", "status"),
        default="status",
    ),
    Param(
        "level",
        "Target volume percentage 0-100 (required when action is 'set').",
        type="integer",
        default=None,
    ),
    triggers={"volume", "mute", "unmute", "louder", "quieter", "sound", "audio"},
)
def control_volume(action: str, level: int | None = None) -> str:
    """Set, mute, unmute or report the Windows master volume."""
    if AudioUtilities is None:
        return "Audio controls are unavailable (pycaw is not installed)."

    try:
        if comtypes is not None:
            # Required on FastAPI worker threads; harmless if already done.
            try:
                comtypes.CoInitialize()
            except Exception:
                pass

        device = AudioUtilities.GetSpeakers()
        vol = getattr(device, "EndpointVolume", None) if device else None
        if vol is None:
            return "Could not acquire the Windows audio endpoint."

        clean_action = (action or "status").strip().lower()

        if clean_action == "mute":
            vol.SetMute(1, None)
            return "System audio muted."

        if clean_action == "unmute":
            vol.SetMute(0, None)
            current = round(vol.GetMasterVolumeLevelScalar() * 100)
            return f"System audio unmuted (currently {current}%)."

        if clean_action in ("set", "change", "adjust") or level is not None:
            if level is None:
                return "Please specify a volume percentage between 0 and 100."
            target = max(0, min(100, int(level)))
            vol.SetMasterVolumeLevelScalar(target / 100.0, None)
            vol.SetMute(0, None)  # Adjusting volume implies you want to hear it.
            return f"System volume set to {target}%."

        current = round(vol.GetMasterVolumeLevelScalar() * 100)
        state = "Muted" if vol.GetMute() else "Unmuted"
        return f"Master volume is {current}% ({state})."

    except Exception as exc:
        log.error("Volume control failed: %s", exc)
        return f"Failed to modify audio settings: {exc}"


# ---------------------------------------------------------------------------
#  Media transport
# ---------------------------------------------------------------------------
@tool(
    "control_media",
    "Control hardware media playback buttons (pause active music, resume, next track, previous track). Call ONLY for general button commands like 'pause', 'resume', 'next track'. NEVER use this when the user names a specific song or artist to play.",
    Param(
        "action",
        "The transport button action to perform.",
        required=True,
        enum=("play_pause", "next", "previous", "stop"),
        aliases=("command",),
        default="play_pause",
        normalize=lambda value: normalize_media_action(value),
    ),
    triggers={"play", "pause", "resume", "skip", "track", "next", "previous"},
)
def control_media(action: str) -> str:
    """Send a media transport key to whichever app owns playback."""
    if os.name != "nt":
        return "Media transport control is only implemented on Windows."

    key = normalize_media_action(action)
    vk = _VK_MEDIA.get(key)
    if vk is None:
        return (
            f"Unknown media action '{action}'. "
            "Use play_pause, next, previous or stop."
        )

    try:
        user32 = ctypes.windll.user32
        user32.keybd_event(vk, 0, 0, 0)
        user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)
    except Exception as exc:
        log.error("Media key %s failed: %s", key, exc)
        return f"Failed to send the media key: {exc}"

    return _MEDIA_LABELS[key]


# ---------------------------------------------------------------------------
#  Documents
# ---------------------------------------------------------------------------
def _resolve_document(raw_path: str) -> Path | None:
    """Locate a document, trying the usual folders for bare filenames."""
    clean = (raw_path or "").strip().strip('"').strip("'")
    if not clean:
        return None

    candidate = Path(clean).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    for base in (Path.cwd(), Path.home() / "Desktop", Path.home() / "Documents"):
        probe = base / clean
        if probe.is_file():
            return probe.resolve()
    return None


@tool(
    "read_document",
    "Open and read the text of a local file or document (.pdf, .py, .txt, .md, .json, etc.). Call ONLY when the user explicitly instructs you to read, open, summarize, or analyze a specific local file.",
    Param(
        "file_path",
        "Path or name of the document, e.g. 'C:\\Users\\me\\Desktop\\report.pdf' or 'main.py'.",
        required=True,
        aliases=("path", "file"),
        grounded=True,
        ground_ignore=("document", "doc", "file", "read", "c", "users", "desktop", "documents"),
    ),
    untrusted_output=True,
    triggers={"read", "document", "pdf"},
)
def read_document(file_path: str, max_pages: int = 30) -> str:
    """Read text out of a local document (.pdf, .txt, source code, etc.)."""
    target = _resolve_document(file_path)
    if target is None:
        return (
            f"Could not find a file at '{file_path}'. "
            "Give me an absolute path, or put it on your Desktop or in Documents."
        )

    ext = target.suffix.lower()

    if ext == ".pdf":
        if pypdf is None:
            return "PDF reading is unavailable (pypdf is not installed)."
        try:
            reader = pypdf.PdfReader(str(target))
            total_pages = len(reader.pages)
            pages_to_read = min(total_pages, max_pages)
            chunks = []
            for idx in range(pages_to_read):
                text = (reader.pages[idx].extract_text() or "").strip()
                if text:
                    chunks.append(f"--- [PAGE {idx + 1}/{total_pages}] ---\n{text}")

            if not chunks:
                return (
                    f"Opened '{target.name}' but found no extractable text — "
                    "it is probably a scanned image PDF."
                )

            body = "\n\n".join(chunks)
            if total_pages > pages_to_read:
                body += f"\n\n[...truncated: read {pages_to_read} of {total_pages} pages...]"
            return f"CONTENTS OF '{target.name}' ({total_pages} pages):\n{body}"
        except Exception as exc:
            log.error("PDF read failed for %s: %s", target, exc)
            return f"Failed to extract PDF contents from '{target.name}': {exc}"

    if ext and ext not in _TEXT_EXTENSIONS:
        return (
            f"'{target.name}' is a {ext} file, which I can't read as text. "
            "I handle PDFs, plain text, and source code."
        )

    try:
        content = target.read_text(encoding="utf-8", errors="ignore")
        total_lines = content.count("\n") + 1
        if len(content) > _MAX_DOC_CHARS:
            content = content[:_MAX_DOC_CHARS] + "\n\n[...truncated...]"
        return f"CONTENTS OF '{target.name}' ({total_lines} lines):\n{content.strip()}"
    except Exception as exc:
        log.error("Text read failed for %s: %s", target, exc)
        return f"Failed to read '{target.name}': {exc}"
