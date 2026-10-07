"""
G.I.D.E.O.N Tray & Global Hotkey

The entry point. Running this is the only thing you should have to do:

    pythonw.exe gideon_tray.py

(a thin launcher for this module). It supervises the engine (run_engine.py), sits in the notification area, and
answers Ctrl+Alt+G anywhere in Windows with a small ask-and-answer window.

Three design decisions are load-bearing:

* **The tray is not the engine.** A Windows service cannot show a tray icon or
  a window — Session 0 isolation blocks all GUI — so the engine is registered
  separately to run at boot (install_autostart.ps1) while this process starts
  at logon and merely attaches to it. If the engine is not up, this starts one
  as a child; if it is, this leaves it alone.

* **RegisterHotKey, not a keyboard hook.** The obvious library for this
  installs a low-level global hook, which is a keylogger by any technical
  definition — antivirus flags it and it needs admin. RegisterHotKey asks
  Windows to deliver one specific chord and nothing else.

* **The window is pre-warmed, never built on demand.** WebView2 cold start is
  half a second, which is the difference between a reflex and a chore, so the
  window is created hidden at logon and the hotkey only calls show().

Thread layout, which the Win32 and GUI toolkits both constrain:

    main      pywebview message loop (must be the main thread)
    hotkey    hidden Win32 window + GetMessage loop (owns the hotkey)
    tray      pystray icon
    engine    health polling and supervision
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import time
import webbrowser
from ctypes import wintypes
from pathlib import Path

import httpx
import pystray
import webview
from PIL import Image, ImageDraw

# Nothing below may log before bind_headless_streams() runs; these imports
# only read .env and build constants.
from gideon.core.config import (
    BASE_DIR,
    ENGINE_START_TIMEOUT,
    HOST,
    HOTKEY_KEY,
    HOTKEY_MODIFIERS,
    PORT,
    QUICK_WINDOW_HEIGHT,
    QUICK_WINDOW_WIDTH,
    STATIC_DIR,
    WEBVIEW_DIR,
)
from gideon.core.logs import bind_headless_streams, get_logger
from gideon.core.security import API_TOKEN, API_TOKEN_HEADER

# Before the first get_logger(). Under pythonw.exe — which is how this is
# normally launched, since it must not depend on a console staying open —
# sys.stderr is None and every log record would be silently discarded. This is
# a no-op when there is a real console.
_LOG_FILE = bind_headless_streams("tray")

log = get_logger("gideon.tray")

BASE_URL = f"http://{HOST}:{PORT}"
QUICK_URL = f"{BASE_URL}/quick"
WINDOW_TITLE = "G.I.D.E.O.N Quick Ask"

# ---------------------------------------------------------------------------
#  Win32
# ---------------------------------------------------------------------------
user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(
    LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN = 0x0001, 0x0002, 0x0004, 0x0008
# Without NOREPEAT, holding the chord down fires continuously.
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_DESTROY = 0x0002
WM_APP_SHOW = 0x8000 + 1  # a second launch asking the first to surface
WM_APP_QUIT = 0x8000 + 2
SW_RESTORE, SW_SHOW = 9, 5
ERROR_ALREADY_EXISTS = 183
HOTKEY_ID = 1

WINDOW_CLASS = "GideonTrayHotkeyWindow"
MUTEX_NAME = "Local\\GIDEON_TRAY_SINGLETON"

_MODIFIER_MAP = {
    "ctrl": MOD_CONTROL,
    "control": MOD_CONTROL,
    "alt": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN,
}


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HICON),
    ]


# ctypes defaults every argument and return value to int, which truncates
# handles on 64-bit Windows. These have to be declared.
user32.DefWindowProcW.restype = LRESULT
user32.DefWindowProcW.argtypes = [
    wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
]
user32.CreateWindowExW.restype = wintypes.HWND
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
]
user32.FindWindowW.restype = wintypes.HWND
user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.RegisterHotKey.argtypes = [
    wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT
]
user32.PostMessageW.argtypes = [
    wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, wintypes.LPVOID]
user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]


def parse_hotkey(modifiers: str, key: str) -> tuple[int, int]:
    """Turn "ctrl+alt" / "G" into the (modifier mask, virtual key) pair.

    Single characters and digits map to their ASCII code, which is also their
    virtual-key code; F1-F24 are handled explicitly.
    """
    mask = MOD_NOREPEAT
    for part in modifiers.lower().replace(" ", "").split("+"):
        if part in _MODIFIER_MAP:
            mask |= _MODIFIER_MAP[part]
        elif part:
            log.warning("Unknown hotkey modifier %r; ignoring", part)

    key = (key or "G").strip().upper()
    if key.startswith("F") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        vk = 0x70 + int(key[1:]) - 1
    elif len(key) == 1:
        vk = ord(key)
    else:
        log.warning("Unusable hotkey key %r; falling back to G", key)
        vk = ord("G")
    return mask, vk


def force_foreground(hwnd: int) -> None:
    """Raise and focus a window from a thread that does not own it.

    Windows refuses SetForegroundWindow across input queues to stop apps
    stealing focus. Attaching to the current foreground thread's input for the
    duration is the sanctioned way around it; without this the window appears
    behind whatever you were working in, which makes the hotkey useless.
    """
    if not hwnd:
        return
    foreground = user32.GetForegroundWindow()
    target_tid = user32.GetWindowThreadProcessId(hwnd, None)
    fg_tid = user32.GetWindowThreadProcessId(foreground, None)
    attached = False
    try:
        if fg_tid and target_tid and fg_tid != target_tid:
            attached = bool(user32.AttachThreadInput(fg_tid, target_tid, True))
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetForegroundWindow(hwnd)
        user32.BringWindowToTop(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(fg_tid, target_tid, False)


# ---------------------------------------------------------------------------
#  Single instance
# ---------------------------------------------------------------------------
def claim_singleton() -> bool:
    """Take the process-wide lock. False means another tray already holds it.

    Without this, clicking the taskbar icon twice starts a second engine that
    cannot bind port 7777, leaving a half-dead process and two hotkey
    registrations fighting over the same chord.
    """
    handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    if not handle:
        # Failing open is right: a broken mutex should not stop G.I.D.E.O.N
        # from starting at all.
        log.warning("Could not create singleton mutex; continuing anyway")
        return True
    if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        return False
    # Deliberately leaked: the handle must outlive this function, and Windows
    # releases it when the process ends.
    globals()["_singleton_handle"] = handle
    return True


def signal_running_instance() -> bool:
    """Ask the tray that already owns the lock to show its window."""
    hwnd = user32.FindWindowW(WINDOW_CLASS, None)
    if not hwnd:
        return False
    user32.PostMessageW(hwnd, WM_APP_SHOW, 0, 0)
    return True


# ---------------------------------------------------------------------------
#  Engine supervision
# ---------------------------------------------------------------------------
class Engine:
    """Tracks the engine process: is it up, and did we start it?"""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.alive = False
        self._client = httpx.Client(timeout=2.0)
        self._lock = threading.Lock()

    def probe(self) -> bool:
        try:
            response = self._client.get(f"{BASE_URL}/api/health")
            self.alive = response.status_code == 200
        except Exception:
            self.alive = False
        return self.alive

    def _python(self) -> str:
        """pythonw.exe, so a supervised engine has no console window.

        Falls back to whatever launched us if the windowless build is absent.
        """
        candidate = Path(sys.executable).with_name("pythonw.exe")
        return str(candidate if candidate.exists() else sys.executable)

    def start(self) -> None:
        """Start an engine if nothing is answering.

        Nothing here forces ownership: if the boot-time scheduled task already
        started one, probe() sees it and this returns immediately. That is why
        the same tray works whether or not autostart is installed.
        """
        with self._lock:
            if self.probe():
                log.info("Engine already running at %s", BASE_URL)
                return
            if self.process and self.process.poll() is None:
                return

            log.info("No engine at %s; starting one", BASE_URL)
            # run_engine.py, not `-m uvicorn`: under pythonw there is no
            # console, so sys.stderr is None and uvicorn's first log line
            # kills the process with exit 1 before it binds. The launcher
            # attaches real streams to the engine log first.
            self.process = subprocess.Popen(
                [self._python(), str(BASE_DIR / "run_engine.py")],
                cwd=str(BASE_DIR),
                # NEW_PROCESS_GROUP is what makes a graceful stop possible at
                # all: it is the precondition for sending CTRL_BREAK, and
                # without it TerminateProcess is the only option — which skips
                # lifespan shutdown and leaves Telegram's getUpdates consumer
                # registered, so the next start hits 409 Conflict.
                creationflags=(
                    subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                ),
            )

        # Cold start loads ChromaDB, Kokoro and Piper; this is genuinely slow.
        deadline = time.time() + ENGINE_START_TIMEOUT
        while time.time() < deadline:
            if self.probe():
                log.info("Engine ready")
                return
            if self.process and self.process.poll() is not None:
                log.error(
                    "Engine exited during startup (code %s). Run run.bat in a "
                    "console to see why.",
                    self.process.returncode,
                )
                return
            time.sleep(1.5)
        log.error("Engine did not become healthy within %ds", ENGINE_START_TIMEOUT)

    def _shutdown_child(self) -> None:
        """Stop our engine child, giving it a chance to shut down properly.

        Graceful matters: lifespan teardown is what stops the Telegram updater
        polling and checkpoints the SQLite WAL. Skip it and the next start
        collides with the still-registered getUpdates consumer (409 Conflict).

        The request goes over HTTP rather than as a signal because it cannot
        go as a signal — see core/lifecycle.py. TerminateProcess remains the
        fallback for a wedged engine, not the first move.
        """
        process = self.process
        if not process or process.poll() is not None:
            return

        try:
            self._client.post(
                f"{BASE_URL}/api/shutdown", headers={API_TOKEN_HEADER: API_TOKEN}
            )
        except Exception as exc:
            # A engine mid-shutdown drops the connection rather than
            # answering, so a transport error here is expected, not a failure.
            log.debug("Shutdown request did not return cleanly: %s", exc)

        try:
            process.wait(timeout=25)
            log.info("Engine shut down cleanly")
            return
        except subprocess.TimeoutExpired:
            log.warning("Engine did not stop when asked; terminating")

        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.error("Engine would not terminate; killing")
            process.kill()

    def restart(self) -> None:
        with self._lock:
            self._shutdown_child()
            self.process = None
        self.start()

    def stop_if_ours(self) -> None:
        """Only stop an engine this process started.

        An engine owned by the boot task belongs to the machine, not to this
        login session; quitting the tray must not take reminders down with it.
        """
        if self.process and self.process.poll() is None:
            log.info("Stopping the engine we started")
            self._shutdown_child()

    def watch(self, stop: threading.Event, on_change=None) -> None:
        """Poll health so the tray can show engine state without being asked."""
        while not stop.wait(20):
            was_alive = self.alive
            if self.probe() != was_alive and on_change:
                on_change()


# ---------------------------------------------------------------------------
#  Quick window
# ---------------------------------------------------------------------------
# The window instance the JS bridge below acts on. Module-level on purpose —
# see QuickApi.
_quick_window: QuickWindow | None = None


class QuickApi:
    """The object exposed to JavaScript as `window.pywebview.api`.

    It deliberately holds **no instance attributes at all**, and that is the
    whole reason it exists as a separate class.

    pywebview introspects whatever it is handed as js_api to decide what to
    expose. Passing the QuickWindow itself means it walks that object's
    attributes, reaches `.window`, and from there `.window.native` — the
    WinForms Form — where it recurses forever through .NET properties:

        Error while processing window.native.AccessibilityObject.Bounds
        .Empty.Empty.Empty.Empty... : maximum recursion depth exceeded

    Non-fatal, but it floods the log and leaves the bridge half-registered, so
    Esc-to-dismiss and the "Full HUD" button silently stop working. Reaching
    the window through a module global instead gives introspection nothing to
    descend into.
    """

    def hide(self) -> None:
        if _quick_window is not None:
            _quick_window.hide_window()

    def open_hud(self) -> None:
        open_hud()


class QuickWindow:
    """The pre-warmed spotlight window."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.window: webview.Window | None = None
        self.ready = threading.Event()
        self._visible = False

    def hide_window(self) -> None:
        self._set_visible(False)

    def create(self) -> webview.Window:
        window = webview.create_window(
            WINDOW_TITLE,
            url=QUICK_URL,
            width=QUICK_WINDOW_WIDTH,
            height=QUICK_WINDOW_HEIGHT,
            hidden=True,
            frameless=True,
            easy_drag=True,
            on_top=True,
            resizable=True,
            background_color="#0a0e14",
            text_select=True,
            # Never `self` — see QuickApi for what that costs.
            js_api=QuickApi(),
        )
        # Alt+F4 and any programmatic close should park the window, not
        # destroy it — rebuilding costs the half second the pre-warm exists to
        # avoid. Returning False cancels the close.
        assert window is not None
        window.events.closing += self._on_closing
        self.window = window
        return window

    def _on_closing(self) -> bool:
        self._set_visible(False)
        return False

    def _set_visible(self, visible: bool) -> None:
        if not self.window or not self.ready.is_set():
            return
        try:
            if visible:
                self.window.show()
                force_foreground(self._hwnd())
                # The page re-focuses its input on every show, not just on
                # load, because the window outlives many showings.
                self.window.evaluate_js(
                    "window.gideonOnShow && window.gideonOnShow()"
                )
            else:
                self.window.hide()
            self._visible = visible
        except Exception:
            log.exception("Could not %s the quick window", "show" if visible else "hide")

    def _hwnd(self) -> int:
        return user32.FindWindowW(None, WINDOW_TITLE)

    def toggle(self) -> None:
        """What the hotkey does: summon, or dismiss if already in front."""
        if self._visible and self._hwnd() == user32.GetForegroundWindow():
            self._set_visible(False)
        else:
            self._set_visible(True)

    def summon(self) -> None:
        self._set_visible(True)


# ---------------------------------------------------------------------------
#  Hotkey thread
# ---------------------------------------------------------------------------
class HotkeyListener(threading.Thread):
    """Hidden window owning the global chord and the second-instance channel."""

    def __init__(self, on_hotkey, on_show) -> None:
        super().__init__(name="gideon-hotkey", daemon=True)
        self.on_hotkey = on_hotkey
        self.on_show = on_show
        self.registered = threading.Event()
        self.ok = False
        self._hwnd = None
        # Held on the instance so CPython cannot collect the callback while
        # Windows still holds the pointer — that crashes the process.
        self._wndproc = WNDPROC(self._handle_message)

    def _handle_message(self, hwnd, msg, wparam, lparam):
        if msg == WM_HOTKEY and wparam == HOTKEY_ID:
            self.on_hotkey()
            return 0
        if msg == WM_APP_SHOW:
            self.on_show()
            return 0
        if msg in (WM_APP_QUIT, WM_DESTROY):
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def run(self) -> None:
        instance = kernel32.GetModuleHandleW(None)

        wndclass = WNDCLASSEXW()
        wndclass.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wndclass.lpfnWndProc = self._wndproc
        wndclass.hInstance = instance
        wndclass.lpszClassName = WINDOW_CLASS
        if not user32.RegisterClassExW(ctypes.byref(wndclass)):
            log.error("RegisterClassExW failed: %s", ctypes.get_last_error())
            self.registered.set()
            return

        self._hwnd = user32.CreateWindowExW(
            0, WINDOW_CLASS, WINDOW_CLASS, 0, 0, 0, 0, 0, None, None, instance, None
        )
        if not self._hwnd:
            log.error("CreateWindowExW failed: %s", ctypes.get_last_error())
            self.registered.set()
            return

        mask, vk = parse_hotkey(HOTKEY_MODIFIERS, HOTKEY_KEY)
        self.ok = bool(user32.RegisterHotKey(self._hwnd, HOTKEY_ID, mask, vk))
        if self.ok:
            log.info("Hotkey %s+%s registered", HOTKEY_MODIFIERS, HOTKEY_KEY)
        else:
            # Almost always another app owning the same chord. The tray menu
            # still works, so this is a degraded start, not a failed one.
            log.error(
                "Could not register %s+%s (error %s) — another application "
                "probably owns it. Set GIDEON_HOTKEY_KEY to something else.",
                HOTKEY_MODIFIERS, HOTKEY_KEY, ctypes.get_last_error(),
            )
        self.registered.set()

        message = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))

        if self.ok:
            user32.UnregisterHotKey(self._hwnd, HOTKEY_ID)

    def stop(self) -> None:
        if self._hwnd:
            user32.PostMessageW(self._hwnd, WM_APP_QUIT, 0, 0)


# ---------------------------------------------------------------------------
#  Tray icon
# ---------------------------------------------------------------------------
def build_icon_image(size: int = 64) -> Image.Image:
    """Draw the tray glyph rather than ship a binary asset."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    accent = (53, 208, 255, 255)
    pad = size // 8
    draw.ellipse([pad, pad, size - pad, size - pad], outline=accent, width=size // 12)
    inner = size // 3
    draw.ellipse(
        [inner, inner, size - inner, size - inner], fill=accent
    )
    return image


def ensure_ico_file() -> Path:
    """Write the icon to disk so the shortcut and the window can reuse it."""
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    path = STATIC_DIR / "gideon.ico"
    if not path.exists():
        try:
            build_icon_image(256).save(
                path, sizes=[(16, 16), (32, 32), (48, 48), (256, 256)]
            )
        except Exception:
            log.exception("Could not write %s", path)
    return path


def open_hud() -> None:
    """Open the full HUD in a chrome-less app window when a browser allows it.

    App mode gives a native-looking window instead of a tab in whatever pile
    of tabs is already open.
    """
    for browser in ("msedge", "chrome"):
        executable = _find_browser(browser)
        if not executable:
            continue
        try:
            subprocess.Popen(
                [executable, f"--app={BASE_URL}"], creationflags=subprocess.CREATE_NO_WINDOW
            )
            return
        except Exception:
            log.debug("Could not launch %s in app mode", browser)
    webbrowser.open(BASE_URL)


def _find_browser(name: str) -> str | None:
    import shutil

    found = shutil.which(name)
    if found:
        return found
    candidates = {
        "msedge": [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ],
        "chrome": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ],
    }
    for path in candidates.get(name, []):
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
#  Wiring
# ---------------------------------------------------------------------------
def main() -> int:
    if not claim_singleton():
        # Second launch. Surfacing the window that already exists is what the
        # user meant by clicking the icon again.
        if signal_running_instance():
            log.info("G.I.D.E.O.N is already running; asked it to surface")
        else:
            log.warning("Another tray holds the lock but its window is not up yet")
        return 0

    global _quick_window

    ensure_ico_file()
    engine = Engine()
    quick = QuickWindow(engine)
    # How QuickApi reaches the window without holding a reference to it.
    _quick_window = quick
    stop_event = threading.Event()

    threading.Thread(target=engine.start, name="gideon-engine", daemon=True).start()

    hotkey = HotkeyListener(on_hotkey=quick.toggle, on_show=quick.summon)
    hotkey.start()

    # Annotated because the Quit item below refers to icon inside its own constructor.
    icon: pystray.Icon = pystray.Icon(
        "gideon",
        icon=build_icon_image(),
        title="G.I.D.E.O.N",
        menu=pystray.Menu(
            pystray.MenuItem(
                f"Quick Ask  ({HOTKEY_MODIFIERS}+{HOTKEY_KEY})",
                lambda: quick.summon(),
                default=True,
            ),
            pystray.MenuItem("Open Full HUD", lambda: open_hud()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda item: f"Engine: {'online' if engine.alive else 'offline'}",
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                "Restart Engine",
                lambda: threading.Thread(target=engine.restart, daemon=True).start(),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda: _quit(icon, hotkey, quick)),
        ),
    )

    # pystray's Win32 backend pumps its own messages, so it is happy off the
    # main thread. pywebview is not — it must own the main thread below.
    threading.Thread(target=icon.run, name="gideon-tray", daemon=True).start()
    threading.Thread(
        target=engine.watch,
        args=(stop_event, icon.update_menu),
        name="gideon-watch",
        daemon=True,
    ).start()

    def on_gui_ready() -> None:
        quick.ready.set()
        log.info("G.I.D.E.O.N tray ready — press %s+%s", HOTKEY_MODIFIERS, HOTKEY_KEY)

    quick.create()
    try:
        webview.start(on_gui_ready, private_mode=False, storage_path=str(WEBVIEW_DIR))
    finally:
        stop_event.set()
        hotkey.stop()
        try:
            icon.stop()
        except Exception:
            pass
        engine.stop_if_ours()
    return 0


def _quit(icon: pystray.Icon, hotkey: HotkeyListener, quick: QuickWindow) -> None:
    log.info("Shutting down tray")
    try:
        icon.stop()
    except Exception:
        pass
    hotkey.stop()
    if quick.window:
        try:
            quick.window.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
