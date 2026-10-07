"""The tray entry point and the hotkey window.

Nothing here opens a window or touches the notification area — those need a
desktop session. What is testable is the logic that decides *what* the hotkey
does and *what* the quick page is served, which is where the mistakes live.
"""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="the tray is Win32-only"
)


# ---------------------------------------------------------------------------
#  Hotkey parsing
# ---------------------------------------------------------------------------
def test_default_chord_is_ctrl_alt_g():
    from gideon.interfaces import tray

    mask, vk = tray.parse_hotkey("ctrl+alt", "G")
    assert mask & tray.MOD_CONTROL
    assert mask & tray.MOD_ALT
    assert vk == ord("G")


def test_norepeat_is_always_set():
    """Without it, holding the chord down fires continuously."""
    from gideon.interfaces import tray

    mask, _ = tray.parse_hotkey("ctrl+shift", "K")
    assert mask & tray.MOD_NOREPEAT


def test_function_keys_resolve_to_vk_codes():
    from gideon.interfaces import tray

    assert tray.parse_hotkey("alt", "F9")[1] == 0x70 + 8


def test_unknown_modifier_is_ignored_not_fatal():
    from gideon.interfaces import tray

    mask, vk = tray.parse_hotkey("ctrl+banana", "G")
    assert mask & tray.MOD_CONTROL
    assert vk == ord("G")


def test_unusable_key_falls_back_to_g():
    """A bad config must still leave a working hotkey."""
    from gideon.interfaces import tray

    assert tray.parse_hotkey("ctrl+alt", "NotAKey")[1] == ord("G")


def test_win_modifier_is_available_even_though_we_avoid_it():
    from gideon.interfaces import tray

    assert tray.parse_hotkey("win", "J")[0] & tray.MOD_WIN


# ---------------------------------------------------------------------------
#  JS bridge
# ---------------------------------------------------------------------------
def test_js_api_exposes_nothing_pywebview_can_recurse_into():
    """pywebview walks the js_api object's attributes to decide what to expose.

    Handing it anything holding a webview.Window means it reaches
    `.window.native` — the WinForms Form — and loops forever through .NET
    properties ("...Bounds.Empty.Empty.Empty...: maximum recursion depth
    exceeded"), flooding the log and leaving the bridge half-registered so
    Esc-to-dismiss stops working. The bridge must stay attribute-free.
    """
    from gideon.interfaces import tray

    api = tray.QuickApi()
    assert api.__dict__ == {}

    public = [name for name in dir(api) if not name.startswith("_")]
    assert public  # it does expose something
    assert all(callable(getattr(api, name)) for name in public)


def test_window_is_not_passed_as_the_js_api():
    """The specific mistake, pinned at the source."""
    from pathlib import Path

    source = (Path(__file__).parent.parent / "gideon" / "interfaces" / "tray.py").read_text(
        encoding="utf-8"
    )
    assert "js_api=QuickApi()" in source
    assert "js_api=self" not in source


def test_quick_page_calls_only_methods_the_bridge_defines():
    """A rename on either side silently breaks Esc and the HUD button."""
    from gideon.core.config import FRONTEND_DIR
    from gideon.interfaces import tray

    html = (FRONTEND_DIR / "quick.html").read_text(encoding="utf-8")
    for method in ("hide", "open_hud"):
        assert f"pywebview.api.{method}" in html
        assert callable(getattr(tray.QuickApi, method))


# ---------------------------------------------------------------------------
#  Quick page
# ---------------------------------------------------------------------------
def _quick_client():
    from fastapi.testclient import TestClient

    from gideon.core.config import PORT
    from gideon.interfaces.api.app import app

    # The Host check that defeats DNS rebinding rejects TestClient's default
    # "testserver" hostname.
    return TestClient(app, base_url=f"http://localhost:{PORT}")


def test_quick_route_is_served_and_not_swallowed_by_the_static_mount():
    """StaticFiles is mounted at "/" and matches everything under it, so this
    route only works because it is registered first."""
    response = _quick_client().get("/quick")
    assert response.status_code == 200
    assert "QUICK ASK" in response.text


def test_quick_route_injects_token_and_session():
    """The window must be able to accept the first keystroke without first
    round-tripping through session creation."""
    from gideon.core.config import QUICK_SESSION_ID
    from gideon.core.security import API_TOKEN

    body = _quick_client().get("/quick").text
    assert f'window.GIDEON_API_TOKEN = "{API_TOKEN}"' in body
    assert f'window.GIDEON_QUICK_SESSION = "{QUICK_SESSION_ID}"' in body


def test_quick_page_is_self_contained():
    """No external assets: this window is shown far more often than loaded, so
    it must not fetch anything at open time."""
    from gideon.core.config import FRONTEND_DIR

    html = (FRONTEND_DIR / "quick.html").read_text(encoding="utf-8")
    assert "<script src=" not in html
    assert "<link rel=\"stylesheet\"" not in html
    assert ".glb" not in html  # no hologram, so no VRAM


def test_quick_page_never_plays_audio():
    """Typing at the desk — in a meeting, at night — should stay silent."""
    from gideon.core.config import FRONTEND_DIR

    html = (FRONTEND_DIR / "quick.html").read_text(encoding="utf-8")
    assert "new Audio" not in html
    assert "play()" not in html


def test_quick_session_is_separate_from_the_hud():
    from gideon.core.config import QUICK_SESSION_ID

    assert QUICK_SESSION_ID
    assert QUICK_SESSION_ID != "default"
