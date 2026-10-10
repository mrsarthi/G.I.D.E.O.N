"""
G.I.D.E.O.N Windows Shortcuts

The Startup-folder shortcut (start the tray at logon) and the Start Menu
entry (find it by typing "GIDEON" in Windows search). Neither needs admin.

The Startup shortcut has the same name and target install_autostart.ps1
creates, so the tray's "Start with Windows" toggle and the installer manage
the same file. The boot-time engine task is separate and still needs the
elevated installer.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from gideon.core.config import BASE_DIR, STATIC_DIR
from gideon.core.logs import get_logger

log = get_logger("gideon.shortcuts")

SHORTCUT_NAME = "G.I.D.E.O.N.lnk"
_PROGRAMS = (
    Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    / "Microsoft" / "Windows" / "Start Menu" / "Programs"
)
STARTUP_SHORTCUT = _PROGRAMS / "Startup" / SHORTCUT_NAME
START_MENU_SHORTCUT = _PROGRAMS / SHORTCUT_NAME
TRAY_SCRIPT = BASE_DIR / "gideon_tray.py"

# Values reach PowerShell through environment variables, never by splicing
# them into the script, so a path with quotes or `$` cannot change its meaning.
_CREATE_SHORTCUT_PS = r"""
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:GIDEON_LNK)
$s.TargetPath = $env:GIDEON_TARGET
$s.Arguments = '"' + $env:GIDEON_ARGS + '"'
$s.WorkingDirectory = $env:GIDEON_WORKDIR
$s.IconLocation = $env:GIDEON_ICON
$s.Description = 'G.I.D.E.O.N - tray, global hotkey and quick ask'
$s.Save()
"""


def pythonw() -> Path:
    """The windowless interpreter: a shortcut to python.exe opens a console
    window, and closing that window would kill the tray."""
    venv = BASE_DIR / ".venv" / "Scripts" / "pythonw.exe"
    if venv.exists():
        return venv
    return Path(sys.executable).with_name("pythonw.exe")


def _write_shortcut(path: Path) -> bool:
    icon = STATIC_DIR / "gideon.ico"
    env = {
        **os.environ,
        "GIDEON_LNK": str(path),
        "GIDEON_TARGET": str(pythonw()),
        "GIDEON_ARGS": str(TRAY_SCRIPT),
        "GIDEON_WORKDIR": str(BASE_DIR),
        "GIDEON_ICON": str(icon if icon.exists() else pythonw()),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _CREATE_SHORTCUT_PS],
            env=env,
            check=True,
            capture_output=True,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        log.error("Could not create shortcut %s: %s", path, exc)
        return False


def autostart_enabled() -> bool:
    return STARTUP_SHORTCUT.exists()


def set_autostart(enabled: bool) -> bool:
    """Add or remove the logon shortcut. Returns the resulting state."""
    if enabled:
        if _write_shortcut(STARTUP_SHORTCUT):
            log.info("Start with Windows: on (%s)", STARTUP_SHORTCUT)
    else:
        try:
            STARTUP_SHORTCUT.unlink(missing_ok=True)
            log.info("Start with Windows: off")
        except OSError as exc:
            log.error("Could not remove %s: %s", STARTUP_SHORTCUT, exc)
    return autostart_enabled()


def ensure_start_menu_entry() -> None:
    """Create the Start Menu entry once; never overwrite one the user edited."""
    if not START_MENU_SHORTCUT.exists() and _write_shortcut(START_MENU_SHORTCUT):
        log.info("Added Start Menu entry %s", START_MENU_SHORTCUT)
