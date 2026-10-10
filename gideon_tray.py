"""
G.I.D.E.O.N Tray launcher.

    pythonw.exe gideon_tray.py

Kept at the project root because install_autostart.ps1 points the logon
shortcut here. The implementation is gideon/interfaces/tray.py.
"""

import sys

# First, before anything else is imported. Under pythonw there is no console;
# importing pywebview then swaps the missing streams for `nul`, and every log
# line after that — the only diagnostics a background tray has — is lost.
from gideon.core.logs import bind_headless_streams

bind_headless_streams("tray")

from gideon.interfaces.tray import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
