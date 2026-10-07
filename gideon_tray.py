"""
G.I.D.E.O.N Tray launcher.

    pythonw.exe gideon_tray.py

Kept at the project root because install_autostart.ps1 points the logon
shortcut here. The implementation is gideon/interfaces/tray.py.
"""

import sys

from gideon.interfaces.tray import main

if __name__ == "__main__":
    sys.exit(main())
