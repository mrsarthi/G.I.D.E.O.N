"""
G.I.D.E.O.N Engine Launcher (headless)

The entry point used by the boot-time scheduled task and by the tray when it
has to start an engine itself. Running
`pythonw -m uvicorn gideon.interfaces.api.app:app` directly does not work, and
the failure is silent and total:

    pythonw.exe has no console, so sys.stdout and sys.stderr are None.
    uvicorn logs to stderr on its very first line, that write raises, and the
    process dies with exit code 1 before the server ever binds.

Nothing surfaces this — there is no console for the traceback to appear in —
so the symptom is just "G.I.D.E.O.N does not start at boot". Binding the
streams to a real file before uvicorn is imported fixes it, and has the side
benefit that a background service finally has a log you can read
(engine.log in the logs directory).

Usage:
    pythonw.exe run_engine.py
"""

from __future__ import annotations

import sys

# Safe before the streams are bound: these read .env and build constants, but
# nothing in them logs at import time.
from gideon.core.logs import bind_headless_streams


def main() -> int:
    # Must happen before importing uvicorn and before the first get_logger().
    log_path = bind_headless_streams("engine")

    # Imported only after the streams exist.
    import signal

    import uvicorn

    from gideon.core.config import HOST, PORT
    from gideon.core.logs import get_logger

    log = get_logger("gideon.launcher")
    log.info("Starting engine on http://%s:%s (log: %s)", HOST, PORT, log_path)

    # No reload: a file watcher doubles memory and restarts the engine on any
    # stray edit, which for a boot service means dropping reminders.
    config = uvicorn.Config(
        "gideon.interfaces.api.app:app", host=HOST, port=PORT, reload=False, log_config=None
    )
    server = uvicorn.Server(config)

    # Registered so POST /api/shutdown can stop this server gracefully. That
    # endpoint, not a signal, is how the tray stops the engine — see
    # core/lifecycle.py for why signals cannot reach a CREATE_NO_WINDOW child.
    from gideon.core import lifecycle

    lifecycle.bind(server)

    # CTRL_BREAK is the only shutdown signal Windows will deliver to another
    # process group, and it arrives as SIGBREAK — which uvicorn does not
    # handle (it installs SIGINT/SIGTERM only). Without this, the tray's
    # "graceful" stop is ignored and falls through to TerminateProcess, which
    # skips lifespan teardown: the Telegram updater never stops polling, so
    # the next start collides with the old getUpdates consumer (409), and the
    # SQLite WAL is left unchecked-pointed.
    #
    # uvicorn's own handler installation only touches SIGINT and SIGTERM, so
    # this one survives server.run().
    def _on_break(signum, frame):
        log.info("CTRL_BREAK received; shutting down gracefully")
        server.should_exit = True

    if hasattr(signal, "SIGBREAK"):  # Windows only
        signal.signal(signal.SIGBREAK, _on_break)

    server.run()
    log.info("Engine stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
