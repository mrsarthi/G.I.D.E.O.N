"""G.I.D.E.O.N — Guided Intelligence Does Everything One Needs.

Dependencies point one way; each layer imports only from those below it:

    interfaces   api, tray, cli                (entry points)
    app          composition root: builds and wires the services
    interfaces   telegram                      (constructed by app)
    brain / reminders / voice
    tools
    storage
    core

Interfaces never import each other: the web API reaches Telegram only through
the services app.py hands it.
"""
