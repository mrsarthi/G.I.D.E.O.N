"""G.I.D.E.O.N terminal chat. Run: python main.py"""

from __future__ import annotations

import sys

from gideon.brain.agent import Agent
from gideon.storage.db import get_db
from gideon.storage.memory import VectorMemory


def run() -> None:
    """Interactive chat in the terminal, sharing the engine's database and memory."""
    # The Windows console defaults to cp1252, which cannot encode the box
    # drawing characters below or most model output (em dashes, degree signs).
    for stream_obj in (sys.stdout, sys.stderr):
        try:
            stream_obj.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    db = get_db()
    memory = VectorMemory()
    agent = Agent(db, memory)

    session_id = db.create_session("Terminal Session")
    print("\n\033[36m╔══════════════════════════════════════╗")
    print("║   G.I.D.E.O.N  Agentic Engine v2.0   ║")
    print("╚══════════════════════════════════════╝\033[0m\n")

    try:
        while True:
            try:
                inp = input("\033[36mYou: \033[0m").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye, boss.")
                break

            if not inp:
                continue
            if inp.lower() in ("/quit", "exit"):
                break

            agent.record_user_message(session_id, inp)
            print("\033[32mG.I.D.E.O.N:\033[0m ", end="", flush=True)
            for token in agent.stream(session_id, inp, channel="cli"):
                print(token, end="", flush=True)
            print()
    finally:
        memory.close()
        db.close()
