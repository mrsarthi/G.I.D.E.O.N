"""G.I.D.E.O.N terminal chat. Run: python main.py"""

from __future__ import annotations

import sys
from typing import Any

from gideon.brain.agent import Agent
from gideon.brain.approvals import ApprovalError, ApprovalQueue
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
    approvals = ApprovalQueue(db)
    agent = Agent(db, memory, approvals)

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
            requested: list[dict[str, Any]] = []
            print("\033[32mG.I.D.E.O.N:\033[0m ", end="", flush=True)
            for token in agent.stream(session_id, inp, channel="cli", on_event=requested.append):
                print(token, end="", flush=True)
            print()
            for event in requested:
                if "approval" in event:
                    _ask_approval(approvals, event["approval"])
    finally:
        memory.close()
        db.close()


def _ask_approval(approvals: ApprovalQueue, action: dict[str, Any]) -> None:
    """The terminal's approval card: the user's keypress, not the model, decides."""
    print(f"\n\033[33mApproval needed #{action['action_id']}:\033[0m {action['summary']}")
    if action.get("reason"):
        print(f"  Why: {action['reason']}.")
    try:
        answer = input("  Run it? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        answer = ""
    try:
        if answer in ("y", "yes"):
            print(f"\033[36m{approvals.approve(action['action_id']).result}\033[0m\n")
        else:
            approvals.deny(action["action_id"])
            print("  Denied.\n")
    except ApprovalError as exc:
        print(f"  {exc}\n")
