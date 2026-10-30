"""One "boss"/"sir" per reply, in its first sentence (brain/style.py)."""

from __future__ import annotations

import random

import pytest

from gideon.brain.agent import Agent
from gideon.brain.style import AddressLimiter, limit_addresses
from gideon.storage.db import get_db

CASES = [
    # The user's complaint (2026-10-27): one at the start, one at the end.
    ("Boss, the weather is clear, sir.", "Boss, the weather is clear."),
    # Stored replies from before the fix.
    (
        'Yes boss. I see that "sendme" is a folder. Would you like VS Code, sir?',
        'Yes boss. I see that "sendme" is a folder. Would you like VS Code?',
    ),
    ("Visual Studio Code is now open, boss. Ready for your work, sir.", "Visual Studio Code is now open, boss. Ready for your work."),
    ("Good evening, sir. How can I assist you today?", "Good evening, sir. How can I assist you today?"),
    ("Sir, the calculator is open. Sir, anything else?", "Sir, the calculator is open. Anything else?"),
    ("Yes boss I will check, sir.", "Yes boss I will check."),
    ("Paused. Done, boss, anything else?", "Paused. Done, anything else?"),
    ("Right away sir. Here it is boss!", "Right away sir. Here it is!"),
    ("Calculator is open.\n\nBoss, need anything else?", "Calculator is open.\n\nNeed anything else?"),
    # Not addresses: left alone.
    ("My boss said hello. Thanks, boss.", "My boss said hello. Thanks."),
    ("Sir Isaac Newton was a physicist, sir.", "Sir Isaac Newton was a physicist, sir."),
    ("That game has a tough final boss, sir.", "That game has a tough final boss, sir."),
    ("No address here at all.", "No address here at all."),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_one_address_in_the_first_sentence(text, expected):
    assert limit_addresses(text) == expected


@pytest.mark.parametrize("text,expected", CASES)
def test_streaming_in_any_chunks_gives_the_same_text(text, expected):
    for seed in range(25):
        rng = random.Random(seed)
        limiter, out, i = AddressLimiter(), "", 0
        while i < len(text):
            step = rng.randint(1, 6)
            out += limiter.feed(text[i:i + step])
            i += step
        assert out + limiter.flush() == expected


def test_what_is_shown_is_what_is_stored():
    def chat(**kwargs):
        return iter([{"message": {"content": piece}} for piece in ("Boss, the ", "weather is ", "clear, sir.")])

    db = get_db()
    session = db.create_session("style")
    agent = Agent(db, None, chat=chat)
    agent.record_user_message(session, "how's the weather looking?")
    reply = agent.respond(session, "how's the weather looking?")
    assert reply == "Boss, the weather is clear."
    stored = [m for m in db.get_sliding_window(session, limit=5) if m.role == "assistant"]
    assert stored[-1].content == reply
