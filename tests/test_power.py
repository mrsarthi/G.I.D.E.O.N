"""Battery safety.

On this laptop, running the chat model on the GPU from battery powers the
machine off. With autostart, booting unplugged would do it with no one at the
keyboard. These tests pin that, on battery, the model never runs on the GPU
unless explicitly allowed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gideon.brain import llm
from gideon.brain.agent import Agent
from gideon.core.config import NUM_CTX, OLLAMA_OPTIONS
from gideon.storage.db import get_db


@pytest.fixture
def unplugged(monkeypatch):
    monkeypatch.setattr(llm, "on_battery", lambda: True)


@pytest.fixture
def plugged_in(monkeypatch):
    monkeypatch.setattr(llm, "on_battery", lambda: False)


# ---------------------------------------------------------------------------
#  Where the model runs
# ---------------------------------------------------------------------------
def test_mains_uses_the_gpu_as_before(plugged_in):
    assert llm.chat_options() == OLLAMA_OPTIONS


def test_battery_keeps_the_model_off_the_gpu(unplugged, monkeypatch):
    monkeypatch.setattr(llm, "BATTERY_MODE", "cpu")
    options = llm.chat_options()
    assert options["num_gpu"] == 0
    assert options["num_thread"] >= 1


def test_context_size_never_changes_with_power(unplugged, monkeypatch):
    """A different num_ctx would force a reload and break the token budgets."""
    for mode in ("cpu", "gpu"):
        monkeypatch.setattr(llm, "BATTERY_MODE", mode)
        assert llm.chat_options()["num_ctx"] == NUM_CTX


def test_gpu_mode_is_an_explicit_opt_in(unplugged, monkeypatch):
    monkeypatch.setattr(llm, "BATTERY_MODE", "gpu")
    assert llm.chat_options() == OLLAMA_OPTIONS


def test_overrides_do_not_undo_battery_placement(unplugged, monkeypatch):
    monkeypatch.setattr(llm, "BATTERY_MODE", "cpu")
    options = llm.chat_options(num_predict=16)
    assert options["num_gpu"] == 0 and options["num_predict"] == 16


def test_agent_passes_battery_options_to_every_call(unplugged, monkeypatch):
    monkeypatch.setattr(llm, "BATTERY_MODE", "cpu")
    chat = MagicMock(return_value=iter([{"message": {"content": "hi"}}]))
    Agent(get_db(), None, chat=chat).respond(get_db().create_session("t"), "hello")
    assert chat.call_args.kwargs["options"]["num_gpu"] == 0


def test_off_mode_never_calls_the_model(unplugged, monkeypatch):
    monkeypatch.setattr(llm, "BATTERY_MODE", "off")
    chat = MagicMock()
    reply = Agent(get_db(), None, chat=chat).respond(get_db().create_session("t"), "hello")
    chat.assert_not_called()
    assert "battery" in reply


# ---------------------------------------------------------------------------
#  The guard
# ---------------------------------------------------------------------------
def _ps(*models):
    return SimpleNamespace(models=[SimpleNamespace(model=m, size_vram=v) for m, v in models])


def test_gpu_resident_model_is_evicted(monkeypatch):
    fake = MagicMock()
    fake.ps.return_value = _ps(("llama3.2:latest", 2_588_346_940))
    monkeypatch.setattr(llm, "ollama", fake)

    assert llm.evict_from_gpu() is True
    fake.generate.assert_called_once_with(model="llama3.2:latest", prompt="", keep_alive=0)


def test_cpu_resident_model_is_left_alone(monkeypatch):
    """The CPU copy is what battery mode runs on; evicting it would reload
    the model on every poll."""
    fake = MagicMock()
    fake.ps.return_value = _ps(("llama3.2:latest", 0))
    monkeypatch.setattr(llm, "ollama", fake)

    assert llm.evict_from_gpu() is False
    fake.generate.assert_not_called()


def test_unplugging_evicts_and_announces(monkeypatch):
    evict = MagicMock()
    monkeypatch.setattr(llm, "evict_from_gpu", evict)
    monkeypatch.setattr(llm, "BATTERY_MODE", "cpu")
    events = []
    guard = llm.PowerGuard(publish=lambda kind, payload: events.append((kind, payload)))

    monkeypatch.setattr(llm, "on_battery", lambda: False)
    guard.check()                      # boot on mains: no event, nothing to evict
    evict.assert_not_called()
    assert events == []

    monkeypatch.setattr(llm, "on_battery", lambda: True)
    guard.check()                      # cable pulled
    evict.assert_called_once()
    assert events == [("power_changed", {"on_battery": True, "battery_mode": "cpu"})]


def test_booting_on_battery_evicts_on_the_first_check(monkeypatch):
    evict = MagicMock()
    monkeypatch.setattr(llm, "evict_from_gpu", evict)
    monkeypatch.setattr(llm, "BATTERY_MODE", "cpu")
    monkeypatch.setattr(llm, "on_battery", lambda: True)

    llm.PowerGuard().check()
    evict.assert_called_once()


def test_gpu_mode_does_not_evict(monkeypatch):
    evict = MagicMock()
    monkeypatch.setattr(llm, "evict_from_gpu", evict)
    monkeypatch.setattr(llm, "BATTERY_MODE", "gpu")
    monkeypatch.setattr(llm, "on_battery", lambda: True)

    llm.PowerGuard().check()
    evict.assert_not_called()


# ---------------------------------------------------------------------------
#  Thinking models on the fast path
# ---------------------------------------------------------------------------
def test_thinking_is_switched_off_only_for_models_that_think(monkeypatch):
    """Qwen3 reasons before every reply unless told not to; llama3.2 errors
    if told anything about thinking at all."""
    fake = MagicMock()
    fake.show.side_effect = lambda model: SimpleNamespace(
        capabilities=["completion", "tools", "thinking"] if "qwen3" in model else ["completion", "tools"]
    )
    monkeypatch.setattr(llm, "ollama", fake)
    monkeypatch.setattr(llm, "_thinking_support", {})

    assert llm.thinking_kwargs("qwen3:4b") == {"think": False}
    assert llm.thinking_kwargs("llama3.2:latest") == {}
    llm.thinking_kwargs("qwen3:4b")
    assert fake.show.call_count == 2  # cached per model
