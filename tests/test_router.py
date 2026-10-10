"""The tool router (F-11): offer the relevant few tools, not all of them."""

from __future__ import annotations

from gideon.brain.router import ToolRouter
from gideon.tools.registry import tools_for_channel

SCHEMAS = tools_for_channel("web")
NAMES = [s["function"]["name"] for s in SCHEMAS]


def _fake_embed(texts):
    """Deterministic stand-in: one dimension per keyword family."""
    families = ["weather", "remind", "volume", "music|song|play", "clipboard|copy", "system|ram|cpu|gpu"]
    vectors = []
    for text in texts:
        low = text.lower()
        vectors.append([1.0 if any(k in low for k in f.split("|")) else 0.0 for f in families] + [0.01])
    return vectors


def _names(selected):
    return [s["function"]["name"] for s in selected]


def test_disabled_router_offers_everything():
    assert ToolRouter(mode="off").select("what's the weather", SCHEMAS) == SCHEMAS


def test_relevant_tool_is_offered_and_the_list_shrinks():
    selected = _names(ToolRouter(top_k=3, mode="on", embed=_fake_embed).select("will it snow in Oslo?", SCHEMAS))
    assert "get_weather" in selected
    assert len(selected) < len(SCHEMAS)


def test_trigger_words_always_get_their_tool():
    """The embedding ranks; a tool's own keyword in the message is evidence."""
    router = ToolRouter(top_k=1, mode="on", embed=lambda texts: [[1.0, 0.0] for _ in texts])
    selected = _names(router.select("set the volume to 30", SCHEMAS))
    assert "control_volume" in selected


def test_selection_keeps_registry_order():
    """Small models are order-biased: play_youtube_music must stay ahead of
    control_media (F-12)."""
    selected = _names(ToolRouter(top_k=4, mode="on", embed=_fake_embed).select("play some music", SCHEMAS))
    assert selected == [n for n in NAMES if n in selected]
    if {"play_youtube_music", "control_media"} <= set(selected):
        assert selected.index("play_youtube_music") < selected.index("control_media")


def test_tool_embeddings_are_computed_once():
    calls = []

    def counting_embed(texts):
        calls.append(len(texts))
        return _fake_embed(texts)

    router = ToolRouter(top_k=3, mode="on", embed=counting_embed)
    router.select("weather please", SCHEMAS)
    router.select("remind me later", SCHEMAS)
    # First call embeds every tool, then one query per message.
    assert calls == [len(SCHEMAS), 1, 1]


def test_any_failure_falls_back_to_every_tool():
    def broken(texts):
        raise ConnectionError("ollama down")

    assert ToolRouter(top_k=3, mode="on", embed=broken).select("weather", SCHEMAS) == SCHEMAS


def test_auto_mode_follows_the_template(monkeypatch):
    """F-16: route for templates that re-read tools every turn (llama3.2),
    not for templates that cache them (Qwen)."""
    from gideon.brain import router as router_module

    monkeypatch.setattr(router_module, "template_caches_tools", lambda model: "qwen" in model)
    router = ToolRouter(top_k=3, mode="auto", embed=_fake_embed)
    assert router.active_for("llama3.2:latest") is True
    assert router.active_for("qwen3:4b-instruct") is False
    assert router.select("weather", SCHEMAS, "qwen3:4b-instruct") == SCHEMAS
    assert len(router.select("weather", SCHEMAS, "llama3.2:latest")) < len(SCHEMAS)
