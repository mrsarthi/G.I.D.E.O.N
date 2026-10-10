"""
G.I.D.E.O.N Tool Router

Offers the model only the tools relevant to this message instead of all of
them (PRD §10.1: "router — small fast classifier selects intent").

Why (F-11, F-13): with llama3.2's chat template the full tool list (~2,300
tokens for 22 tools) is re-read on every tool-enabled turn — about 12 s of
prompt processing on the interim GPU — and a 3B model also picks worse from a
long list. Qwen templates cache a *stable* tool list instead, so for them a
list that changes per message can cost cache hits; whether the router helps
is decided per model from its chat template (`template_caches_tools`), and
the setting can force it on or off (`GIDEON_TOOL_ROUTER`). Measured: F-16.

How: each tool's name, description, parameters and trigger words are
embedded once with the memory embedder (nomic-embed-text, pinned to CPU);
each message is embedded and the closest tools are offered. Tools whose
trigger words appear in the message are always included, and the selection
keeps the registry's order (small models are order-biased — F-12's
play_youtube_music vs control_media). Any failure falls back to every tool.
"""

from __future__ import annotations

import math
import re
import threading
from typing import Any

import ollama

from gideon.brain.llm import template_caches_tools
from gideon.core.config import (
    EMBED_KEEP_ALIVE,
    EMBED_OPTIONS,
    EMBEDDING_MODEL,
    ROUTER_TOP_K,
    TOOL_ROUTER,
)
from gideon.core.logs import get_logger
from gideon.tools.registry import get_tool

log = get_logger("gideon.brain.router")

_WORD = re.compile(r"[a-z']+")


def _tool_text(schema: dict[str, Any]) -> str:
    """What a tool is *for*, as text to embed."""
    fn = schema["function"]
    params = fn.get("parameters", {}).get("properties", {})
    spec = get_tool(fn["name"])
    triggers = " ".join(sorted(spec.triggers)) if spec else ""
    param_text = "; ".join(f"{name}: {p.get('description', '')}" for name, p in params.items())
    return f"{fn['name'].replace('_', ' ')}. {fn['description']} {param_text} Keywords: {triggers}"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class ToolRouter:
    def __init__(self, top_k: int = ROUTER_TOP_K, mode: str = TOOL_ROUTER, embed: Any = None) -> None:
        self.top_k = top_k
        self.mode = mode
        # Injectable so tests need no Ollama.
        self._embed = embed or self._ollama_embed
        self._tool_vectors: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _ollama_embed(texts: list[str]) -> list[list[float]]:
        response = ollama.embed(
            model=EMBEDDING_MODEL, input=texts, options=EMBED_OPTIONS, keep_alive=EMBED_KEEP_ALIVE
        )
        return response["embeddings"]

    def _vectors_for(self, schemas: list[dict[str, Any]]) -> dict[str, list[float]]:
        """Tool embeddings, computed once per tool and cached."""
        with self._lock:
            missing = [s for s in schemas if s["function"]["name"] not in self._tool_vectors]
            if missing:
                vectors = self._embed([_tool_text(s) for s in missing])
                for schema, vector in zip(missing, vectors, strict=True):
                    self._tool_vectors[schema["function"]["name"]] = vector
            return {s["function"]["name"]: self._tool_vectors[s["function"]["name"]] for s in schemas}

    def active_for(self, model: str | None) -> bool:
        """Whether routing helps this model (see config TOOL_ROUTER)."""
        if self.mode != "auto":
            return self.mode == "on"
        return model is None or not template_caches_tools(model)

    def select(
        self, message: str, schemas: list[dict[str, Any]], model: str | None = None
    ) -> list[dict[str, Any]]:
        """The tools to offer for `message`, in registry order."""
        if not self.active_for(model) or len(schemas) <= self.top_k:
            return schemas
        try:
            tool_vectors = self._vectors_for(schemas)
            query = self._embed([message])[0]
        except Exception as exc:
            log.warning("Tool routing failed (%s); offering every tool", exc)
            return schemas

        ranked = sorted(schemas, key=lambda s: _cosine(query, tool_vectors[s["function"]["name"]]), reverse=True)
        chosen = {s["function"]["name"] for s in ranked[: self.top_k]}

        # A tool whose own trigger word is in the message is always offered:
        # the embedding is a ranking, the keyword is evidence.
        words = set(_WORD.findall(message.lower()))
        for schema in schemas:
            spec = get_tool(schema["function"]["name"])
            if spec and spec.triggers & words:
                chosen.add(spec.name)

        selected = [s for s in schemas if s["function"]["name"] in chosen]
        log.debug("Routed %d of %d tools: %s", len(selected), len(schemas), sorted(chosen))
        return selected
