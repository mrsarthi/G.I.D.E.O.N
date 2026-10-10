"""
G.I.D.E.O.N Model Placement

Where the chat model runs, decided per request from the power state.

Measured on this machine (GTX 1650 Ti, 6-core CPU, llama3.2 3B Q4_K_M):
with num_gpu=0 the model holds 0 MiB of VRAM, the GPU stays idle, and it
still generates ~17.7 tokens/s on the CPU. On mains it runs on the GPU as
before. On battery, GIDEON_ON_BATTERY chooses:

    cpu  (default)  run on the CPU only — slower, never touches the GPU
    off             do not run the model at all until plugged in
    gpu             previous behaviour; only for machines whose battery copes

Changing num_gpu makes Ollama reload the model, so plugging in or unplugging
costs one reload (~5s) on the next request. That is the price of the switch.
"""

from __future__ import annotations

import asyncio
from typing import Any

import ollama

from gideon.brain.prompt import STATIC_SYSTEM_PROMPT
from gideon.core.config import (
    BATTERY_MODE,
    BATTERY_THREADS,
    CHAT_KEEP_ALIVE,
    CHAT_MODEL,
    OLLAMA_OPTIONS,
    POWER_POLL_SECONDS,
    WARM_UP_MODEL,
)
from gideon.core.logs import get_logger
from gideon.core.power import on_battery

log = get_logger("gideon.brain.llm")

BATTERY_PAUSED_REPLY = (
    "I'm running on battery, boss, so my language model is paused — loading it "
    "on the graphics card shuts this laptop down. Plug in and ask again. "
    "Reminders and notifications still work."
)


def model_paused() -> bool:
    """True when the model must not run at all (on battery, mode 'off')."""
    return BATTERY_MODE == "off" and on_battery()


def chat_options(**overrides: Any) -> dict[str, Any]:
    """Ollama options for the chat model under the current power state.

    num_ctx is never changed here: a different context size also forces a
    reload, and the token budgets elsewhere assume NUM_CTX.
    """
    options = dict(OLLAMA_OPTIONS)
    if BATTERY_MODE == "cpu" and on_battery():
        options["num_gpu"] = 0
        options["num_thread"] = BATTERY_THREADS
    options.update(overrides)
    return options


def evict_from_gpu() -> bool:
    """Unload the chat model if any of it is resident in VRAM.

    Called when the machine goes onto battery: a model loaded on the GPU while
    plugged in would otherwise stay there for the keep-alive window and run on
    the GPU at the next request. Returns True if something was unloaded.
    """
    try:
        on_gpu = [m.model or CHAT_MODEL for m in ollama.ps().models if m.size_vram]
        for name in on_gpu:
            ollama.generate(model=name, prompt="", keep_alive=0)
            log.info("Unloaded %s from the GPU (on battery)", name)
        return bool(on_gpu)
    except Exception as exc:
        log.warning("Could not check or unload GPU models: %s", exc)
        return False


class PowerGuard:
    """Keeps the model off the GPU while on battery, and says so when it changes.

    Polls rather than subscribing to Windows power broadcasts: the engine has
    no window to receive WM_POWERBROADCAST (it may run before anyone logs
    in), and psutil's check costs microseconds.

    Each poll on battery also evicts anything GPU-resident, not just the
    first one after unplugging — so a model loaded on the GPU by something
    else (an `ollama run` in a terminal, a request that started just before
    the cable came out) is still cleared within one interval.
    """

    def __init__(self, publish: Any = None) -> None:
        self.publish = publish
        self.on_battery: bool | None = None

    async def run(self) -> None:
        log.info(
            "Power guard started (on battery: model %s)",
            {"cpu": "runs on CPU", "off": "paused", "gpu": "stays on GPU"}[BATTERY_MODE],
        )
        while True:
            try:
                await asyncio.to_thread(self.check)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Power check failed")
            await asyncio.sleep(POWER_POLL_SECONDS)

    def check(self) -> None:
        battery = on_battery()
        if battery and BATTERY_MODE != "gpu":
            evict_from_gpu()
        if battery != self.on_battery:
            changed = self.on_battery is not None
            self.on_battery = battery
            log.info("Power: %s", "on battery" if battery else "on mains")
            if changed and self.publish is not None:
                self.publish("power_changed", {"on_battery": battery, "battery_mode": BATTERY_MODE})


# ---------------------------------------------------------------------------
#  Warm-up
# ---------------------------------------------------------------------------
# Ollama's Windows app starts at logon, but the boot task starts the engine
# before anyone logs in, so the model server is often not there yet. Keep
# trying quietly instead of failing once and leaving the first question cold.
_WARM_UP_RETRY_SECONDS = (15, 30, 60, 120)


def _load_model() -> None:
    # The system prompt is the byte-stable prefix of every real request, so
    # prefilling it here means the first question reuses this KV cache.
    ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": STATIC_SYSTEM_PROMPT}],
        options=chat_options(num_predict=1),
        keep_alive=CHAT_KEEP_ALIVE,
        **thinking_kwargs(CHAT_MODEL),
    )


async def warm_up() -> None:
    """Load the model once Ollama is reachable, unless on battery."""
    if not WARM_UP_MODEL:
        return
    attempt = 0
    while True:
        if on_battery():
            log.info("On battery: skipping model warm-up (it loads on first use)")
            return
        try:
            started = asyncio.get_running_loop().time()
            await asyncio.to_thread(_load_model)
            log.info(
                "Model %s warmed up in %.1fs",
                CHAT_MODEL,
                asyncio.get_running_loop().time() - started,
            )
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            delay = _WARM_UP_RETRY_SECONDS[min(attempt, len(_WARM_UP_RETRY_SECONDS) - 1)]
            if attempt == 0:
                log.info("Ollama not ready for warm-up (%s); retrying every %ds+", exc, delay)
            attempt += 1
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
#  Thinking models
# ---------------------------------------------------------------------------
_thinking_support: dict[str, bool] = {}


def thinking_kwargs(model: str) -> dict[str, Any]:
    """`think=False` for models that can think (e.g. Qwen3), nothing otherwise.

    The orchestrator is the fast path (PRD §4): a hidden reasoning pass before
    every reply multiplies latency, and on some setups leaks into the spoken
    text. Thinking belongs to the future reasoner role. Passing `think` to a
    model without the capability is an error, so it is only sent when Ollama
    reports the capability; the answer is cached per model.
    """
    if model not in _thinking_support:
        try:
            _thinking_support[model] = "thinking" in (ollama.show(model).capabilities or [])
        except Exception as exc:
            log.debug("Could not read capabilities of %s: %s", model, exc)
            return {}
    return {"think": False} if _thinking_support[model] else {}
