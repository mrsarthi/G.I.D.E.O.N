# G.I.D.E.O.N — Findings Log

What we measured, what it means, and what we decided because of it.

**Why this file exists:** the interim laptop rig teaches lessons that are easy
to forget and expensive to re-learn. When System 2 (PRD §3.2) arrives, read
this file top to bottom before changing anything, then work through the
**System 2 day checklist** at the end.

**How to keep it useful:** add an entry whenever a test, trace or eval tells
us something we did not know. Newest findings go at the top of §2. Give each
one a date, the evidence (numbers, not impressions), and what changed because
of it. Superseded findings stay, marked as such — the history is the point.

---

## 1. The interim rig (the hardware these numbers come from)

| Part | Detail |
| :--- | :--- |
| Machine | Windows 11 laptop |
| GPU | NVIDIA GTX 1650 Ti, **4 GB VRAM** |
| CPU | 6 cores / 12 threads |
| Ollama | 0.35.1 |
| Default model | `llama3.2:latest` (3.2B, Q4_K_M, 2.0 GB) |
| Context | `num_ctx` 8192 |

Every latency below is from this machine unless stated. Absolute numbers will
not transfer to System 2; **ratios and causes will**.

---

## 2. Findings

### F-13 · 2026-10-09 · Qwen templates cache the tool list; llama's never does — 5–13× faster tool turns
**Evidence** (same GPU, same 22 tools, eval traces): median time to the first
words on turns that called a tool —

| Model | Where the template puts tools | First text, tool turns (p50) |
| :--- | :--- | :--- |
| llama3.2 | inside the **last user message** | **13.7 s** |
| qwen2.5:3b-instruct | in the **system block** (cached) | **1.05 s** |
| qwen3:4b-instruct | in the **system block** (cached) | **2.7 s** |

**Meaning:** F-11's 12-second tool cost is a property of llama3.2's template,
not of tool calling itself. With a template that caches the tool list, the
cost is paid once, not every turn.
**Decision:** template placement of tools is now a first-class criterion when
choosing a model (System 2 checklist, step 4). The router (F-11) still
matters for accuracy and context budget, but for Qwen-family models it is no
longer the main latency fix.

### F-12 · 2026-10-09 · Valid three-way bake-off (supersedes F-05)
Run with the F-09 prompt layout and the non-thinking Qwen3 (F-10). GPU,
plugged in, 95 cases, fake tools, `temperature 0`, `seed 42`.
Reports: `evals/reports/20261009-12*`.

| | llama3.2 | qwen2.5:3b-instruct | qwen3:4b-instruct |
| :--- | :--- | :--- | :--- |
| **Overall** | **73.7%** | 52.6% | 66.3% |
| Tool selection | **89.4%** | 39.4% | 59.1% |
| Slots filled correctly | **82.5%** | 37.5% | 67.5% |
| Chat — no tool when none needed | 80% | **100%** | **100%** |
| Clarify — asks instead of guessing | **0%** | **70%** | 50% |
| Media (play vs. press play) | 57% | 0% | 57% |
| Wrong tool chosen | 5 | 0 | **0** |
| Tool called during plain chat | 3 | 0 | **0** |
| Wrong slot value when it did act | 5 | 2 | **0** |
| "No tool" replies that *promised or claimed* the action | 0 | **31** | 10 |
| First text p50 (all turns) | 13.6 s | 1.0 s | 1.6 s |
| Whole turn p50 | 14.1 s | 1.9 s | 3.2 s |

**Meaning — each model fails differently:**
- **llama3.2** acts readily and usually correctly, but **never asks** (0/10 —
  the PRD's never-assume rule, §5.5), sometimes calls tools during plain chat,
  occasionally invents values ("put on some Arctic Monkeys" → `query:
  "Starboy by The Weeknd"`), and is slow on every tool turn (F-13).
- **qwen2.5** is careful and fast but mostly **talks instead of acting**:
  31 of its 39 misses are "I will check your GPU temperature right away"
  with no tool call — the false-success failure.
- **qwen3:4b-instruct** is the most *precise*: when it acts it never picks the
  wrong tool, never fills a wrong value, never calls tools during chat. Its
  misses are almost all **not acting** (26), 10 of them promising the action.

**Decision:** keep llama3.2 as the default for now — it is the only one that
reliably acts. But qwen3:4b-instruct's failure mode (not acting) is exactly
what Phase 2's **claimed-action check** fixes (retry when a reply promises an
action no tool performed), while llama3.2's failure modes (guessing, wrong
tools, invented values) are harder. **Re-run this bake-off after Phase 2;
qwen3:4b-instruct is the expected winner** — more precise, asks more, and
5× faster on tool turns.

**Eval caveat found:** some llama "slot" failures are the scorer being too
strict — `control_media(action="pause")` is normalised to `play_pause` by the
tool itself, but the eval demands the literal value. Accept tool-level
aliases in the scorer before the next run.

### F-11 · 2026-10-09 · Offering all tools costs ~12 s per tool request on this GPU
**Evidence** (eval traces, llama3.2, GPU, plugged in):

| Turn | 1st model pass: prompt processing | 2nd pass (after the tool ran) |
| :--- | :--- | :--- |
| "check my ram usage" | 12.3 s | 1.1 s |
| "Remind me to drink water in 30 minutes" | 12.1 s | 1.0 s |
| "add buy milk to my to-do list" | 12.1 s | 1.0 s |

Model fully on the GPU (2,468 MiB of 2,468 in VRAM). The ~2,300 tokens of
schemas for 22 tools are re-read on every tool-enabled turn (see F-08), at
roughly 250 tokens/s of prompt processing on this card.

**Meaning:** the time to the first spoken word on a tool request (~13.5 s) is
almost entirely the tool list, not the model thinking or the tool running.
**Decision:** the Phase 2 router — offer the 4–5 relevant tools, not all 22 —
is the single biggest latency fix available on any hardware. Re-measure on
System 2: a 24 GB card processes prompts far faster, but the schemas will
still dominate the fast path's budget (PRD target: < 1 s).

### F-10 · 2026-10-09 · `qwen3:4b` is a thinking-only model; use `qwen3:4b-instruct`
**Evidence:** with `think=False`, `qwen3:4b` still wrote its reasoning into
the reply ("Okay, the user is asking…") and never called a tool; with
`think=True` it spent its whole 200-token budget thinking (949 chars) and
never reached the call. `qwen3:4b-instruct` called `get_system_status`
cleanly with no reasoning, with or without `think=False`.

**Meaning:** a model tag's default variant can change between releases.
"Supports thinking" in Ollama's capabilities does not mean thinking can be
switched off.
**Decision:** fast path uses instruct (non-thinking) variants. Thinking
variants belong to the future reasoner role (PRD §4). The agent sends
`think=False` to any model reporting the capability (`brain/llm.py`).
**System 2:** check the exact tag of any Qwen3 model before trusting it on
the fast path — the same trap applies to `qwen3:32b`.

### F-09 · 2026-10-09 · Where the live context goes decides whether qwen2.5 calls tools
**Evidence** (direct Ollama calls, "check my ram usage", all 22 tools):

| Live context placement | qwen2.5:3b result |
| :--- | :--- |
| As a system message | calls the tool ✅ — but breaks prompt caching (F-07) |
| As a **separate** user message before the request | **no tool**: "Yes boss, I will check your RAM usage for you." ❌ |
| **Appended** to the user's own message | calls the tool ✅ and caching still works |

llama3.2 tolerated the separate message, so only the Qwen models collapsed
(eval tool selection 29% for qwen2.5, from F-07's fix).

**Meaning:** prompt layout changes behaviour differently per model family;
a change that helps one model can silently break another.
**Decision:** live context is appended to the newest user message
(`brain/prompt.py`). Any prompt-layout change must be re-run through the eval
on **every** candidate model, not just the default.

### F-08 · 2026-10-09 · llama3.2's template keeps tool schemas out of the cache
**Evidence:** llama3.2's template renders the tool list *inside the last user
message* ("Given the following functions…"), so the ~2,300 schema tokens sit
after the cache-stable prefix and are re-processed on every tool-enabled
turn. (Qwen's templates put them in the system block, where they *are*
cached — measured in F-13.)

**Decision:** fewer tools per turn (router, Phase 2). Check where each new
model's template puts tools — a template that puts them in the system block
would make them cacheable, if the tool set is stable.

### F-07 · 2026-10-09 · The prompt-cache design never actually worked (fixed)
**Evidence:** traces showed every turn re-processing the whole prompt
(856 of 856 tokens on a plain "hi"). Cause: llama3.2's and Qwen's templates
have no slot for a mid-conversation system message — Ollama joins **all**
system messages into one block at the top, so the volatile live context
(clock, recalled memories) sat in the "stable" prefix.

After moving the live context to the end (final form: F-09), on CPU:

| | Prompt processing |
| :--- | :--- |
| Cold (nothing cached) | 7.5 s |
| Next turn, same minute | 0.44 s |
| Next turn, **after the clock changed minute** | 0.69 s |

**Meaning:** the old layout was a full cache miss whenever the minute
changed; turns are now 10–17× faster to start on CPU.
**System 2:** verify with traces (`prompt_ms`) that a new model's template
keeps the static system prompt cacheable.

### F-06 · 2026-10-09 · Ollama's prompt token count includes cached tokens
**Evidence:** `prompt_eval_count` stayed at 857–858 while `prompt_eval_duration`
fell from 7,501 ms to 442 ms.
**Decision:** judge cache reuse by **`prompt_ms`** in traces, never by the
token count.

### F-05 · 2026-10-09 · First bake-off was invalid (kept for the record)

| | llama3.2 | qwen2.5:3b | qwen3:4b |
| :--- | :--- | :--- | :--- |
| Overall | 63.2% | 45.3% | 32.6% |
| Tool selection | 71.2% | 28.8% | 7.6% |
| Chat (no tool) | 85% | 100% | 100% |
| Clarify (ask, don't guess) | **0%** | 50% | 70% |

**Why invalid:** qwen2.5 was broken by the live-context layout (F-09) and
qwen3:4b was the thinking-only variant (F-10). **Superseded by F-12.**
**Still valid from this run:** llama3.2 never asked a clarifying question
(0/10) — it always guessed. qwen2.5, when it skipped tools, *claimed the
action happened* ("I have set a reminder for you…") — the false-success
failure the Phase 2 claimed-action check targets.

### F-04 · 2026-10-09 · On battery, an eval case takes ~3 minutes
**Evidence:** CPU-only tool turns processed ~3,100 prompt tokens at ~19 tok/s
(≈ 163 s) before F-07's fix.
**Decision:** run evals plugged in. The runner records placement (`gpu`/`cpu`)
in every report; never compare latency across placements.

### F-03 · 2026-10-09 · Eval methodology (what the numbers mean)
- `python -m gideon.evals` — 95 hand-written cases in `evals/cases.toml`
  (tool choice, slot filling, chat without tools, clarification, media).
- **Fake tools**: calls are recorded and answered with canned results; nothing
  real runs. Throwaway data directory per run.
- Fixed decoding: `temperature 0`, `seed 42`, `num_predict 256`. Production
  uses the model's default temperature — evals measure decisions, not variety.
- A case passes only if the **first** tool call is right (and its slots match).
- Reports: `evals/reports/<time>-<model>.{json,md}`; `--models a,b,c` compares.

### F-02 · 2026-10-07 · CPU-only inference is usable and leaves the GPU idle
**Evidence:** `num_gpu=0` → 0 MiB VRAM, GPU ~0%, 17.7 tok/s generation
(llama3.2, 6 threads). Used for battery mode.

### F-01 · 2026-10-07 · This laptop powers off if the GPU runs the model on battery
**Evidence:** reported by the user; a hardware power-delivery limit.
**Decision:** on battery the model runs CPU-only and GPU-resident models are
evicted within 5 s of unplugging (`brain/llm.py`, `GIDEON_ON_BATTERY`).
**Lesson for clients:** never assume a laptop client can run GPU inference on
battery (PRD §8.7).

---

## 3. Decisions log (short form)

| Date | Decision | Because |
| :--- | :--- | :--- |
| 2026-10-07 | Model CPU-only on battery | F-01, F-02 |
| 2026-10-09 | Live context appended to the user's message | F-07, F-09 |
| 2026-10-09 | `think=False` for thinking-capable models on the fast path | F-10 |
| 2026-10-09 | Fast path uses instruct variants (`qwen3:4b-instruct`, not `qwen3:4b`) | F-10 |
| 2026-10-09 | Router (fewer tools per turn) is Phase 2's top priority | F-08, F-11 |
| 2026-10-09 | Judge cache reuse by `prompt_ms`, not token counts | F-06 |
| 2026-10-09 | Keep llama3.2 as default for now; re-run the bake-off after Phase 2, expecting qwen3:4b-instruct to win | F-12 |
| 2026-10-09 | Where a model's template puts tools is a model-selection criterion | F-13 |
| 2026-10-09 | Bigger-model evals on Groq (free, no card) / Kaggle; AWS credits as backup | Free plan blocks GPUs; user prefers no card |

---

## 4. Where to evaluate bigger models before System 2

| Venue | What | Cost | Notes |
| :--- | :--- | :--- | :--- |
| **Groq API** | `qwen/qwen3-32b` with tool calling | Free tier, no card | ~30 req/min, 1,000/day. **Synthetic eval prompts only — never personal data.** Not the same quantization as a home Q4 file. Speed is not representative. |
| **Kaggle Notebooks** | T4 ×2 = 32 GB VRAM, 30 h/week | Free, phone verification | Runs the same Q4 model file as a 24 GB home card — the realistic accuracy test. |
| AWS (backup) | `g5`/`g6`/`g6e` GPU instances | $81.90 credit, expires ~2027-01-12 | Account is on the Free plan, which blocks GPU instances; upgrading needs a card. |

---

## 5. System 2 day checklist

Work through in order. Record every result as a new finding.

1. **Baseline first.** Before changing anything, run the eval on the current
   laptop model and save the report — it is the "before" picture.
2. **Pick the orchestrator by eval, not by reputation.** Run
   `python -m gideon.evals --models <candidates>`. For each candidate:
   - confirm it is an **instruct / non-thinking** variant (F-10);
   - check `ollama show <model>` capabilities and template;
   - verify tools are actually called (no prose-only "I will check…", F-09).
3. **Verify prompt caching** on the new model: two turns a minute apart; the
   second's `prompt_ms` in traces must be a small fraction of the first (F-07,
   F-06).
4. **Check where the template puts tools** (F-08) and re-measure the cost of
   the tool list (F-11) on the new GPU.
5. **Measure the fast path against the PRD target** (< 1 s to first word):
   trace `first_text_ms` and `first_audio_ms` on real requests.
6. **Replay real traces**, not just `cases.toml`: every turn the user rated 👎
   or corrected is a regression case; turn them into eval cases first.
7. **Decide the reasoner design** (PRD §4.1): two models, or one hybrid model
   in two modes. Measure the thinking mode's latency and tool behaviour
   separately.
8. **Revisit battery assumptions** for any laptop client: it is a thin client
   now, so no local GPU inference at all (F-01, PRD §8.7).
9. **Re-check open questions** in PRD §16 that depended on hardware.

---

## 6. Open threads

- **After Phase 2:** re-run the F-12 bake-off (claimed-action check + router
  in place); decide whether qwen3:4b-instruct replaces llama3.2.
- Local models kept (2026-10-10): `llama3.2` (default), `qwen3:4b-instruct`
  (post-Phase-2 re-run), `nomic-embed-text` (memory). Removed to free disk:
  `qwen2.5:3b-instruct` (lost F-12), `qwen2.5-coder:3b`, `gemma4:e4b`.
  To reproduce F-12 exactly: `ollama pull qwen2.5:3b-instruct`.
- Scorer: accept tool-level argument aliases (e.g. `pause` → `play_pause`).
- Groq eval of `qwen3-32b` once the eval set is stable (needs a small adapter).
- Why prompt processing is only ~250 tok/s on the 1650 Ti (F-11): check
  whether `OLLAMA_FLASH_ATTENTION=1` / `OLLAMA_KV_CACHE_TYPE=q8_0` are
  actually set for the Ollama service (they are only set by `run.bat` for its
  own shell).
