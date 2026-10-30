# Gideon OS — Product Requirements Document (PRD)

**Version:** 0.2 (Draft)
**Date:** 2026-10-08 (v0.1: 2026-09-26)
**Status:** Interim build running on a laptop; harness work (build-order steps 1–3) in progress
**Scope:** Architecture, hardware, design principles, and — new in v0.2 — the current state of the build (§13) and how it maps onto the plan. Changes from v0.1 are listed in §19.

---

## 1. Vision

Gideon is a **personal, always-on, privacy-first AI assistant** in the spirit of Jarvis — a single conversational partner that controls, orchestrates, and reports on your digital life. The "OS" is not the product; it is the **delivery mechanism** that makes Gideon feel native, present, and fast.

The core promise:

- **One assistant, many devices.** One brain, many screens (client–server).
- **Local-first and private.** Nothing sensitive leaves the house by default.
- **Converse, don't navigate.** You ask; Gideon finds, does, and reports.
- **Trustworthy by construction.** Gideon never guesses on consequential actions.

### 1.1 What "OS" means here

"AI OS" can mean three different things; only the first two are in scope:

| Interpretation | In scope? | Meaning |
|---|---|---|
| **A. A Linux distribution with Gideon as the interface** | Yes (late) | Boot straight into Gideon; the normal desktop remains as a hidden fallback. Linux is the kernel, Gideon is the identity layer. |
| **B. An AI-first ambient environment** | Yes | Voice-primary, hologram interface, the agent operates your apps on your behalf. |
| **C. A true AI-native OS** (files, windows and apps are agent-managed objects) | No | Research-grade (e.g. AIOS). Not a solo project. |

The target is **A layered on B** — *an environment that feels like an OS*, built on an existing one, with a clean escape hatch back to a normal desktop. Writing a kernel is permanently out of scope.

---

## 2. Goals & Non-Goals

### 2.1 Goals

- Build a **fast, reliable conversational assistant** with dependable tool calling.
- Provide a **single unified interface** over media, files, services, and AI compute.
- Keep all data and inference **local** (offline-first).
- Support **multiple client devices** (server display, gaming rig, laptop) against one brain.
- Preserve a **durable harness + structured data** that survives model upgrades.
- Enable **Jarvis-style proactivity** to be added later without a rewrite.

### 2.2 Non-Goals (for now)

- Writing a custom kernel.
- Building the HUD / compositor / ISO before the assistant loop is proven.
- Supporting multiple independent primary users.
- Cloud dependency or third-party data processing.

---

## 3. System Architecture

### 3.1 The client–server model

Gideon is **not a single-PC OS**. It is **one brain on System 2, with thin clients everywhere else**.

```
        ┌──────────────────────── System 2 (Brain + Vault) ───────────────────────┐
        │  AI compute (orchestrator + reasoner)  │  Home server │  Media archive  │
        │  Memory / RAG  │  Harness / Broker  │  Hosting  │  ZFS storage          │
        └───────────────▲─────────────────────▲─────────────────────▲────────────┘
                        │ (LAN, private)       │                     │
          ┌─────────────┴──────┐   ┌───────────┴────────┐   ┌────────┴─────────┐
          │ Monitor on S2      │   │ System 1 (Gaming)  │   │ Laptop (client)  │
          │ (display + I/O)    │   │ thin Gideon client │   │ Gideon OS client │
          └────────────────────┘   └────────────────────┘   └──────────────────┘
```

### 3.2 System 2 — the brain and the vault

The single physical machine that runs **everything**:

- The **local AI models** (orchestrator + reasoner).
- The **home server** (hosting, services).
- The **media library** (movies, shows, music, games, personal/family archives).
- The **Gideon harness**, memory, and RAG store.
- The **Gideon OS** layer that wraps it all.

Runs 24/7. This is the only place running the full stack.

### 3.3 System 1 — the gaming rig

- A **separate, independent build** optimized purely for AAA gaming at high settings.
- Does **no heavy Gideon work**. Only runs a **light thin client** to query the brain.
- Fully decoupled so gaming and AI/server workloads never compete for resources.

### 3.4 Laptop

- Runs a **Gideon OS client** (HUD, voice, interface).
- Hardware only **displays and listens**; System 2 thinks.

### 3.5 Direct display on System 2

- A monitor + keyboard/mouse connected directly to System 2 (the server), presenting the **Gideon OS** interface.

### 3.6 Optional: dedicated "Gideon Terminal"

- A Raspberry Pi 5 or mini-PC as an always-on thin client (microphone, speaker, small screen) — a room-level endpoint for the brain. Optional and last; it adds nothing System 2 cannot do.

### 3.7 Interim development rig (today)

Until System 2 exists, everything runs on the **laptop**: Windows 11, GTX 1650 Ti (4 GB VRAM), 6-core CPU, `llama3.2` 3B via Ollama. It is a stand-in for the harness work, not the target. Two consequences shape the current code:

- **4 GB VRAM** — one small model at a time; no second "reasoner" model, no always-on vision.
- **Battery power cannot feed the GPU.** Running the model on the GPU while unplugged shuts the laptop down. On battery the model runs CPU-only (measured: 0 MiB VRAM, ~17.7 tok/s), and GPU-resident models are evicted when the cable comes out (§13.6).

---

## 4. Model Strategy — The Two-Brain Design

### 4.1 The key insight

Reasoning models are **worse** at the Jarvis job than instruct/agentic models: slow, verbose, and weak at tool calling — the exact thing Gideon depends on. Therefore: **two roles, not one.**

| Role | Purpose | Behavior | Model class |
|---|---|---|---|
| **Orchestrator (fast brain)** | Conversation, intent, tool calls, replies | Always on, ~1 s latency | Instruct / agentic (e.g., Qwen3-32B) |
| **Reasoner (deep brain)** | Hard thinking, planning, analysis | Asleep by default; woken on demand | Reasoning distill (e.g., R1-Distill-Qwen-32B, QwQ) |

**v0.2 note — roles, not necessarily models.** Hybrid models such as Qwen3 switch between a fast non-thinking mode and a thinking mode in the same weights. On a single 24 GB card that may be the better design: one resident model, two modes, no second load. The harness should therefore treat "orchestrator" and "reasoner" as **roles in config** that can map to two models *or* one model in two modes. Decide when System 2's GPU is bought.

### 4.2 Candidate models

**Orchestrator:**
- **Qwen3-32B** — recommended primary: strong tool calling, fast, fits one 24 GB card at 4-bit, easy to fine-tune.
- **Qwen2.5-72B-Instruct** — upgrade target for depth.
- **Llama 3.3 70B** — solid alternative.
- **Mistral / Command R+** — strong tool/RAG options.

**Reasoner:**
- R1-Distill-Qwen-32B or QwQ — invoked only for hard problems.

**Interim (laptop):** `llama3.2` 3B is in use; `qwen2.5:3b-instruct` is installed and is the first comparison to run once evals exist (§10).

> Re-check this list when the GPU is purchased — open models move fast and the trace-based eval (§6.3) is what should pick the winner, not this table.

### 4.3 Handoff rules (enforced in code, not by model self-report)

**Escalate to the reasoner when:**
- User explicitly requests deep thinking ("think hard," "plan this," "analyze this").
- Router labels the intent as *hard* (planning, debugging, math, multi-step).
- Orchestrator plan exceeds N steps (target: 3+).
- Tool results conflict or error.
- Requires interpretation of large datasets (e.g., "why did revenue drop?").

**Stay on the fast path when:**
- Simple lookups, reminders, watchlist, media queries, email drafts.
- Any single-tool-call-plus-sentence task.

### 4.4 Why this feels like Jarvis

- The **feeling is not the model** — it is the harness.
- Swap the model: ~10% improvement. Build the harness: the Jarvis feeling.
- Bright line: **fast path must complete under ~1 second**; humans feel tools slow above ~2 seconds.

### 4.5 Cloud models

A hybrid "local by default, cloud for hard tasks" design was considered and is **rejected by default**: it breaks the local-first promise. If ever added, it is opt-in per request and on the sacred list (data leaves the machine).

---

## 5. The Harness (the heart of the system)

### 5.1 Components

1. **Orchestrator (fast brain)** — voice of Gideon.
2. **Reasoner (deep brain)** — rare consultant.
3. **Harness / Broker (the hands)** — pure code. Holds tools, policy gate, and memory. **Both brains go through this. Neither gets a raw shell.**

### 5.2 Request flow

```
You speak
   │
   ▼
[Voice] → text
   │
   ▼
[Orchestrator]  ← memory / RAG
   │  "easy or hard?"
   ├── easy → pick tool → run → reply        (fast path, ~1s)
   └── hard → [Reasoner] → plan/answer → back to Orchestrator  (slow path)
   │
   ▼
[Policy Gate] → allowed? → yes: do it / no: ask you
   │
   ▼
[Tools] (read, act)
   │
   ▼
Reply + log to memory
```

### 5.3 The Broker — three jobs

1. **Validate** — are all required tool-slot fields filled? If not → Gideon **asks**.
2. **Policy** — is this a sacred action? If yes → **ask first**. If free → just do it.
3. **Log** — record request → intent → tool → result → **your correction**.

> Mental model: **the reasoner plans, the orchestrator executes, the broker allows.**

### 5.4 Tool schemas

Every tool is a typed schema: `name`, required fields, optional fields, returns.

Priority tools for v1:
- `set_reminder` — *exists*
- `draft_email` — drafting only; *sending* is a sacred action
- `add_to_watchlist`
- `get_project_metrics`

### 5.5 The "never assume" rule (enforced in code)

The model is **never permitted to invent a required slot value**. If a value is not in the current turn or retrieved memory, the slot is `null` → Gideon asks.

Three deterministic states:

| State | Definition | Action |
|---|---|---|
| **Under-specified** | A required slot is empty | **Ask** (always) |
| **Ambiguous** | Two valid interpretations, both complete | **Ask, offering candidates** |
| **Resolved** | All required slots filled from explicit context | **Act** |

Handling of missing fields:
- **Required missing → always ask.**
- **Optional missing → use a stated default** ("I've set it for 9am — change if you want").
- **Low-stakes irreversible-free action (watchlist) → just do it and report.**

> **Enforced since Phase 2:** the broker blocks any call whose required slot is missing or holds words the user never said (`tools/validation.py`), and invented optional values are dropped. The old `set_reminder` defaults ("in 1 day", unparseable time → +1 day) are gone; it asks instead.

### 5.6 Policy — the Sacred List

**Requires explicit confirmation, always, never voice-only:**
- Sending any email or message
- Any payment / financial action
- Deleting or overwriting files
- Changing system settings or installing software
- Running shell commands
- Anything that leaves the machine (see §16 for how this is interpreted)

**Gideon may do without asking:**
- Add to watchlist
- Set reminders / timers
- Read and summarize metrics, files, messages
- Retrieve and report

> Stored as an **auditable config file** the broker reads (editable without touching code).

Confirmation follows a **permission model like Android's**: start with harmless capabilities (launch apps, play music, read folder listings), escalate carefully, and allow **standing rules** ("always allow opening Notepad") once trust exists. An **off switch** — stop all agent actions — must always be one keypress away.

### 5.7 Clarification vs. Proactivity

These are **separate subsystems** and must be kept apart:

- **Clarification** happens *inside a turn*, while the user is present.
- **Proactivity** happens *unprompted* — Gideon must decide silently whether to speak.

Current stance: **Gideon does not interrupt; it responds.** Reminders the user explicitly set are *responses delivered later*, not interruptions, and are allowed. Everything else waits. The architecture must support later proactivity via:
- A **scheduler**
- An **event bus** — the system's nervous system: notifications, reminders, task progress, hardware events (battery, GPU temperature) and app events all flow through it, and the agent both listens and speaks on it
- A **relevance filter** ("is this worth saying unprompted, or should it wait for the morning briefing?")

Start at zero interruptions; lower the threshold as trust grows. The first proactive feature, when its time comes, is a **morning briefing** (weather, today's reminders and tasks, what changed overnight).

---

## 6. Memory, RAG & Data Strategy

### 6.1 Principles

- **RAG = knowledge.** Fine-tuning is bad at facts.
- **Fine-tuning = style/format.** Not facts, not capability at scale.
- **RAG beats fine-tuning for personalization**, decisively.
- Retrieval stays **outside the model, always**.

### 6.2 RAG layer

- Embed documents → store vectors → fetch relevant chunks at query time → inject into context.
- What *can* be fine-tuned is the model's **behavior around retrieval**: when to search, how to cite, how to say "I couldn't find it."
- Build **RAG first**; fine-tune **last**.
- Today's memory retrieves **past conversation messages** and **project facts** (§13.4). Document RAG and personal **fact memory** ("my sister is Priya", preferences) are still to build — projects are the first kind of fact and set the pattern: SQLite as the source of truth, a summary in vector memory, named facts injected directly.

### 6.3 Structured trace logger (from day one)

For every interaction, record:

`request → intent → slots filled → tool → outcome → user correction`

Plus the context needed to reproduce it: model and options used, tools offered, approval decisions, and per-stage latency (needed for the < 1 s target).

This one dataset serves three purposes:
1. **Eval suite** immediately (replay real cases across model changes).
2. **Future training data** (labeled and balanced).
3. **Regression safety net** when swapping models.

> Rule: **structured traces > raw logs.**

Correction signals to capture: denied approvals, the user rephrasing or retrying ("no, I meant…"), and an explicit thumbs up/down in the interface.

### 6.4 Fine-tuning strategy

- **Full fine-tuning of a 30B is out of reach** for an individual (≈480 GB of states at FP16). Use **LoRA/QLoRA only.**
- QLoRA on 30B needs roughly 24–48 GB VRAM and many GPU-hours.
- A 30B base is already good; fine-tuning **narrows** it toward your style/format — done badly it causes **catastrophic forgetting**.
- **Your fine-tune does not travel** across base-model upgrades; the **harness does**. Hence: build harness first.
- Realistic goal: **base model + LoRA (voice/format/tool-calling) + RAG (all facts)** — not "a brain trained on my life."

### 6.5 Decision rule

> **Finetune for *how* Gideon talks and *what shape* answers take. Retrieve for *what* Gideon knows. Evaluate with everything you capture.**

---

## 7. Interfaces & Sensory Integration

### 7.1 Visual Interface (the HUD)

- Tony Stark–inspired **holographic HUD**: transparent, interactive overlays.
- Rendering approach (candidate): a Wayland compositor + layer-shell clients; WebGL/Three.js for overlays.
- **Caution:** Three.js implies a browser/JS runtime. A native HUD is better served by **wgpu (Rust)** or **bgfx/raw Vulkan** rendered into a fullscreen client.
- **Do not write a compositor from scratch.** Build on **wlroots**, or run Hyprland/Sway + a **layer-shell HUD client**. For a first "boot to Gideon" proof, a **kiosk compositor** (`cage` or `labwc`) running the existing web HUD fullscreen is enough.
- The final HUD is a **deferred deliverable** — not built until the assistant loop is proven. (An interim web HUD exists today; §13.)
- Planned shell elements, when the time comes: command palette, widgets (time, weather, system), notification tray fed by the event bus.

### 7.2 Voice

- Always-on audio processing: **wake word → STT → orchestrator → TTS**.
- Components: VAD + wake word (e.g. openWakeWord), STT (whisper.cpp / faster-whisper class), TTS (Piper / Kokoro class — both already in use locally).
- Barge-in: the user can interrupt Gideon mid-sentence.
- **Latency target:** fast path under ~1 second end-to-end.
- **Voice = identification, not authentication.** Voice may answer "who is this," but consequential actions require a second factor and are restricted to the primary user.
- **Trusted others** are limited to low-stakes reads (timers, weather, playlist) with **no state mutation**.

### 7.3 Vision / Spatial Awareness

- Do **not** rely on screen capture + vision as the primary sense (lossy, slow, expensive).
- Prefer **structured window state** (`wlr-foreign-toplevel-management`) and **accessibility trees (AT-SPI)**.
- Use vision only as a **fallback** for apps with no accessibility tree.
- On current hardware, always-on vision is not feasible; treat vision as **load-on-demand**.

---

## 8. Hardware Specification

### 8.1 Constraints driving the build

- Target model: ~70B distill at 4-bit.
- Transcoding: up to **3 simultaneous 4K streams**.
- Always-on 24/7 server: power, thermals, and acoustics are first-class.
- Media + family archive storage is the **actual core** of System 2.

### 8.2 Model VRAM sizing (4-bit reference)

| Model | Weights | Realistic VRAM | Feel |
|---|---|---|---|
| R1-Distill-Qwen-32B | ~20 GB | 24 GB | Good, fast |
| R1-Distill-Llama-70B | ~40 GB | 48 GB | Better, usable |
| R1 full (671B MoE) | ~400 GB | 512 GB+ RAM offload | Slow (~few tok/s) — not target |

### 8.3 The critical design move

- The **LLM GPU** is idle most of the time and spikes when Gideon thinks.
- **Transcoding** happens exactly when the GPU would otherwise be free.
- **Do transcoding on the CPU iGPU (Intel QuickSync) or a cheap dedicated card** — far more power-efficient for 24/7, and it keeps the expensive AI GPU purely for Gideon.

### 8.4 Recommended Build — "Sweet Spot" (70B, 3× 4K)

**Compute**
- GPU: **2× RTX 3090** (48 GB total) — primary recommendation
  - Alternatives: **1× RTX 5090** (32 GB, partial offload) or **1× RTX 6000 Ada / used A6000** (48 GB, single card, quieter, less power per GB)
- CPU: **Ryzen 9 or Threadripper** (PCIe lanes matter for 2 GPUs + storage)
- RAM: **128 GB ECC** (target), **256 GB** if larger-model offload is desired

**Storage**
- **ZFS pool** (or RAID + checksums) for media + archive
- **6–8× large HDDs** + **NVMe special/cache vdev**
- ECC RAM strongly preferred
- **Backups:** 3-2-1 strategy mandatory

**Transcoding**
- **Intel Arc A310/A380** (cheap, low-power, AV1) or **Intel QuickSync iGPU**

**Power**
- PSU: **1000–1200W Gold/Platinum**, good efficiency at idle
- **UPS** for graceful shutdown / surge protection

**Motherboard & Expansion**
- Enough PCIe lanes for 2 GPUs + HBA/storage controller
- ECC support

**Networking**
- **2.5GbE minimum, 10GbE preferred** (streaming 4K + clients + backups)

**Cooling, Case & Acoustics**
- Adequate airflow for dual GPUs + stacked HDDs
- Quiet operation (machine runs 24/7 near users)

**Peripherals**
- Monitor + keyboard/mouse directly on System 2 (Gideon OS surface)

### 8.5 Alternative Tiers

**Budget / entry (32B distill)**
- GPU: 1× used RTX 3090 (24 GB)
- CPU: Ryzen 9 (or Intel with QuickSync)
- RAM: 64 GB
- Storage: 4× large HDDs in ZFS mirror/stripe + NVMe cache

**Serious (70B fast / flirt with larger)**
- GPU: 2× RTX 5090 (64 GB) or 4× 3090 (96 GB)
- RAM: 256 GB+
- Power and heat become the real constraint

### 8.6 System 1 — Gaming Rig

- Separate build, optimized for AAA gaming at high settings.
- Gaming-focused GPU; no AI/server role.
- Runs only a **thin Gideon client**.
- No shared dependencies with System 2.

### 8.7 Laptop

- Standard laptop running the **Gideon OS client**.
- No heavy inference, storage, or transcoding role.
- **Lesson from the interim rig:** a laptop's battery may not be able to feed its GPU under model load (§3.7). Clients must never assume local GPU inference is safe on battery.

### 8.8 Power Reality (24/7)

- Dual-3090 server idles roughly 150–250W; peaks past 700W.
- Over years, power and heat are real costs. Prefer **fewer, larger cards** over many small ones when possible.

### 8.9 Buying Strategy

- **Do not buy the AI GPU early** — fastest-depreciating part.
- Buy the **case, PSU, storage, motherboard foundation** first (age slowly).
- Purchase the **GPU last**, just before needed.

---

## 9. Security & Privacy

### 9.1 Threat model

- If Gideon can read screen/files/audio **and** execute, any **prompt injection** (webpage, email, indexed file) becomes a path to code execution as the user.

### 9.2 Defense: the Capability Broker

- Split capabilities into **observe** (read files, screenshot, window state), **mutate** (write files, change settings), **execute** (run commands).
- Route **every** tool call through the broker: enforce policy, log intent + diff, require confirmation for `mutate`/`execute` unless a standing rule covers it.
- **Taint tracking:** once outside content (web pages, documents, clipboard, command output) has entered a turn, `mutate`/`execute` tools in that turn require confirmation regardless of standing rules. Outside content is marked as data, not instructions, in the model's context. *(Implemented in v0.2 — §13.3.)*
- Run the orchestrator as a **dedicated unprivileged user** with a narrow IPC surface — **not a shell**.
- Make rollback **per-action** where possible (action log with inverse ops + snapshots around sessions), not only whole-FS time travel.
- File access is **sandboxed and read-only by default**; deletion and overwriting are sacred actions.

### 9.3 Secrets

- Connector keys (Stripe, analytics, etc.) stored in the **OS keyring / secret service**.
- **Never** in the model's context. Model sees **summarized results**, not keys or raw payloads.
- Use **restricted/read-only** credentials per connector.
- *(v0.2: secrets currently live in a `.env` file — to migrate.)*

### 9.4 Network & Remote Access

- **Local network:** fast and private.
- **Outside home:** require a **secure tunnel (WireGuard/VPN)** — otherwise the offline-first, private promise breaks.
- Remote access must never silently degrade privacy.
- Every client authenticates to the brain individually (per-device credentials), so one lost device can be revoked without re-keying the rest.

### 9.5 Voice Authentication

- Voice is **identification, not authentication** (spoofable with a recording).
- Consequential actions require a **second factor**.

---

## 10. Data Collection & Evaluation

- **Start collecting traces now**, even on weak hardware — data **compounds**, hardware waits don't.
- The dataset is hardware-independent; the OS work is not.
- Two independent decisions:
  - **The OS can wait.**
  - **The harness and data should not.**

### 10.1 First artifacts to build

1. **Tool schemas** — reminder, email draft, watchlist, metrics (required vs optional). *(Schema system exists; three of the four tools do not.)*
2. **Trace logger** — request → intent → slots → outcome → correction. *(Done in Phase 1: every turn traced with per-stage latency; corrections from 👍/👎, denied approvals and "no, I meant…" messages.)*
3. **Router** — small fast classifier (or regex/embedding) selects intent; ensures constrained tool calls (grammar/JSON-schema forced output). *(Phase 2: embedding router offers the ~5 relevant tools, 98% recall (FINDINGS F-14). Constrained JSON output not yet built.)*
4. **RAG prototype** — chunking + embeddings + retrieval on real data.
5. **Connector #1** — read-only, cached source for "how's my project doing" (the killer demo).

### 10.2 Evaluation

> Every measurement and lesson from the interim rig — and the checklist to work through when System 2 arrives — is kept in **`FINDINGS.md`**.

- Unit tests (about 410 today) exercise the harness with the model mocked out; they cannot tell whether the **model** chose well.
- A **real-model eval set** — prompts with the expected tool, arguments and clarification behavior — replayed against Ollama is required before any model or prompt change can be judged. It starts hand-written and grows from traces. *(Phase 1: 95 hand-written cases in `evals/cases.toml`, runner `python -m gideon.evals`, fake tools, per-model reports and comparisons. The same eval will run against Groq for 32B-class models.)*

---

## 11. Project Metrics Feature (Killer Demo)

### 11.1 Why it matters

A single question — "how's my project doing?" — returning **visitors, revenue, conversions, top pages, recent signups, error rates** in one spoken sentence is what earns the Jarvis comparison. This is an **integration problem**, not an ML problem.

### 11.2 Design

- **Connectors are the product.** Plausible/GA/Umami (visitors), Stripe/Paddle/LemonSqueezy (revenue), own DB (signups), Sentry (errors).
- **Read-only + cached, always.** Never give Gideon write access to billing.
- Pull on a schedule into a **local store** — fast, rate-limit-safe, key-safe.
- **Anomalies first, not raw numbers** — surface deltas and what changed ("revenue up 40% WoW, 12 signups overnight").

---

## 12. Deployment Strategy

### 12.1 Phase A — Live USB (testing)

- RAM-first architecture (Alpine diskless-style) on a pendrive.
- **OverlayFS** for persistence without burning flash.
- Keep the **index on tmpfs**; persist only raw files.
- Be realistic: always-on vision + STT + TTS + chat on CPU will not feel good. Preload only VAD/wake-word/STT + small router; load planner lazily.

### 12.2 Phase B — Bare metal (System 2)

- Compile environment into a **custom ISO** (e.g., Archiso, or Debian `live-build`).
- Boot path: autologin → kiosk compositor → Gideon fullscreen, with the standard desktop as a fallback session.
- **Snapshot-capable filesystem** for automated AI safety rollbacks — **Btrfs** (pragmatic) on root; ZFS for the media pool.
- Consider **NixOS** (atomic generations, trivial rollback) or a **ChromeOS-style A/B + dm-verity** scheme for immutability.
- Decide authority: **the FS rollback vs. the AI action rollback** — pick one as source of truth.
- **Pin versions.** Kernel, graphics-driver and compositor updates will break a kiosk setup occasionally; budget for it.

### 12.3 Build order (discipline)

| # | Step | Status (v0.2) |
|---|---|---|
| 1 | Tool schemas | **Mostly done** — typed `@tool` declarations; return types not yet declared |
| 2 | Broker (validate / policy / log) | **Done (v1)** — never-assume validation, `policy.toml`, approvals, claimed-action check, trace logging |
| 3 | Orchestrator + tools, **fast path only** (terminal or voice) | **Working, measured** — agent loop runs over web, quick window, Telegram and terminal; latency traced per stage; real-model eval in place |
| 4 | Add reasoner handoff | Not started (blocked on VRAM; design the role config first) |
| 5 | Add RAG | Message recall + project facts (first fact memory); document RAG not started |
| 6 | Add metrics connector | Not started |
| 7 | Build HUD / client | Interim web HUD exists; native HUD deferred |
| 8 | Build ISO / bare metal | Not started |

> **Do not touch the OS, HUD, or compositor until steps 1–3 feel genuinely useful.**

---

## 13. Current Capabilities

State of the interim build as of 2026-10-30 (Windows laptop, GTX 1650 Ti, `qwen3:4b-instruct` on Ollama 0.35.1, pinned — FINDINGS F-20). Code lives in the `gideon/` package; see `GIDEON_ARCHITECTURE.md` for internals and `FINDINGS.md` for every measurement.

### 13.1 Interfaces

- **Web HUD** — 3D hologram dashboard with streamed text and sentence-by-sentence speech.
- **Quick-ask window** — global **Ctrl+Alt+G**, silent, pre-warmed.
- **Tray** — starts at logon, supervises and auto-restarts the engine, "Start with Windows" toggle, Start Menu entry.
- **Telegram bot** — remote chat (user-ID allowlist; no shell or app-launch tools), replies stream in.
- **Terminal chat** (`main.py`).
- One shared agent behind all of them; one engine process (FastAPI) as the server.

### 13.2 Agent & tools

- 22 tools declared once each with `@tool` (schema, argument aliases, channel policy, confirmation flags, intent triggers):
  - **System:** status/telemetry, launch apps (built-ins + any Start Menu app; new windows; a project folder in VS Code or Explorer, in a new or the current window), file search, clipboard, volume, media keys, guarded shell.
  - **Information:** web search, weather, markets, Hacker News, IP lookup, Wolfram Alpha, document reading (PDF/text).
  - **Personal:** reminders (set/list), notes and to-do list (save/list/complete), YouTube Music.
- **Tool gating** by keyword triggers, plus an embedding **router** that offers only the relevant tools to models whose chat template re-reads the tool list every turn (llama); off for Qwen, whose template caches it (F-13, F-16).
- Cache-stable prompt layout for Ollama's prefix cache; one tool list per turn so retries reuse the cache (F-20); token budget measured from real schema size; history trimmed so tool output can never push the system prompt out of the window.
- History replays earlier tool calls with short, dated results (outside content is not repeated), so commands after a few chat turns still get tool calls (F-20, F-23).
- **No false "done":** until a tool has acted, a reply's opening is held back and checked for claims — promises, reported results, invented live facts or absences, false refusals. A claim is replaced by a JSON-schema-constrained decision (one tool or a question) or cut; after a decline it is only cut (F-12 to F-23).
- One "boss"/"sir" per reply, enforced on the stream (F-21).
- Workarounds for small-model failures: text-emitted tool calls are intercepted; "play X" is disambiguated deterministically.

### 13.3 Broker & security (partial)

- **Never-assume validation:** a required value that is missing, invented or only pointing words ("copy that") blocks the call and G.I.D.E.O.N asks; an invented optional value is dropped. A value counts as the user's when they said it — shortenings, words run together and close typos included (F-15, F-17, F-18, F-23).
- **Policy file** (`policy.toml`): each tool's class (observe / mutate / execute), approval rule and allowed channels; a malformed file stops the engine.
- **Approval queue:** the shell always waits for the user's click; app launching waits whenever outside content entered the turn (taint tracking). Approve/Deny cards in HUD and quick window, y/N in the terminal. Requests expire; replays are refused.
- Outside content is fenced as data; forged fence markers are neutralised.
- Irreversible-command denylist (including encoded/obfuscated execution); shell audit log.
- Loopback-only server with API token and DNS-rebinding protection.

### 13.4 Memory

- SQLite: sessions, messages, profile, reminders, notes, approvals, audit.
- ChromaDB vector recall of past messages (filtered on write, distance-thresholded on read), embedder pinned to CPU.
- **Fact memory, first kind: projects.** The user's code projects (search folders + VS Code history) with their state — kind, git branch, uncommitted changes, last commit, last opened — in SQLite and as recallable summaries; refreshed every 30 minutes. A project named in a message reaches the model as a known fact.
- Sessions auto-titled after the first exchange.

### 13.5 Reminders & delivery

- Delivered to HUD (spoken) and Telegram; reminders missed while offline are summarised in one digest instead of firing in a burst.
- Optional always-on Cloudflare Worker fires Telegram reminders while the laptop is off (exclusive per-reminder ownership, no double sends).

### 13.6 Runtime & power

- Engine autostart (boot task, needs admin once) and tray autostart (logon).
- **Battery safety:** on battery the model runs CPU-only and GPU-resident models are evicted within 5 s of unplugging; the HUD throttles its hologram. Optional "pause the model on battery" mode.
- Model warm-up at engine start on mains power (system prompt pre-filled).
- Local TTS (Kokoro, Piper fallback); cloud TTS off by default.

### 13.7 Engineering

- 618 unit tests (model mocked), ruff + mypy clean, pre-commit hooks; tests run against an isolated data directory.

### 13.8 Measurement

- Every turn traced (§6.3) with per-stage latency and correction signals; 👍/👎 in the HUD.
- Real-model eval: 108 cases in 7 categories (tool choice, slots, chat, clarify, media, projects, history), fake tools, isolated data, reports and model comparisons. Baseline on the restored runtime (2026-10-30, `evals/reports/20261030-111552-*`): **95.4% (103/108)**, first word 1.8 s median / 4.2 s p90, generation 37.7 tok/s (F-24).
- Found and fixed through measurement: the prompt-cache layout never actually reused Ollama's cache (the live context was hoisted into the system block by the chat template). Turns now reuse the cached prefix — prompt processing ~7.5 s → ~0.5 s on CPU.
- Measured cost of offering all tools: ~2,300 extra prompt tokens on every tool-enabled turn with llama3.2 (its template places tool schemas in the last user message, so they are never cached).
- Found through measurement: Ollama 0.40.x's llama.cpp runs ~6× slower on this GPU (no tensor cores); pinned to 0.35.1 — prompts 178 tok/s, generation 46 tok/s (F-20).

### 13.9 Not yet built

Standing approval rules · speech input (STT/wake word) · document RAG and personal fact memory (people, preferences) · reasoner role · `draft_email`, `add_to_watchlist`, `get_project_metrics` · LAN thin-client auth · keyring secrets · Linux port.

---

## 14. Success Metrics

- **Fast-path latency:** < 1 second end-to-end (voice).
- **Tool-call accuracy:** high enough that the user trusts actions without double-checking.
- **Clarification behavior:** zero invented required slots.
- **Sacred-list compliance:** 100% confirmation on sensitive actions.
- **Trace coverage:** every interaction logged with correction signal.
- **Model-swap portability:** swapping models requires config change, not rewrite.

---

## 15. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 3B/weak-model tool hallucination | Unusable assistant | Router + constrained decoding + deterministic fallback. Observed on the interim rig: claiming an action happened when no tool ran; paraphrasing slot values; inventing paths. |
| Prompt injection → code execution | Severe security | Capability broker, taint tracking, unprivileged orchestrator, policy gate |
| Fine-tuning overfitting / wasted effort | Time lost, model worse | RAG first; fine-tune last, style/format only |
| Waiting for hardware = never starting | No data, no progress | Start harness + trace collection now |
| Fine-tune doesn't port across models | Wasted investment | Invest in harness, not weights |
| "Gideon controls everything" = raw root | Security disaster | Capabilities through a gate, not a shell |
| Proactivity becomes nagging | Assistant muted in days | Separate subsystem, relevance filter, quiet hours, batching |
| Power/heat/noise 24/7 | Cost + annoyance | Efficient components, transcoding offload, good cooling |
| **Scope creep** ("it's an OS!") | Project never ships | The build order in §12.3; v1 is a useful assistant, not a distro |
| **Platform lock-in** to the interim OS | Rewrite when moving to Linux | Keep OS-specific tools behind a platform layer; harness code stays portable |
| **Laptop power delivery** | Machine shuts down under GPU load on battery | Power-aware model placement (§3.7) |

---

## 16. Open Questions

1. **Timeline to System 2** — months or years? (Determines how long the interim rig carries the work.)
2. **Which connector is source #1** — Stripe, analytics, or own DB?
3. **Budget ceiling** for System 2.
4. **Exact live-USB role** — pure test rig or eventual portable OS?
5. **Reasoner trigger tuning** — what counts as "hard" in practice?
6. **Where the "brain" (preferences, relationships, history) lives** and how it is backed up separately from the OS.
7. **Data-at-rest policy** — encryption, auditability, deletion of captured interaction traces.
8. **"Anything that leaves the machine"** — does it cover outbound *lookups* (web search, weather, market quotes) or only *the user's data* (messages, emails, files)? Working assumption: only the user's data.
9. **Telegram and the Cloudflare reminder node** route data through third parties (messages; reminder text). Keep as deliberate exceptions, or replace with WireGuard + a self-hosted channel when System 2 exists?
10. **Two models or one hybrid model** for the orchestrator/reasoner roles (§4.1)?

---

## 17. Design Principles (Summary)

- **RAG > fine-tuning** for personalization.
- **Structured traces > raw logs** for training.
- **Small useful loop > impressive OS** for progress.
- **Harness > model** for the Jarvis feel.
- **Never assume** — ask when unsure (enforced in code).
- **Capabilities through a gate**, never a raw shell.
- **One brain (System 2), many thin clients.**
- **Local-first, private by default; tunnel for remote.**
- **Upgrade System 2 once → every device gets smarter.**
- **Build an environment that feels like an OS — never an OS.**

---

## 18. Beyond v1 — Toward the Full Jarvis

These are **not** in the v1 build order (§12.3) and are not built until steps 1–3 are genuinely useful. They are written down now for one reason: so the harness is **designed with hooks for them**, the same way §5.7 keeps proactivity possible without building it. Each item lists the hook that must exist early.

### 18.1 Home automation — Gideon runs the house

- Integrate through **Home Assistant** (local, open-source, runs on System 2), not device-by-device. Gideon reads entity state and calls services; Home Assistant owns the devices.
- Capability classes apply unchanged (§9.2): reading a sensor is **observe**; lights and climate are **mutate**; **locks, alarms, garage doors and cameras are sacred** — always confirmed, never voice-only.
- Home events (door opened, motion, smoke, power cut) arrive on the **event bus**, where the proactivity rules decide whether they are worth saying.
- **Hook now:** keep tools grouped by capability class and keep the event bus generic, so a connector can publish device events without special cases.

### 18.2 Background tasks — work that outlives the conversation

- Jobs that keep running after the user stops talking: "watch this price and tell me if it drops below X", "research this tonight and have a summary in the morning", "keep an eye on the build".
- Each task is a **persisted record** (goal, schedule or trigger, steps taken, status, result) that survives a restart — the same way reminders do today.
- Every step goes through the broker exactly as a live turn would. A background task **never gains permissions** it would not have in conversation; a sacred step waits for the user's approval like any other.
- Results are delivered as **responses**, not interruptions — at the next interaction or in the morning briefing — unless the user asked to be told immediately.
- **Hook now:** keep the agent loop callable without a live user (it already serves web, Telegram and the terminal through one entry point), and log background steps in the same trace format.

### 18.3 Mobile client — Gideon away from home

- A phone client (installable web app first; native only if needed) that talks to System 2 over **WireGuard** — the same API every other thin client uses, with its own revocable device credential (§9.4).
- Push notifications for reminders and approvals, so an approval request can be answered from the phone.
- This is the path to retiring the third-party channels (Telegram, the Cloudflare reminder node — §16 Q9).
- **Hook now:** per-device credentials and an API that does not assume a browser on the same machine.

### 18.4 Presence & context — knowing where you are and what you are doing

- Which device the user is at (last active client), whether they are home (phone on Wi-Fi, Home Assistant presence), and whether they are busy (in a meeting, gaming on System 1).
- Used to **route** replies and notifications to the right place, and to stay **quiet** when it would intrude (gaming, meetings, night).
- Presence data is personal: stored locally, never sent out, and deletable (§16 Q7).
- **Hook now:** every request records which client it came from; delivery picks a target instead of broadcasting to everything.

### 18.5 Learning from corrections — getting better from use

- Traces (§6.3) already capture corrections. This closes the loop **without fine-tuning**:
  - A correction that states a fact ("no, my sister's name is Priya") is written to **fact memory**.
  - A correction to a tool choice or slot is added to the **eval set**, so the same mistake is caught on the next model or prompt change.
  - Repeated approvals of the same action become a **suggested standing rule**, which the user accepts or rejects (§5.6) — never adopted silently.
- **Hook now:** the trace logger stores the correction alongside the turn it corrects.

### 18.6 Personality — sounding like Gideon

- Today's personality is a handful of prompt rules (forms of address, brevity). A consistent voice — dry, concise, unflappable — is a **style** concern and therefore the job of the LoRA described in §6.4, trained on traces the user has rated well, and done last.
- Personality never overrides the harness: a witty reply that invents a slot value is still a failure.

### 18.7 Not planned

- **Physical sensing beyond Home Assistant** (always-on cameras, room-level tracking of people) — the privacy cost outweighs the benefit for now.
- **General intelligence.** Jarvis is fiction with a mind that out-thinks Tony Stark. Local 32–70B models are capable assistants, not that. The plan's bet is unchanged: the Jarvis *feeling* comes from speed, trust and context — the harness — not from the model.

---

## 19. Changes from v0.1

- **§1.1** — added the three meanings of "AI OS" and the A-on-B target (from the OSIdea brainstorm).
- **§3.6–3.7** — optional Gideon Terminal; the interim laptop rig and its VRAM/battery constraints.
- **§4.1, §4.2, §4.5** — roles vs. models (hybrid thinking models), interim model, rejected cloud hybrid.
- **§5.5–5.7** — flagged the current never-assume violation; shell commands added to the sacred list; Android-style permissions, standing rules and an always-available off switch; reminders clarified as allowed; event bus described as the nervous system; morning briefing named as the first proactive feature.
- **§6.2–6.3** — fact memory gap; trace fields extended (model, tools offered, approvals, latency) and correction signals listed.
- **§7** — kiosk compositor for the first boot-to-Gideon proof; wake-word/STT examples; barge-in; planned shell elements.
- **§8.7** — laptop battery lesson.
- **§9.2–9.4** — taint tracking (implemented), sandboxed read-only file access, `.env` secrets noted, per-device client credentials.
- **§10.1–10.2** — status per artifact; real-model eval set.
- **§12.2–12.3** — boot path, version pinning; build order now shows status.
- **§13** — authored (was a placeholder).
- **§15–16** — new risks (scope creep, platform lock-in, power delivery) and open questions 8–10.
- **§18** — new "Beyond v1" section: home automation, background tasks, mobile client, presence and context, learning from corrections, personality — each with the hook the v1 harness must leave for it.
- Formatting: inline LaTeX (`$$24/7$$`) replaced with plain text.
- **§6.2, §13.2, §13.4** (2026-10-12) — fact memory started with project knowledge; `launch_app` opens new windows and project folders.
- **§12.3, §13** (2026-10-30) — status refreshed: model, router, never-assume, policy file, false-claim handling, history replay, addressing rule, eval size and baseline, the pinned Ollama version; constrained decoding moved out of "not yet built".
