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
| Default model | `qwen3:4b-instruct` (4.0B, Q4_K_M, 2.5 GB) since 2026-10-10; `llama3.2:latest` before |
| Context | `num_ctx` 8192 |

Every latency below is from this machine unless stated. Absolute numbers will
not transfer to System 2; **ratios and causes will**.

---

## 2. Findings

### F-24 · 2026-10-30 · Baseline on the restored runtime: 95.4%, after finding a regression I caused
The first full eval on Ollama 0.35.1 scored **79.6%**: only 22 of 76 tool
cases got a native first call (58 on 2026-10-27), 42 decisions had to be
forced, and replies opening "Yes boss / Certainly" rose from 12 to 45.
Several were invented and slipped past the detector ("your battery is at
87%", "opening dumbpipe…", made-up Hacker News headlines).

**Cause: my F-21 edit**, not the runtime. It replaced the live context's
4-line addressing block with *"Address: boss or sir, once, at the start of
the reply"*, and the model started replying instead of calling tools. A/B on
the same runtime (tool + projects, 26 cases):

| Live-context addressing | Score | Forced |
| :--- | ---: | ---: |
| "…once, at the start of the reply" (F-21) | 65% | — |
| Old 4-line block | 88.5% | 3 |
| **None** (RULE 7 in the cached system prompt + `brain/style.py`) | **88.5%** | **1** |

The line is gone. That also saves ~70 tokens per turn. The miss: after F-21
I ran only the unit tests. **Any change to text the model reads needs an eval
run before it ships.**

The detector also learnt the holes this run exposed: an action gerund after a
command ("opening dumbpipe…"), a reading of the user's machine with a number
and no tool ("your battery is at 87%"), and time-sensitive results presented
as fetched ("here are the top trending stories…").

**Baseline, `20261030-111552`** (108 cases, GPU, Ollama 0.35.1):

| Metric | Result |
| :--- | ---: |
| **Overall** | **95.4% (103/108)** |
| chat · clarify · history · media · projects | 100% each |
| slots | 95.0% |
| tool | 83.3% |
| Forced decisions | 6 |
| First text p50 / p90 | **1.8 s / 4.2 s** |
| Whole turn p50 / p90 | 2.5 s / 5.9 s |
| Generation | 37.7 tok/s |

Remaining failures are the long-standing model limits: Wolfram (2), "where
am I?", "turn it up" (gated), and "what's my CPU temperature?".

### F-23 · 2026-10-30 · "No its ok" → the same question again; "no meetings" without looking; a misspelled city → the wrong forecast
A real quick-ask morning, read from its traces:

- **"Do I have any meetings or pending tasks for today?"** No tool was
  called, yet the answer was "I don't see any scheduled meetings or pending
  tasks". Invented absence wasn't a known claim shape. It is now ("I don't
  see any / you have no / no … meetings, reminders, tasks, notes, items").
- **"No its ok"** (declining G.I.D.E.O.N's offer) came back as the same
  offer, which looked like lost context. The history was there. A replay of
  the turn (3 runs) showed the first pass sometimes claims "I've checked
  your schedule…", and the forced decision, whose only options are a tool or
  a question, re-asked. After a **decline** (`is_decline`: no / it's ok /
  never mind / thanks…, short, no action verb) a claim is now cut and never
  forced. Re-run: 3/3 without the repeat; one cut to "Let me know if you'd
  like to add anything."
- **A weather question with the city misspelled**: never-assume dropped
  the model's correctly spelled city as invented (a typo isn't a prefix), so
  the forecast came for the current location instead. The reply then
  quoted a 3-day-old temperature for the city, replayed in the history.
  Grounding now accepts close typos (≥5 letters, similarity ≥ 0.8: barcelone,
  madird; not linden → London), and replayed tool results carry their time
  ("[result from 2026-10-27 18:34] …").

### F-22 · 2026-10-30 · A greeting set off the false-claim check and pulled in the repo's facts
A probe of "good morning gideon" on the restored runtime returned a greeting
about the G.I.D.E.O.N repository ("latest updates committed on October
27th"), then a forced question ("What would you like me to do for you
today?"). Two causes:
- **"I'm ready" matched "read"**: the claim patterns allowed any ending after
  an action verb. Words that only start like one (ready, really, address,
  listen, settle…) are now excluded (`_NOT_ACTIONS`).
- **"gideon" named the G.I.D.E.O.N project**, so every message addressed to
  the assistant loaded the repo's facts into the prompt. A project named like
  the assistant now only counts when the message is about code (project,
  repo, branch, VS Code…), for vector recall too.

### F-21 · 2026-10-27 · "Boss, … sir": the addressing rule is now enforced in code
User report: G.I.D.E.O.N says "boss"/"sir" too often, often at both ends of
a reply. In the last 300 stored replies, **53 had two or more** and **103
ended with one**, although the rule ("at most once, near the beginning,
never at the end") was in the system prompt *and* in the live context every
turn.

`brain/style.py` now keeps one address, the first, and only in the first
sentence, and removes the rest as the text streams. So what is shown,
spoken and stored agree, and history no longer teaches the habit back.
Non-addresses stay ("my boss", "the final boss", "Sir Isaac Newton").
Streaming any chunking gives the same text as filtering the whole reply
(tested on 13 real-style replies, 25 random chunkings each).

The live context's 4-line addressing block (≈70 tokens, re-read on every
turn because it sits after the cached prefix) is one line now. At today's
GPU speed that alone saves ≈2 s per reply.

Latency seen by the user the same evening ("good evening gideon", 68.6 s):
9.8 s model load (an A/B test of mine had unloaded the model) and 56.3 s
reading a cold 1,668-token prompt at ~30 tok/s (the F-20 slowdown).

### F-20 · 2026-10-27 · The "7-second penalty" was a cache miss we caused; separately, the GPU runs at a tenth of its speed
**The penalty.** When the model answers a command in prose, the harness
forces a decision (F-15). Timing every call and reading Ollama's own log
(`llama-server` prints how many prompt tokens it actually *read*, not
restored from cache) showed where the time went:

| Call | Prompt | Re-read before | Why |
| :--- | ---: | ---: | :--- |
| Forced decision | 1,357 tok | **676** | sent with `tools=[]`: Qwen renders tools at the top, so the prompt differed from token ~680 on |
| Reply after the tool | 1,898 tok | **1,217** | sent with the router's 5 tools instead of the turn's 22: same effect |

About 1,900 tokens re-read per forced command. At this GPU's normal speed
(~250 tok/s, F-11) that is ≈7.5 s: the penalty F-17 measured.

**Fix: one tool list per turn.** The forced decision now sends the turn's
own tools and messages with the claim and the instruction appended. The
schema's enum still limits the choice to the router's candidates. The reply
pass keeps the same list too. Measured after (same history, real model):

| Call | Re-read after |
| :--- | ---: |
| Forced decision | **142–144** |
| Reply after the tool | **40–41** |

≈185 tokens instead of ≈1,900: ≈0.7 s instead of ≈7.5 s at normal speed.
The forced call is now traced as its own pass (`forced_candidates`), so the
cost shows up in traces and evals.

**Also found:**
- "Paused the music, sir." with no tool call: a bare past tense opening a
  sentence wasn't a known claim shape. It is now (it needs an object, so
  "Set theory is…" isn't caught).
- **History replays tool calls.** Same conversation (a weather turn and a
  "open spotify" turn, both done with tools), then a command, 2 runs each:

  | Command | Replay off (shipped until now) | Replay on |
  | :--- | :--- | :--- |
  | open calculator | 0/2 tool calls; "Calculator is open, sir." | 2/2 native |
  | Open notepad | 0/2; "Notepad is open, sir." | 2/2 native |
  | pause the music | 0/2; "Music paused, sir." | 2/2 forced (caught, then done) |

  With replay off, **all six replies were false and none was caught**. The
  patterns knew "is *now* open", not "is open" or "Music paused". The fix is
  that **after a command** (tools offered), any statement reporting a state
  (open, paused, playing, done, set…) counts as a claim; questions don't. In
  conversation the same words stay innocent ("the shop is open until 9").

**Eval, three full runs** (108 cases, 5 new `history` cases; on the slowed
GPU, so the latency numbers are not comparable with earlier runs):

| Run | Overall | chat | clarify | history | slots | tool | Forced calls |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `144906` (cache + replay + command claims) | 88.9% | 95% | 80% | 4/5 | 92.5% | 83.3% | 19 |
| `154127` (+ fixes below) | 86.1% | 100% | 70% | 4/5 | 90.0% | 72.2% | 16 |
| `163622` (+ fixes below) | **88.9%** | 95% | 80% | 4/5 | 92.5% | 83.3% | 16 |

Each run exposed false positives or holes in today's own changes:
- **Advice taken for a claim.** "Any tips for running?" offered tools (trigger
  words), and the tips were caught by the new "state words after a command"
  rule. That rule now needs an imperative request (`is_command`). Later the
  same answer was caught by "2. **Set** a consistent time" (bare past tense)
  and by "**Running** in the mornings…" (gerund). "Set" is out of the
  past-tense list, gerunds need an object, and in conversation only the
  opening is checked.
- **"Pause the music" became a web search about octopuses.** The forced
  decision's candidates were re-routed on the previous exchange (F-19). When
  tools were offered, it now chooses among those same tools; context routing
  is kept for turns where gating withheld them.
- **A premature claim beside a real tool call** ("Playing Arctic Monkeys
  now") was cut to "I haven't done that yet" and glued to the real answer.
  It is now dropped quietly.
- **`ask_user` without a question** left only "I haven't done that yet". The
  question is now required in the schema, with a plain fallback.
- **The request repeated as the value:**
  `set_clipboard(text="Copy that to your clipboard")` ran. A required value
  made only of the tool's instruction words (`Param.filler`, default
  `ground_ignore`) is now missing; `search_web` keeps "latest news" as a
  query.

After the last fix (advice patterns), `170139` re-ran the affected
categories: **chat 20/20**, clarify 80%, history 4/5, media 5/7.

Steady failures across all runs (model limits, not regressions): no tool for
the Wolfram cases (3), "where am I?", tech news, "play Arctic Monkeys" →
search instead of play, and two "which file/command?" clarify cases where
the model answers instead of asking.

**Next for the forced decision:** it often names the right tool with empty
arguments (`control_media({})` ×2 and `ask_user({})` ×2 in `170139`). The
broker then refuses and asks, which is safe but a wasted turn. A second
constrained call with the chosen tool's own parameter schema (required
fields, enums) would fill them for a few cached tokens.

**Separately, the runtime is ~8× slower than on 2026-10-12.** Uncached
prompt processing ~27–30 tok/s (was ~190–250), generation ~10.5 tok/s (was
23). Every penalty is multiplied by that. Investigated, on a temporary second
Ollama server so the real one's settings stayed untouched:

| Tried | Prompt tok/s | Gen tok/s |
| :--- | ---: | ---: |
| Current: flash attention on, KV q8_0, `cuda_v13` | 30 | 10.5 |
| Flash attention off, KV f16 | 28 | 4.0 (VRAM 3.9 GB, nearly full) |
| Flash attention on, KV f16 | 30 | 10.8 |
| `cuda_v12` library | 30 | 11.0 |
| CPU only (Vulkan request fell back) | 27 | 7.9 |

- **Not the settings:** flash attention and the q8 cache are fine; keep
  them (they save VRAM, and turning flash attention off made generation
  worse). This closes F-11's open thread: both are set for the user account.
- **Not throttling:** at "100% utilisation" the card draws ~15 W of its
  30 W limit, sits at 1,350 MHz with the memory clock at full speed, and
  reports no power, thermal or power-brake limit, only *GPU idle*. It is
  waiting, not working: the GPU does no better than the CPU.
- **Not the model file:** the Q4_K_M blob is unchanged since 2026-10-09.
- **What changed:** Ollama 0.40.4 (new `llama-server` engine) and four
  Windows updates, both on 2026-10-14. The laptop has not fully restarted
  since 2026-10-15: 12.4 days, **50 sleep/resume cycles**, and the NVIDIA
  driver logged an error (event 153) on 2026-10-15 09:43. Every fresh Ollama
  process tested today was equally slow, so the problem is below Ollama.
  Upstream llama.cpp couldn't be benchmarked: GitHub downloads ran at
  ~0.3 KB/s today.
- **Restart and update did not fix it.** The user restarted the laptop
  (2026-10-27 17:59), and Ollama's updater installed 0.40.11 at logon.
  Same numbers: prompts 29 tok/s, generation 11 tok/s.
- **Ruled out since:**
  - **sleep/resume state and Ollama 0.40.4:** both changed, nothing changed;
  - **batch size:** prompt speed is the same at batch 64 and 2048 (29
    tok/s). It does not scale with batching, as if each token went through
    alone;
  - **model-specific:** `llama3.2` is as slow (41 tok/s prompts, 15 tok/s
    generation, ~25 ms per token, in proportion to its size);
  - **the embedding table in system RAM:** pinned to the GPU with
    `LLAMA_ARG_OVERRIDE_TENSOR=token_embd\.weight=CUDA0` (Ollama passes
    `LLAMA_ARG_*` through to `llama-server`): one graph split, same speed;
  - **runtime compilation (PTX JIT):** with `CUDA_DISABLE_PTX_JIT=1` both
    bundled CUDA libraries still load and run, so both carry native sm_75
    code;
  - **Windows power mode and GPU scheduling:** "Best performance" on mains,
    scheduling at its default, NVIDIA driver unchanged since March.
- **Seen during runs:** the GPU alternates between busy (1,350 MHz, ~16 W)
  and fully idle (300 MHz, 3.8 W), and GPU discovery at Ollama start times
  out at 90 s before succeeding on retry.
- **Root cause: current llama.cpp on a GPU without tensor cores.** The
  upstream benchmark (b11919, run by the user) on the same file: **pp512
  32.3 tok/s, tg64 11.8 tok/s**, with the warning *"suboptimal performance
  due to a lack of tensor cores … compile with
  CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and GGML_CUDA_FORCE_MMQ to
  force the use of the Pascal code for Turing"*. The GTX 16 series is
  Turing (compute 7.5) **without tensor cores**, and current llama.cpp sends
  7.5 down its tensor-core path. Ollama switched to that engine with 0.40.x,
  installed 2026-10-14. Before that it was **0.35.1** (app log, 2026-10-03),
  whose older engine gave the F-11/F-16 speeds.
- **No runtime switch fixes it** (all verified on the GPU, 3.1 GB VRAM in use):

  | Back-end, setting | Prompt tok/s | Gen tok/s |
  | :--- | ---: | ---: |
  | CUDA as shipped | 30 | 10.4 |
  | CUDA, `GGML_CUDA_FORCE_MMQ=1` | 30 | 10.2 |
  | CUDA, `CUDA_FORCE_PTX_JIT=1` (older code) | 16 | 2.9 |
  | Vulkan, NVIDIA driver only, FA + q8 | 66 | 8.7 |
  | **Vulkan, NVIDIA only, FA + f16 KV** | **90** | 9.3 |
  | CPU only | 33 | 10.1 |
  | *Ollama 0.35.1, 2026-10-12* | *~190–250* | *~23* |

- **Vulkan hangs unless restricted to the NVIDIA driver:** the AMD iGPU's
  Vulkan driver (2021) hangs at start-up. That is also Ollama's 90-second
  GPU-discovery stall (`OLLAMA_VULKAN` is on). With
  `VK_DRIVER_FILES=…
v-vk64.json` it starts in ~10 s.
- **Fix:** roll Ollama back to **0.35.1** (`OllamaSetup.exe`, 1.58 GB) and
  turn off its automatic updates (Ollama's `auto_update_enabled` setting).
  Interim: Vulkan via Ollama (`OLLAMA_LLM_LIBRARY=vulkan` +
  `VK_DRIVER_FILES`) triples prompt reading but not generation.
- **For System 2:** prefer a GPU with tensor cores (RTX); current llama.cpp
  is not tuned for the GTX 16 series.
- **Resolved 2026-10-30: rolled back to Ollama 0.35.1, auto-update off.**
  Same benchmark: **prompts 178 tok/s, generation 46 tok/s**; the GPU boosts
  to 1,950 MHz at ~35 W, against 1,350 MHz at 15 W before. Correction to the
  cause above: 0.35.1 *also* runs `llama-server` and prints the same
  tensor-core warning, so the regression is in the newer llama.cpp bundled
  with 0.40.x, not a change of engine. The cached 0.40.13 update was deleted
  first (with the user's OK), or Ollama would have re-upgraded itself at
  start-up, as it did on 2026-10-27.
- **The 2-minute stall at every Ollama start:** GPU discovery waits ~120 s
  for the hung AMD Vulkan probe. The user's first message after the rollback
  sat 77 s in memory recall (embedding through Ollama), and the window gave
  up. `OLLAMA_VULKAN=0` (set for the user account) cuts discovery to **4 s**;
  CUDA, which is what Ollama uses here, is unaffected. It applies from the
  next Ollama start.

### F-19 · 2026-10-12 · Live red flag: "The sendme folder is now open in VS Code", and nothing opened
User session (quick-ask, 18:28):

| Turn | What went wrong |
| :--- | :--- |
| "Open sendme" | `launch_app(app_name="sendme", target=<sendme's path>)`; the launcher refused it as "not an app" instead of seeing a project with no app named |
| "Open it in file explorer" | No native call; the forced retry asked a confused question |
| "Vs code please" | The forced retry picked **`set_clipboard`** (routed on "Vs code please" alone, among all 22 tools). It was blocked, then the **next** pass said "The sendme folder is now open in Visual Studio Code" — spoken, false |

Root causes and fixes:
1. **The gate covered only the first answer.** Any block or refusal turned it
   off. It is now armed until a tool has **acted**: it ran and did not answer
   "Not …". Launcher refusals now all start with "Not opened:". It is also
   armed when gating withheld tools ("vs code please" offers none).
2. **A held claim with no decision left** (already forced once, or after a
   refusal) is cut sentence by sentence (`strip_claims`). The question that
   follows is kept; a bare claim becomes "I haven't done that yet — it didn't
   go through."
3. **Forced retry routing** now uses the last exchange plus the message, not
   the message alone.
4. **Launcher:** a project named as the app (with or without its path as the
   target) means "no app named" (remembered app, else ask). `file_explorer`
   and `vs-code` are normalised.

Replay of the same conversation (real model, real launcher, launches
recorded): no false claims in 2 runs. "Open it in file explorer" opens sendme
in Explorer; "open sendme in vs code" gives
`Code.exe "…\OPEN SOURCE\sendme"`.

Still weak:
- "Open sendme" alone gets a model-invented app ("vs code"), which
  never-assume blocks, so G.I.D.E.O.N asks; correct but roundabout.
- A bare follow-up ("Vs code please") can drop the folder from the decision:
  it then just launches VS Code, honestly reported.

Replay caveat: the Start Menu path (`os.startfile`) was not intercepted, so
one replay opened a real VS Code window. Replays must patch it too.

### F-18 · 2026-10-12 · Project knowledge and window control: 8/8 on the new cases, no regressions
Built because of the user's requests: "open a new instance of vs code",
"open sendme in vs code", "open dumbpipe in a new / this vs code window", and
"knowledge about my projects in permanent memory".

- **Fact memory, first kind: projects** (`storage/projects.py`). Projects
  come from the search roots (marker files) and VS Code's history
  (`workspaceStorage/*/workspace.json`, which names every folder ever opened).
  On this machine: **35 projects in 24 s** (26 from VS Code). Git status on
  big repos is most of that time, and it runs in the background every
  30 min. Stored in SQLite and as summaries in Chroma (`kind="project"`).
- **Named projects are injected directly, not left to vector recall.**
  "open dumbpipe" must bring dumbpipe's folder every time; vector recall only
  covers questions that name no project.
- **`launch_app` gained `target` and `window`.** The model picks meanings;
  `APP_PROFILES` maps them to fixed flags. VS Code runs as `Code.exe`
  (`--new-window` / `--reuse-window`), never through `code.cmd`. `window` is
  grounded by **evidence words** (new ← new/another/fresh/instance…, this ←
  this/current/same); invented values are dropped.
- **Eval:** 8 new `projects` cases (the user's own sentences, explorer, and
  two knowledge questions scored with `expect`). The eval database is seeded
  with three projects. Full run `20261012-095459`: **87.4% (90/103)**,
  projects 8/8, chat 100%. The original 95 cases: 86.3% (F-17: 84.2%).
- Grounding fixes found by the new cases:
  - "vscode" typed by the user ↔ "vs code" written by the model
  - pointing words ("that") were evidence, so `set_clipboard` could copy the
    model's own sentence "Yes boss, I'll copy that…"

  After both: clarify **80%** (best so far), projects 100%, slots 87.5%
  (subset run).

**Not done (next):**
- verify the window opened (count VS Code windows before and after)
- a deterministic fast path for "open X (in Y)"
- personal facts (people, preferences) in the same fact store

### F-17 · 2026-10-12 · Live: qwen3 reports results it never produced — "Notepad is now open", invented weather
User report: "open calculator" and "Open notepad" (quick-ask, 2026-10-10) got
"Yes boss. The calculator is now open." with nothing opened. The traces for
that whole session show **zero tool calls in 7 turns**. `launch_app` was
offered every time, and the weather answer ("clear, 24.6 °C, no chance of
rain") was invented too.

Three separate faults, each reproduced by replaying the turn through the real
agent with fake tools (scratch replay, 2–3 runs each):

1. **The claim detector only knew promises** ("I'll open…", "Opening…"), not
   reported results ("X is now open", "has been set"), nor invented live
   facts (weather, "chance of rain", "high of 28°C"). Nothing triggered the
   forced decision. Added both pattern families (`brain/intent.py`).
2. **Conversation history suppresses native tool calls.** The same prompt
   gave a native `launch_app` call 2/2 times with no history, and 0/6 times
   after three earlier plain-text answers. Memory recall is *not* the cause:
   same result with memory on or off. One long phrasing ("What does the
   weather look like in <city> today, also is there any chance of rain?")
   gets no tool call even without history (4/4). Phase 2's claim → forced
   decision catches all of these now, at ~7 s extra per affected turn.
3. **Never-assume rejected the model's shortening of the user's word**:
   user "open calculator", model `app_name="calc"`, blocked as ungrounded,
   so G.I.D.E.O.N asked "What app would you like me to open?". Grounding now
   accepts a 4+-letter prefix either way (`calc` ↔ `calculator`).

Also fixed the **spoken false claim** (F-16's known UX cost): when tools were
offered, the opening of the reply (first sentence ≥ 80 chars, max 240) is
held until checked. A claim is never shown; the forced call's real answer or
question replaces it, and it is kept out of the next pass's messages. Honest
replies lose at most one sentence of streaming, which sentence-level TTS
waits for anyway.

Replay after the fix: calculator 2/2, notepad 2/2, city weather 2/2 —
each a single truthful reply ("Calculator has been opened, sir."), the tool
actually called.

The eval run (`20261012-084712`, 78.9%) showed two more ways the model fakes
work when tools are offered:
- an offer instead of the call: "I can check your GPU temperature right
  now. Let me retrieve that…"
- a false refusal: "I don't have access to real-time weather data"

Both now count as claims. Grounding also accepts two spoken words run
together ("vs code" → `vscode`). A required value made only of pointing
words (`set_clipboard(text="That")` for "copy that") counts as missing.

| | F-16 (qwen3, no router) | **F-17 final (`20261012-090126`)** |
| :--- | :--- | :--- |
| Overall | 80.0% | **84.2%** |
| Tool selection | 80.3% | **86.4%** |
| chat | 100% | 100% |
| tool | 61.1% | **77.8%** |
| slots | 80.0% | **85.0%** |
| clarify | 70.0% | 60.0% → 70.0% after the pointing-word rule (subset re-run) |
| First text p50 | 1.6 s | 3.1 s (the held opening plus forced calls) |

Run-to-run variance on this GPU is real, despite temperature 0 and a fixed
seed: four cases flipped between runs with an unchanged first-pass prompt.
The prompt differs only in the date. Read single-case flips as noise.

**Open:** fault 2 means most tool turns after a few chat turns pay the
forced-decision latency. Worth a case family in `cases.toml` with history,
and a look at whether stored tool turns should be replayed in history.

### F-16 · 2026-10-10 · Phase 2 result: qwen3:4b-instruct without the router is best; the router is right only for llama
Full Phase 2 harness (never-assume, schema-constrained retry after a claimed
action, wider claim detector, policy file), each model with the router on
and off. GPU, 95 cases. Reports: `evals/reports/20261010-*`.

| | llama3.2 + router | llama3.2 | qwen3 + router | **qwen3 (no router)** |
| :--- | :--- | :--- | :--- | :--- |
| **Overall** | 77.9% | 77.9% | 74.7% | **80.0%** |
| Tool selection | 87.9% | 89.4% | 75.8% | 80.3% |
| Slots | 82.5% | 77.5% | 82.5% | 80.0% |
| Chat — no tool | 70% | 70% | **100%** | **100%** |
| Clarify — asks | 60% | 60% | 50% | **70%** |
| Media | **100%** | 86% | 71% | 86% |
| First text p50 | 4.6 s | 14.8 s | 4.8 s | **1.6 s** |
| Whole turn p50 | 5.3 s | 15.7 s | 6.9 s | **5.0 s** |

Against the Phase 1 baseline (F-12): llama3.2 73.7% → 77.9%, clarify 0% → 60%;
qwen3:4b-instruct 66.3% → **80.0%**, tool selection 59% → 80%.

**Meaning:**
- **The router is template-dependent, as F-13 predicted.** llama3.2: same
  accuracy, **3× faster** (it re-reads the tool list every turn). qwen3:
  slower *and* less accurate (a changing tool list breaks its cached prefix).
  llama's per-case results shuffled (6 lost, 6 gained) but the total held.
- **Constraining beats asking** (F-15): the forced retry turned qwen3's main
  weakness — not acting — into its 80%.
- llama3.2 still calls tools during plain chat 30% of the time; qwen3 never.

**Decisions:**
- Router mode **`auto`** (default): on when the model's template renders tools
  inside the conversation (llama), off when it renders them first (Qwen) —
  detected from the template (`brain/llm.py: template_caches_tools`).
- **Default model switched to `qwen3:4b-instruct`** (2026-10-10, user approved)
  — best accuracy, never calls tools during chat, asks most, fastest to first
  word.
- Known UX cost: when the forced retry fires, the model's original claim
  ("I'll set that…") has already been spoken before the real answer. Fixing
  that means holding back short replies until classified — deferred.

### F-15 · 2026-10-10 · Never-assume works on llama; asking politely does not work on qwen3
**Step 1 of Phase 2** — scorer fixed to judge values the way the tool receives
them, never-assume validation, and a *plain-text* nudge when a reply claimed
an action with no tool. No router. GPU, 95 cases.

| | llama3.2 before → after | qwen3:4b-instruct before → after |
| :--- | :--- | :--- |
| Overall | 73.7% → **77.9%** | 66.3% → 67.4% |
| Clarify (asks instead of guessing) | **0% → 50%** | 50% → 70% |
| Media | 57% → **86%** | 57% → 57% |
| Chat (no tool) | 80% → 75% | 100% → 100% |

**What happened to qwen3's claims:** the detector caught 8 claimed actions
and nudged — and qwen3 **repeated the same false claim word for word** ("Yes
sir, I've set a reminder… Yes sir, I've set a reminder…") without a tool.
It also **invented data** the detector did not cover: "Here are your current
reminders: 1. Meeting with the team at 3 PM…" with no tool call.

**Decisions:**
- The retry after a claimed action is now **schema-constrained** (Ollama
  `format`): the model must return one tool + arguments, or `ask_user` + a
  question — no prose possible (PRD §10.1 "constrained tool calls"). A forced
  call still passes validation, policy and approvals.
- The detector also catches "Here are your reminders / to-do list / notes…"
  and "I've made a note…".
- Live check, qwen3, four cases it failed: all four now right (two forced
  calls, one native call, one invented reminder time blocked → it asked
  "when would you like to be reminded?"). Cost: the forced retry adds a few
  seconds when it fires.
- Lesson: with small models, **don't ask — constrain.** Instructions in the
  prompt are suggestions; a JSON schema in the decoder is a guarantee.

### F-14 · 2026-10-10 · The embedding router almost never hides the right tool
**Evidence:** for the 66 eval cases that expect a tool, is that tool among the
ones the router offers? (nomic-embed-text on CPU, trigger-word tools always
included, 22 tools total)

| `top_k` | Right tool offered | Tools offered (avg) | Routing time |
| :--- | :--- | :--- | :--- |
| 3 | 65/66 (98%) | 3.3 | ~90 ms |
| 5 | 65/66 (98%) | 5.1 | ~80 ms |
| 7 | 66/66 (100%) | 7.1 | ~80 ms |

The single miss at 3 and 5 (`tricky-play-chat`) is a case where answering
without a tool also counts as correct.

**Meaning:** routing costs ~80 ms and removes ~75% of the tool schemas with
essentially no loss of the right tool. Its effect on the *model's* choices
and on latency per template is measured end-to-end in F-15.
**Decision:** router on by default with `top_k = 5` (margin over 3); the
claimed-action retry offers the full list in case the router was wrong.

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
| 2026-10-10 | Never-assume enforced in the broker; invented optional values dropped | F-12, F-15 |
| 2026-10-10 | After a claimed action, retry with schema-constrained output (don't ask — constrain) | F-15 |
| 2026-10-10 | Tool router on by default in `auto` mode: per model, from its template | F-14, F-16 |
| 2026-10-10 | **Default model is now `qwen3:4b-instruct`** (approved by the user) | F-16 |
| 2026-10-12 | Hold back a reply's opening until checked for false claims; detect reported results and invented live facts | F-17 |
| 2026-10-12 | Fact memory starts with projects: SQLite source of truth, summary in Chroma, named facts injected directly | F-18 |
| 2026-10-12 | Apps get meanings (target, window) from the model; flags come from fixed profiles; VS Code via Code.exe | F-18 |
| 2026-10-12 | False-claim gate armed until a tool acts (not just on the first pass); unreplaceable claims are cut | F-19 |
| 2026-10-27 | One tool list per turn; the forced decision extends the cached prompt | F-20 |
| 2026-10-27 | History replays tool calls (capped; outside content omitted; grounding from spoken text only) | F-20 |
| 2026-10-27 | Keep flash attention + q8_0 KV cache (A/B: no speed cost, less VRAM) | F-20 |
| 2026-10-30 | No addressing line in the live context; any change to text the model reads needs an eval run first | F-24 |
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

- Ollama is pinned at 0.35.1 with auto-update off (F-20). Before updating,
  check that a newer Ollama is fast on a GTX 16-series card (re-run the F-20
  benchmark) — or wait for System 2's GPU.
- A deterministic fast path for the commonest commands ("open X", volume,
  media) would skip the model's decision entirely: proposed in F-20, not built.
- Forced decisions with empty arguments (F-20): add a second constrained call
  for the chosen tool's own parameter schema.
- Grow `cases.toml` from real 👎 traces; today's eval is hand-written only.
- Local models kept (2026-10-10): `qwen3:4b-instruct` (default), `llama3.2`
  (previous default, kept for comparison), `nomic-embed-text` (memory). Removed to free disk:
  `qwen2.5:3b-instruct` (lost F-12), `qwen2.5-coder:3b`, `gemma4:e4b`.
  To reproduce F-12 exactly: `ollama pull qwen2.5:3b-instruct`.
- Groq eval of `qwen3-32b` once the eval set is stable (needs a small adapter).
