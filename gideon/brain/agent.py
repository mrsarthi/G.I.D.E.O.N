"""
G.I.D.E.O.N Agent

The one conversational loop every interface shares. The web API, Telegram,
the quick-ask window and the CLI all call Agent.stream / Agent.respond; none
of them talk to Ollama, the prompt builder or the tool registry directly.

Tool calls pass through one policy before they run:

* `confirm` tools (the shell) wait for the user's approval when
  GIDEON_REQUIRE_CONFIRMATION is on — the default.
* `privileged` tools (shell, app launching) wait for approval whenever a tool
  that returns outside content (web search, documents, clipboard) already ran
  in this turn. That content may carry instructions written by someone else;
  the user's click, not the model, is what lets it touch the machine.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any

import ollama

from gideon.brain.approvals import ApprovalQueue, action_payload
from gideon.brain.intent import (
    claims_action,
    correct_media_tool_choice,
    is_command,
    should_offer_tools,
    strip_claims,
)
from gideon.brain.llm import BATTERY_PAUSED_REPLY, chat_options, model_paused, thinking_kwargs
from gideon.brain.prompt import (
    build_context_window,
    fit_tool_result,
    format_messages_for_ollama,
    spoken_messages,
    trim_to_budget,
)
from gideon.brain.router import ToolRouter
from gideon.brain.toolcalls import (
    PSEUDO_CALL_CHAR_LIMIT,
    extract_chunk_content,
    extract_tool_calls,
    is_tool_call_candidate,
    json_start_index,
    parse_pseudo_tool_call,
)
from gideon.brain.traces import TraceRecorder, TraceStore, ollama_stats
from gideon.core.config import (
    CHAT_KEEP_ALIVE,
    CHAT_MODEL,
    MAX_TOOL_TURNS,
    REQUIRE_CONFIRMATION,
)
from gideon.core.context import current_channel, current_session
from gideon.core.logs import get_logger
from gideon.storage.db import DatabaseManager
from gideon.storage.memory import VectorMemory
from gideon.tools.registry import dispatch_tool_call, get_tool, tools_for_channel
from gideon.tools.validation import check_call

log = get_logger("gideon.brain.agent")

OFFLINE_REPLY = (
    "I've lost contact with my local inference engine, boss. "
    "Check that Ollama is running."
)
EXHAUSTED_REPLY = "I wasn't able to finish that one — the tool chain ran too long."

# Sent back to the model when its reply promised or reported an action but no
# tool ran (F-12). Marked as coming from the harness, not the user.
CLAIMED_ACTION_NUDGE = (
    "[HARNESS] Your reply says you are doing or have done something, but you did "
    "not call any tool, so nothing has happened. If you have everything you need, "
    "call the tool now. If a detail is missing, ask the user for it in one short "
    "question. Do not say it is done unless a tool has done it."
)
# Asked when a forced decision chose ask_user but wrote no question.
CLARIFY_REPLY = "Could you tell me a bit more about what you'd like me to do?"
# Tools start a refusal this way ("Not opened: …", "Not run: …"): the tool
# was called but did nothing.
REFUSAL_PREFIX = "Not "
# How much recent conversation counts as "what the user said" when checking
# that tool arguments were not invented (tools/validation.py).
GROUNDING_MESSAGES = 6
# How much of a reply is held back for checking when tools were offered.
OPENING_MIN_CHARS = 80
OPENING_MAX_CHARS = 240
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


class OpeningGate:
    """Holds back the start of a reply until it is known not to be a false claim.

    Until a tool has acted, a reply that reports doing something is often
    false — "Yes boss. Notepad is now open." or an invented forecast — and
    once streamed it is spoken, so the forced decision's real answer arrives
    as a contradiction. Claims come first, so holding the opening sentence or
    two (cut at a sentence end once OPENING_MIN_CHARS have arrived, or at
    OPENING_MAX_CHARS) is enough; sentence-level TTS waits for a sentence end
    anyway, so an honest reply loses little. After a claim nothing is
    released, so the claim can be replaced instead of spoken.
    """

    def __init__(self, active: bool, command: bool = False) -> None:
        self.active = active
        # The user's message was a command: state words count as claims.
        self.command = command
        self.claimed = False
        self._buffer = ""

    def feed(self, text: str) -> str:
        """Text that may be shown now ("" while holding)."""
        if not self.active:
            return text
        self._buffer += text
        if self.claimed or claims_action(self._buffer, self.command):
            self.claimed = True
            return ""
        if len(self._buffer) >= OPENING_MAX_CHARS or (
            len(self._buffer) >= OPENING_MIN_CHARS and _SENTENCE_END.search(self._buffer)
        ):
            return self.release()
        return ""

    def release(self) -> str:
        """Everything held so far; the gate stays open from here on."""
        text, self._buffer, self.active = self._buffer, "", False
        return text


# Per-turn events an interface may want to render (an approval card).
OnEvent = Callable[[dict[str, Any]], None]


def approval_reason(name: str, untrusted_seen: bool) -> str | None:
    """Why this call must wait for the user, or None if it may run now."""
    spec = get_tool(name)
    if spec is None:
        return None
    if spec.confirm and REQUIRE_CONFIRMATION:
        return "this action always needs your approval"
    if spec.privileged and untrusted_seen:
        return "outside content was read earlier in this request"
    return None


def execute_tool(name: str, args: dict[str, Any], channel: str = "web") -> str:
    """Run a tool, converting failures to text. Policy is the caller's job."""
    try:
        return dispatch_tool_call(name, args, channel=channel)
    except Exception as exc:
        log.exception("Tool %s raised", name)
        return f"The {name} tool failed: {exc}"


# (name, args, channel) -> result text. Swapped out by the eval runner so a
# test turn never launches an app or runs a command.
ToolExecutor = Callable[[str, dict[str, Any], str], str]


class Agent:
    def __init__(
        self,
        db: DatabaseManager,
        memory: VectorMemory | None,
        approvals: ApprovalQueue | None = None,
        chat: Callable[..., Any] = ollama.chat,
        *,
        model: str = CHAT_MODEL,
        traces: TraceStore | None = None,
        tool_executor: ToolExecutor | None = None,
        option_overrides: dict[str, Any] | None = None,
        router: ToolRouter | None = None,
    ) -> None:
        self.db = db
        self.memory = memory
        self.approvals = approvals or ApprovalQueue(db)
        # Injectable so tests can drive the loop without a running Ollama.
        self._chat = chat
        # Configurable so the eval can compare models without touching config.
        self.model = model
        self.traces = traces or TraceStore(db)
        self._execute = tool_executor or (lambda name, args, channel: execute_tool(name, args, channel))
        self._option_overrides = dict(option_overrides or {})
        self.router = router or ToolRouter()

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------
    def _remember(self, message_id: int, content: str, session_id: str, role: str) -> None:
        if self.memory is not None:
            self.memory.add_async(
                message_id, content, session_id, role, datetime.now().isoformat()
            )

    def record_user_message(self, session_id: str, text: str) -> int:
        """Persist the incoming message before generating.

        stream() builds its context from SQLite. If the prompt is not stored
        first, the model answers the previous turn instead. A message that
        reads like a correction ("no, I meant…") is also recorded against the
        previous turn's trace.
        """
        self.traces.note_user_message(session_id, text)
        message_id = self.db.insert_message(session_id, "user", text)
        self._remember(message_id, text, session_id, "user")
        return message_id

    # ------------------------------------------------------------------
    #  The loop
    # ------------------------------------------------------------------
    def stream(
        self,
        session_id: str,
        user_prompt: str,
        channel: str = "web",
        on_event: OnEvent | None = None,
    ) -> Iterator[str]:
        """Stream a reply, executing tools in between passes when the model asks.

        Yields plain text deltas. Out-of-band events go to `on_event`: the
        turn's trace id first (so an interface can attach feedback to it),
        then any action waiting for approval. Raises nothing: transport
        failures are converted into a spoken apology so an SSE stream always
        terminates cleanly. Every turn is traced, including failed ones.
        """
        trace = TraceRecorder(session_id, channel, user_prompt, self.model)
        if on_event is not None:
            on_event({"trace_id": trace.trace_id})
        try:
            for text in self._run(trace, session_id, user_prompt, channel, on_event):
                trace.text_emitted(text)
                yield text
        finally:
            # Also reached when the client disconnects mid-stream (outcome
            # stays "aborted"), which is itself worth knowing.
            self.traces.save(trace)

    def _run(
        self,
        trace: TraceRecorder,
        session_id: str,
        user_prompt: str,
        channel: str,
        on_event: OnEvent | None,
    ) -> Iterator[str]:
        if model_paused():
            # On battery with GIDEON_ON_BATTERY=off. Not stored as an
            # assistant turn, so it never replays as conversation.
            trace.outcome = "paused"
            yield BATTERY_PAUSED_REPLY
            return

        db = self.db
        context_messages = build_context_window(db, session_id)
        trace.stage("history")
        messages = format_messages_for_ollama(
            db,
            self.memory,
            context_messages,
            user_prompt,
            actions=self.approvals.recent(session_id),
        )
        trace.stage("prompt")  # includes memory recall
        # Set once a tool returns text from outside the user's control.
        untrusted_seen = False
        ran_tool = requested_approval = blocked_any = False
        # A tool ran and did not refuse ("Not opened: …", "Not run: …"). Until
        # then, any reply that reports success is false (seen live: a blocked
        # call followed by "The sendme folder is now open in VS Code").
        acted = False
        nudged = False
        # What the user actually said, recently: tool arguments must come from
        # here, not from the model's imagination (never-assume, PRD §5.5).
        # Only what was said counts — not replayed tool results (F-20).
        spoken = spoken_messages(context_messages)
        grounding = "\n".join([m["content"] for m in spoken[-GROUNDING_MESSAGES:]] + [user_prompt])
        # What the forced decision routes on: this message and the exchange
        # before it, so a short answer ("in file explorer") keeps its subject.
        routing_text = "\n".join([m["content"] for m in spoken[-3:-1]] + [user_prompt])

        # Read by tools (set_reminder) to learn where this turn came from.
        current_channel.set(channel)
        current_session.set(session_id)

        offer_tools = should_offer_tools(user_prompt)
        # Schemas are filtered per channel: a remote channel never even sees
        # the tools it is not allowed to run.
        channel_tools = tools_for_channel(channel)
        # Of those, only the ones relevant to this message (F-11).
        offered_tools = self.router.select(user_prompt, channel_tools, self.model) if offer_tools else []
        trace.stage("routing")
        trace.tools_offered = offer_tools
        if offer_tools:
            trace.tools_offered_names = [t["function"]["name"] for t in offered_tools]
        log.debug("Tools %s for this turn", "offered" if offer_tools else "withheld")
        # After a command, any reported state counts as a claim ("Music
        # paused, sir."); after a question that merely mentions an action
        # ("any tips for running?") the answer is just an answer.
        command = offer_tools and is_command(user_prompt)

        for turn in range(MAX_TOOL_TURNS):
            full_ans = ""
            collected_tool_calls: list[dict[str, Any]] = []
            messages = trim_to_budget(messages, tools_offered=offer_tools)
            # Re-read per pass: unplugging mid-turn moves the next pass onto
            # the CPU (brain/llm.py).
            options = {**chat_options(), **self._option_overrides}
            trace.placement = "cpu" if options.get("num_gpu") == 0 else "gpu"
            record = trace.start_pass()
            stats: dict[str, Any] = {}
            # Armed until a tool has acted, tools offered or not: a reply to a
            # message gating misjudged ("vs code please") can claim too.
            gate = OpeningGate(not acted and not requested_approval, command=command)
            try:
                stream = self._chat(
                    model=self.model,
                    messages=messages,
                    tools=offered_tools if offer_tools else [],
                    stream=True,
                    options=options,
                    keep_alive=CHAT_KEEP_ALIVE,
                    **thinking_kwargs(self.model),
                )

                held = ""          # Buffered text from the first brace onwards.
                suppress = False   # True after intercepting a text tool call.

                for chunk in stream:
                    trace.first_chunk(record, record["started_ms"], trace.elapsed_ms())
                    stats = ollama_stats(chunk) or stats
                    calls = extract_tool_calls(chunk)
                    if calls:
                        collected_tool_calls.extend({**c, "source": "native"} for c in calls)
                        continue

                    token = extract_chunk_content(chunk)
                    if not token or suppress:
                        continue
                    # Some GGUF chat templates emit a stray role header first.
                    if not full_ans and not held and token.strip().lower() == "assistant":
                        continue

                    # Prose streams straight through; only text from a '{' or
                    # a code fence onwards is withheld, and only until it can
                    # be classified. Buffering every reply would delay the
                    # first token by the whole buffer and defeat sentence TTS.
                    if held:
                        held += token
                    else:
                        split_at = json_start_index(token)
                        if split_at is None:
                            full_ans += token
                            if shown := gate.feed(token):
                                yield shown
                            continue
                        prefix, held = token[:split_at], token[split_at:]
                        if prefix:
                            full_ans += prefix
                            if shown := gate.feed(prefix):
                                yield shown

                    pseudo = parse_pseudo_tool_call(held)
                    if is_tool_call_candidate(pseudo):
                        log.info("Intercepted text-emitted tool call: %s", pseudo["name"])
                        collected_tool_calls.append({**pseudo, "source": "text"})
                        held, suppress = "", True
                    elif len(held) >= PSEUDO_CALL_CHAR_LIMIT:
                        # Too long to be a tool call — release it as speech.
                        full_ans += held
                        if shown := gate.feed(held):
                            yield shown
                        held = ""

                # Anything still withheld when the stream ends.
                if held and not suppress:
                    pseudo = parse_pseudo_tool_call(held)
                    if is_tool_call_candidate(pseudo):
                        log.info("Intercepted end-of-stream tool call: %s", pseudo["name"])
                        collected_tool_calls.append({**pseudo, "source": "text"})
                    else:
                        full_ans += held
                        if shown := gate.feed(held):
                            yield shown
                if gate.active and not gate.claimed and (shown := gate.release()):
                    yield shown

            except Exception as exc:
                log.exception("Ollama stream failed on turn %d", turn)
                trace.end_pass(record, stats)
                trace.outcome, trace.error = "offline", str(exc)
                yield OFFLINE_REPLY
                return
            trace.end_pass(record, stats)

            # No tools requested: a final answer — unless it claims an action
            # that never happened (F-12). Then send it back once.
            if (
                not collected_tool_calls
                and not nudged
                and not (ran_tool or requested_approval or blocked_any)
                and turn < MAX_TOOL_TURNS - 1
                # Claims come first; in conversation only the opening is
                # checked, so a long answer's advice ("Set a consistent
                # time") is not taken for one. After a command, all of it.
                and claims_action(full_ans if command else full_ans[:OPENING_MAX_CHARS], command)
            ):
                nudged = True
                log.info("Reply claimed an action without a tool; forcing a decision")
                trace.stages["claimed_action_nudge"] = trace.elapsed_ms()
                if offer_tools:
                    # The model had these tools; it chooses among the same
                    # ones. Re-routing on the conversation once sent "pause
                    # the music" to a web search about the previous topic.
                    candidates = offered_tools
                else:
                    # Gating withheld the tools: the router picks candidates
                    # (98% recall, F-14), from this message and the exchange
                    # before it — "Vs code please" means nothing alone (F-19).
                    candidates = self.router.select(routing_text, channel_tools)
                forced = self._force_decision(
                    messages,
                    full_ans,
                    candidates,
                    options,
                    prompt_tools=offered_tools if offer_tools else [],
                    trace=trace,
                )
                if forced is not None and forced["tool"] == "ask_user":
                    question = str(forced.get("question") or "").strip() or CLARIFY_REPLY
                    if question:
                        trace.stages["forced_question"] = trace.elapsed_ms()
                        blocked_any = True  # outcome: clarified
                        if gate.claimed:
                            # The claim was held back: the question replaces it.
                            gate.release()
                            full_ans = question
                            yield question
                        else:
                            addition = f"\n\n{question}"
                            full_ans += addition
                            yield addition
                elif forced is not None:
                    trace.stages["forced_tool_call"] = trace.elapsed_ms()
                    if gate.claimed:
                        # Held back, never shown: drop it, so neither the user
                        # nor the next pass sees the false claim.
                        gate.release()
                        full_ans = ""
                    elif full_ans.strip():
                        # The claim was already streamed; keep the real answer
                        # that follows from running into it.
                        yield "\n\n"
                    # The tool list stays what this turn started with: a
                    # different list re-reads the whole prompt (F-20).
                    collected_tool_calls = [
                        {"name": forced["tool"], "arguments": forced["arguments"], "source": "forced"}
                    ]
                # Otherwise: no usable decision; the reply stands as given.
            # Anything still held (no decision was made) is the answer after all
            # — minus any sentence claiming what no tool did.
            if gate.active:
                held_text = gate.release()
                if gate.claimed and collected_tool_calls:
                    # Said alongside a real tool call ("Playing Arctic Monkeys
                    # now"): premature, not false. Dropped — the next pass
                    # reports what actually happened.
                    full_ans = held_text = ""
                elif gate.claimed:
                    honest = strip_claims(held_text, command)
                    log.info("Cut a false claim from the reply: %r", held_text[:120])
                    trace.stages["claim_cut"] = trace.elapsed_ms()
                    full_ans = honest
                    held_text = honest
                if held_text:
                    yield held_text

            # No tools requested: this was the final answer.
            if not collected_tool_calls:
                if full_ans.strip():
                    asst_id = db.insert_message(session_id, "assistant", full_ans)
                    self._remember(asst_id, full_ans, session_id, "assistant")
                trace.outcome = (
                    "approval_requested" if requested_approval
                    else "tool" if ran_tool
                    else "clarified" if blocked_any
                    else "answered"
                )
                return

            # Tools requested: execute, feed results back, and loop.
            for tool_call in collected_tool_calls:
                requested_name = tool_call.get("name", "")
                args = tool_call.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}

                # A 3B model cannot reliably tell "press play" from "play <song>".
                name, args = correct_media_tool_choice(requested_name, args, user_prompt)
                log.info("Tool call: %s(%s)", name, args)

                started_ms = trace.elapsed_ms()
                approval: dict[str, Any] | None = None
                blocked: str | None = None
                dropped: list[str] = []
                reason: str | None = None
                spec = get_tool(name)

                # Validate first (PRD §5.3): a call with a missing or invented
                # required value never runs; the model is told to ask.
                check = check_call(spec, args, grounding) if spec is not None else None
                if check is not None:
                    args, dropped = check.arguments, check.dropped
                    if dropped:
                        log.info("Dropped invented optional value(s) %s for %s", dropped, name)
                if check is not None and not check.ok:
                    blocked = check.problem()
                    blocked_any = True
                    log.info("Blocked %s: %s", name, blocked)
                    tool_result = check.instruction()
                elif reason := approval_reason(name, untrusted_seen):
                    action = self.approvals.request(
                        session_id, channel, name, args, trace_id=trace.trace_id
                    )
                    approval = {"action_id": action.action_id, "reason": reason}
                    requested_approval = True
                    if on_event is not None:
                        on_event({"approval": {**action_payload(action), "reason": reason}})
                    tool_result = (
                        f"Not run yet: approval required because {reason}. Approval request "
                        f"#{action.action_id} for {action.describe()} is now waiting for the user "
                        f"in the interface. Tell them briefly what it will do and that it needs "
                        f"their approval."
                    )
                else:
                    tool_result = self._execute(name, args, channel)
                    ran_tool = True
                    acted = acted or not tool_result.lstrip().startswith(REFUSAL_PREFIX)

                trace.tool_call(
                    record,
                    name=name,
                    args=args,
                    source=tool_call.get("source", "native"),
                    corrected_from=requested_name if requested_name != name else None,
                    approval=approval,
                    result=tool_result,
                    ms=trace.elapsed_ms() - started_ms,
                    blocked=blocked,
                    dropped=dropped,
                )

                untrusted = bool(spec and spec.untrusted_output and not reason and not blocked)
                untrusted_seen = untrusted_seen or untrusted

                # Stored as 'tool_call'/'tool' so build_context_window never
                # replays them as something G.I.D.E.O.N said out loud.
                db.insert_message(
                    session_id,
                    "tool_call",
                    f"[Tool Call: {name}]",
                    tool_name=name,
                    tool_args=json.dumps(args),
                )
                db.insert_message(session_id, "tool", tool_result, tool_name=name)

                messages.append(
                    {
                        "role": "assistant",
                        "content": full_ans,
                        "tool_calls": [{"function": {"name": name, "arguments": args}}],
                    }
                )
                messages.append(
                    {"role": "tool", "content": fit_tool_result(tool_result, untrusted), "name": name}
                )

        log.warning("Tool loop exhausted %d turns without a final answer", MAX_TOOL_TURNS)
        trace.outcome = "exhausted"
        yield EXHAUSTED_REPLY

    def _force_decision(
        self,
        messages: list[dict[str, Any]],
        claim: str,
        candidates: list[dict[str, Any]],
        options: dict[str, Any],
        *,
        prompt_tools: list[dict[str, Any]],
        trace: TraceRecorder,
    ) -> dict[str, Any] | None:
        """Ask again, with output constrained to a tool call or a question.

        F-15: a polite reminder did not work — qwen3:4b-instruct repeated its
        false claim word for word. Constrained decoding (PRD §10.1) leaves the
        model no prose to fall back on: it must name one of `candidates` and
        its arguments, or `ask_user` with a question. The result still goes
        through validation, policy and approvals like any other call.

        F-20: the call reuses the pass it follows — same tool list
        (`prompt_tools`) and same messages, with the claim and the instruction
        appended — so Ollama reads only those new tokens. With `tools=[]` the
        prompt changed from its first token (Qwen renders tools at the top)
        and the whole conversation was read again: ~680 tokens here and
        ~1,200 more in the reply pass, the "7-second penalty". The choice is
        still limited to `candidates`, by the schema's enum.
        """
        names = [t["function"]["name"] for t in candidates]
        if not names:
            return None
        if prompt_tools:
            # The schemas are already in the prompt; naming the choices is enough.
            choices = "Choose one of: " + ", ".join(names) + "."
        else:
            # No tools in the prompt (gating withheld them): describe them.
            choices = "Available tools:\n" + "\n".join(
                f"- {t['function']['name']}("
                + ", ".join(t["function"].get("parameters", {}).get("properties", {}))
                + f"): {t['function']['description'][:180]}"
                for t in candidates
            )
        schema = {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": [*names, "ask_user"]},
                "arguments": {"type": "object"},
                "question": {"type": "string"},
            },
            # Required: an ask_user without its question once left nothing
            # to say but "I haven't done that yet" (eval, 2026-10-27).
            "required": ["tool", "arguments", "question"],
        }
        prompt = (
            f"{CLAIMED_ACTION_NUDGE}\n\n{choices}\n\n"
            'Reply with JSON only: {"tool": <tool name, or "ask_user" if a detail '
            'is missing>, "arguments": {...}, "question": <the question for ask_user, '
            'else "">}.'
        )
        record = trace.start_pass()
        record["forced_candidates"] = names
        stats: dict[str, Any] = {}
        try:
            response = self._chat(
                model=self.model,
                messages=[*messages, {"role": "assistant", "content": claim}, {"role": "user", "content": prompt}],
                tools=prompt_tools,
                stream=False,
                format=schema,
                options=options,
                keep_alive=CHAT_KEEP_ALIVE,
                **thinking_kwargs(self.model),
            )
            stats = ollama_stats(response)
            decision = json.loads(extract_chunk_content(response) or "{}")
        except Exception as exc:
            log.warning("Forced tool decision failed: %s", exc)
            return None
        finally:
            trace.end_pass(record, stats)
        if not isinstance(decision, dict) or decision.get("tool") not in (*names, "ask_user"):
            return None
        if not isinstance(decision.get("arguments"), dict):
            decision["arguments"] = {}
        log.info("Forced decision: %s(%s)", decision["tool"], decision["arguments"])
        return decision

    def respond(
        self,
        session_id: str,
        user_prompt: str,
        channel: str = "web",
        on_event: OnEvent | None = None,
    ) -> str:
        """Run a whole turn and return the full reply text."""
        return "".join(self.stream(session_id, user_prompt, channel, on_event))
