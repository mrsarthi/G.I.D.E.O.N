"""
G.I.D.E.O.N Eval Runner

Sends every case in evals/cases.toml through the real agent and model, and
scores what the model *decided*: which tool it chose, which slots it filled,
whether it asked when it should have. Tools are never executed — each call is
recorded and answered with a canned result — so an eval can run on any
machine without opening apps, running commands or touching real data.

Everything the agent records goes to a throwaway data directory (set up by
`python -m gideon.evals` before anything is imported), so evals never write
traces, sessions or approvals into the real database.

Usage:
    python -m gideon.evals                          # current model
    python -m gideon.evals --model qwen2.5:3b-instruct
    python -m gideon.evals --only media,clarify     # categories or case ids
    python -m gideon.evals --compare a.json b.json  # side-by-side
"""

from __future__ import annotations

import json
import statistics
import time
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gideon.brain.agent import Agent
from gideon.brain.llm import chat_options
from gideon.core.config import BASE_DIR, CHAT_MODEL
from gideon.core.power import on_battery
from gideon.storage.db import get_db

CASES_FILE = BASE_DIR / "evals" / "cases.toml"
REPORTS_DIR = BASE_DIR / "evals" / "reports"

# Fixed so two runs of the same model are comparable. Production uses the
# model's default temperature; evals measure decisions, not variety.
EVAL_OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 256}


# ---------------------------------------------------------------------------
#  Cases
# ---------------------------------------------------------------------------
@dataclass
class Case:
    id: str
    prompt: str
    category: str
    tools: list[str] = field(default_factory=list)   # acceptable first tools; "none" allowed
    args: dict[str, str] = field(default_factory=dict)
    ask: bool = False
    forbid: list[str] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)


def load_cases(path: Path = CASES_FILE) -> list[Case]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    cases = []
    for entry in raw.get("case", []):
        tools = entry.get("tools") or ([entry["tool"]] if "tool" in entry else [])
        cases.append(
            Case(
                id=entry["id"],
                prompt=entry["prompt"],
                category=entry.get("category", "tool"),
                tools=list(tools),
                args={k: str(v) for k, v in entry.get("args", {}).items()},
                ask=bool(entry.get("ask", False)),
                forbid=list(entry.get("forbid", [])),
                history=list(entry.get("history", [])),
            )
        )
    ids = [c.id for c in cases]
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise ValueError(f"Duplicate case ids: {duplicates}")
    return cases


# ---------------------------------------------------------------------------
#  Fake tools
# ---------------------------------------------------------------------------
_CANNED = {
    "get_system_status": "CPU Utilization: 18%\nGPU Utilization: 9% at 52°C\nRAM Usage: 7.1 GB / 15.8 GB (45%)\nBattery: 88% (Plugged in)",
    "list_reminders": "Active Reminders:\n#4 — [Fri, Oct 10 at 09:00 AM] submit the assignment",
    "list_notes": "Task #3: renew passport\nNote #2: wifi password is on the router sticker",
    "get_weather": "Clear sky, 24°C, humidity 61%. Today: high 27°C, low 19°C, 10% chance of rain.",
    "get_financial_data": "Last price 182.40 USD, +1.2% today.",
    "get_hackernews": "1. Show HN: A tiny database in 500 lines (312 points)\n2. Why SQLite is everywhere (288 points)",
    "get_ip_info": "Public IP 203.0.113.7 — Pune, Maharashtra, India (ISP: Example Broadband)",
    "query_wolfram_alpha": "Result: 1702519",
    "search_web": "1. Example result: the answer is described on example.com.",
    "get_clipboard": "def add(a, b):\n    return a + b",
    "read_document": "Q3 report: revenue up 12%, costs flat, two new hires.",
    "search_files": "Found 1 file(s):\n— C:\\Users\\me\\Documents\\budget.xlsx",
    "search_youtube_music": "1. Lofi Girl — beats to study to (playlist)",
}


def fake_tool(name: str, args: dict[str, Any], channel: str) -> str:
    """Stand-in for every tool: plausible output, no side effects."""
    if name in _CANNED:
        return _CANNED[name]
    return f"Done: {name} completed successfully."


# ---------------------------------------------------------------------------
#  Scoring
# ---------------------------------------------------------------------------
def _contains(value: Any, expected: str) -> bool:
    return expected.lower() in str(value if value is not None else "").lower()


def score(case: Case, trace: dict[str, Any]) -> tuple[bool, str]:
    """(passed, reason). Reasons are stable labels used in the report."""
    calls = [c for p in trace["detail"].get("passes", []) for c in p["tool_calls"]]
    names = [c["name"] for c in calls]
    reply = trace.get("reply") or ""

    if case.ask:
        forbidden = [n for n in names if n in case.forbid]
        if forbidden:
            return False, f"guessed:{forbidden[0]}"
        if "?" not in reply:
            return False, "no_question"
        return True, "asked"

    if case.tools == ["none"] or (not case.tools):
        return (not calls), ("ok" if not calls else f"unexpected:{names[0]}")

    if not calls:
        if "none" in case.tools:
            return True, "ok"
        return False, "gated" if not trace.get("tools_offered") else "no_tool"

    first = calls[0]
    if first["name"] not in case.tools:
        return False, f"wrong_tool:{first['name']}"

    for slot, expected in case.args.items():
        if not _contains(first["args"].get(slot), expected):
            return False, f"slot:{slot}"
    return True, "ok"


# ---------------------------------------------------------------------------
#  Running
# ---------------------------------------------------------------------------
def _percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def run(model: str = CHAT_MODEL, only: list[str] | None = None, progress: bool = True) -> dict[str, Any]:
    cases = load_cases()
    if only:
        wanted = set(only)
        cases = [c for c in cases if c.id in wanted or c.category in wanted]

    db = get_db()
    agent = Agent(db, None, model=model, tool_executor=fake_tool, option_overrides=EVAL_OPTIONS)

    # Load the model once so the first case's latency is not a cold load.
    warm_session = db.create_session("eval warm-up")
    agent.respond(warm_session, "hi")

    results = []
    started = time.perf_counter()
    for index, case in enumerate(cases, 1):
        session = db.create_session(f"eval {case.id}")
        for turn in case.history:
            db.insert_message(session, turn["role"], turn["content"])
        events: list[dict[str, Any]] = []
        agent.record_user_message(session, case.prompt)
        agent.respond(session, case.prompt, channel="web", on_event=events.append)

        trace_id = next(e["trace_id"] for e in events if "trace_id" in e)
        trace = db.get_trace(trace_id)
        assert trace is not None
        passed, reason = score(case, trace)
        calls = [c for p in trace["detail"]["passes"] for c in p["tool_calls"]]
        passes = trace["detail"]["passes"]
        results.append(
            {
                "id": case.id,
                "category": case.category,
                "prompt": case.prompt,
                "expected": case.tools or ("ask" if case.ask else ["none"]),
                "passed": passed,
                "reason": reason,
                "calls": [{"name": c["name"], "args": c["args"], "source": c["source"]} for c in calls],
                "tools_offered": trace["tools_offered"],
                "reply": trace["reply"],
                "first_text_ms": trace["first_text_ms"],
                "total_ms": trace["total_ms"],
                "tokens_per_sec": next(
                    (p["tokens_per_sec"] for p in passes if "tokens_per_sec" in p), None
                ),
                "outcome": trace["outcome"],
            }
        )
        if progress:
            mark = "PASS" if passed else "FAIL"
            print(f"[{index:3}/{len(cases)}] {mark} {case.id:28} {reason}", flush=True)

    return _summarise(model, results, time.perf_counter() - started)


def _rate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"passed": 0, "total": 0, "pct": None}
    passed = sum(1 for r in items if r["passed"])
    return {"passed": passed, "total": len(items), "pct": round(100 * passed / len(items), 1)}


def _summarise(model: str, results: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    expects_tool = [r for r in results if r["expected"] not in (["none"], "ask")]
    first_text = [r["first_text_ms"] for r in results if r["first_text_ms"] is not None]
    totals = [r["total_ms"] for r in results]
    speeds = [r["tokens_per_sec"] for r in results if r["tokens_per_sec"]]
    options = {**chat_options(), **EVAL_OPTIONS}

    return {
        "model": model,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "placement": "cpu" if options.get("num_gpu") == 0 else "gpu",
        "on_battery": on_battery(),
        "options": EVAL_OPTIONS,
        "seconds": round(seconds),
        "overall": _rate(results),
        "categories": {name: _rate(items) for name, items in sorted(by_category.items())},
        "tool_selection": _rate(
            [{"passed": not r["reason"].startswith(("wrong_tool", "no_tool", "gated"))} for r in expects_tool]
        ),
        "gated_misses": sum(1 for r in results if r["reason"] == "gated"),
        "text_tool_calls": sum(1 for r in results for c in r["calls"] if c["source"] == "text"),
        "failure_reasons": dict(
            Counter(r["reason"].split(":")[0] for r in results if not r["passed"]).most_common()
        ),
        "latency_ms": {
            "first_text_p50": _percentile(first_text, 50),
            "first_text_p90": _percentile(first_text, 90),
            "total_p50": _percentile(totals, 50),
            "total_p90": _percentile(totals, 90),
        },
        "tokens_per_sec_mean": round(statistics.mean(speeds), 1) if speeds else None,
        "results": results,
    }


# ---------------------------------------------------------------------------
#  Reports
# ---------------------------------------------------------------------------
def _pct(rate: dict[str, Any]) -> str:
    return "—" if rate["pct"] is None else f"{rate['pct']}% ({rate['passed']}/{rate['total']})"


def render_markdown(report: dict[str, Any]) -> str:
    lat = report["latency_ms"]
    lines = [
        f"# Eval — {report['model']}",
        "",
        f"Run {report['run_at']} on **{report['placement'].upper()}**"
        f"{' (battery)' if report['on_battery'] else ''}, {report['seconds']}s, "
        f"options {report['options']}.",
        "",
        "| Metric | Result |",
        "| :--- | :--- |",
        f"| **Overall** | **{_pct(report['overall'])}** |",
        f"| Tool selection | {_pct(report['tool_selection'])} |",
    ]
    lines += [f"| {name} | {_pct(rate)} |" for name, rate in report["categories"].items()]
    lines += [
        f"| Tools withheld by gating (missed) | {report['gated_misses']} |",
        f"| Tool calls typed as text | {report['text_tool_calls']} |",
        f"| First text p50 / p90 | {lat['first_text_p50']} / {lat['first_text_p90']} ms |",
        f"| Whole turn p50 / p90 | {lat['total_p50']} / {lat['total_p90']} ms |",
        f"| Generation speed | {report['tokens_per_sec_mean']} tok/s |",
        "",
        "## Failures",
        "",
        "| Case | Expected | Got | Reason |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for r in report["results"]:
        if r["passed"]:
            continue
        got = ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in r["calls"]) or "no tool"
        expected = r["expected"] if isinstance(r["expected"], str) else " / ".join(r["expected"])
        lines.append(f"| `{r['id']}` | {expected} | {got.replace('|', '/')} | {r['reason']} |")
    return "\n".join(lines) + "\n"


def save(report: dict[str, Any]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = report["model"].replace(":", "_").replace("/", "_")
    path = REPORTS_DIR / f"{stamp}-{slug}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    return path


def compare(paths: list[Path]) -> str:
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    names = [f"{r['model']} ({r['placement']})" for r in reports]
    lines = ["| Metric | " + " | ".join(names) + " |", "| :--- |" + " :--- |" * len(reports)]

    def row(label: str, values: list[str]) -> None:
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    row("**Overall**", [f"**{_pct(r['overall'])}**" for r in reports])
    row("Tool selection", [_pct(r["tool_selection"]) for r in reports])
    for category in sorted({c for r in reports for c in r["categories"]}):
        row(category, [_pct(r["categories"].get(category, {"pct": None})) for r in reports])
    row("Gating misses", [str(r["gated_misses"]) for r in reports])
    row("First text p50", [f"{r['latency_ms']['first_text_p50']} ms" for r in reports])
    row("Whole turn p50", [f"{r['latency_ms']['total_p50']} ms" for r in reports])
    row("tok/s", [str(r["tokens_per_sec_mean"]) for r in reports])
    return "\n".join(lines) + "\n"
