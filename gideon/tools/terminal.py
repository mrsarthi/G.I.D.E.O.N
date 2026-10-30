"""
G.I.D.E.O.N Terminal Execution Tool

Executes shell commands on the local system and streams the result to the
Terminal drawer via the event bus.

Three controls sit in front of execution, because the caller is a 3B model
that has demonstrably hallucinated tool names and misrouted arguments:

1. A denylist of irreversible commands, applied to every segment of a chained
   command line so `dir && format c:` cannot slip a second command through.
2. Approval: the agent will not run this tool until the user approves it in
   the interface (brain/approvals.py). The HUD's terminal drawer calls this
   function directly — there the user typed the command, which is approval.
3. A durable audit row for every attempt, allowed or refused.

None of this is a sandbox. It blocks the catastrophic tail; it does not make
arbitrary execution safe.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from gideon.core.config import (
    BASE_DIR,
    ENFORCE_COMMAND_DENYLIST,
    TERMINAL_TIMEOUT_SECONDS,
)
from gideon.core.events import bus
from gideon.core.logs import get_logger
from gideon.core.security import find_destructive_pattern
from gideon.storage.db import get_db
from gideon.tools.base import Param, tool

log = get_logger("gideon.tools.terminal")

MAX_OUTPUT_CHARS = 2000


def _audit(command, cwd, channel, allowed, reason=None, code=None, output=None) -> None:
    """Record the attempt. Never let auditing failure break execution."""
    try:
        get_db().record_command(
            command=command,
            cwd=str(cwd) if cwd else None,
            channel=channel,
            allowed=allowed,
            block_reason=reason,
            return_code=code,
            output_excerpt=output,
        )
    except Exception as exc:
        log.warning("Could not write command audit row: %s", exc)


def check_command_safety(command: str) -> str | None:
    """Return a refusal reason, or None when the command may run.

    Each chained segment is checked separately: an allowed prefix must not be
    able to smuggle a destructive suffix past the filter.
    """
    if not ENFORCE_COMMAND_DENYLIST:
        return None
    pattern = find_destructive_pattern(command)
    return f"matches blocked pattern /{pattern}/" if pattern else None


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Terminate the shell and everything it spawned.

    Killing only the shell leaves orphans behind — a timed-out `pip install`
    keeps holding file locks and network handles.
    """
    try:
        import psutil

        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(children, timeout=3)
    except Exception as exc:
        log.debug("psutil tree kill failed (%s); falling back", exc)
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def _publish(command, code, output, cwd) -> None:
    try:
        # Tools run in FastAPI worker threads, not the event loop.
        bus.publish_threadsafe(
            "terminal",
            {"command": command, "returncode": code, "output": output, "cwd": str(cwd)},
        )
    except Exception as exc:
        log.warning("Could not publish terminal event: %s", exc)


@tool(
    "run_terminal_command",
    "Execute a shell command line (PowerShell/CMD) on the host system and capture stdout/stderr output. Call when user explicitly asks to run a CLI command, install a package, check git status, or execute terminal commands.",
    Param(
        "command",
        "The exact CLI command string to execute (e.g., 'git status', 'pip list', 'dir').",
        required=True,
        aliases=("cmd", "line"),
    ),
    Param("cwd", "Optional working directory path.", aliases=("dir",), default=None),
    # Telegram is a remote channel guarded only by a user-ID allowlist.
    blocked_channels={"telegram"},
    confirm=True,
    privileged=True,
    # Command output can contain anything a file or website put there.
    untrusted_output=True,
    triggers={
        "terminal", "command", "cmd", "powershell", "shell", "exec", "execute",
        "pip", "npm", "git", "pytest", "cli",
    },
    wants_channel=True,
)
def run_terminal_command(
    command: str,
    cwd: str | None = None,
    timeout: int = TERMINAL_TIMEOUT_SECONDS,
    channel: str = "web",
) -> str:
    """Execute a shell command and return formatted stdout/stderr output."""
    cmd_str = (command or "").strip()
    if not cmd_str:
        return "Error: No command specified for execution."

    work_dir = Path(cwd).resolve() if cwd else BASE_DIR
    if not work_dir.exists():
        work_dir = BASE_DIR

    # 1. Irreversible-command denylist.
    reason = check_command_safety(cmd_str)
    if reason:
        log.warning("REFUSED destructive command %r (%s)", cmd_str, reason)
        _audit(cmd_str, work_dir, channel, allowed=False, reason=reason)
        _publish(cmd_str, -1, f"REFUSED: {reason}", work_dir)
        return (
            f"Refused to run `{cmd_str}`: this command is on the irreversible-action "
            f"denylist ({reason}). If you genuinely intend it, run it yourself in a "
            f"terminal."
        )

    log.info("Executing (%s) %r in %s", channel, cmd_str, work_dir)

    proc = None
    try:
        proc = subprocess.Popen(
            cmd_str,
            shell=True,
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            # New group so the whole tree can be signalled on timeout.
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        return_code = proc.returncode

    except subprocess.TimeoutExpired:
        if proc is not None:
            _kill_process_tree(proc)
        msg = f"Command `{cmd_str}` timed out after {timeout} seconds."
        log.warning(msg)
        _audit(cmd_str, work_dir, channel, allowed=True, reason="timeout", code=-1)
        _publish(cmd_str, -1, msg, work_dir)
        return f"Error: {msg}"

    except Exception as err:
        msg = f"Failed to execute command `{cmd_str}`: {err}"
        log.error(msg)
        _audit(cmd_str, work_dir, channel, allowed=True, reason=str(err), code=1)
        _publish(cmd_str, 1, msg, work_dir)
        return f"Error: {msg}"

    stdout = (stdout or "").strip()
    stderr = (stderr or "").strip()

    combined = []
    if stdout:
        combined.append(stdout)
    if stderr:
        combined.append(f"[STDERR]\n{stderr}")
    output_text = "\n\n".join(combined) if combined else "(No output returned)"

    if len(output_text) > MAX_OUTPUT_CHARS:
        dropped = len(output_text) - MAX_OUTPUT_CHARS
        output_text = output_text[:MAX_OUTPUT_CHARS] + f"\n... [Truncated {dropped} characters]"

    _audit(cmd_str, work_dir, channel, allowed=True, code=return_code, output=output_text)
    _publish(cmd_str, return_code, output_text, work_dir)

    return f"Command: `{cmd_str}`\nReturn Code: {return_code}\nOutput:\n{output_text}"
