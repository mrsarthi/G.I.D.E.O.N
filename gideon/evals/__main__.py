"""python -m gideon.evals — see gideon/evals/runner.py."""

import os
import sys
import tempfile

# Before anything from gideon is imported: configuration resolves its paths
# at import time, and evals must never write into the real database.
_DATA_DIR = tempfile.mkdtemp(prefix="gideon-eval-")
os.environ["GIDEON_DATA_DIR"] = _DATA_DIR
os.environ["GIDEON_TELEGRAM_BOT_TOKEN"] = ""
os.environ["GIDEON_REMOTE_SCHEDULER_URL"] = ""
os.environ["GIDEON_REMOTE_SCHEDULER_TOKEN"] = ""

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    from gideon.core.config import CHAT_MODEL
    from gideon.evals import runner

    parser = argparse.ArgumentParser(prog="python -m gideon.evals", description=runner.__doc__)
    parser.add_argument("--model", default=CHAT_MODEL, help=f"Ollama model (default {CHAT_MODEL})")
    parser.add_argument("--models", help="comma-separated models: run each, then compare")
    parser.add_argument("--only", help="comma-separated case ids or categories")
    parser.add_argument("--compare", nargs="+", type=Path, metavar="REPORT.json",
                        help="print a side-by-side table of saved reports and exit")
    args = parser.parse_args()

    if args.compare:
        print(runner.compare(args.compare))
        return 0

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    models = [m.strip() for m in args.models.split(",")] if args.models else [args.model]
    saved = []
    for model in models:
        print(f"\n=== {model} ===", flush=True)
        report = runner.run(model, only=only)
        path = runner.save(report)
        saved.append(path)
        print()
        print(runner.render_markdown(report))
        print(f"Saved: {path}  (and {path.with_suffix('.md').name})")
    if len(saved) > 1:
        print("\n=== Comparison ===\n")
        print(runner.compare(saved))
    return 0


def _cleanup() -> None:
    """Remove the throwaway data directory (reports are saved elsewhere)."""
    import shutil

    try:
        from gideon.storage.db import close_db

        close_db()  # Windows will not delete a database file that is still open
    except Exception:
        pass
    shutil.rmtree(_DATA_DIR, ignore_errors=True)


if __name__ == "__main__":
    try:
        code = main()
    finally:
        _cleanup()
    sys.exit(code)
