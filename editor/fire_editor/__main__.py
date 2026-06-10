"""Entry point: ``python -m fire_editor --port <p>``.

Launched by the VS Code / Cursor extension from the game repo's ``.venv``.
Logs to stderr (the extension pipes stderr into its output channel); the bound
port is announced on stdout as a JSON line (see :meth:`Daemon.run`).

Run ``--port 0`` to let the OS pick a free port (recommended; the extension
reads the announced port back).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .daemon import Daemon


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fire_editor", description="Fire Editor daemon")
    parser.add_argument("--port", type=int, default=0, help="TCP port (0 = OS-assigned)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (localhost only)")
    parser.add_argument("--log-level", default="info", help="debug|info|warning|error")
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    daemon = Daemon(host=args.host)
    try:
        asyncio.run(daemon.run(args.port))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
