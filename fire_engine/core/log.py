"""
core/log.py — Thin logging wrapper for the Torn Apart engine.

Provides a single entry-point ``get_logger(name)`` that ensures a sane
console handler is installed exactly once (on first call), then returns a
named ``logging.Logger``.

All engine modules should obtain their logger via this module rather than
calling ``logging.getLogger`` directly, so that log format and handler setup
are consistent and centralised.

Format
------
``[LEVEL] fire_engine.subsystem — message``

Example
-------
    from fire_engine.core.log import get_logger

    log = get_logger(__name__)
    log.info("Chunk (0, 0, 0) generated in 4.2 ms")
    log.warning("RNG domain 'npc' called before set_world_seed — using seed 0")
"""

from __future__ import annotations

import logging

_LOG_FORMAT = "[%(levelname)s] %(name)s — %(message)s"
_handler_installed: bool = False


def get_logger(name: str) -> logging.Logger:
    """
    Return a ``logging.Logger`` for the given name, with a sane default
    console handler set up on the root logger (once, idempotently).

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.
        Should be a dotted Python module path (e.g. ``fire_engine.core.rng``).

    Returns
    -------
    logging.Logger

    Example
    -------
    >>> log = get_logger("fire_engine.terrain.chunk_manager")
    >>> log.debug("Streaming chunk (1, 2, 0)")
    """
    global _handler_installed
    if not _handler_installed:
        _setup_root_handler()
        _handler_installed = True
    return logging.getLogger(name)


def _setup_root_handler() -> None:
    """
    Install a StreamHandler on the root logger if none exists.

    Sets level to DEBUG so individual loggers can filter as needed.
    Uses the engine-standard one-line format.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(handler)
    # Default to WARNING so tests aren't noisy; callers can raise as needed.
    if root.level == logging.NOTSET:
        root.setLevel(logging.WARNING)
