"""
render/_impl/app_profiler.py — Profiler setup and snapshot helpers for App.

Extracted from render/app.py to keep that module under 500 lines (C0302).
Functions take the App instance as their first argument (``self_obj``) and are
called from the class as ``_func(self, ...)``.  Not part of the public API.

Docs: docs/systems/render.md
"""

from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fire_engine.render.app import App


def setup_profiler(self_obj: App) -> None:
    """
    Wire the render-side profiler pieces when ``profiler_enabled``.

    Constructs nothing when the profiler is off (truly free).  When on:
      - builds the PStats bridge + (optionally) connects to a PStats server
        so the standalone ``pstats`` GUI can attach (config.profiler_pstats);
      - builds the F3 in-game overlay (config.profiler_overlay_enabled) and
        binds F3 to toggle it;
      - arms the rolling JSON snapshot writer (config.profiler_snapshot_*).

    Each piece is wrapped in try/except so a profiler failure never takes
    down the game — it logs and disables that piece.
    """
    cfg = self_obj._config
    if not getattr(cfg, "profiler_enabled", False):
        return

    # PStats bridge: mirror core scopes/counters into PStatCollectors and,
    # if requested, connect so the pstats GUI shows the App/Cull/Draw split
    # alongside our custom collectors.
    if getattr(cfg, "profiler_pstats", False):
        try:
            from fire_engine.render.bridges.profiler_bridge import PStatsBridge

            self_obj._profiler_bridge = PStatsBridge(self_obj._profiler, connect=True)
        except Exception as exc:
            from fire_engine.core.log import get_logger

            get_logger("profiler").warning("PStats bridge unavailable: %s", exc)

    # In-game overlay (F3).
    if getattr(cfg, "profiler_overlay_enabled", True):
        try:
            from fire_engine.render.bridges.profiler_overlay import ProfilerOverlay

            self_obj._profiler_overlay = ProfilerOverlay(self_obj, self_obj._profiler, cfg)
            self_obj.accept("f3", self_obj._profiler_overlay.toggle)
        except Exception as exc:
            from fire_engine.core.log import get_logger

            get_logger("profiler").warning("Profiler overlay unavailable: %s", exc)

    # Rolling JSON snapshot (the AI-agent contract).
    if getattr(cfg, "profiler_snapshot_enabled", False):
        self_obj._snapshot_path = getattr(cfg, "profiler_snapshot_path", "profiling/latest.json")


def maybe_write_snapshot(self_obj: App) -> None:
    """Write the rolling profiler JSON snapshot if the interval elapsed."""
    if self_obj._snapshot_path is None:
        return
    now = _time.perf_counter()
    if now - self_obj._last_snapshot_t < self_obj._snapshot_interval_s:
        return
    self_obj._last_snapshot_t = now
    try:
        self_obj._profiler.write_snapshot(self_obj._snapshot_path)
    except OSError as exc:
        from fire_engine.core.log import get_logger

        get_logger("profiler").warning("snapshot write failed: %s", exc)
