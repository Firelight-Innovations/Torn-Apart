"""
tests/test_profiler_window.py — window-marked tests for the profiler's
render-side pieces (PStats bridge + F3 overlay).

These need Panda3D (and a graphics context), so they are excluded from the
default headless run via ``addopts = -m "not window"`` and must be invoked with
the venv interpreter explicitly::

    .venv/Scripts/python.exe -m pytest tests/test_profiler_window.py -m window -q

The core profiler itself is covered headlessly in tests/test_profiler.py; here
we only check that the panda3d mirrors wire up and run without error.
"""

from __future__ import annotations

import pytest

from fire_engine.core.config import Config
from fire_engine.core.profiler import Profiler


def _drive(prof: Profiler, frames: int = 6) -> None:
    """Run a few frames of scoped work + a counter, then flush the last one."""
    for _ in range(frames):
        prof.begin_frame()
        with prof.scope("Update"), prof.scope("Update:Weather"):
            pass
        with prof.scope("Lighting"):
            pass
        prof.set_counter("draw_calls", 123)
        prof.end_frame()
    prof.begin_frame()        # commit the last frame


@pytest.mark.window
def test_pstats_bridge_mirrors_scopes_and_counters():
    """The bridge lazily creates a PStatCollector per scope/counter name."""
    pytest.importorskip("panda3d.core")
    from fire_engine.render.profiler_bridge import PStatsBridge

    prof = Profiler(enabled=True, history_frames=32, hitch_window=8)
    bridge = PStatsBridge(prof, connect=False)   # no server needed
    _drive(prof)

    # Every scope that ran got a collector; counters too.
    assert "Update" in bridge._timers
    assert "Update:Weather" in bridge._timers
    assert "Lighting" in bridge._timers
    assert "draw_calls" in bridge._counters


@pytest.mark.window
def test_overlay_builds_toggles_and_refreshes():
    """The F3 overlay constructs against a real ShowBase, toggles, and refreshes
    from the ring buffer without raising."""
    pytest.importorskip("panda3d.core")
    from panda3d.core import loadPrcFileData  # type: ignore[import]
    loadPrcFileData("", "window-type offscreen\naudio-library-name null")
    from direct.showbase.ShowBase import ShowBase  # type: ignore[import]

    base = ShowBase()
    try:
        from fire_engine.render.profiler_overlay import ProfilerOverlay
        prof = Profiler(enabled=True, history_frames=64, hitch_window=8)
        cfg = Config(profiler_enabled=True, profiler_overlay_hz=1000.0)
        overlay = ProfilerOverlay(base, prof, cfg)

        # Hidden by default; cheap update is a no-op.
        assert overlay.visible is False
        overlay.update()

        # Drive frames so the ring has data, then show + refresh.
        _drive(prof, frames=10)
        overlay.toggle()
        assert overlay.visible is True
        overlay.update()          # forces a refresh (hz is huge)
        # Text node reflects the data (frame ms string is non-empty).
        assert overlay._t_frame.getText() != ""
        # Hide again.
        overlay.toggle()
        assert overlay.visible is False
        overlay.destroy()
    finally:
        base.destroy()
