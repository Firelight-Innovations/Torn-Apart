"""
render/profiler_overlay.py — the in-game performance overlay (F3).

The human-visible half of the profiler: a compact HUD drawn from the core
profiler's ring buffer.  Toggle with **F3**.  It shows, refreshed at a low rate
(``profiler_overlay_hz``, default ~8 Hz) so the overlay itself barely costs
anything:

  - current frame ms + FPS and **ms vs the budget** (green under budget,
    yellow approaching, red over);
  - a **scrolling frame-time graph** (last ``profiler_overlay_graph_frames``
    frames) with a budget line and a hitch line — flat = smooth, spikes =
    stutter;
  - **1% low (p99) and 0.1% low (p999)** ms, and a **HITCHES** counter (count +
    per-second), which flashes red briefly when a new hitch lands and names the
    prime-suspect scope;
  - a **top-N scopes** list by ms and % of frame — so the owner instantly sees,
    e.g., ``Update:WeatherMapComponent  11.4 ms (62%)``.

Panda3D imports are allowed here per ARCHITECTURE.md §3.  The overlay reuses its
text nodes and only rebuilds strings / graph geometry on a refresh tick — never
every frame.  It does NOT depend on PStats (that is a separate, optional view).

Example
-------
    overlay = ProfilerOverlay(app, get_profiler(), config)
    app.accept("f3", overlay.toggle)
    # ... each frame: overlay.update()  (throttled internally)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from direct.gui.OnscreenText import OnscreenText  # type: ignore[import]
from panda3d.core import (  # type: ignore[import]
    LineSegs,
    NodePath,
    TextNode,
    Vec4,
)

if TYPE_CHECKING:
    from fire_engine.core.config import Config
    from fire_engine.core.profiler import Profiler


# Colour bands (RGBA) for the frame-time readout.
_GREEN = Vec4(0.45, 0.95, 0.45, 1.0)
_YELLOW = Vec4(1.0, 0.85, 0.30, 1.0)
_RED = Vec4(1.0, 0.35, 0.30, 1.0)
_WHITE = Vec4(0.92, 0.92, 0.92, 1.0)
_DIM = Vec4(0.70, 0.74, 0.80, 1.0)
_FLASH = Vec4(1.0, 0.15, 0.15, 1.0)

_TOP_N_SCOPES = 6


def _budget_color(ratio: float) -> Vec4:
    """Green under budget, yellow approaching (>=80%), red over budget."""
    if ratio > 1.0:
        return _RED
    if ratio >= 0.8:
        return _YELLOW
    return _GREEN


class ProfilerOverlay:
    """
    F3 frame-time / stutter HUD driven by a core :class:`Profiler`.

    Parameters
    ----------
    base : world.app.App (a Panda3D ShowBase)
        Provides ``aspect2d`` / ``getAspectRatio`` and the task clock.
    profiler : Profiler
        The core profiler to read (ring buffer + snapshot).
    config : Config
        Reads ``profiler_frame_budget_ms``, ``profiler_overlay_hz``,
        ``profiler_overlay_graph_frames``.

    Notes
    -----
    Starts hidden; :meth:`toggle` shows/hides it.  :meth:`update` is cheap when
    hidden (early return) and throttled to ``profiler_overlay_hz`` when shown.
    """

    def __init__(self, base, profiler: Profiler, config: Config) -> None:
        self._base = base
        self._prof = profiler
        self._budget = float(getattr(config, "profiler_frame_budget_ms", 5.0))
        hz = float(getattr(config, "profiler_overlay_hz", 8.0))
        self._refresh_period = 1.0 / hz if hz > 0.0 else 0.125
        self._graph_frames = int(getattr(config, "profiler_overlay_graph_frames", 240))

        self._visible = False
        self._last_refresh = -1.0e9
        self._last_hitch_count = 0
        self._flash_ticks = 0  # refresh ticks left to keep the flash on

        # Root under aspect2d; hidden until toggled on.
        self._root: NodePath = base.aspect2d.attach_new_node("profiler_overlay")
        self._root.hide()

        ar = base.getAspectRatio() if hasattr(base, "getAspectRatio") else 1.6
        left = -ar + 0.06
        top = 0.92

        def _text(y: float, scale: float = 0.05) -> OnscreenText:
            return OnscreenText(
                text="",
                parent=self._root,
                scale=scale,
                pos=(left, y),
                align=TextNode.ALeft,
                fg=_WHITE,
                shadow=(0, 0, 0, 0.85),
                mayChange=True,
            )

        self._t_frame = _text(top, 0.060)
        self._t_lows = _text(top - 0.075)
        self._t_hitch = _text(top - 0.135)
        self._t_scopes = _text(top - 0.205, 0.045)

        # Frame-time graph: a LineSegs polyline rebuilt each refresh under this
        # node.  Lay it out below the text block.
        self._graph_root: NodePath = self._root.attach_new_node("ptimer_graph")
        self._graph_left = left
        self._graph_right = min(ar - 0.06, left + 1.2)
        self._graph_top = top - 0.62
        self._graph_height = 0.34

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def toggle(self) -> None:
        """Show/hide the overlay (F3).  Forces a refresh when shown."""
        self._visible = not self._visible
        if self._visible:
            self._root.show()
            self._last_refresh = -1.0e9  # force an immediate refresh
            self.update()
        else:
            self._root.hide()

    @property
    def visible(self) -> bool:
        return self._visible

    # ------------------------------------------------------------------
    # Per-frame entry (throttled)
    # ------------------------------------------------------------------

    def update(self) -> None:
        """
        Called every frame by the app.  Cheap when hidden; refreshes the text +
        graph only at ``profiler_overlay_hz``.
        """
        if not self._visible or not self._prof.enabled:
            return
        import time as _time

        now = _time.perf_counter()
        if now - self._last_refresh < self._refresh_period:
            return
        self._last_refresh = now
        self._refresh()

    # ------------------------------------------------------------------
    # Refresh (rebuild strings + graph) — runs at low Hz only
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        snap = self._prof.snapshot()
        fm = snap["frame_ms"]
        budget = self._budget
        frame_ms = self._prof.last_frame_ms
        ratio = (frame_ms / budget) if budget > 0.0 else 0.0
        fps = (1000.0 / frame_ms) if frame_ms > 0.0 else 0.0

        # Line 1: frame ms / FPS / budget headroom.
        self._t_frame.setText(
            f"{frame_ms:5.2f} ms   {fps:4.0f} FPS   budget {budget:.1f} ms ({ratio * 100:3.0f}%)"
        )
        self._t_frame.setFg(_budget_color(ratio))

        # Line 2: 1% / 0.1% lows + over-budget %.
        self._t_lows.setText(
            f"1% low {fm['p99']:5.1f} ms   0.1% low {fm['p999']:5.1f} ms   "
            f"over-budget {snap['over_budget_pct']:3.0f}%"
        )
        self._t_lows.setFg(_DIM)

        # Line 3: hitch counter (flashes red on a new hitch, names the suspect).
        h = snap["hitches"]
        if h["count"] > self._last_hitch_count:
            self._flash_ticks = 4
        self._last_hitch_count = h["count"]
        suspect = ""
        if h["recent"]:
            r = h["recent"][0]
            suspect = f"  last: {r['prime_suspect']} @ {r['ms']:.1f} ms"
        self._t_hitch.setText(
            f"HITCHES {h['count']} ({h['per_second']:.1f}/s)  "
            f"thr {h['threshold_ms']:.1f} ms{suspect}"
        )
        if self._flash_ticks > 0:
            self._t_hitch.setFg(_FLASH)
            self._flash_ticks -= 1
        else:
            self._t_hitch.setFg(_RED if h["count"] else _DIM)

        # Lines 4+: top-N scopes by mean ms.
        lines = []
        for s in snap["scopes"][:_TOP_N_SCOPES]:
            lines.append(
                f"{s['name']:<28} {s['mean_ms']:6.2f} ms "
                f"({s['pct_of_frame']:3.0f}%)  x{s['calls_per_frame']:.0f}"
            )
        self._t_scopes.setText("\n".join(lines) if lines else "(no scopes yet)")
        self._t_scopes.setFg(_WHITE)

        self._redraw_graph()

    def _redraw_graph(self) -> None:
        """Rebuild the frame-time polyline + budget/hitch reference lines."""
        self._graph_root.node().removeAllChildren()
        frames = self._prof.recent_frame_ms(self._graph_frames)
        if frames.size < 2:
            return

        budget = self._budget
        thr = self._prof._last_threshold_ms
        # Display scale: fit the worse of (4× budget, observed max) so spikes
        # are visible but the budget line isn't squashed flat.
        ymax = max(budget * 4.0, float(frames.max()), thr * 1.2, 1e-3)

        x0, x1 = self._graph_left, self._graph_right
        ytop = self._graph_top
        h = self._graph_height
        n = frames.size

        def x_of(i: int) -> float:
            return x0 + (x1 - x0) * (i / (n - 1))

        def y_of(ms: float) -> float:
            return ytop - h + h * min(ms / ymax, 1.0)

        # Background box.
        box = LineSegs()
        box.set_color(0.30, 0.32, 0.38, 0.7)
        box.set_thickness(1.0)
        box.move_to(x0, 0.0, ytop - h)
        box.draw_to(x1, 0.0, ytop - h)
        box.draw_to(x1, 0.0, ytop)
        box.draw_to(x0, 0.0, ytop)
        box.draw_to(x0, 0.0, ytop - h)
        self._graph_root.attach_new_node(box.create())

        # Budget reference line (green) + hitch threshold line (orange).
        ref = LineSegs()
        ref.set_thickness(1.0)
        ref.set_color(0.45, 0.95, 0.45, 0.8)
        yb = y_of(budget)
        ref.move_to(x0, 0.0, yb)
        ref.draw_to(x1, 0.0, yb)
        if thr > 0.0:
            ref.set_color(1.0, 0.6, 0.25, 0.8)
            yt = y_of(thr)
            ref.move_to(x0, 0.0, yt)
            ref.draw_to(x1, 0.0, yt)
        self._graph_root.attach_new_node(ref.create())

        # Frame-time polyline (white, red segments above the hitch line).
        poly = LineSegs()
        poly.set_thickness(1.5)
        poly.set_color(0.92, 0.92, 0.92, 1.0)
        poly.move_to(x_of(0), 0.0, y_of(float(frames[0])))
        for i in range(1, n):
            ms = float(frames[i])
            if thr > 0.0 and ms > thr:
                poly.set_color(1.0, 0.35, 0.30, 1.0)
            else:
                poly.set_color(0.92, 0.92, 0.92, 1.0)
            poly.draw_to(x_of(i), 0.0, y_of(ms))
        self._graph_root.attach_new_node(poly.create())

    def destroy(self) -> None:
        """Tear down the overlay nodes."""
        self._root.remove_node()
