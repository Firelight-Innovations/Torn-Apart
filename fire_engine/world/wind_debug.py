"""
world/wind_debug.py — dev-only wind-field physics seam proof (the "wind ball").

``WindBallDebugComponent`` is a **developer diagnostic**, gated behind the
``[debug] debug_wind_ball`` config flag, that proves the wind field is genuinely
CPU-sampleable for physics: a small bright ball rests on the flat ground near
spawn and each fixed step is shoved by the *same*
:meth:`~fire_engine.wind.WindField.sample` the future physics/audio systems will
call.  When a gust band sweeps across it the ball visibly scoots downwind; in a
storm (``force_weather(STORM)``, F6) it rolls hard.  It is the in-engine
counterpart of the headless ``tests/test_wind_ball.py`` assertion.

Design
------
- The physics is the panda3d-free pure function
  :func:`fire_engine.wind.debug_ball_step` (headless-tested); this component is
  only the glue: it samples the field at the ball's position, steps the
  integrator in ``fixed_update`` (fixed 50 Hz, so the motion is frame-rate
  independent), and writes the result onto a NodePath.
- The ball geometry is a procedural UV-sphere built in code — **no asset** — and
  rendered **unlit, full-bright, emissive-looking** (``set_light_off`` + a
  bright flat colour) so it is obviously visible on either lighting backend and
  reads as a debug gizmo, not scene content.
- Unlike grass/motes this component does **not** require the GPU lighting
  pipeline: it only needs a ``WindField`` to sample.  It disables itself (with a
  log line) when the flag is off or no field was built.

Why a separate dev component (not folded into the wind renderer): the wind
renderer is the production upload path on every run; this is throwaway
diagnostic geometry that must not exist unless a developer asks for it.

Example (wired by main.py, behind the flag)
-------------------------------------------
    if cfg.debug_wind_ball:
        ball_go = instantiate()
        ball_go.add_component(WindBallDebugComponent,
                              base=app, clock=clock, wind_field=wind_field)
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (  # type: ignore[import]
    GeomNode,
    NodePath,
)

from fire_engine.core import get_logger
from fire_engine.wind import BallParams, debug_ball_step
from fire_engine.world.component import Component
from fire_engine.world.primitives import build_sphere_geom as _build_sphere_geom

__all__ = ["WindBallDebugComponent"]

_log = get_logger("world.wind_debug")

# Where the ball starts, relative to spawn.  Spawn camera is at (0, -20, 10)
# looking +Y; the demo grass volume is x∈[-12,12], y∈[-5,25].  Drop the ball a
# few meters into that field, dead ahead, so a gust crossing the grass crosses
# the ball too.
_BALL_START_XY = (0.0, 2.0)
# A bright, debug-obvious colour (warm orange, full-bright).
_BALL_COLOR = (1.0, 0.45, 0.1, 1.0)


class WindBallDebugComponent(Component):
    """
    Dev-only ball pushed by the wind field — a physics-sampling seam proof.

    Parameters (pass as ``add_component`` kwargs)
    ---------------------------------------------
    base : world.app.App
        The application — provides ``render`` (the ball is parented here, not
        under ``terrain_root``, since it is unlit debug geometry) and
        ``_config`` (ground height, ball tuning).
    clock : fire_engine.core.Clock
        The shared clock; ``game_day`` + ``game_time_of_day`` give the monotonic
        absolute game time the wind field is evaluated at (same convention as
        ``WindSystemComponent``).
    wind_field : fire_engine.wind.WindField | None
        The headless field to sample.  ``None`` disables the component.
    radius_m : float, default 0.4
        Ball radius in meters.
    sky_system : fire_engine.sky.SkySystem | None
        Optional weather source; its ``state`` (duck-typed) is passed to
        ``WindField.update`` so a forced storm gusts the ball.  If a separate
        ``WindSystemComponent`` is already calling ``update`` each frame this may
        be left ``None`` (this component then only ever *samples*).

    Units: meters, seconds, m/s.  World-space Z-up.
    """

    def __init__(self, base: Any = None, clock: Any = None,
                 wind_field: Any = None, radius_m: float = 0.4,
                 sky_system: Any = None) -> None:
        super().__init__()
        self.base = base
        self.clock = clock
        self.wind_field = wind_field
        self.sky_system = sky_system
        self._radius_m = float(radius_m)

        self._node: NodePath | None = None
        self._params: BallParams | None = None
        self._pos = np.zeros(3, dtype=np.float64)
        self._vel = np.zeros(3, dtype=np.float64)
        self._wind_time: float | None = None   # seeded on first CPU-path update

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build the ball geometry and seat it on the ground near spawn."""
        if self.base is None or self.wind_field is None:
            _log.warning("WindBallDebugComponent: missing base/wind_field — "
                         "disabled")
            self.enabled = False
            return

        cfg = self.base._config
        ground_z = float(cfg.ground_height_m)
        self._params = BallParams(
            ground_z=ground_z,
            radius_m=self._radius_m,
        )

        # Rest the ball on the ground at the start XY (centre = ground + radius).
        self._pos = np.array(
            [_BALL_START_XY[0], _BALL_START_XY[1], ground_z + self._radius_m],
            dtype=np.float64)
        self._vel = np.zeros(3, dtype=np.float64)

        geom_node = GeomNode("wind_debug_ball")
        geom_node.add_geom(_build_sphere_geom(self._radius_m))
        node = self.base.render.attach_new_node(geom_node)
        # Unlit + full-bright so it reads as a debug gizmo on either backend.
        node.set_light_off()
        node.set_color(*_BALL_COLOR)
        node.set_shader_off()                 # ignore any inherited GPU shader
        node.set_pos(float(self._pos[0]), float(self._pos[1]),
                     float(self._pos[2]))
        self._node = node

        _log.info("Wind debug ball online at (%.1f, %.1f, %.1f), r=%.2f m — "
                  "watch it scoot when a gust crosses (storm = rolls hard)",
                  self._pos[0], self._pos[1], self._pos[2], self._radius_m)

    def fixed_update(self, dt: float) -> None:
        """Sample the wind at the ball and step the pure integrator (50 Hz)."""
        if self._node is None or self.wind_field is None \
                or self._params is None:
            return

        # If we own a sky_system we also drive the field's update; otherwise we
        # assume the WindSystemComponent already published this frame's snapshot
        # and we just sample it.  Sampling needs a published snapshot — guard so
        # a frame before the first wind update is a harmless no-op.
        if self.sky_system is not None and self.clock is not None:
            # Wind clock: real seconds × wind_time_scale, independent of the
            # game timescale (mirror of wind_renderer.py — gusts are an
            # aesthetic real-time effect).  Seeded once from the game clock so
            # a loaded save resumes at a deterministic phase.
            rate = float(self.base._config.wind_time_scale)
            if self._wind_time is None:
                game_s = (float(self.clock.game_day) * 86400.0
                          + float(self.clock.game_time_of_day))
                scale = max(float(self.clock.game_time_scale), 1e-6)
                self._wind_time = game_s / scale * rate
            self._wind_time += float(dt) * rate
            sky_state = getattr(self.sky_system, "state", None)
            self.wind_field.update(dt, self._wind_time, sky_state,
                                   (float(self._pos[0]), float(self._pos[1]),
                                    float(self._pos[2])))
        try:
            v_wind = self.wind_field.sample(self._pos[None])[0]
        except RuntimeError:
            return  # field not updated yet this run — nothing to sample

        self._pos, self._vel = debug_ball_step(
            self._pos, self._vel, v_wind, dt, self._params)
        self._node.set_pos(float(self._pos[0]), float(self._pos[1]),
                           float(self._pos[2]))

    def on_destroy(self) -> None:
        """Remove the ball geometry."""
        if self._node is not None:
            self._node.remove_node()
            self._node = None


# Sphere geometry moved to world/primitives.py (shared with scene visuals);
# imported above as _build_sphere_geom to keep call sites unchanged.
