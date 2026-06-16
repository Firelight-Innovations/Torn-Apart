"""
world/sky_renderer.py — SkyRendererComponent: the render half of the sky system.

Draws everything the headless ``fire_engine.world.sky`` simulation describes through
its per-frame ``SkyState``:

* **Sky dome** — an inverted UV-sphere (radius ~800 m) camera-centred via
  translation-only follow, painted by ``sky_shaders.SKY_DOME_FRAGMENT``:
  physical atmosphere, sun disc + halo, moon with phase terminator, the
  ``"night_sky_cube"`` star/galaxy CUBE MAP (per-star twinkle, no pole
  distortion) rotating about a seed-derived TILTED celestial axis, and
  deterministic shooting stars.
* **Volumetric clouds** — a second camera-centred inverted sphere whose
  fragment shader (``sky_shaders.CLOUD_VOLUMETRIC_FRAGMENT``) raymarches the
  cloud slab sampling the baked tileable 3-D noise (``sky.cloud_noise``):
  self-shadowed (Beer + powder), HG forward-scatter silver lining, premultiplied
  OVER the sky so a bright sun bleeds through thin cloud.  Gated by ``gfx_clouds``.
* **Fog + global light** — exponential ``panda3d.core.Fog`` on the terrain
  root (density ``SkyState.fog_density`` 1/m, colour ``fog_color``), window
  clear colour blended toward fog, and ``terrain_root.set_color_scale`` driven
  by ``SkyState.terrain_light_scale`` (day/night modulation of the baked
  vertex sunlight).

Units: meters / seconds / radians, world Z-up.  All scene-graph geometry is
built ONCE in ``start()`` with bulk numpy → memoryview writes (Hard Rule 7);
per-frame work is a fixed handful of ``set_shader_input`` / ``set_pos`` /
``set_color_scale`` calls.

This module imports panda3d and therefore lives in ``world/`` (Hard Rule 1).
``fire_engine.render.__init__`` exports ``SkyRendererComponent`` behind the same
try/except guard as the other panda3d bridges.

Example
-------
::

    from fire_engine.world.sky import SkySystem
    from fire_engine.render import instantiate
    from fire_engine.render.sky.sky_renderer import SkyRendererComponent

    sky_system = SkySystem(cfg, clock, bus)
    sky_go = instantiate()
    sky_go.name = "Sky"
    sky_go.add_component(
        SkyRendererComponent,
        base=app,                       # the world.app.App (ShowBase)
        sky_system=sky_system,          # drives + reads SkyState
        terrain_root=app.terrain_root,  # receives fog + light colour scale
        clock=clock,                    # optional; shooting-star scheduling
    )

Docs: docs/systems/render.sky.md
"""

from __future__ import annotations

from typing import Any

# Panda3D imports are allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (
    Fog,
    NodePath,
)

from fire_engine.core import get_logger
from fire_engine.render.component import Component
from fire_engine.render.sky._impl.sky_build import (
    build_clouds,
    build_dome,
)
from fire_engine.render.sky._impl.sky_geom import (
    _CAMERA_FAR_M,
    _DEFAULT_STAR_COUNT,
    _DOME_RADIUS_M,
    _DOME_SLICES,
    _DOME_STACKS,
    _VCLOUD_ALT_M,
    _VCLOUD_THICK_M,
    _build_dome_node,
    _clamp01,
    _fallback_moon,
    _fallback_star_cube,
    _load_or_bake_cloud_noise,
    _make_geom_node,
    _sky_texture,
)
from fire_engine.render.sky._impl.sky_update import (
    update_clouds,
    update_dome,
    update_fog_and_light,
    update_shooting_star,
)

__all__ = ["SkyRendererComponent"]

_log = get_logger("world.sky_renderer")

# Re-export constants so any existing code that imports them from this module
# still works (no public API break).
__all__ += [
    "_CAMERA_FAR_M",
    "_DOME_RADIUS_M",
    "_DOME_SLICES",
    "_DOME_STACKS",
    "_VCLOUD_ALT_M",
    "_VCLOUD_THICK_M",
    "_build_dome_node",
    "_clamp01",
    "_fallback_moon",
    "_fallback_star_cube",
    "_load_or_bake_cloud_noise",
    "_make_geom_node",
    "_sky_texture",
]

# Shooting stars: deterministic schedule (see _update_shooting_star).
_SS_SLOT_GAME_S: float = 1800.0  # one slot = 30 game-minutes (game seconds)
_SS_DURATION_REAL_S: float = 1.2  # streak animation length (real seconds)
_SS_SPAWN_P: float = 0.5  # spawn probability per slot
_SS_MIN_STAR_VIS: float = 0.5  # only spawn when stars are visible

_GAME_SECONDS_PER_DAY: float = 86400.0


# ---------------------------------------------------------------------------
# SkyRendererComponent
# ---------------------------------------------------------------------------


class SkyRendererComponent(Component):
    """
    Render component for the procedural sky + weather system.

    Lifecycle
    ---------
    * ``start()`` — builds all scene-graph nodes ONCE (dome, cloud quads, Fog
      object), compiles the GLSL shaders, uploads textures, and extends the
      camera far plane to cover the sky geometry.  (Rain now lives in
      ``world/rain_renderer.py`` — M6.)
    * ``update(dt)`` — drives ``sky_system.update()`` (the registry runs all
      updates before any late_update, so the state is fresh for every reader
      this frame without touching App).
    * ``late_update(dt)`` — reads the frame's ``SkyState`` and writes the
      per-frame render state: a fixed handful of ``set_shader_input`` /
      ``set_pos`` / ``set_color_scale`` calls (never per-element loops).

    Parameters (pass as ``add_component`` kwargs)
    ---------------------------------------------
    base : world.app.App (ShowBase)
        The application — provides ``render``, ``camLens``, ``camera_go``,
        ``set_background_color`` and (optionally) ``_config``.
    sky_system : fire_engine.world.sky.SkySystem (or duck-typed stub)
        ``update() -> SkyState`` once per frame; only SkyState attributes are
        read, so any object with the contract fields works (see
        ``tools/screenshot.py --stub-sky``).
    terrain_root : panda3d NodePath
        Receives the exponential Fog and the day/night ``set_color_scale``
        (baked vertex sunlight × global scale = the lighting integration).
    clock : core.clock.Clock | None
        Optional; used for star rotation + the deterministic shooting-star
        schedule.  Defaults to ``sky_system.clock`` / ``sky_system._clock``
        when available, else those features idle.

    Units: meters, seconds, radians.  All directions world-space Z-up.

    Docs: docs/systems/render.sky.md
    """

    # Class-level annotations for attributes read/written by _impl functions.
    # These are instance attributes set in __init__ / start(); annotating them
    # here lets mypy --strict resolve them without inspecting the _impl modules.
    base: Any
    sky_system: Any
    terrain_root: Any
    clock: Any
    external_lighting: bool
    _state: Any
    _time_s: float
    _wind_x_m: float
    _wind_y_m: float
    _dome_np: NodePath | None
    _cloud_np: NodePath | None
    _fog: Fog | None
    _ss_slot: tuple[int, int] | None
    _ss_progress: float
    _fog_tex_bound: bool

    def __init__(
        self,
        base: Any = None,
        sky_system: Any = None,
        terrain_root: Any = None,
        clock: Any = None,
        external_lighting: bool = False,
    ) -> None:
        super().__init__()
        self.base = base
        self.sky_system = sky_system
        self.terrain_root = terrain_root
        self.clock = clock
        # True when the GPU volumetric lighting pipeline owns terrain
        # shading: the sky renderer must then NOT apply Panda3D Fog, NOT
        # set terrain_root colour-scale, and NOT blend the clear colour
        # toward fog (the froxel fog composites in the terrain/sky shaders).
        self.external_lighting = bool(external_lighting)
        if self.clock is None and sky_system is not None:
            self.clock = getattr(sky_system, "clock", None) or getattr(sky_system, "_clock", None)

        # Per-frame state
        self._state: Any = None  # last SkyState from sky_system.update()
        self._time_s: float = 0.0  # real seconds since start (twinkle)
        self._wind_x_m: float = 0.0  # accumulated wind drift (meters)
        self._wind_y_m: float = 0.0

        # Scene-graph nodes (built in start())
        self._dome_np: NodePath | None = None
        self._cloud_np: NodePath | None = None
        self._fog: Fog | None = None

        # Shooting-star animation state
        self._ss_slot: tuple[int, int] | None = None  # (game_day, slot)
        self._ss_progress: float = -1.0  # < 0 → inactive
        self._fog_tex_bound: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build all sky scene-graph nodes, shaders, and textures (once)."""
        if self.base is None:
            _log.warning("SkyRendererComponent.start: no base — disabled")
            self.enabled = False
            return

        cfg = getattr(self.base, "_config", None)
        star_count = int(getattr(cfg, "sky_star_count", _DEFAULT_STAR_COUNT))

        # Far plane must cover the sky dome (800 m); the volumetric cloud slab
        # is intersected analytically in the shader, so it needs no geometry.
        lens = self.base.camLens
        if lens.get_far() < _CAMERA_FAR_M:
            lens.set_far(_CAMERA_FAR_M)

        build_dome(self, star_count)
        build_clouds(self)

        # Exponential fog on the terrain (density/colour driven per frame).
        # Skipped under external (GPU volumetric) lighting — the froxel fog
        # composites inside the terrain shader instead.
        if not self.external_lighting:
            self._fog = Fog("sky_fog")
            self._fog.set_exp_density(0.0008)
            if self.terrain_root is not None:
                self.terrain_root.set_fog(self._fog)

        _log.info(
            "Sky renderer ready (volumetric cloud slab z=[%.0f, %.0f] m)",
            _VCLOUD_ALT_M,
            _VCLOUD_ALT_M + _VCLOUD_THICK_M,
        )

    def update(self, dt: float) -> None:
        """
        Advance the sky simulation for this frame.

        The component registry runs ALL update() calls before any
        late_update(), so calling ``sky_system.update()`` here guarantees a
        fresh ``SkyState`` for ``late_update`` without modifying App.

        This is the single driver of ``sky_system.update(player_pos)`` — it
        threads the camera world XY through so distant storms sample at the
        camera (M4).  Other readers (wind, the WeatherMapComponent's raster)
        consume the already-advanced weather system, so nothing else calls
        ``update`` (no double-advance).
        """
        self._time_s += dt
        if self.sky_system is not None:
            cx, cy, _ = self._camera_pos()
            self._state = self.sky_system.update((cx, cy))

    def late_update(self, dt: float) -> None:
        """Write this frame's SkyState to the GPU (bulk uniform/state writes)."""
        st = self._state if self._state is not None else getattr(self.sky_system, "state", None)
        if st is None or self._dome_np is None:
            return

        cx, cy, cz = self._camera_pos()

        # Wind drift accumulates in meters; the cloud shader shifts its grid by it.
        wx, wy = st.wind_dir
        self._wind_x_m += float(wx) * float(st.wind_speed) * dt
        self._wind_y_m += float(wy) * float(st.wind_speed) * dt

        update_dome(self, st, cx, cy, cz)
        update_shooting_star(self, st, dt)
        update_clouds(self, st, cx, cy, cz)
        update_fog_and_light(self, st)

    def on_destroy(self) -> None:
        """Detach all sky nodes and clear the terrain fog."""
        for np_node in (self._dome_np, self._cloud_np):
            if np_node is not None:
                np_node.remove_node()
        self._dome_np = self._cloud_np = None
        if self.terrain_root is not None and self._fog is not None:
            self.terrain_root.clear_fog()
            self.terrain_root.clear_color_scale()

    # ------------------------------------------------------------------
    # Per-frame helpers
    # ------------------------------------------------------------------

    def _camera_pos(self) -> tuple[float, float, float]:
        """World-space camera position in meters (engine transform authority)."""
        go = getattr(self.base, "camera_go", None)
        if go is not None:
            p = go.transform.position
            return float(p.x), float(p.y), float(p.z)
        cp = self.base.camera.get_pos(self.base.render)
        return float(cp.x), float(cp.y), float(cp.z)

    @staticmethod
    def _fog_blend(st: Any) -> float:
        """
        Map ``fog_density`` (1/m, ~0.0008 clear … 0.025 heavy) to a 0-1 factor
        for blending the horizon band / clear colour toward the fog colour.
        """
        return _clamp01((float(st.fog_density) - 0.0008) / (0.020 - 0.0008))
