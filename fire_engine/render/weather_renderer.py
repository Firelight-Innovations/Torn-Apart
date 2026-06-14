"""
world/weather_renderer.py — weather-map GPU upload + uniform binding (render component).

``WeatherMapComponent`` is the world-side half of the M4 GPU weather contract.
It owns the headless :class:`~fire_engine.world.weather.WeatherMap` (``weather/`` is
panda3d-free per Hard Rule 1), re-rasters the four spatial weather channels
(coverage, density, precip, fog) around the player a few times a second, packs
the raster into a small 2-D float16 texture with
:func:`~fire_engine.world.sky.pack_weather_map`, and binds the **weather-map uniform
contract** on ``base.render`` so the volumetric-cloud raymarch (and later rain /
fog / wetness passes) can sample one spatially-varying weather field instead of
the flat ``SkyState`` scalars.

This is the exact structural twin of ``world/wind_renderer.py`` — same
committed-origin discipline, same fp16-BGRA upload path — for the weather map
instead of the wind field.

The contract (bound here, inherited by every node under ``render``):

    sampler2D u_weather_map        RGBA16F — R=coverage, G=density, B=precip,
                                   A=fog (FT_linear, WM_clamp)
    vec2  u_wmap_origin            world XY (m) of the map's MIN corner (texel
                                   (0,0)'s corner) — refreshed ONLY together
                                   with a texture upload
    float u_wmap_cell_m            texel edge in meters
    float u_wmap_cells             texels per axis
    int   u_weather_map_enabled    0 (boot default / kill-switch off) / 1 once
                                   the first upload has landed
    vec2  u_weather_ambient        (coverage, density) fallback beyond the map
                                   edge — the local SkyState values, so the
                                   edge-fade is seamless
    int   u_virga_enabled          0 / 1 — gray rain shafts below storm bases

Decode in any shader::

    float span = u_wmap_cell_m * u_wmap_cells;
    vec2 uv = (world_xy - u_wmap_origin) / span;     // RAW world XY — see below
    vec4 wm = texture(u_weather_map, uv);            // R=cov G=den B=precip A=fog

CRITICAL — sample the map at the **raw world XY**, never ``world_xy + u_wind``:
the weather map already encodes cell motion (storm cells drift on the synoptic
flow, baked into the raster each re-raster), so adding the wind drift would
double-advect the storm.  ``u_wind`` belongs only on the procedural noise
lookups in the cloud shader.

Committed-origin discipline
---------------------------
``u_wmap_origin`` is refreshed **only in the same frame as a texture upload**,
never on a bare recenter — the same discipline ``wind_renderer.py`` and
``lighting/gpu.py`` follow.  The map recenters (and re-uploads) only when the
player has moved more than ~half the map span from the committed center, so the
common per-frame case is a single cheap ``texture()`` bind with no work.

Driver coordination
--------------------
The :class:`~fire_engine.render.sky_renderer.SkyRendererComponent` is the single
driver of ``sky_system.update(player_pos)`` (threading the camera XY through so
distant storms sample at the camera).  This component therefore NEVER calls
``update`` — it only reads the already-advanced weather system through
:meth:`WeatherMap.rasterize`, a pure function of (weather seed, center, t_abs).
Because the component registry runs all ``update()`` before any
``late_update()``, the weather system is fresh by the time we raster here.

Texture format
--------------
``Texture.T_half_float`` + ``Texture.F_rgba16`` — the same true-half-float path
``wind_renderer.py`` pioneered; :func:`~fire_engine.world.sky.pack_weather_map`
emits exactly the matching fp16 BGRA buffer, so the upload is a no-repack
``set_ram_image``.  ``FT_linear`` (smooth weather field) + ``WM_clamp`` (the
shader's own edge-fade takes over outside, but clamping keeps the border
texels well-defined).

Master kill switch
------------------
``config.gfx_weather_map`` (False/absent) ⇒ the component binds
``u_weather_map_enabled = 0`` and does no per-frame work, so the cloud shader
falls back to the flat ambient scalars (the pre-M4 look).  ``config.gfx_clouds``
False ⇒ there is no cloud pass to feed, but the contract is still bound (cheap)
so future passes work.

Example (wired by main.py)
--------------------------
    weather_go = instantiate()
    weather_go.add_component(
        WeatherMapComponent,
        base=app, sky_system=sky_system)
"""

from __future__ import annotations

from typing import Any

# Panda3D imports allowed in world/ per ARCHITECTURE §3.
from panda3d.core import (  # type: ignore[import]
    LVecBase2f,
    SamplerState,
    Texture,
)

from fire_engine.core import get_logger
from fire_engine.world.sky import pack_weather_map
from fire_engine.world.weather import WeatherMap
from fire_engine.render.component import Component

__all__ = ["WeatherMapComponent"]

_log = get_logger("world.weather_map")

#: Monotonic game seconds per day (mirrors weather/system.py::_DAY_S and
#: clock.py's per-day constant) — used to build the absolute sample time the
#: raster is a pure function of.
_GAME_SECONDS_PER_DAY: float = 24.0 * 3600.0

#: Re-raster cadence while the player is stationary: every N frames (~a few Hz
#: at 60 FPS), so drifting storm cells animate without per-frame raster cost.
_RERASTER_EVERY_N_FRAMES: int = 12


class WeatherMapComponent(Component):
    """
    Render component that uploads the weather map and binds its uniforms.

    Parameters (pass as ``add_component`` kwargs)
    ---------------------------------------------
    base : world.app.App
        The application — provides ``render``, ``camera_go`` and ``_config``.
    sky_system : fire_engine.world.sky.SkySystem
        Read-only weather source.  Its ``.weather`` (the
        :class:`~fire_engine.world.weather.WeatherSystem`) is rasterised here; its
        ``.state`` supplies the ambient coverage/density fallback.  The
        ``SkyRendererComponent`` owns the per-frame ``update()`` call.
    clock : fire_engine.core.Clock | None
        Optional; supplies ``game_day`` / ``game_time_of_day`` for the raster's
        absolute sample time.  Defaults to ``sky_system._clock`` when omitted.

    Units: meters, game seconds.  World-space Z-up.
    """

    def __init__(self, base: Any = None, sky_system: Any = None,
                 clock: Any = None) -> None:
        super().__init__()
        self.base = base
        self.sky_system = sky_system
        self.clock = clock
        if self.clock is None and sky_system is not None:
            self.clock = getattr(sky_system, "clock", None) or \
                getattr(sky_system, "_clock", None)

        self._map: WeatherMap | None = None
        self._tex: Texture | None = None
        self._enabled_feature: bool = False     # gfx_weather_map
        self._virga: bool = False               # gfx_cloud_virga
        self._uploaded_once: bool = False
        # Committed center the current texels were rastered around (None until
        # the first upload); recenter only when the player leaves half a span.
        self._center: tuple[float, float] | None = None
        self._half_span_m: float = 0.0
        self._frame: int = 0                    # for the few-Hz re-raster cadence

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Allocate the weather texture and bind the uniform contract (once)."""
        if self.base is None or self.sky_system is None:
            _log.warning("WeatherMapComponent: missing base/sky_system — "
                         "disabled (clouds use flat ambient weather)")
            self.enabled = False
            return

        cfg = self.base._config
        self._enabled_feature = bool(getattr(cfg, "gfx_weather_map", False))
        self._virga = bool(getattr(cfg, "gfx_cloud_virga", False))

        root = self.base.render
        # Bind the static half of the contract + the boot-default kill state so
        # every shader has every uniform defined from frame 0.  u_weather_map
        # needs a bound sampler even when disabled (an unbound sampler2D is UB).
        self._map = WeatherMap(cfg)
        self._half_span_m = 0.5 * self._map.span_m

        tex = Texture("weather_map")
        tex.setup_2d_texture(self._map.cells, self._map.cells,
                             Texture.T_half_float, Texture.F_rgba16)
        tex.set_minfilter(SamplerState.FT_linear)
        tex.set_magfilter(SamplerState.FT_linear)
        tex.set_wrap_u(SamplerState.WM_clamp)
        tex.set_wrap_v(SamplerState.WM_clamp)
        tex.set_clear_color((0.0, 0.0, 0.0, 0.0))
        tex.set_keep_ram_image(False)
        self._tex = tex

        root.set_shader_input("u_weather_map", tex)
        root.set_shader_input("u_wmap_cell_m", float(self._map.cell_m))
        root.set_shader_input("u_wmap_cells", float(self._map.cells))
        root.set_shader_input("u_wmap_origin", LVecBase2f(0.0, 0.0))
        root.set_shader_input("u_weather_map_enabled", 0)
        root.set_shader_input("u_weather_ambient", LVecBase2f(0.0, 0.0))
        root.set_shader_input("u_virga_enabled", 1 if self._virga else 0)

        if not self._enabled_feature:
            _log.info("Weather map disabled (gfx_weather_map=false) — clouds "
                      "use flat ambient weather (pre-M4 look)")
            return

        _log.info("Weather map online: %dx%d raster, %.0f m texels (%.0f m / "
                  "%.1f km span), virga=%s",
                  self._map.cells, self._map.cells, self._map.cell_m,
                  self._map.span_m, self._map.span_m / 1000.0,
                  "on" if self._virga else "off")

    def late_update(self, dt: float) -> None:
        """Re-raster around the player (on recenter), upload, refresh origin."""
        if not self._enabled_feature or self._tex is None or self._map is None:
            return

        cam_x, cam_y = self._camera_xy()

        # Recenter only when the player has left half a span from the committed
        # center (committed-origin discipline) — the first frame always rasters.
        if self._center is None or \
                abs(cam_x - self._center[0]) > self._half_span_m or \
                abs(cam_y - self._center[1]) > self._half_span_m:
            self._reraster(cam_x, cam_y)
            return

        # Otherwise refresh a few times a second so drifting cells animate even
        # while the player stands still (the raster is a pure fn of t_abs).
        self._frame += 1
        if (self._frame % _RERASTER_EVERY_N_FRAMES) == 0:
            self._reraster(self._center[0], self._center[1])

    def on_destroy(self) -> None:
        """Drop the texture reference (the render graph owns the binding)."""
        self._tex = None
        self._map = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reraster(self, center_x: float, center_y: float) -> None:
        """Raster around (center_x, center_y), upload, and commit the origin."""
        weather = getattr(self.sky_system, "weather", None)
        if weather is None:
            return
        t_abs = self._t_abs()
        raster = self._map.rasterize(weather, (center_x, center_y), t_abs)
        self._tex.set_ram_image(pack_weather_map(raster))

        # Committed-origin discipline: refresh u_wmap_origin ONLY here, in the
        # same frame as the upload, so texels and origin never disagree.  The
        # origin is the map's MIN corner = center − half-span.
        root = self.base.render
        ox = center_x - self._half_span_m
        oy = center_y - self._half_span_m
        root.set_shader_input("u_wmap_origin", LVecBase2f(float(ox), float(oy)))

        # Ambient fallback beyond the map edge = the local SkyState weather, so
        # the shader's edge-fade lands on the same values the sky is showing.
        st = getattr(self.sky_system, "state", None)
        amb_cov = float(getattr(st, "cloud_coverage", 0.0)) if st else 0.0
        amb_den = float(getattr(st, "cloud_density", 0.0)) if st else 0.0
        root.set_shader_input("u_weather_ambient",
                              LVecBase2f(amb_cov, amb_den))

        self._center = (center_x, center_y)
        if not self._uploaded_once:
            root.set_shader_input("u_weather_map_enabled", 1)
            self._uploaded_once = True

    def _camera_xy(self) -> tuple[float, float]:
        """World-space camera XY in meters (engine transform authority)."""
        go = getattr(self.base, "camera_go", None)
        if go is not None:
            p = go.transform.position
            return float(p.x), float(p.y)
        cp = self.base.camera.get_pos(self.base.render)
        return float(cp.x), float(cp.y)

    def _t_abs(self) -> float:
        """Absolute game seconds for the raster (day·86400 + time-of-day)."""
        if self.clock is None:
            return 0.0
        return (float(self.clock.game_day) * _GAME_SECONDS_PER_DAY
                + float(self.clock.game_time_of_day))
