"""
sky/weather_map_pack.py — Pack a weather-map raster into GPU texture bytes.

The headless byte-packer for the M4 GPU weather-map contract.  It is the exact
twin of :func:`fire_engine.world.wind.pack_wind_field`: it turns the
``(cells, cells, 4)`` float32 raster that
:meth:`fire_engine.world.weather.WeatherMap.rasterize` produces into a **float16**
buffer in Panda3D's 2-D RAM layout, ready for a single
``Texture(F_rgba16).set_ram_image`` on the render thread
(:class:`fire_engine.render.sky.weather_renderer.WeatherMapComponent`).

Stays in ``sky/`` (headless) rather than ``world/`` per Hard Rule 1: the
panda3d-free packer returns plain ``bytes``; only ``world/`` constructs the
panda3d ``Texture`` and uploads them.

LAYOUT IS PINNED (``tests/test_weather_map_pack.py`` asserts it)
---------------------------------------------------------------
Mirrors ``pack_wind_field``'s transpose + channel-swap discipline exactly:

* **Row-major ``(row=Y, col=X)``** — the raster is already stored
  ``out[row=Y, col=X, channel]`` (see ``weather_map.py``), which is *already*
  Panda3D's row-outer 2-D RAM order, so — unlike ``pack_wind_field``, whose
  field is ``[x, y]`` and therefore transposed — **no transpose is needed
  here**.  We keep the comment explicit because the two packers look like they
  should share a transpose and they deliberately do not.
* **BGRA** channel order (the known engine quirk: ``F_rgba16`` data textures
  upload BGRA even for float data — same swap ``pack_wind_field`` and
  ``lighting/volume.pack_volume`` apply).  The raster's logical RGBA channels
  are ``R=coverage, G=density, B=precip, A=fog`` (matching
  :data:`fire_engine.world.weather.MAP_CHANNELS`); after the BGRA swap the bytes are
  ``B=precip, G=density, R=coverage, A=fog``.

The GPU decode (``cloud_volumetric.frag``) binds this as ``sampler2D
u_weather_map`` and reads ``texture(...).rgba == (coverage, density, precip,
fog)`` — i.e. Panda3D un-swizzles BGRA back to RGBA on sampling, so the shader
sees the logical channel order.  If you change the channel mapping or add a
transpose you MUST update that shader decode and the pinned test together.

No panda3d.  No per-texel Python loops.

Example
-------
>>> import numpy as np
>>> from fire_engine.core import EventBus, load_config, set_world_seed
>>> from fire_engine.world.weather import WeatherSystem, WeatherMap
>>> from fire_engine.world.sky.weather_map_pack import pack_weather_map
>>> set_world_seed(1337)
>>> ws = WeatherSystem(load_config(), EventBus())
>>> wm = WeatherMap(load_config())
>>> raster = wm.rasterize(ws, center_xy=(0.0, 0.0), t_abs=12 * 3600.0)
>>> data = pack_weather_map(raster)
>>> len(data) == raster.shape[0] * raster.shape[1] * 4 * 2   # fp16 RGBA
True

Docs: docs/systems/world.sky.md
"""

from __future__ import annotations

import numpy as np

__all__ = ["pack_weather_map"]


def pack_weather_map(raster: np.ndarray) -> bytes:
    """
    Pack a weather-map raster into Panda3D 2-D-texture RAM bytes.

    Produces a **float16** buffer in Panda3D's 2-D RAM layout: **row-major
    ``(row=Y, col=X)``** with **BGRA** channel order.  The input raster is
    already stored ``[row=Y, col=X, channel]`` (Panda3D's row-outer order), so
    only the RGBA→BGRA channel swap is applied — no transpose (this is the one
    deliberate difference from :func:`fire_engine.world.wind.pack_wind_field`, whose
    field is ``[x, y]`` and therefore transposed).

    Mirrors ``pack_wind_field``'s + ``lighting/volume.pack_volume``'s
    channel-swap convention so the upload is just
    ``Texture(F_rgba16).set_ram_image(bytes)`` on the render thread.  Pure and
    thread-safe (no shared state) — safe to call off the main thread.

    Parameters
    ----------
    raster : numpy.ndarray
        ``(cells, cells, 4)`` float32 raster from
        :meth:`fire_engine.world.weather.WeatherMap.rasterize`.  Channels are
        ``R=coverage, G=density, B=precip, A=fog`` (the
        :data:`fire_engine.world.weather.MAP_CHANNELS` order), all dimensionless
        ``[0, 1]``.  ``out[row=Y, col=X, channel]`` layout.

    Returns
    -------
    bytes
        ``cells * cells * 4 * 2`` bytes of little-endian float16, BGRA, ready
        for ``Texture.T_half_float`` + ``Texture.F_rgba16``
        ``set_ram_image``.

    Raises
    ------
    ValueError
        If *raster* is not ``(N, N, 4)``.

    Example
    -------
    >>> import numpy as np
    >>> r = np.zeros((4, 4, 4), dtype=np.float32)
    >>> r[..., 0] = 0.5            # coverage
    >>> data = pack_weather_map(r)
    >>> len(data) == 4 * 4 * 4 * 2
    True

    Docs: docs/systems/world.sky.md
    """
    arr = np.asarray(raster)
    if arr.ndim != 3 or arr.shape[2] != 4 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"raster must be (N, N, 4); got {arr.shape}")

    # Logical RGBA per texel: R=coverage, G=density, B=precip, A=fog.  Swap to
    # BGRA (the F_rgba16 data-texture quirk) — the raster is already row-major
    # (row=Y, col=X), so NO transpose (cf. pack_wind_field, which transposes its
    # [x, y] field).  Panda3D un-swizzles BGRA→RGBA on sample, so the shader
    # reads texture(...).rgba == (coverage, density, precip, fog).
    bgra = arr[..., [2, 1, 0, 3]]  # B, G, R, A
    data = np.ascontiguousarray(bgra.astype(np.float16))  # (Y, X, 4) fp16
    return data.tobytes()
