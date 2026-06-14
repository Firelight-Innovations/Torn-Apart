"""
weather/weather_map.py — Derived raster of the local weather field.

The storm-cell model (:mod:`fire_engine.world.weather.system`) is the simulation
truth, but sampling it per-march-step on the GPU or per-particle on the CPU
would mean re-evaluating every active cell's Gaussian footprint thousands of
times a frame.  The **weather map** is the cache that avoids that: a small
``(N, N, 4)`` raster of the four spatial channels (coverage, density, precip,
fog) over a square region around the player, re-rastered a few times a second
and uploaded once (M4) for the cloud/rain/fog shaders to sample cheaply.

It is a pure *derivation* of the sim — never saved, recomputed each tick — so
it holds no authoritative state.  Because it calls
:meth:`WeatherSystem.sample_fields` (the same routine ``sample_local`` uses),
a texel's rasterised value equals ``sample_local`` at that texel's center by
construction.

Layout: ``out[row, col, channel]`` with ``row`` indexing world **+Y** and
``col`` indexing world **+X** (matches the wind-field ``[y, x, …]`` convention).
The region spans ``cells · cell_m`` meters centered on the supplied position;
texel ``(row, col)`` covers the center of its cell (half-texel offset).

Units: meters, game seconds.
"""

from __future__ import annotations

import numpy as np

from fire_engine.core.config import Config

__all__ = ["WeatherMap", "MAP_CHANNELS"]

#: Channel order of the raster's last axis.
MAP_CHANNELS: tuple[str, ...] = ("coverage", "density", "precip", "fog")


class WeatherMap:
    """
    Square raster of the spatial weather channels around a moving center.

    Parameters
    ----------
    config : Config — reads ``weather_map_cells`` (resolution) and
        ``weather_map_cell_m`` (texel size, meters).

    Attributes
    ----------
    cells : int — N, the square resolution.
    cell_m : float — texel edge length in meters.
    span_m : float — total covered extent ``cells · cell_m`` (meters).

    Example
    -------
    >>> from fire_engine.core import EventBus, load_config, set_world_seed
    >>> from fire_engine.world.weather import WeatherSystem, WeatherMap
    >>> set_world_seed(1337)
    >>> ws = WeatherSystem(load_config(), EventBus())
    >>> wm = WeatherMap(load_config())
    >>> raster = wm.rasterize(ws, center_xy=(0.0, 0.0), t_abs=12 * 3600.0)
    >>> raster.shape, raster.dtype
    ((128, 128, 4), dtype('float32'))
    """

    def __init__(self, config: Config) -> None:
        self.cells: int = int(config.weather_map_cells)
        self.cell_m: float = float(config.weather_map_cell_m)
        self.span_m: float = self.cells * self.cell_m
        # Precomputed per-axis texel-center offsets from the region's min corner.
        self._offsets: np.ndarray = (
            np.arange(self.cells, dtype=np.float64) + 0.5
        ) * self.cell_m - 0.5 * self.span_m  # (N,)

    def texel_centers(self, center_xy: tuple[float, float]) -> np.ndarray:
        """
        World-XY centers of every texel for a map centered on *center_xy*.

        Returns
        -------
        np.ndarray — shape ``(cells*cells, 2)`` in row-major ``(row=Y, col=X)``
        order, so reshaping a result back to ``(cells, cells)`` matches
        :meth:`rasterize`'s layout.
        """
        xs = center_xy[0] + self._offsets  # (N,) along X
        ys = center_xy[1] + self._offsets  # (N,) along Y
        gx, gy = np.meshgrid(xs, ys)  # (N, N): gx[row,col]=xs[col]
        return np.stack([gx.ravel(), gy.ravel()], axis=1)  # (N*N, 2)

    def rasterize(
        self,
        system,
        center_xy: tuple[float, float],
        t_abs: float,
    ) -> np.ndarray:
        """
        Raster the weather channels around *center_xy* at absolute time *t_abs*.

        Parameters
        ----------
        system : WeatherSystem — the sim to sample (uses ``sample_fields``).
        center_xy : tuple[float, float] — world XY the map is centered on.
        t_abs : float — absolute game seconds.

        Returns
        -------
        np.ndarray — shape ``(cells, cells, 4)`` float32, channels in
        :data:`MAP_CHANNELS` order.  Pure function of (system seed, center,
        t_abs): the result does not depend on when it is called.
        """
        pts = self.texel_centers(center_xy)
        cov, den, rain, fog, _ = system.sample_fields(pts, t_abs)
        out = np.stack([cov, den, rain, fog], axis=1)  # (N*N, 4)
        return out.reshape(self.cells, self.cells, 4).astype(np.float32)
