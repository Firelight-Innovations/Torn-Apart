"""
lighting/exposure.py — Auto-exposure / eye adaptation for the HDR pipeline.

keywords: exposure, auto-exposure, eye adaptation, adaptation, aperture,
    tonemap, luminance, dark cave, night vision, light meter, ExposureMeter

Estimates the incident luminance at the camera every frame (sky/sun openness
via a fixed deterministic ray set marched through the voxel field, plus moon
and nearby point/area lights) and smooths a single exposure *multiplier*
toward ``exposure_key / luminance``.  The render wiring multiplies this value
into ``config.light_exposure`` before tonemapping.

Behaviour
---------
- Standing in an open field at noon the multiplier holds ~1.0 — the daytime
  look is unchanged by design.
- Walking into a cave (or night falling) the "aperture" opens slowly
  (time constant ``exposure_tau_dark_s``, default 4.0 s) toward
  ``exposure_max``.
- Stepping back into bright light stops down fast
  (``exposure_tau_bright_s``, default 0.7 s).

All smoothing happens in log space so up/down adaptation feels symmetric in
stops.  Everything here is headless (numpy only, no panda3d) and fully
deterministic: the ray set is a fixed module-level constant — no randomness.

Units: world space in meters, time in seconds, Z-up.  Radiance/irradiance
inputs are linear HDR RGB; luminance uses Rec.709 luma weights.

Example
-------
    from fire_engine.core.config import load_config
    from fire_engine.lighting.exposure import ExposureMeter

    cfg = load_config("config.toml")
    meter = ExposureMeter(cfg)
    # each frame:
    mult = meter.update(camera_pos, sky_state, chunks, lights.pack(64), dt)
    effective_exposure = cfg.light_exposure * mult
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["ExposureMeter"]

# ----------------------------------------------------------------------
# Rec.709 luma weights (linear RGB -> relative luminance).
# ----------------------------------------------------------------------
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# ----------------------------------------------------------------------
# Tuning constants.  Chosen so that an unoccluded noon sky
# (sky_ambient ~ (0.21, 0.40, 0.71), luma ~ 0.382;
#  sun_radiance ~ (3.2, 3.0, 2.6), luma ~ 3.014) yields
#     L = 0.382 * A + 3.014 * B ~ 0.180 = exposure_key
# i.e. a target multiplier of ~1.0 in open daylight, while the moonless
# night ambient floor (luma ~ 0.0123) gives L ~ 0.0031 -> target clamps to
# exposure_max, and a full moon (luma ~ 0.070) lands the open-field night
# target around 4 (just under max).
# ----------------------------------------------------------------------
_A_SKY: float = 0.25  # weight on sky-ambient luma * sky openness
_B_SUN: float = 0.028  # weight on direct-sun luma * sun-cone openness
_C_MOON: float = 0.60  # weight on moon luma * sky openness

#: Sharpness of the sun-direction cone used to weight rays for the direct
#: sun term (``max(0, dot(ray, sun_dir)) ** _SUN_CONE_POW``).
_SUN_CONE_POW: float = 4.0

#: Ray-march step length in meters and number of steps (28 m total reach).
_STEP_M: float = 1.0
_N_STEPS: int = 28


def _build_ray_set() -> tuple[np.ndarray, np.ndarray]:
    """Build the fixed 13-ray sky-openness probe set.

    Returns ``(dirs, weights)`` where ``dirs`` is float64 ``(13, 3)`` unit
    vectors (Z-up) and ``weights`` is float64 ``(13,)``:

    - 1 ray straight up (weight 2.0),
    - 6 rays tilted up at 50 deg elevation, azimuths every 60 deg (weight 1.0),
    - 6 near-horizontal rays at 15 deg elevation, azimuths offset by 30 deg
      (weight 0.5).

    Upward rays are weighted higher because sky irradiance is dominated by
    the upper hemisphere.  The set is a deterministic constant — no RNG.
    """
    dirs: list[tuple[float, float, float]] = [(0.0, 0.0, 1.0)]
    weights: list[float] = [2.0]
    for elev_deg, count, az_off, w in ((50.0, 6, 0.0, 1.0), (15.0, 6, 30.0, 0.5)):
        z = math.sin(math.radians(elev_deg))
        r = math.cos(math.radians(elev_deg))
        for i in range(count):
            az = math.radians(az_off + 360.0 * i / count)
            dirs.append((r * math.cos(az), r * math.sin(az), z))
            weights.append(w)
    return (np.asarray(dirs, dtype=np.float64), np.asarray(weights, dtype=np.float64))


_RAY_DIRS, _RAY_WEIGHTS = _build_ray_set()
#: Sample distances along each ray, meters: 1.0, 2.0, ..., 28.0.
_RAY_DISTS = (np.arange(_N_STEPS, dtype=np.float64) + 1.0) * _STEP_M


def _as_vec3(v: Any) -> np.ndarray:
    """Coerce a position/direction to a float64 ``(3,)`` numpy array.

    Accepts an object with ``.x/.y/.z`` float attributes (panda3d-style or
    ``SimpleNamespace``) or any 3-sequence / array.  World meters, Z-up.
    """
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
        return np.array([float(v.x), float(v.y), float(v.z)], dtype=np.float64)
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    return arr


def _luma(rgb: Any) -> float:
    """Rec.709 relative luminance of a linear HDR RGB triple (float)."""
    arr = np.asarray(rgb, dtype=np.float64).reshape(3)
    return float(arr @ _LUMA)


class ExposureMeter:
    """
    Auto-exposure ("eye adaptation") meter producing one smoothed multiplier.

    Each :meth:`update` call estimates incident luminance at the camera from
    sky/sun openness (fixed ray set marched through the voxel field), the
    moon, and nearby point/area lights, then eases the exposure multiplier
    toward ``exposure_key / luminance`` asymmetrically in log space: slow
    when adapting to darkness, fast when adapting to brightness.

    Config keys (read with ``getattr(config, key, default)`` so older
    configs work unchanged):

    - ``exposure_adapt_enabled`` : bool, default True — False pins the
      multiplier (it decays back toward 1.0).
    - ``exposure_min`` : float, default 0.55 — lower clamp (bright scenes).
    - ``exposure_max`` : float, default 5.0 — upper clamp (dark scenes).
    - ``exposure_key`` : float, default 0.18 — target mid-gray luminance.
    - ``exposure_tau_dark_s`` : float, default 4.0 — time constant in
      seconds for adapting *to darkness* (multiplier rising).
    - ``exposure_tau_bright_s`` : float, default 0.7 — time constant in
      seconds for adapting *to brightness* (multiplier falling).
    - ``chunk_size`` (32 voxels) and ``voxel_size`` (0.5 m) — voxel grid.

    Determinism: identical inputs always produce identical outputs — the
    probe ray set is a fixed constant and no randomness is used.

    Example
    -------
        meter = ExposureMeter(cfg)
        mult = meter.update((0.0, 0.0, 12.0), sky.state, chunk_mgr.chunks,
                            light_set.pack(cfg.light_max_point_lights),
                            dt=0.016)
        shader_exposure = cfg.light_exposure * mult
    """

    def __init__(self, config: Any) -> None:
        """
        Parameters
        ----------
        config : Config
            Engine config dataclass (``fire_engine.core.config.Config``) or
            any object exposing the keys documented on the class.  Missing
            keys fall back to the documented defaults.
        """
        self._enabled: bool = bool(getattr(config, "exposure_adapt_enabled", True))
        self._min: float = float(getattr(config, "exposure_min", 0.55))
        self._max: float = float(getattr(config, "exposure_max", 5.0))
        self._key: float = float(getattr(config, "exposure_key", 0.18))
        self._tau_dark: float = float(getattr(config, "exposure_tau_dark_s", 4.0))
        self._tau_bright: float = float(getattr(config, "exposure_tau_bright_s", 0.7))
        self._chunk_size: int = int(getattr(config, "chunk_size", 32))
        self._voxel_size: float = float(getattr(config, "voxel_size", 0.5))
        self._exposure: float = 1.0

    # ------------------------------------------------------------------
    @property
    def exposure(self) -> float:
        """Current smoothed exposure multiplier (dimensionless, starts 1.0).

        Multiply into ``config.light_exposure`` before tonemapping.
        Always within ``[min(exposure_min, 1.0), max(exposure_max, 1.0)]``.
        """
        return self._exposure

    # ------------------------------------------------------------------
    def update(
        self,
        camera_pos: Any,
        sky_state: Any,
        chunks: Mapping[tuple[int, int, int], Any] | None,
        lights_packed: tuple[np.ndarray, int] | None,
        dt: float,
    ) -> float:
        """
        Advance the meter by ``dt`` seconds and return the new multiplier.

        Parameters
        ----------
        camera_pos : object with .x/.y/.z floats, or 3-sequence
            Camera position in world meters (Z-up).
        sky_state : object or None
            Duck-typed sky state read via ``getattr`` with fallbacks:
            ``sun_radiance`` (linear HDR RGB, noon ~ (3.2, 3.0, 2.6)),
            ``sky_ambient`` (noon ~ (0.21, 0.40, 0.71)),
            ``moon_radiance`` (full moon ~ (0.06, 0.07, 0.10)),
            ``sun_dir`` (unit vector toward the sun, .x/.y/.z or
            3-sequence).  ``None`` decays the multiplier back to 1.0.
        chunks : dict[(cx, cy, cz) -> chunk] or None
            Loaded chunks; each chunk exposes ``.materials``, a uint8
            ``(32, 32, 32)`` array indexed ``[x, y, z]`` in local voxel
            coords, 0 = air.  Missing keys are treated as air.  Never
            mutated.
        lights_packed : (np.ndarray (N, 12) float32, int count) or None
            ``LightSet.pack()`` output: rows ``[0:3]`` world position (m),
            ``[3]`` falloff radius (m), ``[4:7]`` color*intensity
            (linear HDR).  Rows past ``count`` are zero.
        dt : float
            Real seconds since the last call.  ``dt <= 0`` returns the
            current value unchanged; very large ``dt`` saturates the blend
            at 1 (jump straight to target).

        Returns
        -------
        float
            The new smoothed exposure multiplier (same as :attr:`exposure`).

        Example
        -------
            mult = meter.update(cam.get_pos(), sky.state,
                                chunk_manager.chunks, light_set.pack(64),
                                globalClock.get_dt())
        """
        if not self._enabled or sky_state is None:
            target = 1.0
            tau = self._tau_bright
        else:
            cam = _as_vec3(camera_pos)
            luminance = self._estimate_luminance(
                cam, sky_state, chunks if chunks is not None else {}, lights_packed
            )
            target = self._key / max(luminance, 1e-4)
            target = min(max(target, self._min), self._max)
            # Rising multiplier == adapting to darkness == slow.
            tau = self._tau_dark if target > self._exposure else self._tau_bright

        dt = float(dt)
        if dt > 0.0 and target != self._exposure:
            blend = 1.0 if tau <= 1e-6 else 1.0 - math.exp(-dt / tau)
            blend = min(max(blend, 0.0), 1.0)
            log_cur = math.log(self._exposure)
            log_tgt = math.log(target)
            self._exposure = math.exp(log_cur + (log_tgt - log_cur) * blend)
        return self._exposure

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _estimate_luminance(
        self,
        cam: np.ndarray,
        sky_state: Any,
        chunks: Mapping[tuple[int, int, int], Any],
        lights_packed: tuple[np.ndarray, int] | None,
    ) -> float:
        """Estimate incident luminance (Rec.709, linear HDR) at ``cam``.

        ``L = openness * luma(sky_ambient) * A
            + sun_vis * luma(sun_radiance) * B   (sun above horizon only)
            + openness * luma(moon_radiance) * C
            + sum_lights luma(color*intensity) * window / (d^2 + 1)``
        """
        open_per_ray = self._ray_openness(cam, chunks)
        wsum = float(_RAY_WEIGHTS.sum())
        openness = float((open_per_ray * _RAY_WEIGHTS).sum()) / wsum

        sky_l = _luma(getattr(sky_state, "sky_ambient", (0.0, 0.0, 0.0)))
        sun_l = _luma(getattr(sky_state, "sun_radiance", (0.0, 0.0, 0.0)))
        moon_l = _luma(getattr(sky_state, "moon_radiance", (0.0, 0.0, 0.0)))

        # Direct-sun visibility: openness of the rays nearest sun_dir.
        sun_vis = 0.0
        sun_dir_raw = getattr(sky_state, "sun_dir", None)
        if sun_dir_raw is not None and sun_l > 0.0:
            sun_dir = _as_vec3(sun_dir_raw)
            if sun_dir[2] > 0.0:
                w = np.maximum(_RAY_DIRS @ sun_dir, 0.0) ** _SUN_CONE_POW
                w_total = float(w.sum())
                if w_total > 1e-6:
                    sun_vis = float((open_per_ray * w).sum()) / w_total

        lum = (
            openness * sky_l * _A_SKY
            + sun_vis * sun_l * _B_SUN
            + openness * moon_l * _C_MOON
            + self._light_luminance(cam, lights_packed)
        )
        return lum

    def _ray_openness(
        self, cam: np.ndarray, chunks: Mapping[tuple[int, int, int], Any]
    ) -> np.ndarray:
        """Per-ray sky openness: 1.0 if a probe ray reaches 28 m without
        hitting a solid voxel, else 0.0.  Returns float64 ``(13,)``.

        Vectorized: builds all (rays x steps) sample positions at once,
        converts to voxel/chunk indices, and gathers occupancy with one
        small Python loop over the handful of distinct chunks touched
        (missing chunks count as air).
        """
        # (R, S, 3) world-space sample positions.
        pos = cam[None, None, :] + _RAY_DIRS[:, None, :] * _RAY_DISTS[None, :, None]
        vox = np.floor(pos / self._voxel_size).astype(np.int64)
        cnk = np.floor_divide(vox, self._chunk_size)
        loc = vox - cnk * self._chunk_size

        flat_cnk = cnk.reshape(-1, 3)
        flat_loc = loc.reshape(-1, 3)
        solid = np.zeros(flat_cnk.shape[0], dtype=bool)

        uniq, inv = np.unique(flat_cnk, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        for i in range(uniq.shape[0]):  # ~a dozen chunks max
            key = (int(uniq[i, 0]), int(uniq[i, 1]), int(uniq[i, 2]))
            chunk = chunks.get(key)
            if chunk is None:
                continue  # unloaded == air
            mask = inv == i
            loc = flat_loc[mask]
            solid[mask] = chunk.materials[loc[:, 0], loc[:, 1], loc[:, 2]] != 0

        blocked = solid.reshape(_RAY_DIRS.shape[0], _N_STEPS).any(axis=1)
        return np.asarray((~blocked).astype(np.float64))

    @staticmethod
    def _light_luminance(cam: np.ndarray, lights_packed: tuple[np.ndarray, int] | None) -> float:
        """Summed luminance contribution of packed point/area lights.

        Per light: ``luma(color*intensity) * window / (d^2 + 1)`` where
        ``d`` is the camera distance in meters and
        ``window = clamp(1 - d / radius, 0, 1)^2`` fades smoothly to zero
        at the falloff radius.  Lights beyond their radius contribute 0.
        """
        if lights_packed is None:
            return 0.0
        arr, count = lights_packed
        if count <= 0:
            return 0.0
        rows = np.asarray(arr[:count], dtype=np.float64)
        delta = rows[:, 0:3] - cam[None, :]
        d = np.sqrt((delta * delta).sum(axis=1))
        radius = np.maximum(rows[:, 3], 1e-6)
        window = np.clip(1.0 - d / radius, 0.0, 1.0) ** 2
        lum = rows[:, 4:7] @ _LUMA
        return float((lum * window / (d * d + 1.0)).sum())
