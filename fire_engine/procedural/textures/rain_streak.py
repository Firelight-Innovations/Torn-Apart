"""
procedural/textures/rain_streak.py — "rain_streak" tiling rain texture.

Produces a 128×512 RGBA texture of sparse bright vertical streaks with a
vertical motion-blur falloff, **tileable in both U and V**.  The renderer
scrolls it downward over camera-facing cards/cones, scaled by
``SkyState.rain_intensity``; alpha = streak intensity so it can be
alpha-blended or additively blended.

Generation algorithm (vectorised — no per-pixel Python loops)
-------------------------------------------------------------
1. Draw ``streak_count`` streaks in a coarse "column domain": each streak
   has a random column, start row, length (pixels), and a brightness tier
   sampled from three discrete levels (distant faint / mid / near bright).
2. For every (row, streak) pair, compute the distance below the streak head
   **modulo H** — the modulo makes every streak wrap vertically, which is
   what guarantees V-tileability.  Within a streak the intensity falls off
   as ``(1 − rel/length)^1.6``: bright head, motion-blurred tail.
3. Scatter the (H × K) value grid into the (H, W) canvas with
   ``np.maximum.at`` (overlapping streaks keep the brighter one rather than
   blooming).  A second scatter one column to the right at 35 % intensity
   gives the heavier streaks a soft 2-px body; the column index wraps
   modulo W (U-tileability).
4. Color is a cool blue-white ``(0.78, 0.84, 0.95)``; alpha = intensity.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(1337)
    arr = get("rain_streak")                    # (512, 128, 4) uint8
    arr = get("rain_streak", streak_count=48)   # heavier rain sheet
    # Preview: python tools/preview_texture.py rain_streak
"""

from __future__ import annotations

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef

__all__ = ["RainStreakDef"]


#: Streak color, linear RGB — cool blue-white rain highlight.
_STREAK_RGB = np.array([0.78, 0.84, 0.95], dtype=np.float32)

#: Brightness tiers and their sampling probabilities: most streaks are faint
#: background drops; a few are bright near-camera streaks.
_TIERS = np.array([0.30, 0.55, 1.00], dtype=np.float64)
_TIER_PROBS = np.array([0.50, 0.35, 0.15], dtype=np.float64)


@register_def
class RainStreakDef(ProceduralTextureDef):
    """
    Tiling rain-streak texture (tileable in U and V).

    Registered name
    ---------------
    ``"rain_streak"``

    Output
    ------
    ``numpy.ndarray`` of shape ``(512, 128, 4)``, dtype ``uint8`` by default
    (H = 512, W = 128 — tall and narrow; rain falls along V).  Alpha channel
    = streak intensity (0 between streaks); RGB is a premultiplied-looking
    cool blue-white scaled by the same intensity.

    Parameters (via ``get("rain_streak", ...)``)
    --------------------------------------------
    - ``width`` (int): output width in pixels.  Default 128.
    - ``height`` (int): output height in pixels.  Default 512.
    - ``streak_count`` (int): number of streaks.  Default 28.

    Example
    -------
    ::

        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import get

        set_world_seed(7)
        arr = get("rain_streak")
        assert arr.shape == (512, 128, 4) and arr.dtype == "uint8"
        # V-tileability: rolling the array vertically is still a valid sheet.
    """

    name = "rain_streak"

    DEFAULT_WIDTH = 128
    DEFAULT_HEIGHT = 512
    DEFAULT_STREAK_COUNT = 28

    def generate(self, rng: np.random.Generator, **params) -> np.ndarray:
        """
        Generate the rain-streak texture.

        Parameters
        ----------
        rng : numpy.random.Generator
            Seeded generator injected by the registry — sole randomness
            source (streak columns, heads, lengths, tiers).
        **params : any
            ``width`` / ``height`` / ``streak_count`` overrides.

        Returns
        -------
        numpy.ndarray — shape ``(H, W, 4)``, dtype ``uint8``, RGBA,
        tileable in both axes.
        """
        W = int(params.get("width", self.DEFAULT_WIDTH))
        H = int(params.get("height", self.DEFAULT_HEIGHT))
        K = int(params.get("streak_count", self.DEFAULT_STREAK_COUNT))

        # --- Streak attributes (coarse column domain) ---
        cols = rng.integers(0, W, K)  # (K,)
        heads = rng.integers(0, H, K)  # (K,) head row
        lengths = rng.integers(H // 12, H // 3, K)  # (K,) ~43..170 px
        tiers = _TIERS[rng.choice(len(_TIERS), K, p=_TIER_PROBS)]  # (K,)

        # --- Intensity profile per (row, streak) — wraps in V via modulo ---
        rows = np.arange(H, dtype=np.int64)[:, None]  # (H, 1)
        rel = (rows - heads[None, :]) % H  # (H, K) px below head
        inside = rel < lengths[None, :]  # (H, K) bool
        frac = np.clip(rel / np.maximum(lengths[None, :], 1), 0.0, 1.0)  # (H, K)
        profile = np.where(inside, (1.0 - frac) ** 1.6 * tiers[None, :], 0.0).astype(
            np.float32
        )  # (H, K)

        # --- Scatter into the canvas (max keeps overlaps crisp) ---
        canvas = np.zeros((H, W), dtype=np.float32)
        row_idx = np.broadcast_to(rows, (H, K)).ravel()
        col_idx = np.broadcast_to(cols[None, :], (H, K)).ravel()
        np.maximum.at(canvas, (row_idx, col_idx), profile.ravel())

        # Soft 2-px body for the brighter tiers (wraps in U via modulo).
        heavy = tiers >= _TIERS[1]
        if heavy.any():
            col2 = (cols[heavy] + 1) % W
            row2 = np.broadcast_to(rows, (H, int(heavy.sum()))).ravel()
            col2_idx = np.broadcast_to(col2[None, :], (H, int(heavy.sum()))).ravel()
            np.maximum.at(
                canvas,
                (row2, col2_idx),
                (profile[:, heavy] * 0.35).ravel(),
            )

        # --- Assemble RGBA: rgb = color · intensity, alpha = intensity ---
        np.clip(canvas, 0.0, 1.0, out=canvas)
        rgba = np.empty((H, W, 4), dtype=np.uint8)
        rgba[..., :3] = (canvas[..., None] * _STREAK_RGB[None, None, :] * 255.0).astype(np.uint8)
        rgba[..., 3] = (canvas * 255.0).astype(np.uint8)
        return rgba
