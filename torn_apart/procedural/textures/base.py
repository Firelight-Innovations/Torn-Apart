"""
procedural/textures/base.py — ProceduralTextureDef and shared noise helpers.

This module provides:

``ProceduralTextureDef``
    Domain subclass of ``ProceduralDef`` whose ``generate`` method is
    contractually required to return a ``numpy.ndarray`` of shape
    ``(H, W, 4)`` and dtype ``uint8`` (RGBA, 8 bits per channel).

``value_noise(rng, shape, octaves, persistence, lacunarity, tile) -> ndarray``
    Reusable layered value-noise helper.  Pure numpy — no per-pixel Python
    loops.  Returns a ``float32`` array in ``[0, 1]``.  Phase 3 terrain code
    imports this function directly for heightmap and cave generation.

Implementation notes
--------------------
Value noise is generated as follows for each octave:

1. Draw a small random coarse grid of shape ``(base_h, base_w)`` from *rng*
   using ``rng.random()``.
2. Bilinearly upsample it to ``(H, W)`` using vectorised numpy operations
   (``np.meshgrid`` + weighted sums of four corner values).
3. Scale by the octave weight (``persistence ** octave_index``) and accumulate.

No ``scipy``, no ``PIL``, no per-pixel loops.  The bilinear interpolation is
a single broadcast expression:

::

    # Given corner arrays TL, TR, BL, BR of shape (H, W):
    result = (TL * (1 - wx) * (1 - wy)
            + TR *      wx  * (1 - wy)
            + BL * (1 - wx) *      wy
            + BR *      wx  *      wy)

where ``wx, wy`` are sub-cell fractional offsets computed via ``np.meshgrid``.
"""

from __future__ import annotations

import numpy as np

from torn_apart.procedural.defs import ProceduralDef

__all__ = ["ProceduralTextureDef", "value_noise"]


# ---------------------------------------------------------------------------
# Noise helper — reusable by terrain (Phase 3) and any texture def
# ---------------------------------------------------------------------------

def value_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    base_freq: int = 4,
) -> np.ndarray:
    """
    Generate layered 2-D value noise in ``[0, 1]`` (float32).

    Each octave draws a fresh random coarse grid from *rng* and bilinearly
    upsamples it to *shape*.  Successive octaves are scaled by *persistence*
    and their grid frequency is multiplied by *lacunarity*.  The result is
    the normalised weighted sum across all octaves.

    This function is **pure numpy** — it contains no per-pixel Python loops.

    Parameters
    ----------
    rng : numpy.random.Generator
        A seeded generator (from ``core.rng.for_domain``).  Several random
        draws will be consumed — one ``(freq_h, freq_w)`` grid per octave.
    shape : tuple[int, int]
        Output array shape ``(H, W)``.
    octaves : int, optional
        Number of noise octaves to sum.  More octaves → more detail.
        Default 4.
    persistence : float, optional
        Amplitude scale per octave.  ``0.5`` means each octave has half
        the amplitude of the previous.  Default 0.5.
    lacunarity : float, optional
        Frequency multiplier per octave.  ``2.0`` means each octave has
        twice as many coarse-grid cells as the previous.  Default 2.0.
    base_freq : int, optional
        Number of coarse-grid cells along each axis for the first octave.
        Default 4.

    Returns
    -------
    numpy.ndarray
        Shape ``(H, W)``, dtype ``float32``, values in ``[0.0, 1.0]``.
        The output is normalised so the maximum possible weighted sum maps
        to 1.0 (using the geometric series of persistence weights).

    Example
    -------
    ::

        from torn_apart.core.rng import set_world_seed, for_domain
        from torn_apart.procedural.textures.base import value_noise

        set_world_seed(42)
        rng = for_domain("terrain", "heightmap")
        h = value_noise(rng, shape=(256, 256), octaves=5)
        assert h.shape == (256, 256)
        assert h.dtype == np.float32
        assert 0.0 <= h.min() and h.max() <= 1.0

    Notes for Phase 3
    -----------------
    Terrain heightmap usage::

        rng = for_domain("terrain", "height", (cx, cy))
        heights = value_noise(rng, shape=(32, 32), octaves=6, base_freq=2)
        # Scale to amplitude: (heights * 24.0).astype(np.float32)

    3-D noise hint: call ``value_noise`` independently per Z-slice with
    deterministic per-slice RNGs (e.g. ``for_domain("terrain","cave",cz)``).
    """
    H, W = shape

    # Pre-compute meshgrid of normalised cell coordinates for the full output.
    # These are reused across octaves (the coarse-grid positions shift with freq).
    # iy, ix are integer cell indices; fy, fx are fractional offsets in [0,1).
    accumulated = np.zeros((H, W), dtype=np.float64)
    weight_total = 0.0
    amplitude = 1.0
    freq = float(base_freq)

    for _ in range(octaves):
        freq_h = max(1, int(round(freq)))
        freq_w = max(1, int(round(freq)))

        # Draw random values at coarse-grid corners.
        # Shape: (freq_h + 1, freq_w + 1) — +1 so every cell has all 4 corners.
        grid = rng.random((freq_h + 1, freq_w + 1))  # float64 in [0,1)

        # Map output pixel (row r, col c) → coarse cell + sub-cell fraction.
        # Scale: one coarse cell covers (H/freq_h) rows and (W/freq_w) cols.
        # We use numpy broadcasting: shape (H,) and (W,) → broadcast to (H, W).
        r_f = np.linspace(0.0, freq_h, H, endpoint=False)  # shape (H,)
        c_f = np.linspace(0.0, freq_w, W, endpoint=False)  # shape (W,)

        r0 = r_f.astype(np.int32)          # floor row  (H,)
        c0 = c_f.astype(np.int32)          # floor col  (W,)
        r1 = np.minimum(r0 + 1, freq_h)    # ceil row   (H,)
        c1 = np.minimum(c0 + 1, freq_w)    # ceil col   (W,)

        wy = (r_f - r0).astype(np.float64)  # vertical fraction   (H,)
        wx = (c_f - c0).astype(np.float64)  # horizontal fraction (W,)

        # Bilinear interpolation via broadcasting — shape (H, W)
        # TL = grid[r0, c0], TR = grid[r0, c1], BL = grid[r1, c0], BR = grid[r1, c1]
        # Each corner array is (H, W) via index broadcasting.
        TL = grid[r0[:, None], c0[None, :]]   # (H, W)
        TR = grid[r0[:, None], c1[None, :]]   # (H, W)
        BL = grid[r1[:, None], c0[None, :]]   # (H, W)
        BR = grid[r1[:, None], c1[None, :]]   # (H, W)

        wy2d = wy[:, None]   # (H, 1) — broadcast over W
        wx2d = wx[None, :]   # (1, W) — broadcast over H

        octave_val = (
            TL * (1.0 - wx2d) * (1.0 - wy2d)
            + TR * wx2d * (1.0 - wy2d)
            + BL * (1.0 - wx2d) * wy2d
            + BR * wx2d * wy2d
        )

        accumulated += amplitude * octave_val
        weight_total += amplitude
        amplitude *= persistence
        freq *= lacunarity

    # Normalise to [0, 1]
    result = (accumulated / weight_total).astype(np.float32)
    # Clamp for floating-point edge cases
    np.clip(result, 0.0, 1.0, out=result)
    return result


# ---------------------------------------------------------------------------
# ProceduralTextureDef domain base class
# ---------------------------------------------------------------------------

class ProceduralTextureDef(ProceduralDef):
    """
    Base class for all procedural texture definitions.

    A ``ProceduralTextureDef`` is a ``ProceduralDef`` whose ``generate``
    method is contractually required to return a ``numpy.ndarray`` of shape
    ``(H, W, 4)`` and dtype ``uint8`` (RGBA, premultiplied-alpha not assumed).
    Alpha 255 = fully opaque.

    How to author a new texture
    ---------------------------
    1. Create a new file in ``torn_apart/procedural/textures/``, e.g.
       ``cracked_rock.py``.
    2. Subclass ``ProceduralTextureDef``, set ``name``, and decorate with
       ``@register_def``.
    3. Implement ``generate(self, rng, **params) -> np.ndarray``.
       *Use only numpy — no per-pixel Python loops, no panda3d imports.*
    4. Import your module in ``torn_apart/procedural/textures/__init__.py``
       so it auto-registers at package import time.
    5. Add a determinism test in ``tests/test_procedural.py``.
    6. Run ``python tools/preview_texture.py <name>`` to confirm the output
       looks correct.

    See ``procedural/textures/wasteland_ground.py`` for a complete example.

    Parameters
    ----------
    (none — use class attributes only; construction is implicit via
    ``@register_def`` decorator or by passing an instance to ``register()``.)

    generate() contract
    -------------------
    ::

        def generate(self, rng: np.random.Generator, **params) -> np.ndarray:
            '''Returns (H, W, 4) uint8 RGBA.'''
            H = params.get('height', 256)
            W = params.get('width',  256)
            rgba = np.zeros((H, W, 4), dtype=np.uint8)
            rgba[..., 3] = 255  # fully opaque
            # … fill R G B channels …
            return rgba

    Example
    -------
    ::

        from torn_apart.core.rng import set_world_seed, for_domain
        from torn_apart.procedural import get
        import numpy as np

        set_world_seed(1337)
        arr = get("wasteland_ground")
        assert arr.shape == (256, 256, 4)
        assert arr.dtype == np.uint8
        assert (arr[..., 3] == 255).all()   # fully opaque
    """

    def generate(self, rng: np.random.Generator, **params) -> np.ndarray:
        """
        Generate and return an RGBA texture array.

        Must be overridden by concrete subclasses.  The base implementation
        raises ``NotImplementedError`` pointing at ARCHITECTURE.md §5.2.

        Parameters
        ----------
        rng : numpy.random.Generator
            Seeded generator; consume it for all randomness.
        **params : any
            E.g. ``width=512, height=512, octaves=6``.

        Returns
        -------
        numpy.ndarray
            Shape ``(H, W, 4)``, dtype ``uint8``, RGBA channel order.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.generate() not implemented. "
            "See ARCHITECTURE.md §5.2 for the ProceduralTextureDef contract."
        )
