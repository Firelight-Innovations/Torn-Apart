"""
procedural/textures/moon_surface.py — "moon_surface" texture definition.

Produces a 256×256 RGBA disc texture for the moon: pale regolith base,
darker maria (the "seas" — large low-frequency blotches), and a field of
impact craters with bright rims and shadowed floors.  Every world seed grows
a different moon (crater layout and maria pattern are drawn from
``for_domain``-seeded RNG), satisfying the "completely new sky each world"
requirement.

The texture maps the visible lunar disc directly: UV (0,0)-(1,1) spans the
disc's bounding square; the sky-dome shader samples it with disc-local
coordinates and applies the phase terminator on top (lighting is NOT baked
here — phases stay dynamic).

Generation algorithm
--------------------
1. **Base regolith** — warm-gray base modulated by 4-octave ``value_noise``
   (subtle large-scale albedo variation).
2. **Maria** — a low-frequency noise field thresholded into large dark
   patches, slightly blue-gray, blended softly.
3. **Craters** — ``crater_count`` craters with power-law radii.  For each
   crater a radial profile is evaluated **vectorised over all craters at
   once** (one ``(N, H, W)`` distance tensor): floor darkening inside
   ~0.8 r, a bright ejecta rim near r, fading rapidly outside.  Crater
   centres are biased fully inside the disc so rims don't clip.
4. Pixels outside the unit disc get alpha 0 (the shader masks anyway).

This definition is registered as ``"moon_surface"`` at import time via the
``@register_def`` decorator.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(42)
    arr = get("moon_surface")    # np.ndarray (256, 256, 4) uint8
    # python tools/preview_texture.py moon_surface

Docs: docs/systems/procedural.textures.sky.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, value_noise

__all__ = ["MoonSurfaceDef"]

# Linear-ish RGB tones (0-255).
_REGOLITH = np.array([168.0, 166.0, 158.0])  # pale warm-gray highlands
_MARIA = np.array([96.0, 98.0, 104.0])  # dark blue-gray seas
_RIM_BOOST = 36.0  # ejecta rim brightening
_FLOOR_DARKEN = 42.0  # crater floor shadowing


@register_def
class MoonSurfaceDef(ProceduralTextureDef):
    """
    Procedural lunar disc texture ("moon_surface").

    Parameters (``generate`` kwargs)
    --------------------------------
    size : int, default 256
        Output square edge in pixels.
    crater_count : int, default 56
        Number of impact craters.

    Returns ``(size, size, 4) uint8`` RGBA; alpha 255 inside the unit disc,
    0 outside.  Deterministic per world seed.

    Docs: docs/systems/procedural.textures.sky.md
    """

    name = "moon_surface"

    def generate(self, rng: np.random.Generator, **params: Any) -> np.ndarray:
        """
        Generate the lunar disc texture.

        Parameters
        ----------
        rng : numpy.random.Generator
            Seeded generator (``for_domain``-derived by the registry).
        **params
            ``size`` (default 256), ``crater_count`` (default 56).

        Returns
        -------
        numpy.ndarray — ``uint8 (size, size, 4)`` RGBA.

        Docs: docs/systems/procedural.textures.sky.md
        """
        size = int(params.get("size", 256))
        crater_count = int(params.get("crater_count", 56))

        # Disc-local coordinates in [-1, 1].
        ax = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        rr = np.sqrt(xx * xx + yy * yy)
        disc = rr <= 1.0

        # 1. Regolith base with large-scale albedo variation.
        base_var = value_noise(rng, (size, size), octaves=4, base_freq=3)
        rgb = _REGOLITH[None, None, :] * (0.92 + 0.16 * base_var[..., None].astype(np.float64))

        # 2. Maria: soft-thresholded low-frequency blotches.
        maria_field = value_noise(rng, (size, size), octaves=3, base_freq=2)
        maria_w = np.clip((maria_field - 0.55) / 0.18, 0.0, 1.0)[..., None]
        rgb = rgb * (1.0 - maria_w) + _MARIA[None, None, :] * maria_w

        # 3. Craters — vectorised radial profiles over an (N, H, W) tensor.
        theta = rng.random(crater_count) * 2.0 * np.pi
        rad_pos = np.sqrt(rng.random(crater_count)) * 0.82  # stay inside disc
        cx = (rad_pos * np.cos(theta)).astype(np.float32)
        cy = (rad_pos * np.sin(theta)).astype(np.float32)
        cr = (0.02 + 0.13 * rng.random(crater_count) ** 2.2).astype(np.float32)

        d = (
            np.sqrt(
                (xx[None, :, :] - cx[:, None, None]) ** 2
                + (yy[None, :, :] - cy[:, None, None]) ** 2
            )
            / cr[:, None, None]
        )  # (N, H, W) in radii

        floor = np.clip(1.0 - d / 0.8, 0.0, 1.0) ** 0.7  # 1 at centre → 0 at 0.8r
        rim = np.exp(-(((d - 1.0) / 0.18) ** 2))  # gaussian ring at r
        shade = rim.sum(axis=0) * _RIM_BOOST - floor.sum(axis=0) * _FLOOR_DARKEN  # (H, W)
        rgb = rgb + shade[..., None]

        out = np.zeros((size, size, 4), dtype=np.uint8)
        out[..., :3] = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        out[..., 3] = np.where(disc, 255, 0).astype(np.uint8)
        return out
