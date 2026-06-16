"""
lighting/palette.py — Material id → (albedo, emission) lookup for the GPU light volume.

The volumetric lighting pipeline needs a per-material *light response*: what
colour a surface bounces (albedo) and what light it emits on its own
(emission).  Terrain stores materials as ``uint8`` ids (0 = air), so both
responses are 256-entry lookup tables that vectorised volume assembly can
fancy-index with a whole material array at once (Hard Rule 4 — no per-voxel
loops).

Albedo is *derived* from the registered procedural textures (mean RGB of the
def named for each material), keeping "environment textures are 100 %
procedural" intact — there is no hand-authored colour data here.

No panda3d imports.  Fully headless-testable.

Example
-------
>>> from fire_engine.core.rng import set_world_seed
>>> import fire_engine.procedural  # registers texture defs
>>> from fire_engine.lighting.palette import build_default_palette
>>> set_world_seed(1337)
>>> pal = build_default_palette()
>>> pal.albedo.shape, pal.emission.shape
((256, 3), (256, 3))
>>> bool((pal.albedo[0] == 0).all())   # air bounces nothing
True

Docs: docs/systems/lighting.md
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fire_engine.core import get_logger

_log = get_logger("lighting.palette")

# Material id → procedural texture def name used to derive its albedo.
# Mirrors the material → texture mapping in terrain/generation.py.
_MATERIAL_TEXTURE_DEFS: dict[int, str] = {
    1: "dirt_ground",  # MATERIAL_DIRT
    2: "grass_ground",  # MATERIAL_GRASS
}

# Fallback albedo for any solid material with no registered texture def —
# a neutral mid-grey so unknown materials still bounce *something*.
_FALLBACK_ALBEDO: tuple[float, float, float] = (0.45, 0.42, 0.38)


@dataclass(frozen=True)
class MaterialPalette:
    """
    Per-material light-response lookup tables.

    Attributes
    ----------
    albedo : numpy.ndarray
        ``float32 (256, 3)`` linear RGB in ``[0, 1]``.  Row *i* is the bounce
        colour of material id *i*; row 0 (air) is all zeros.
    emission : numpy.ndarray
        ``float32 (256, 3)`` linear RGB radiance (HDR, unbounded ≥ 0).  Row
        *i* is the self-emitted light of material id *i*; all zeros for
        non-emissive materials.

    Both arrays are safe to fancy-index with whole ``uint8`` material arrays:
    ``palette.albedo[materials]`` → ``float32 (..., 3)``.

    Docs: docs/systems/lighting.md
    """

    albedo: np.ndarray = field(default_factory=lambda: np.zeros((256, 3), dtype=np.float32))
    emission: np.ndarray = field(default_factory=lambda: np.zeros((256, 3), dtype=np.float32))

    def with_emission(self, material: int, rgb: tuple[float, float, float]) -> MaterialPalette:
        """
        Return a copy of this palette with ``material``'s emission set.

        Use to register glowing materials (lava, magic crystal) without
        mutating the shared default palette.

        Parameters
        ----------
        material : int
            Material id in ``[1, 255]``.
        rgb : tuple[float, float, float]
            Linear RGB emitted radiance (HDR — values above 1 are normal for
            bright sources; a torch-like glow is ~(2.0, 1.2, 0.4)).

        Docs: docs/systems/lighting.md
        """
        emission = self.emission.copy()
        emission[material] = np.asarray(rgb, dtype=np.float32)
        return MaterialPalette(albedo=self.albedo, emission=emission)


def _mean_texture_rgb(def_name: str) -> tuple[float, float, float] | None:
    """
    Mean linear RGB of a registered procedural texture def, or ``None``.

    Generates (cached by the registry) the texture and averages its RGB
    channels, converting sRGB-ish ``uint8`` to linear via a gamma-2.2
    approximation so bounce colours live in the same linear space as the
    radiance volume.
    """
    try:
        from fire_engine.procedural import get as get_procedural

        rgba = get_procedural(def_name)  # (H, W, 4) uint8
    except Exception as exc:
        _log.warning("Palette: no texture def %r (%s)", def_name, exc)
        return None
    srgb = rgba[..., :3].astype(np.float32) / 255.0
    linear = srgb**2.2
    mean = linear.reshape(-1, 3).mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def build_default_palette() -> MaterialPalette:
    """
    Build the engine's default material palette from registered texture defs.

    Albedo per material id comes from the mean linear RGB of the texture def
    in the material → texture mapping (``dirt_ground``, ``grass_ground``);
    every other solid id (3..255) gets a neutral fallback grey.  Air (0) is
    zero.  Emission is all zeros — register emissive materials with
    :meth:`MaterialPalette.with_emission`.

    Returns
    -------
    MaterialPalette

    Notes
    -----
    Deterministic: texture defs are pure functions of the world seed, so the
    same seed always produces byte-identical palettes.  Call after
    ``import fire_engine.procedural`` (registration side-effect) and
    ``set_world_seed``.

    Docs: docs/systems/lighting.md
    """
    albedo = np.empty((256, 3), dtype=np.float32)
    albedo[:] = np.asarray(_FALLBACK_ALBEDO, dtype=np.float32)
    albedo[0] = 0.0  # air
    for material, def_name in _MATERIAL_TEXTURE_DEFS.items():
        rgb = _mean_texture_rgb(def_name)
        if rgb is not None:
            albedo[material] = np.asarray(rgb, dtype=np.float32)
    emission = np.zeros((256, 3), dtype=np.float32)
    return MaterialPalette(albedo=albedo, emission=emission)
