"""
fire_engine.procedural — Procedural content registry and generation.

This is the **Foundation layer** for all deterministic content in the engine.
It is callable from any other layer (see ARCHITECTURE.md §4a.2).  It must
never import panda3d — everything here is headless-testable.

Public API summary
------------------
ProceduralDef
    Abstract base class.  Subclass this (or ``ProceduralTextureDef``) to
    author new procedural content.

ProceduralTextureDef
    Domain base for texture defs.  ``generate(rng, **params)`` returns
    ``np.ndarray (H, W, 4) uint8``.

register(def_instance)
    Register a ``ProceduralDef`` instance by name.

get(name, **params)
    Generate (or return cached) content for a named def.

clear_cache()
    Flush the generated-result cache (keeps the def registry intact).

value_noise(rng, shape, ...)
    Reusable layered 2-D value-noise helper (float32, [0,1]).
    Imported by terrain (Phase 3) for heightmap and cave generation.

Built-in registrations
----------------------
Importing this package automatically registers:
  * ``"wasteland_ground"``  (256×256 RGBA dirt/dead-grass ground texture)
  * ``"night_sky"``         (1024×512 RGBA equirect star field + galaxy band)
  * ``"rain_streak"``       (128×512 RGBA tiling rain streaks, U+V tileable)

Quick-start example
-------------------
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get, value_noise
    from fire_engine.core.rng import for_domain
    import numpy as np

    set_world_seed(1337)

    # Get a pre-registered texture
    arr = get("wasteland_ground")    # np.ndarray (256, 256, 4) uint8
    assert arr.shape == (256, 256, 4)
    assert arr.dtype == np.uint8

    # Use the noise helper directly (e.g. terrain heightmap)
    rng = for_domain("terrain", "height")
    h = value_noise(rng, shape=(64, 64), octaves=5)
    assert h.shape == (64, 64) and h.dtype == np.float32
"""

# Importing the flora sub-package registers the built-in tree/bush species
# (TreeSpeciesDef subclasses returning TreeVariantSet — see flora/__init__.py).
import fire_engine.procedural.flora

# Importing the textures sub-package triggers registration of all built-in defs.
import fire_engine.procedural.textures  # noqa: F401
from fire_engine.procedural.defs import ProceduralDef, register_def
from fire_engine.procedural.registry import clear_cache, get, register, reset_registry
from fire_engine.procedural.textures.base import ProceduralTextureDef, value_noise

__all__ = [
    "ProceduralDef",
    "ProceduralTextureDef",
    "clear_cache",
    "get",
    "register",
    "register_def",
    "reset_registry",
    "value_noise",
]
