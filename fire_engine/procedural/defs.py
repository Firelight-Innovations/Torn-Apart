"""
procedural/defs.py — Base class and registration decorator for ProceduralDef.

A ``ProceduralDef`` is a named, deterministic content definition that generates
some output (a texture, a biome description, a building layout, …) from an RNG
seeded by the world seed and any caller-supplied parameters.

All environment content in the engine is authored as subclasses of
``ProceduralDef`` (or one of its domain subclasses such as
``ProceduralTextureDef``).  The definition is an *instance*, registered by
name so any layer can ask for it via ``procedural.get("my_def_name")``.

See ARCHITECTURE.md §5.2 and §8 for the design rationale.

Quick example — define and register a custom texture:
::

    from fire_engine.procedural.defs import ProceduralDef, register_def
    import numpy as np

    @register_def
    class MyTextureDef(ProceduralDef):
        name = "my_texture"

        def generate(self, rng: np.random.Generator, **params) -> np.ndarray:
            height = params.get("height", 256)
            width  = params.get("width",  256)
            # … synthesise RGBA array …
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            return rgba

    # Now accessible anywhere via:
    #   from fire_engine.procedural import get
    #   arr = get("my_texture")          # returns (256, 256, 4) uint8 ndarray
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

__all__ = ["ProceduralDef", "register_def"]


class ProceduralDef(ABC):
    """
    Abstract base class for all procedural content definitions.

    A ``ProceduralDef`` is an *instance* (not a class) that knows how to
    generate a particular piece of content deterministically from an RNG
    and optional keyword parameters.

    Subclass conventions
    --------------------
    - Set the class attribute ``name`` to a unique string identifier.
      This is the key used for ``procedural.get("name")`` and
      ``procedural.register(instance)``.
    - Override ``generate(rng, **params) -> <result>``.  The result type is
      left to the subclass (textures return ``np.ndarray (H,W,4) uint8``;
      future content types return their own structures).
    - Do NOT generate content in ``__init__``.  ``generate`` is called lazily
      by the registry, which also handles caching.
    - Do NOT import panda3d.  ``ProceduralDef`` and all its subclasses must be
      headless-importable so the test suite and CLI tools can run without a GPU.

    Parameters
    ----------
    (none — ``name`` is a class attribute, not a constructor argument)

    Example
    -------
    See module docstring for a minimal working example.
    """

    #: Unique string key for this definition.  Set as a class attribute in each
    #: concrete subclass.  Must not contain whitespace.
    name: str

    @abstractmethod
    def generate(self, rng: np.random.Generator, **params):
        """
        Generate content from *rng* and optional keyword *params*.

        Parameters
        ----------
        rng : numpy.random.Generator
            A deterministic generator seeded by the registry from
            ``core.rng.for_domain("procedural", self.name, params_digest)``.
            Do NOT re-seed or discard it; using it here is what guarantees
            determinism.
        **params : any
            Caller-supplied overrides (e.g. ``width=512``, ``octaves=4``).
            The registry turns these into a canonical digest that forms part
            of the cache key, so changing params busts the cache correctly.

        Returns
        -------
        Subclass-defined result type.  ``ProceduralTextureDef`` returns
        ``np.ndarray (H, W, 4) uint8``.

        Raises
        ------
        NotImplementedError
            Always from the abstract base — override this method.
            See ARCHITECTURE.md §5.2 for the ProceduralDef contract.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.generate() not implemented. "
            "See ARCHITECTURE.md §5.2 for the ProceduralDef contract."
        )


def register_def(cls: type[ProceduralDef]) -> type[ProceduralDef]:
    """
    Class decorator that instantiates *cls* and registers it in the global
    procedural registry immediately at import time.

    Usage::

        @register_def
        class WastelandGroundDef(ProceduralTextureDef):
            name = "wasteland_ground"
            ...

    The decorator is a no-op if the class has not yet defined ``name``
    (e.g. if you decorate an intermediate abstract base class); in that case
    import will raise ``AttributeError`` — always set ``name`` before
    decorating.

    Returns
    -------
    type[ProceduralDef]
        The original class, unchanged (so the class can still be subclassed
        or instantiated directly in tests).
    """
    # Defer the import to avoid a circular import at module load time
    # (registry imports defs; defs should not import registry at the top level).
    from fire_engine.procedural import registry as _reg

    _reg.register(cls())
    return cls
