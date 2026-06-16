"""Trivial support types for the ``.asset`` prefab file format.

Frozen dataclasses (:class:`Transform`, :class:`AssetSource`) and exception
classes (:class:`AssetError`, :class:`AssetVersionError`) shared across the
:mod:`fire_engine.assets` modules.

Docs: docs/systems/assets.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Vec3T = tuple[float, float, float]
QuatT = tuple[float, float, float, float]  # (w, x, y, z)

_IDENTITY_QUAT: QuatT = (1.0, 0.0, 0.0, 0.0)
_ONE: Vec3T = (1.0, 1.0, 1.0)
_ZERO: Vec3T = (0.0, 0.0, 0.0)


class AssetError(ValueError):
    """A malformed, missing, or unreadable .asset file (or blob).

    Docs: docs/systems/assets.md
    """


class AssetVersionError(AssetError):
    """An .asset whose ``fire_asset`` spec version this build cannot load.

    Raised when the file's version is newer than
    :data:`~fire_engine.assets.constants.FIRE_ASSET_VERSION` (forward-incompatible).

    Docs: docs/systems/assets.md
    """


@dataclass(frozen=True)
class Transform:
    """A local TRS placement for materialising a prefab root.

    Attributes:
        position: translation in meters ``(x, y, z)``, Z-up.
        rotation: rotation quaternion ``(w, x, y, z)``.
        scale: unitless scale factors ``(x, y, z)``.

    Docs: docs/systems/assets.md
    """

    position: Vec3T = _ZERO
    rotation: QuatT = _IDENTITY_QUAT
    scale: Vec3T = _ONE


@dataclass(frozen=True)
class AssetSource:
    """Provenance for a generated asset — enables a future "regenerate".

    Attributes:
        def_name: registry name of the ``ProceduralDef`` that produced the asset
            (serialised as ``"def"`` in the envelope, since ``def`` is a keyword).
        params: keyword params passed to the def's ``generate``.
        seed: the RNG seed used, or ``None`` if not seeded.

    Docs: docs/systems/assets.md
    """

    def_name: str
    params: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form ``{"def", "params", "seed"}``.

        Docs: docs/systems/assets.md
        """
        return {"def": self.def_name, "params": dict(self.params), "seed": self.seed}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AssetSource:
        """Inverse of :meth:`to_dict`.

        Docs: docs/systems/assets.md
        """
        return cls(
            def_name=str(d["def"]),
            params=dict(d.get("params", {})),
            seed=None if d.get("seed") is None else int(d["seed"]),
        )
