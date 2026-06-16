"""
buildings/defs.py — procedural building definitions (the tag→building seam).

A ``BuildingDef`` is a :class:`~fire_engine.procedural.defs.ProceduralDef` whose
``generate`` returns a fully-authored :class:`~fire_engine.buildings.model.Building`
(every generator action is one of the imperative authoring-API calls).  This is
the slot the future tag/description→building generator plugs into: it will read
``params`` (footprint, storey count, room program, style tags) and emit walls /
openings / rooms / slabs accordingly.

The concrete demo implementation lives in
``fire_engine.buildings._impl.demo_house`` (split to satisfy the
one-public-class-per-module rule); it is imported here as a re-export so
``from fire_engine.buildings.defs import DemoHouseDef`` remains valid.

Determinism: ``generate`` uses no RNG for the demo (a fixed layout), so
``get("building_demo_house")`` twice yields byte-identical ``to_dict()`` — the
property the manager's clone-on-add and the save round-trip both rely on.

Imports allowed: procedural, core (Hard Rule 1 — no panda3d).

Docs: docs/systems/buildings.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fire_engine.buildings.model import Building
from fire_engine.procedural.defs import ProceduralDef

__all__ = ["BuildingDef", "DemoHouseDef"]


class BuildingDef(ProceduralDef):
    """
    Abstract base for procedural buildings: ``generate(rng, **params) -> Building``.

    Subclasses author a building through the imperative API on
    :class:`Building` / :class:`Storey` and return it.  The registry caches the
    result by ``(name, params digest)``; callers hand it to
    ``BuildingManager.add`` (which clones it and assigns a world id), so a
    ``generate`` MUST be a pure function of ``rng`` + ``params`` and must not
    keep references to its return value.

    Contract for the (future) tag→building generator
    -----------------------------------------------
    - Read placement from ``params`` (``position``, ``ground_z``, ``yaw_rad``)
      and program from style ``params`` (``footprint``, ``storeys``, room
      ``tags``).  Use ``rng`` (seeded by the registry) for every random choice —
      never ``random``/unseeded numpy (Hard Rule 2).
    - Build dimensions from :meth:`BuildingDefaults.from_config` (config is the
      single number source), overridable per call via ``params``.
    - Return the ``Building``; the manager assigns its id, the renderer meshes
      and draws it, and the lighting provider (future) voxelizes it.

    Example
    -------
        from fire_engine.procedural import get
        house = get("building_demo_house", ground_z=8.0)   # a Building

    Docs: docs/systems/buildings.md
    """

    def generate(self, rng: np.random.Generator, **params: Any) -> Building:
        raise NotImplementedError(
            "BuildingDef subclasses must implement generate() -> Building. "
            "See ARCHITECTURE.md §5.7 and docs/systems/buildings.md."
        )


# Import the concrete demo implementation so @register_def fires on package
# load — and to re-export DemoHouseDef for backward-compatible imports.
# NOTE: this is a deliberate late import placed AFTER BuildingDef is defined
# to avoid a circular-import bootstrap problem (demo_house.py imports BuildingDef
# from this module).
from fire_engine.buildings._impl.demo_house import DemoHouseDef  # noqa: E402
