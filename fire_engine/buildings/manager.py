"""
buildings/manager.py — BuildingManager: the runtime registry of buildings.

One BuildingManager per world owns every placed :class:`~fire_engine.buildings.model.Building`,
assigns their world ids, publishes :class:`~fire_engine.core.event_bus.BuildingChangedEvent`
on every change (so the renderer rebuilds and lighting can invalidate), and
implements the ``Saveable`` protocol (``save_key="buildings"``) following the
ZoneStore pattern: a baseline snapshot taken at boot, a **full building list**
delta when the set deviates from it, and ``{}`` when nothing changed.

``add()`` always **clones** the incoming spec (``Building.from_dict(spec.to_dict())``)
before assigning an id — so a building handed straight from ``procedural.get``
(a cached, shared def output) is never mutated in place, and two ``add`` calls
on the same spec yield two independent buildings.

Old saves written before buildings existed simply lack the ``"buildings"``
key; ``SaveManager.load`` never calls ``apply_delta`` for absent keys, so the
manager keeps its fresh boot state — no migration needed.

Example
-------
    from fire_engine.buildings import BuildingManager
    from fire_engine.procedural import get as get_def

    mgr = BuildingManager(config, bus)
    house = mgr.add(get_def("building_demo_house"))   # clone + id + event
    mgr.mark_baseline()                                # boot set = baseline
    save_manager.register(mgr)                         # joins F5/F9 saves
"""

from __future__ import annotations

from typing import Any

from fire_engine.buildings.model import Building
from fire_engine.core import get_logger
from fire_engine.core.event_bus import BuildingChangedEvent

__all__ = ["BuildingManager"]

_log = get_logger("buildings.manager")

_DELTA_VERSION = 1


class BuildingManager:
    """
    Mutable registry of placed buildings; Saveable with ``save_key="buildings"``.

    Attributes
    ----------
    save_key : str
        ``"buildings"`` — the delta-save envelope key.
    version : int
        Monotonic change counter — bumped on every add/remove/modify.  The
        renderer compares it against the value it last rebuilt from.

    Example
    -------
    >>> mgr = BuildingManager(config, bus=None)
    >>> b = mgr.add(spec)            # doctest: +SKIP
    >>> mgr.get(b.id) is b           # doctest: +SKIP
    True
    """

    save_key: str = "buildings"

    def __init__(self, config: Any, bus: Any | None) -> None:
        self._config = config
        self._bus = bus
        self._buildings: dict[int, Building] = {}
        self._next_id: int = 1
        self.version: int = 0
        self._baseline: list[dict] | None = None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, spec: Building) -> Building:
        """
        Clone ``spec``, assign it a fresh world id, register it, and publish a
        ``"added"`` :class:`BuildingChangedEvent`.  Returns the managed clone
        (not ``spec``) — mutate the return value, never the argument.
        """
        clone = Building.from_dict(spec.to_dict())
        clone.id = self._next_id
        self._next_id += 1
        self._buildings[clone.id] = clone
        self.version += 1
        self._publish(clone, "added")
        return clone

    def remove(self, building_id: int) -> bool:
        """Remove a building by id; publish ``"removed"`` with its last bounds.
        Returns True when it existed."""
        b = self._buildings.pop(building_id, None)
        if b is None:
            return False
        self.version += 1
        self._publish(b, "removed")
        return True

    def notify_changed(self, building_id: int) -> None:
        """
        Announce that a managed building was edited in place (call after
        mutating the object returned by :meth:`add`/:meth:`get`).  Bumps the
        version and publishes a ``"modified"`` event.

        Raises
        ------
        KeyError — no building with this id.
        """
        b = self._buildings.get(building_id)
        if b is None:
            raise KeyError(f"no building id={building_id}")
        self.version += 1
        self._publish(b, "modified")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, building_id: int) -> Building | None:
        """The managed building with this id, or None."""
        return self._buildings.get(building_id)

    def buildings(self) -> tuple[Building, ...]:
        """All managed buildings, ordered by id."""
        return tuple(self._buildings[k] for k in sorted(self._buildings))

    # ------------------------------------------------------------------
    # Saveable protocol (ZoneStore pattern)
    # ------------------------------------------------------------------

    def mark_baseline(self) -> None:
        """
        Snapshot the current building set as the procedural baseline.

        Call once at boot after placing the world's default/procedural
        buildings — :meth:`get_delta` then returns ``{}`` until something
        actually changes, so an untouched world costs ~0 save bytes.
        """
        self._baseline = self._snapshot()

    def get_delta(self) -> dict:
        """
        Full building list when it deviates from the baseline, else ``{}``.

        Returns
        -------
        dict
            ``{}`` when unchanged; otherwise ``{"version": 1, "next_id": int,
            "buildings": [building.to_dict(), ...]}``.
        """
        snap = self._snapshot()
        if self._baseline is not None and snap == self._baseline:
            return {}
        return {"version": _DELTA_VERSION,
                "next_id": int(self._next_id),
                "buildings": snap}

    def apply_delta(self, delta: dict) -> None:
        """
        Replace the building set with the saved one and republish ``"added"``
        for each so the renderer rebuilds.  An empty delta means "baseline
        saved unchanged" — the fresh boot set already IS the baseline.
        """
        if not delta:
            return
        version = int(delta.get("version", 0))
        if version > _DELTA_VERSION:
            _log.warning("buildings delta version %d newer than supported %d "
                         "— ignoring", version, _DELTA_VERSION)
            return
        self._buildings = {}
        for d in delta.get("buildings", ()):
            b = Building.from_dict(d)
            self._buildings[b.id] = b
        self._next_id = int(delta.get("next_id",
                                      max(self._buildings, default=0) + 1))
        self.version += 1
        for b in self.buildings():
            self._publish(b, "added")

    # ------------------------------------------------------------------

    def _snapshot(self) -> list[dict]:
        """Serialised, id-ordered building list (comparison + delta payload)."""
        return [b.to_dict() for b in self.buildings()]

    def _publish(self, building: Building, change: str) -> None:
        if self._bus is None:
            return
        mn, mx = building.world_aabb()
        self._bus.publish(BuildingChangedEvent(
            building_id=building.id, change=change,
            bounds_min=mn, bounds_max=mx))
