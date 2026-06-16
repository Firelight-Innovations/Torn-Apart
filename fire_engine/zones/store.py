"""
zones/store.py — ZoneStore: the runtime registry of ZoneVolumes (Saveable).

One ZoneStore per world holds every :class:`~fire_engine.zones.volume.ZoneVolume`
(grass regions, future biome regions, ...).  It implements the ``Saveable``
protocol (``save_key="zones"``): the delta is the **full volume list** when it
differs from the boot-time baseline (volumes are few and tiny — a handful of
dicts — so a full snapshot beats per-volume diffing), and ``{}`` when nothing
changed since :meth:`mark_baseline` was called.

Old saves written before zones existed simply lack the ``"zones"`` key;
``SaveManager.load`` never calls ``apply_delta`` for absent keys, so the store
keeps its fresh boot defaults — no migration needed.

Example
-------
    from fire_engine.zones import ZoneStore

    zones = ZoneStore()
    vol = zones.add("grass", (-12.0, -5.0, 6.0), (12.0, 25.0, 10.0),
                    params={"density": 12.0})
    zones.mark_baseline()              # boot defaults = baseline, delta == {}
    zones.volumes("grass")             # (vol,)
    save_manager.register(zones)       # participates in F5/F9 delta saves

Docs: docs/systems/zones.md
"""

from __future__ import annotations

from typing import Any

from fire_engine.core import get_logger
from fire_engine.zones.volume import ZoneVolume

__all__ = ["ZoneStore"]

_log = get_logger("zones.store")

_DELTA_VERSION = 1


class ZoneStore:
    """
    Mutable registry of ZoneVolumes; Saveable with ``save_key="zones"``.

    Attributes
    ----------
    save_key : str
        ``"zones"`` — the delta-save envelope key.
    version : int
        Monotonic change counter — bumped on every add/remove/replace.
        Renderers compare it against the value they last built from to know
        when to rebuild instancing nodes.

    Example
    -------
    >>> store = ZoneStore()
    >>> v = store.add("grass", (0.0, 0.0, 0.0), (8.0, 8.0, 4.0))
    >>> store.volumes() == (v,)
    True

    Docs: docs/systems/zones.md
    """

    save_key: str = "zones"

    def __init__(self) -> None:
        self._volumes: dict[int, ZoneVolume] = {}
        self._next_id: int = 1
        self.version: int = 0
        self._baseline: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        tag: str,
        min_corner: tuple[float, float, float],
        max_corner: tuple[float, float, float],
        *,
        biome: str | None = None,
        params: dict[str, float] | None = None,
    ) -> ZoneVolume:
        """
        Create, register and return a new volume (id assigned by the store).

        Parameters
        ----------
        tag : str
            Volume meaning — ``"grass"`` or ``"biome"``.
        min_corner / max_corner : tuple[float, float, float]
            World-space AABB corners in meters (``min < max`` per axis).
        biome : str | None
            Biome name for ``tag="biome"`` volumes.
        params : dict[str, float] | None
            Per-volume tuning (e.g. grass ``"density"`` blades/m²).

        Docs: docs/systems/zones.md
        """
        vol = ZoneVolume(
            id=self._next_id,
            tag=tag,
            min_corner=min_corner,
            max_corner=max_corner,
            biome=biome,
            params=dict(params or {}),
        )
        self._volumes[vol.id] = vol
        self._next_id += 1
        self.version += 1
        return vol

    def remove(self, volume_id: int) -> bool:
        """Remove a volume by id; returns True when it existed.

        Docs: docs/systems/zones.md
        """
        if self._volumes.pop(volume_id, None) is None:
            return False
        self.version += 1
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def volumes(self, tag: str | None = None) -> tuple[ZoneVolume, ...]:
        """
        All volumes (ordered by id), optionally filtered by tag.

        Parameters
        ----------
        tag : str | None
            When given, only volumes whose ``tag`` matches are returned.

        Docs: docs/systems/zones.md
        """
        vols = sorted(self._volumes.values(), key=lambda v: v.id)
        if tag is not None:
            vols = [v for v in vols if v.tag == tag]
        return tuple(vols)

    def get(self, volume_id: int) -> ZoneVolume | None:
        """The volume with this id, or None.

        Docs: docs/systems/zones.md
        """
        return self._volumes.get(volume_id)

    # ------------------------------------------------------------------
    # Saveable protocol
    # ------------------------------------------------------------------

    def mark_baseline(self) -> None:
        """
        Snapshot the current volume list as the procedural baseline.

        Call once at boot after registering the world's default volumes —
        :meth:`get_delta` then returns ``{}`` until something actually
        changes, keeping untouched worlds at ~0 save bytes.

        Docs: docs/systems/zones.md
        """
        self._baseline = self._snapshot()

    def get_delta(self) -> dict[str, Any]:
        """
        Full volume list when it deviates from the baseline, else ``{}``.

        Returns
        -------
        dict
            ``{}`` when unchanged; otherwise ``{"version": 1,
            "volumes": [vol.to_dict(), ...], "next_id": int}``.

        Docs: docs/systems/zones.md
        """
        snap = self._snapshot()
        if self._baseline is not None and snap == self._baseline:
            return {}
        return {"version": _DELTA_VERSION, "volumes": snap, "next_id": int(self._next_id)}

    def apply_delta(self, delta: dict[str, Any]) -> None:
        """
        Replace the volume set with the saved one (post-baseline overlay).

        An empty delta means "baseline was saved unchanged" — the freshly
        registered boot defaults already ARE the baseline, so nothing happens.

        Docs: docs/systems/zones.md
        """
        if not delta:
            return
        version = int(delta.get("version", 0))
        if version > _DELTA_VERSION:
            _log.warning(
                "zones delta version %d newer than supported %d — ignoring", version, _DELTA_VERSION
            )
            return
        self._volumes = {int(d["id"]): ZoneVolume.from_dict(d) for d in delta.get("volumes", ())}
        self._next_id = int(delta.get("next_id", max(self._volumes, default=0) + 1))
        self.version += 1

    # ------------------------------------------------------------------

    def _snapshot(self) -> list[dict[str, Any]]:
        """Serialised, id-ordered volume list (comparison + delta payload)."""
        return [v.to_dict() for v in self.volumes()]
