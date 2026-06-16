"""SceneObjectStore — the authoring scene graph shared by editor and game.

A headless, deterministic model of the placeable objects in an open world: the
Unity-style hierarchy the Fire Editor's Scene View and sidebar tree both read,
and the schema the game's :class:`~fire_engine.scene.runtime.SceneRuntime`
consumes when it loads an authored scene. Each object is an id, a name, a kind,
an optional parent, and a TRS transform. The store is pure Python (no panda3d,
no RNG — ids come from a monotonic counter so the same sequence of edits always
yields the same ids), so it is fully headless-testable and participates in
delta saves via the :class:`~fire_engine.save.saveable.Saveable` protocol.

This module lives in the ENGINE (not the editor) so the placed-object schema
has exactly one definition: the editor daemon imports it via the
``fire_editor.scene_objects`` shim, the game via ``fire_engine.scene``
(DECISIONS.md 2026-06-12 — editor imports engine, never the reverse).

The baseline (procedural) scene is empty, so an untouched world saves ~0 bytes;
every object the user creates is a deviation captured in :meth:`get_delta`.

These authoring objects map onto runtime ``fire_engine.render.GameObject`` instances
when a world is built for play (see ``fire_engine.scene.runtime``); here they are
plain data the editor manipulates.

Example::

    store = SceneObjectStore()
    cube = store.create("cube", name="Crate")
    child = store.create("empty", parent=cube["id"], name="Pivot")
    store.set_transform(cube["id"], position=(4.0, 0.0, 2.0))
    store.reparent(child["id"], parent=None)   # promote to a root
    tree = store.tree()                          # flat, DFS-ordered list of dicts
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from fire_engine.scene.components import (
    COMPONENT_CATALOG,
    coerce_params,
    default_components_for_kind,
    is_known,
    make_component,
)

# Object kinds the editor can place. "empty" is a bare transform (a grouping
# node, like an empty Unity GameObject); the rest carry a default visual gizmo.
KINDS: frozenset[str] = frozenset({"empty", "cube", "sphere", "light", "spawn"})

Vec3T = tuple[float, float, float]
QuatT = tuple[float, float, float, float]  # (w, x, y, z)

_IDENTITY_QUAT: QuatT = (1.0, 0.0, 0.0, 0.0)
_ONE: Vec3T = (1.0, 1.0, 1.0)
_ZERO: Vec3T = (0.0, 0.0, 0.0)


class SceneError(ValueError):
    """Invalid scene operation (unknown id/kind, or a reparent that would cycle)."""


@dataclass
class SceneObject:
    """One node in the authoring hierarchy.

    Attributes:
        id: Stable integer id, unique within a session (monotonic counter).
        name: Display name (not required unique).
        kind: One of :data:`KINDS`.
        parent: Parent object id, or ``None`` for a root.
        position: Local translation in meters ``(x, y, z)``, Z-up.
        rotation: Local rotation quaternion ``(w, x, y, z)``.
        scale: Local scale factors ``(x, y, z)``.
        components: Built-in components beyond the Transform — each a dict
            ``{"type", "enabled", "params"}`` (see
            :mod:`fire_engine.scene.components`). ``kind`` seeds these on
            creation; thereafter the list is the source of truth for visuals.
            The Transform is intrinsic (the TRS fields) and is NOT in this list.
    """

    id: int
    name: str
    kind: str
    parent: int | None = None
    position: Vec3T = _ZERO
    rotation: QuatT = _IDENTITY_QUAT
    scale: Vec3T = _ONE
    components: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Wire/serialisation form: plain JSON-friendly primitives."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "parent": self.parent,
            "position": list(self.position),
            "rotation": list(self.rotation),
            "scale": list(self.scale),
            "components": copy.deepcopy(self.components),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SceneObject:
        kind = str(d["kind"])
        # Migration seam (the ONLY one): pre-component saves lack "components",
        # so synthesise the kind's defaults. New saves carry them verbatim.
        raw = d.get("components")
        components = default_components_for_kind(kind) if raw is None else copy.deepcopy(list(raw))
        return SceneObject(
            id=int(d["id"]),
            name=str(d["name"]),
            kind=kind,
            parent=None if d.get("parent") is None else int(d["parent"]),
            position=tuple(float(v) for v in d.get("position", _ZERO)),  # type: ignore[arg-type]
            rotation=tuple(float(v) for v in d.get("rotation", _IDENTITY_QUAT)),  # type: ignore[arg-type]
            scale=tuple(float(v) for v in d.get("scale", _ONE)),  # type: ignore[arg-type]
            components=components,
        )


class SceneObjectStore:
    """Mutable, ordered store of :class:`SceneObject` with hierarchy operations.

    Sibling order is insertion order (Python dict preserves it); :meth:`tree`
    returns a depth-first flattening (roots first) that the tree view and the
    viewport both consume.

    Implements the ``Saveable`` protocol (``save_key`` + ``get_delta`` /
    ``apply_delta``) so the scene persists inside the world's ``.ta`` save.
    """

    save_key: str = "editor_scene"

    def __init__(self) -> None:
        self._objects: dict[int, SceneObject] = {}
        self._next_id: int = 1

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._objects)

    def get(self, obj_id: int) -> SceneObject:
        obj = self._objects.get(int(obj_id))
        if obj is None:
            raise SceneError(f"no scene object with id {obj_id}")
        return obj

    def _children(self, parent: int | None) -> list[SceneObject]:
        return [o for o in self._objects.values() if o.parent == parent]

    def _descendants(self, obj_id: int) -> set[int]:
        """All ids beneath ``obj_id`` (exclusive), via breadth-first walk."""
        out: set[int] = set()
        frontier = [obj_id]
        while frontier:
            cur = frontier.pop()
            for child in self._children(cur):
                if child.id not in out:
                    out.add(child.id)
                    frontier.append(child.id)
        return out

    def tree(self) -> list[dict[str, Any]]:
        """Flat, depth-first list of object dicts (roots first, siblings ordered)."""
        out: list[dict[str, Any]] = []

        def walk(parent: int | None) -> None:
            for child in self._children(parent):
                out.append(child.to_dict())
                walk(child.id)

        walk(None)
        return out

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    def create(
        self,
        kind: str,
        *,
        parent: int | None = None,
        name: str | None = None,
        position: Vec3T = _ZERO,
    ) -> dict[str, Any]:
        """Create a new object; returns its dict form. Raises on bad kind/parent."""
        k = str(kind).lower()
        if k not in KINDS:
            raise SceneError(f"unknown kind {kind!r}; expected one of {sorted(KINDS)}")
        if parent is not None and int(parent) not in self._objects:
            raise SceneError(f"no parent object with id {parent}")
        obj = SceneObject(
            id=self._next_id,
            name=name or _default_name(k),
            kind=k,
            parent=None if parent is None else int(parent),
            position=tuple(float(v) for v in position),  # type: ignore[arg-type]
            components=default_components_for_kind(k),
        )
        self._objects[obj.id] = obj
        self._next_id += 1
        return obj.to_dict()

    def rename(self, obj_id: int, name: str) -> dict[str, Any]:
        obj = self.get(obj_id)
        obj.name = str(name)
        return obj.to_dict()

    def reparent(self, obj_id: int, parent: int | None) -> dict[str, Any]:
        """Move ``obj_id`` under ``parent`` (``None`` = root). Rejects cycles."""
        obj = self.get(obj_id)
        if parent is not None:
            pid = int(parent)
            if pid == obj.id:
                raise SceneError("cannot parent an object to itself")
            if pid not in self._objects:
                raise SceneError(f"no parent object with id {parent}")
            if pid in self._descendants(obj.id):
                raise SceneError("cannot parent an object beneath its own descendant")
            obj.parent = pid
        else:
            obj.parent = None
        return obj.to_dict()

    def set_transform(
        self,
        obj_id: int,
        *,
        position: Vec3T | None = None,
        rotation: QuatT | None = None,
        scale: Vec3T | None = None,
    ) -> dict[str, Any]:
        obj = self.get(obj_id)
        if position is not None:
            obj.position = tuple(float(v) for v in position)  # type: ignore[assignment]
        if rotation is not None:
            obj.rotation = tuple(float(v) for v in rotation)  # type: ignore[assignment]
        if scale is not None:
            obj.scale = tuple(float(v) for v in scale)  # type: ignore[assignment]
        return obj.to_dict()

    # ------------------------------------------------------------------ #
    # Components
    # ------------------------------------------------------------------ #
    def add_component(self, obj_id: int, type_name: str) -> dict[str, Any]:
        """Attach a built-in component of ``type_name`` to ``obj_id``.

        Components are independent of ``kind`` (Unity-style) — an ``empty`` can
        be given a Light. Singletons (every built-in) reject a second instance.
        Raises :class:`SceneError` on unknown type or singleton violation.
        """
        obj = self.get(obj_id)
        t = str(type_name)
        if not is_known(t):
            raise SceneError(
                f"unknown component type {type_name!r}; expected one of {sorted(COMPONENT_CATALOG)}"
            )
        if not COMPONENT_CATALOG[t].multiple and any(c.get("type") == t for c in obj.components):
            raise SceneError(f"object {obj_id} already has a {t} component")
        obj.components.append(make_component(t))
        return obj.to_dict()

    def remove_component(self, obj_id: int, index: int) -> dict[str, Any]:
        """Remove the component at ``index`` (0-based). Raises on a bad index."""
        obj = self.get(obj_id)
        i = int(index)
        if i < 0 or i >= len(obj.components):
            raise SceneError(f"object {obj_id} has no component at index {index}")
        obj.components.pop(i)
        return obj.to_dict()

    def set_component(
        self,
        obj_id: int,
        index: int,
        *,
        params: dict[str, Any] | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any]:
        """Edit the component at ``index``: merge validated ``params`` and/or
        toggle ``enabled``. Unknown/extraneous param keys are dropped and values
        are coerced to the catalog field types. Raises on a bad index."""
        obj = self.get(obj_id)
        i = int(index)
        if i < 0 or i >= len(obj.components):
            raise SceneError(f"object {obj_id} has no component at index {index}")
        comp = obj.components[i]
        if enabled is not None:
            comp["enabled"] = bool(enabled)
        if params:
            clean = coerce_params(str(comp.get("type")), params)
            comp.setdefault("params", {}).update(clean)
        return obj.to_dict()

    def delete(self, obj_id: int) -> list[int]:
        """Delete ``obj_id`` and all its descendants; returns removed ids."""
        obj = self.get(obj_id)
        removed = self._descendants(obj.id) | {obj.id}
        for rid in removed:
            self._objects.pop(rid, None)
        return sorted(removed)

    def clear(self) -> None:
        """Drop all objects (e.g. on ``world.open``); resets the id counter."""
        self._objects.clear()
        self._next_id = 1

    # ------------------------------------------------------------------ #
    # Saveable protocol
    # ------------------------------------------------------------------ #
    def get_delta(self) -> dict[str, Any]:
        """Deviation from the empty baseline: the full object list (or ``{}``)."""
        if not self._objects:
            return {}
        return {
            "objects": [o.to_dict() for o in self._objects.values()],
            "next_id": self._next_id,
        }

    def apply_delta(self, delta: dict[str, Any]) -> None:
        """Restore objects saved by :meth:`get_delta` onto the empty baseline."""
        self.clear()
        objs = delta.get("objects", [])
        for d in objs:
            obj = SceneObject.from_dict(d)
            self._objects[obj.id] = obj
        # Keep ids monotonic past anything loaded, even if next_id was absent.
        max_id = max((o.id for o in self._objects.values()), default=0)
        self._next_id = max(int(delta.get("next_id", 0)), max_id + 1)


def _default_name(kind: str) -> str:
    return {
        "empty": "GameObject",
        "cube": "Cube",
        "sphere": "Sphere",
        "light": "Light",
        "spawn": "Spawn Point",
    }.get(kind, kind.capitalize())
