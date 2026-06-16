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

Docs: docs/systems/scene.md
"""

from __future__ import annotations

from typing import Any

from fire_engine.scene.components import (
    COMPONENT_CATALOG,
    coerce_params,
    default_components_for_kind,
    is_known,
    make_component,
)
from fire_engine.scene.types import (
    _ZERO,
    QuatT,
    SceneError,
    SceneObject,
    Vec3T,
)

# Re-export support types so every existing import path keeps working.
__all__ = [
    "KINDS",
    "QuatT",
    "SceneError",
    "SceneObject",
    "SceneObjectStore",
    "Vec3T",
]

# Object kinds the editor can place. "empty" is a bare transform (a grouping
# node, like an empty Unity GameObject); the rest carry a default visual gizmo.
KINDS: frozenset[str] = frozenset({"empty", "cube", "sphere", "light", "spawn"})


class SceneObjectStore:
    """Mutable, ordered store of :class:`SceneObject` with hierarchy operations.

    Sibling order is insertion order (Python dict preserves it); :meth:`tree`
    returns a depth-first flattening (roots first) that the tree view and the
    viewport both consume.

    Implements the ``Saveable`` protocol (``save_key`` + ``get_delta`` /
    ``apply_delta``) so the scene persists inside the world's ``.ta`` save.

    Docs: docs/systems/scene.md
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
        """Return the :class:`SceneObject` for ``obj_id``; raises :class:`SceneError` if missing.

        Docs: docs/systems/scene.md
        """
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
        """Flat, depth-first list of object dicts (roots first, siblings ordered).

        Docs: docs/systems/scene.md
        """
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
        """Create a new object; returns its dict form. Raises on bad kind/parent.

        Docs: docs/systems/scene.md
        """
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
        """Rename the object at ``obj_id``; returns its updated dict form.

        Docs: docs/systems/scene.md
        """
        obj = self.get(obj_id)
        obj.name = str(name)
        return obj.to_dict()

    def reparent(self, obj_id: int, parent: int | None) -> dict[str, Any]:
        """Move ``obj_id`` under ``parent`` (``None`` = root). Rejects cycles.

        Docs: docs/systems/scene.md
        """
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
        """Set the local TRS of ``obj_id``; returns the updated dict form.

        All keyword arguments are optional — omit to keep the current value.
        Positions are in meters (local to parent); rotation is a quaternion
        ``(w, x, y, z)``; scale is a unitless multiplier ``(x, y, z)``.

        Docs: docs/systems/scene.md
        """
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

        Docs: docs/systems/scene.md
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
        """Remove the component at ``index`` (0-based). Raises on a bad index.

        Docs: docs/systems/scene.md
        """
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
        are coerced to the catalog field types. Raises on a bad index.

        Docs: docs/systems/scene.md
        """
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
        """Delete ``obj_id`` and all its descendants; returns removed ids.

        Docs: docs/systems/scene.md
        """
        obj = self.get(obj_id)
        removed = self._descendants(obj.id) | {obj.id}
        for rid in removed:
            self._objects.pop(rid, None)
        return sorted(removed)

    def clear(self) -> None:
        """Drop all objects (e.g. on ``world.open``); resets the id counter.

        Docs: docs/systems/scene.md
        """
        self._objects.clear()
        self._next_id = 1

    # ------------------------------------------------------------------ #
    # Saveable protocol
    # ------------------------------------------------------------------ #
    def get_delta(self) -> dict[str, Any]:
        """Deviation from the empty baseline: the full object list (or ``{}``).

        Docs: docs/systems/scene.md
        """
        if not self._objects:
            return {}
        return {
            "objects": [o.to_dict() for o in self._objects.values()],
            "next_id": self._next_id,
        }

    def apply_delta(self, delta: dict[str, Any]) -> None:
        """Restore objects saved by :meth:`get_delta` onto the empty baseline.

        Docs: docs/systems/scene.md
        """
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
