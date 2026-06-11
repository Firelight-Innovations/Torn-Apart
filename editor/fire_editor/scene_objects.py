"""SceneObjectStore — the editor's authoring scene graph (EDITOR_PRD Phase E2).

A headless, deterministic model of the placeable objects in an open world: the
Unity-style hierarchy the Scene View and the sidebar tree both read. Each object
is an id, a name, a kind, an optional parent, and a TRS transform. The store is
pure Python (no panda3d, no RNG — ids come from a monotonic counter so the same
sequence of edits always yields the same ids), so it is fully headless-testable
and participates in delta saves via the :class:`~fire_engine.save.saveable.Saveable`
protocol.

The baseline (procedural) scene is empty, so an untouched world saves ~0 bytes;
every object the user creates is a deviation captured in :meth:`get_delta`.

These authoring objects map onto runtime ``fire_engine.world.GameObject`` instances
when a world is built for play; here they are plain data the editor manipulates.

Example::

    store = SceneObjectStore()
    cube = store.create("cube", name="Crate")
    child = store.create("empty", parent=cube["id"], name="Pivot")
    store.set_transform(cube["id"], position=(4.0, 0.0, 2.0))
    store.reparent(child["id"], parent=None)   # promote to a root
    tree = store.tree()                          # flat, DFS-ordered list of dicts
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
    """

    id: int
    name: str
    kind: str
    parent: int | None = None
    position: Vec3T = _ZERO
    rotation: QuatT = _IDENTITY_QUAT
    scale: Vec3T = _ONE

    def to_dict(self) -> dict:
        """Wire/serialisation form: plain JSON-friendly primitives."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "parent": self.parent,
            "position": list(self.position),
            "rotation": list(self.rotation),
            "scale": list(self.scale),
        }

    @staticmethod
    def from_dict(d: dict) -> "SceneObject":
        return SceneObject(
            id=int(d["id"]),
            name=str(d["name"]),
            kind=str(d["kind"]),
            parent=None if d.get("parent") is None else int(d["parent"]),
            position=tuple(float(v) for v in d.get("position", _ZERO)),  # type: ignore[arg-type]
            rotation=tuple(float(v) for v in d.get("rotation", _IDENTITY_QUAT)),  # type: ignore[arg-type]
            scale=tuple(float(v) for v in d.get("scale", _ONE)),  # type: ignore[arg-type]
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

    def tree(self) -> list[dict]:
        """Flat, depth-first list of object dicts (roots first, siblings ordered)."""
        out: list[dict] = []

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
    ) -> dict:
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
        )
        self._objects[obj.id] = obj
        self._next_id += 1
        return obj.to_dict()

    def rename(self, obj_id: int, name: str) -> dict:
        obj = self.get(obj_id)
        obj.name = str(name)
        return obj.to_dict()

    def reparent(self, obj_id: int, parent: int | None) -> dict:
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
    ) -> dict:
        obj = self.get(obj_id)
        if position is not None:
            obj.position = tuple(float(v) for v in position)  # type: ignore[assignment]
        if rotation is not None:
            obj.rotation = tuple(float(v) for v in rotation)  # type: ignore[assignment]
        if scale is not None:
            obj.scale = tuple(float(v) for v in scale)  # type: ignore[assignment]
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
    def get_delta(self) -> dict:
        """Deviation from the empty baseline: the full object list (or ``{}``)."""
        if not self._objects:
            return {}
        return {
            "objects": [o.to_dict() for o in self._objects.values()],
            "next_id": self._next_id,
        }

    def apply_delta(self, delta: dict) -> None:
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
