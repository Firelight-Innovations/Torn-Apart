"""Shared trivial support types for the fire_engine.scene package.

Frozen dataclasses, dataclasses, and exception classes used across scene
submodules: :class:`FieldSpec`, :class:`ComponentSpec`, :class:`SceneError`,
:class:`SceneObject`.

Docs: docs/systems/scene.md
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Component catalog support types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """One editable parameter of a component.

    Attributes:
        name: Param key inside the component's ``params`` dict.
        ui_type: How the inspector renders/edits it — one of
            ``"float"``, ``"color"`` (rgb 0..1), ``"vec3"``, ``"enum"``, ``"bool"``.
        default: Default value (floats for float, ``[r, g, b]`` for color/vec3,
            a choice string for enum, a bool for bool).
        min, max: Optional inclusive clamp for ``float`` fields.
        choices: Allowed values for an ``enum`` field.
        label: Human label for the inspector (defaults to ``name``).
    """

    name: str
    ui_type: str
    default: Any
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] = ()
    label: str | None = None


@dataclass(frozen=True)
class ComponentSpec:
    """A built-in component type.

    Attributes:
        type: Stable type id stored in each component dict's ``"type"``.
        label: Inspector section title.
        multiple: Whether an object may carry more than one of this type
            (all current built-ins are singletons).
        fields: Editable parameters, in display order.
    """

    type: str
    label: str
    multiple: bool
    fields: tuple[FieldSpec, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Scene object support types
# ---------------------------------------------------------------------------

Vec3T = tuple[float, float, float]
QuatT = tuple[float, float, float, float]  # (w, x, y, z)

_IDENTITY_QUAT: QuatT = (1.0, 0.0, 0.0, 0.0)
_ONE: Vec3T = (1.0, 1.0, 1.0)
_ZERO: Vec3T = (0.0, 0.0, 0.0)


class SceneError(ValueError):
    """Invalid scene operation (unknown id/kind, or a reparent that would cycle).

    Docs: docs/systems/scene.md
    """


@dataclass
class SceneObject:
    """One node in the authoring hierarchy.

    Attributes:
        id: Stable integer id, unique within a session (monotonic counter).
        name: Display name (not required unique).
        kind: One of :data:`~fire_engine.scene.objects.KINDS`.
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
        """Deserialise from a wire/save dict; migrates pre-component saves."""
        # Local import to avoid a circular dependency: types <- components <- types.
        from fire_engine.scene.components import default_components_for_kind

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
