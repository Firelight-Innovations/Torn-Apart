"""
devtools/introspect.py — reflect a GameObject into editable inspector sections.

Given a GameObject, :func:`describe_object` produces the list of
:class:`~fire_engine.devtools.fields.Section`s the Inspector shows: identity,
Transform, and one section per attached Component.  Component fields are
discovered generically from ``__slots__`` across the type's MRO and typed by the
*runtime value* (bool/int/float/str/Vec3), so a newly written Component is
inspectable and editable with **zero** extra wiring — just give its tunables
plain typed attributes.

Editing round-trips through the engine's own public state: each editable Field's
``set`` writes the attribute (or calls the proper setter, e.g. ``set_active`` or
``transform.local_rotation =``), so the inspector can never reach past the public
surface (mirrors the engine's "public APIs only" discipline).

Reflection is intentionally duck-typed: this module reads ``.transform``,
``._components``, ``.name`` etc. directly and imports nothing from ``world/``,
so it (and its tests) stay clear of panda3d.

No panda3d imports — headless-testable.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from fire_engine.core.math3d import Vec3, Quat
from fire_engine.devtools.fields import Field, FieldKind, Section

if TYPE_CHECKING:
    from fire_engine.world.gameobject import GameObject
    from fire_engine.world.component import Component
    from fire_engine.terrain.chunk import Chunk


# Attributes every Component carries for the framework's own bookkeeping — never
# shown as tunable inspector rows (``enabled`` is surfaced explicitly below).
_COMPONENT_INTERNAL = {"game_object", "transform", "enabled", "_started"}

_RAD2DEG = 180.0 / math.pi
_DEG2RAD = math.pi / 180.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def describe_object(go: "GameObject") -> list[Section]:
    """
    Build the inspector sections for a GameObject.

    Parameters
    ----------
    go : GameObject
        The selected object.

    Returns
    -------
    list[Section]
        ``[GameObject identity, Transform, <one per component>]``.  Each
        editable Field's ``set`` applies straight back to the live object.

    Example
    -------
        sections = describe_object(selected)
        # sections[1].title == "Transform"
    """
    sections: list[Section] = [_identity_section(go), _transform_section(go)]
    for comp in list(go._components):
        sections.append(_component_section(comp))
    return sections


# ---------------------------------------------------------------------------
# Terrain chunks (a non-GameObject the picker can also select)
# ---------------------------------------------------------------------------

def is_chunk(obj: Any) -> bool:
    """
    True when ``obj`` is a terrain :class:`~fire_engine.terrain.chunk.Chunk`.

    Duck-typed (never imports terrain at runtime, so this package stays panda3d-
    *and* terrain-free): a chunk is anything carrying the voxel-array trio
    ``materials`` / ``coord`` / ``chunk_meters``.  Used by the Inspector to route
    a picked chunk to :func:`describe_chunk` instead of :func:`describe_object`.
    """
    return (
        hasattr(obj, "materials")
        and hasattr(obj, "coord")
        and hasattr(obj, "chunk_meters")
    )


def describe_chunk(chunk: "Chunk") -> list[Section]:
    """
    Build read-only inspector sections for a terrain chunk.

    Chunks are not GameObjects — they have no components or editable Transform —
    so this surfaces their identity (coord / world origin / size) and a live
    read-out of their voxel contents (solid count, fill, material ids, the
    ``dirty`` / ``edited`` streaming + save flags).  All rows are read-only:
    voxels are edited with the brush, not typed into the inspector.

    Parameters
    ----------
    chunk : Chunk
        The selected chunk (duck-typed: ``coord``, ``world_origin``,
        ``chunk_meters``, ``materials``, ``dirty``, ``edited``).

    Returns
    -------
    list[Section]
        ``[Chunk identity, Voxels]`` — fed to the Inspector panel.

    Example
    -------
        sections = describe_chunk(chunk)
        # sections[0].title == "Chunk"
    """
    def origin_str() -> str:
        o = chunk.world_origin
        return f"({o.x:.1f}, {o.y:.1f}, {o.z:.1f}) m"

    def solid_count() -> int:
        return int((chunk.materials > 0).sum())

    def fill_pct() -> str:
        total = int(chunk.materials.size)
        if total == 0:
            return "0.0%"
        return f"{100.0 * solid_count() / total:.1f}%"

    def material_ids() -> str:
        ids = [int(m) for m in np.unique(chunk.materials).tolist()]
        return ", ".join(str(m) for m in ids)

    return [
        Section("Chunk", [
            Field("coord", FieldKind.LABEL, lambda: str(tuple(chunk.coord))),
            Field("world origin", FieldKind.LABEL, origin_str),
            Field("size", FieldKind.LABEL, lambda: f"{chunk.chunk_meters:.1f} m"),
        ]),
        Section("Voxels", [
            Field("solid", FieldKind.LABEL, solid_count),
            Field("total", FieldKind.LABEL, lambda: int(chunk.materials.size)),
            Field("fill", FieldKind.LABEL, fill_pct),
            Field("material ids", FieldKind.LABEL, material_ids),
            Field("dirty", FieldKind.LABEL, lambda: chunk.dirty),
            Field("edited", FieldKind.LABEL, lambda: chunk.edited),
        ]),
    ]


# ---------------------------------------------------------------------------
# Identity + Transform
# ---------------------------------------------------------------------------

def _identity_section(go: "GameObject") -> Section:
    """Name / tag / layer / active flag for the GameObject itself."""

    def set_name(v: str) -> None:
        go.name = str(v)

    def set_tag(v: str) -> None:
        go.tag = str(v)

    def set_layer(v: int) -> None:
        go.layer = int(v)

    def set_active(v: bool) -> None:
        go.set_active(bool(v))

    return Section(
        "GameObject",
        [
            Field("id", FieldKind.LABEL, lambda: str(go.id)[:8]),
            Field("name", FieldKind.STRING, lambda: go.name, set_name),
            Field("tag", FieldKind.STRING, lambda: go.tag, set_tag),
            Field("layer", FieldKind.INT, lambda: go.layer, set_layer),
            Field("active", FieldKind.BOOL, lambda: go.active_self, set_active),
        ],
    )


def _transform_section(go: "GameObject") -> Section:
    """
    Position (local), rotation (euler degrees view of the quaternion), and scale.

    Rotation is shown/edited in **degrees** for usability but stored as a
    quaternion (ARCHITECTURE.md §5.4 — eulers are display-only): the setter
    composes ``Quat.from_euler`` from the entered HPR degrees.
    """
    tf = go.transform

    def get_pos() -> tuple[float, float, float]:
        p = tf.local_position
        return (p.x, p.y, p.z)

    def set_pos(v: tuple[float, float, float]) -> None:
        tf.local_position = Vec3(float(v[0]), float(v[1]), float(v[2]))

    def get_rot_deg() -> tuple[float, float, float]:
        h, p, r = tf.local_rotation.as_euler()
        return (h * _RAD2DEG, p * _RAD2DEG, r * _RAD2DEG)

    def set_rot_deg(v: tuple[float, float, float]) -> None:
        tf.local_rotation = Quat.from_euler(
            float(v[0]) * _DEG2RAD,
            float(v[1]) * _DEG2RAD,
            float(v[2]) * _DEG2RAD,
        )

    def get_scale() -> tuple[float, float, float]:
        s = tf.local_scale
        return (s.x, s.y, s.z)

    def set_scale(v: tuple[float, float, float]) -> None:
        tf.local_scale = Vec3(float(v[0]), float(v[1]), float(v[2]))

    return Section(
        "Transform",
        [
            Field("position", FieldKind.VEC3, get_pos, set_pos, step=0.5, units="m"),
            Field("rotation", FieldKind.VEC3, get_rot_deg, set_rot_deg, step=5.0, units="deg"),
            Field("scale", FieldKind.VEC3, get_scale, set_scale, step=0.1),
        ],
    )


# ---------------------------------------------------------------------------
# Components (generic reflection)
# ---------------------------------------------------------------------------

def _component_section(comp: "Component") -> Section:
    """
    One section per component, with a row per public tunable attribute.

    ``enabled`` is always shown first (toggle); the remaining rows come from the
    component's ``__slots__`` (public names only), typed by their current value.
    """
    fields: list[Field] = [
        Field(
            "enabled",
            FieldKind.BOOL,
            lambda c=comp: c.enabled,
            lambda v, c=comp: setattr(c, "enabled", bool(v)),
        )
    ]
    for name in _public_slots(comp):
        fld = _field_for_attr(comp, name)
        if fld is not None:
            fields.append(fld)
    return Section(type(comp).__name__, fields)


def _public_slots(obj: Any) -> list[str]:
    """
    Ordered, de-duplicated public ``__slots__`` names across the type's MRO.

    Excludes private names (leading underscore) and the framework-internal
    bookkeeping attributes in ``_COMPONENT_INTERNAL``.
    """
    seen: set[str] = set()
    out: list[str] = []
    for klass in type(obj).__mro__:
        for name in getattr(klass, "__slots__", ()):
            if name in seen or name.startswith("_") or name in _COMPONENT_INTERNAL:
                continue
            seen.add(name)
            out.append(name)
    return out


def _field_for_attr(obj: Any, name: str) -> "Field | None":
    """
    Build an editable Field for ``obj.<name>`` by inspecting its runtime value.

    Returns ``None`` if the attribute is unset/unreadable.  Unknown value types
    fall back to a read-only ``repr`` label so nothing is silently hidden.
    """
    try:
        value = getattr(obj, name)
    except AttributeError:
        return None

    # bool must be checked before int (bool is a subclass of int).
    if isinstance(value, bool):
        return Field(
            name, FieldKind.BOOL,
            lambda o=obj, n=name: getattr(o, n),
            lambda v, o=obj, n=name: setattr(o, n, bool(v)),
        )
    if isinstance(value, int):
        return Field(
            name, FieldKind.INT,
            lambda o=obj, n=name: getattr(o, n),
            lambda v, o=obj, n=name: setattr(o, n, int(v)),
        )
    if isinstance(value, float):
        return Field(
            name, FieldKind.FLOAT,
            lambda o=obj, n=name: getattr(o, n),
            lambda v, o=obj, n=name: setattr(o, n, float(v)),
        )
    if isinstance(value, str):
        return Field(
            name, FieldKind.STRING,
            lambda o=obj, n=name: getattr(o, n),
            lambda v, o=obj, n=name: setattr(o, n, str(v)),
        )
    if isinstance(value, Vec3):
        return Field(
            name, FieldKind.VEC3,
            lambda o=obj, n=name: tuple(getattr(o, n)),
            lambda v, o=obj, n=name: setattr(o, n, Vec3(float(v[0]), float(v[1]), float(v[2]))),
        )

    # Unknown type: show it, don't hide it (read-only repr).
    return Field(name, FieldKind.LABEL, lambda o=obj, n=name: repr(getattr(o, n)))
