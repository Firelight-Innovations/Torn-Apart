"""Component catalog — the authoring component types the Fire Editor's inspector
edits and the game's ``SceneVisualFactory`` materialises.

Pure data (no panda3d, no RNG): this is the SINGLE source of truth for the
built-in component set, read by both the headless editor daemon (via the
``fire_editor.scene_objects`` shim) and the game. The inspector fetches it over
the ``scene.catalog`` RPC instead of duplicating field lists in TypeScript.

A component is a plain JSON-friendly dict::

    {"type": str, "enabled": bool, "params": {<field name>: <value>}}

The **Transform is not a catalog component** — it is the
:class:`~fire_engine.scene.objects.SceneObject`'s intrinsic TRS, rendered as a
synthetic, non-removable section in the inspector. ``kind`` is a creation
archetype: :func:`default_components_for_kind` seeds a new object's components,
after which the ``components`` list is the source of truth (an ``empty`` can be
given a Light; a ``cube``'s Mesh can be removed) — DECISIONS.md 2026-06-13.

Built-in types:

    Mesh        primitive: cube|sphere       (one renderer; singleton)
    Light       color, intensity, radius     (a real PointLight in-game; singleton)
    SpawnPoint  (no params)                  (marker; player start stays kind-based)

Example::

    from fire_engine.scene.components import make_component, default_components_for_kind
    default_components_for_kind("light")        # -> [{"type": "Light", ...}]
    c = make_component("Mesh")                   # -> a Mesh with default params
    c["params"]["primitive"] = "sphere"
"""

from __future__ import annotations

import copy
from typing import Any

from fire_engine.scene.types import ComponentSpec, FieldSpec

# Re-export so every existing import path keeps working.
__all__ = [
    "COMPONENT_CATALOG",
    "ComponentSpec",
    "FieldSpec",
    "catalog_payload",
    "coerce_params",
    "default_components_for_kind",
    "default_params",
    "is_known",
    "make_component",
]

# Warm torch defaults for a Light component — the single source the game's
# scene_visuals.SceneVisualFactory reads (it imports default_params("Light")),
# so editor, save and runtime never drift.
_LIGHT_COLOR: tuple[float, float, float] = (1.0, 0.62, 0.28)
_LIGHT_INTENSITY: float = 8.0
_LIGHT_RADIUS_M: float = 16.0

COMPONENT_CATALOG: dict[str, ComponentSpec] = {
    "Mesh": ComponentSpec(
        "Mesh",
        "Mesh",
        multiple=False,
        fields=(
            FieldSpec("primitive", "enum", "cube", choices=("cube", "sphere"), label="Primitive"),
        ),
    ),
    "Light": ComponentSpec(
        "Light",
        "Light",
        multiple=False,
        fields=(
            FieldSpec("color", "color", list(_LIGHT_COLOR), label="Color"),
            FieldSpec("intensity", "float", _LIGHT_INTENSITY, min=0.0, max=64.0, label="Intensity"),
            FieldSpec("radius", "float", _LIGHT_RADIUS_M, min=0.0, max=128.0, label="Radius (m)"),
        ),
    ),
    "SpawnPoint": ComponentSpec("SpawnPoint", "Spawn Point", multiple=False),
}


def is_known(type_name: str) -> bool:
    """True if ``type_name`` is a registered component type."""
    return type_name in COMPONENT_CATALOG


def default_params(type_name: str) -> dict[str, Any]:
    """Fresh default ``params`` dict for ``type_name`` (deep-copied defaults).

    Raises:
        KeyError: if ``type_name`` is not a registered component type.
    """
    spec = COMPONENT_CATALOG[type_name]
    return {f.name: copy.deepcopy(f.default) for f in spec.fields}


def make_component(type_name: str, *, enabled: bool = True) -> dict[str, Any]:
    """Build a component dict of ``type_name`` with its default params.

    Raises:
        KeyError: if ``type_name`` is not a registered component type.
    """
    return {"type": type_name, "enabled": bool(enabled), "params": default_params(type_name)}


def default_components_for_kind(kind: str) -> list[dict[str, Any]]:
    """The components a freshly created (or migrated) object of ``kind`` carries.

    ``kind`` is only the creation archetype; the returned list becomes editable
    and the source of truth thereafter.
    """
    k = str(kind).lower()
    if k == "cube":
        return [_mesh("cube")]
    if k == "sphere":
        return [_mesh("sphere")]
    if k == "light":
        return [make_component("Light")]
    if k == "spawn":
        return [make_component("SpawnPoint")]
    return []  # "empty" (and any unknown kind): a bare transform


def coerce_params(type_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Validate+coerce a partial ``params`` dict against the catalog.

    Returns only recognised keys, each coerced to its field's type (floats
    clamped to min/max, enums snapped to a valid choice, colors/vec3 to a
    3-list of floats). Unknown keys are dropped. Unknown ``type_name`` -> ``{}``.
    """
    spec = COMPONENT_CATALOG.get(type_name)
    if spec is None:
        return {}
    by_name = {f.name: f for f in spec.fields}
    out: dict[str, Any] = {}
    for key, value in params.items():
        fspec = by_name.get(key)
        if fspec is None:
            continue
        out[key] = _coerce_value(fspec, value)
    return out


def catalog_payload() -> dict[str, Any]:
    """JSON-friendly catalog for the ``scene.catalog`` RPC / the inspector."""
    return {
        "types": [
            {
                "type": s.type,
                "label": s.label,
                "multiple": s.multiple,
                "fields": [_field_payload(f) for f in s.fields],
            }
            for s in COMPONENT_CATALOG.values()
        ]
    }


# ---------------------------------------------------------------------------- #
# Internals
# ---------------------------------------------------------------------------- #
def _mesh(primitive: str) -> dict[str, Any]:
    c = make_component("Mesh")
    c["params"]["primitive"] = primitive
    return c


def _coerce_value(fspec: FieldSpec, value: Any) -> Any:
    if fspec.ui_type == "float":
        x = float(value)
        if fspec.min is not None:
            x = max(fspec.min, x)
        if fspec.max is not None:
            x = min(fspec.max, x)
        return x
    if fspec.ui_type in ("color", "vec3"):
        vals = [float(c) for c in value][:3]
        while len(vals) < 3:
            vals.append(0.0)
        return vals
    if fspec.ui_type == "bool":
        return bool(value)
    if fspec.ui_type == "enum":
        s = str(value)
        return s if s in fspec.choices else fspec.default
    return value


def _field_payload(f: FieldSpec) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": f.name,
        "ui_type": f.ui_type,
        "default": copy.deepcopy(f.default),
        "label": f.label or f.name,
    }
    if f.min is not None:
        out["min"] = f.min
    if f.max is not None:
        out["max"] = f.max
    if f.choices:
        out["choices"] = list(f.choices)
    return out
