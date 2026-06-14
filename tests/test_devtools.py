"""
tests/test_devtools.py — headless tests for the developer-overlay engine.

Covers the panda3d-free half of the dev tools (everything in
``fire_engine/devtools/``): the selection counter, CPU ray/AABB picking, the
GameObject → editable-sections introspection (including the edit round-trip),
and the tool/manager plumbing.  The DirectGUI renderer
(``world/devtools_overlay.py``) is panda3d-backed and out of scope here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core.math3d import Quat, Vec3
from fire_engine.devtools import (
    ActionsTool,
    Button,
    CallbackTool,
    ClockTool,
    DevToolsManager,
    Field,
    FieldKind,
    Gizmo,
    GizmoMode,
    HandleType,
    InspectorTool,
    PerformanceTool,
    Section,
    Selectable,
    Selection,
    describe_chunk,
    describe_object,
    is_chunk,
    pick,
    ray_aabb,
    update_drag,
)
from fire_engine.devtools.gizmo import closest_on_axis, ray_plane_intersect
from fire_engine.render.component import Component
from fire_engine.render.gameobject import GameObject
from fire_engine.render.registry import ComponentRegistry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the component registry between tests (objects/buckets leak otherwise)."""
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


# A component with a tunable of every reflected kind.
class TunableComponent(Component):
    __slots__ = ("flag", "label", "offset", "speed")

    def __init__(self) -> None:
        super().__init__()
        self.speed: float = 10.0
        self.label: str = "hello"
        self.flag: bool = False
        self.offset: Vec3 = Vec3(1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_selection_revision_only_moves_on_change():
    sel = Selection()
    assert sel.current is None and sel.revision == 0

    a = GameObject(name="A")
    sel.set(a)
    assert sel.current is a and sel.revision == 1

    sel.set(a)  # same object → no-op
    assert sel.revision == 1

    sel.clear()
    assert sel.current is None and sel.revision == 2


def test_selection_listener_fires():
    sel = Selection()
    seen = []
    sel.on_change(lambda go: seen.append(go))
    go = GameObject(name="X")
    sel.set(go)
    sel.clear()
    assert seen == [go, None]


# ---------------------------------------------------------------------------
# Ray / AABB picking
# ---------------------------------------------------------------------------


def test_ray_aabb_hit_miss_and_inside():
    bmin = np.array([-1.0, -1.0, -1.0])
    bmax = np.array([1.0, 1.0, 1.0])

    # Straight down +Y at the box centred ahead → hit at t=4 (origin y=-5, face y=-1).
    t = ray_aabb(Vec3(0, -5, 0), Vec3(0, 1, 0), bmin, bmax)
    assert t is not None and abs(t - 4.0) < 1e-6

    # Pointing away → miss.
    assert ray_aabb(Vec3(0, -5, 0), Vec3(0, -1, 0), bmin, bmax) is None

    # Parallel and offset (misses the slab) → miss.
    assert ray_aabb(Vec3(5, -5, 0), Vec3(0, 1, 0), bmin, bmax) is None

    # Origin inside the box → t == 0.
    assert ray_aabb(Vec3(0, 0, 0), Vec3(0, 1, 0), bmin, bmax) == 0.0


def test_pick_returns_nearest():
    near = GameObject(name="near")
    near.transform.local_position = Vec3(0, 5, 0)
    far = GameObject(name="far")
    far.transform.local_position = Vec3(0, 20, 0)

    sels = [
        Selectable(far, Vec3(1, 1, 1)),
        Selectable(near, Vec3(1, 1, 1)),
    ]
    hit = pick(Vec3(0, 0, 0), Vec3(0, 1, 0), sels)
    assert hit is near

    # A ray that misses both.
    assert pick(Vec3(50, 0, 0), Vec3(0, 1, 0), sels) is None


def test_selectable_world_aabb_follows_transform_and_scale():
    go = GameObject(name="box")
    go.transform.local_position = Vec3(10, 0, 0)
    go.transform.local_scale = Vec3(2, 2, 2)
    sel = Selectable(go, Vec3(0.5, 0.5, 0.5))
    bmin, bmax = sel.world_aabb()
    # half-extent 0.5 * scale 2 = 1.0 around centre (10,0,0)
    assert np.allclose(bmin, [9.0, -1.0, -1.0])
    assert np.allclose(bmax, [11.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Introspection + edit round-trip
# ---------------------------------------------------------------------------


def test_describe_object_sections_layout():
    go = GameObject(name="Hero", tag="player")
    go.add_component(TunableComponent)
    sections = describe_object(go)

    titles = [s.title for s in sections]
    assert titles[0] == "GameObject"
    assert titles[1] == "Transform"
    assert "TunableComponent" in titles


def test_inspector_edit_roundtrip_transform_and_component():
    go = GameObject(name="Hero")
    comp = go.add_component(TunableComponent)
    sections = describe_object(go)

    fields = {s.title: {f.label: f for f in s.fields} for s in sections}

    # Transform position (VEC3) edit applies to local_position.
    pos_field = fields["Transform"]["position"]
    assert pos_field.kind == FieldKind.VEC3
    pos_field.set((4.0, 5.0, 6.0))
    assert go.transform.local_position.approx_eq(Vec3(4.0, 5.0, 6.0))

    # Component float edit.
    speed_field = fields["TunableComponent"]["speed"]
    assert speed_field.kind == FieldKind.FLOAT
    assert speed_field.get() == pytest.approx(10.0)
    speed_field.set(42.5)
    assert comp.speed == pytest.approx(42.5)

    # Component bool edit (enabled is surfaced explicitly).
    enabled_field = fields["TunableComponent"]["enabled"]
    assert enabled_field.kind == FieldKind.BOOL
    enabled_field.set(False)
    assert comp.enabled is False

    # Component string + Vec3 edits.
    fields["TunableComponent"]["label"].set("world")
    assert comp.label == "world"
    fields["TunableComponent"]["offset"].set((7.0, 8.0, 9.0))
    assert comp.offset.approx_eq(Vec3(7.0, 8.0, 9.0))


def test_inspector_rotation_is_euler_degrees_view():
    go = GameObject(name="Hero")
    sections = describe_object(go)
    rot = next(
        f for s in sections if s.title == "Transform" for f in s.fields if f.label == "rotation"
    )
    # Set 90° heading (about +Z); stored as a quaternion, read back ~90.
    rot.set((90.0, 0.0, 0.0))
    h, p, r = rot.get()
    assert h == pytest.approx(90.0, abs=1e-3)
    # forward (+Y) rotated 90° about +Z → −X, confirming it stored as a real rotation.
    assert go.transform.forward.approx_eq(Vec3(-1, 0, 0), eps=1e-5)


def test_identity_section_active_toggle():
    go = GameObject(name="Hero")
    active = next(
        f
        for s in describe_object(go)
        if s.title == "GameObject"
        for f in s.fields
        if f.label == "active"
    )
    assert active.get() is True
    active.set(False)
    assert go.active_self is False


# ---------------------------------------------------------------------------
# Tools + manager
# ---------------------------------------------------------------------------


def test_performance_tool_reads_providers_live():
    counter = {"n": 0}

    def provider():
        counter["n"] += 1
        return counter["n"]

    tool = PerformanceTool({"ticks": provider})
    p1 = tool.build()
    field = p1.sections[0].fields[0]
    assert field.label == "ticks" and field.read_only
    assert field.get() == 1
    assert field.get() == 2  # live each call


def test_inspector_tool_tracks_selection_revision():
    mgr = DevToolsManager()
    tool = InspectorTool(mgr.selection)
    assert "nothing selected" in tool.build().sections[0].fields[0].label

    go = GameObject(name="Sel")
    mgr.selection.set(go)
    assert tool.revision == mgr.selection.revision
    panel = tool.build()
    assert "Sel" in panel.title


def test_is_chunk_distinguishes_chunk_from_gameobject():
    from fire_engine.world.terrain.chunk import Chunk

    assert is_chunk(Chunk((0, 0, 0))) is True
    assert is_chunk(GameObject(name="Obj")) is False


def test_describe_chunk_reports_voxel_stats():
    from fire_engine.world.terrain.chunk import Chunk

    chunk = Chunk((1, 0, -1))  # all-air baseline
    chunk.materials[0, 0, 0] = 1  # one solid voxel
    chunk.materials[1, 0, 0] = 2  # a second material id
    sections = describe_chunk(chunk)
    titles = [s.title for s in sections]
    assert titles == ["Chunk", "Voxels"]

    rows = {f.label: f for s in sections for f in s.fields}
    # Every chunk row is read-only (voxels are edited with the brush).
    assert all(f.set is None for f in rows.values())
    assert rows["coord"].get() == "(1, 0, -1)"
    assert rows["solid"].get() == 2
    assert rows["material ids"].get() == "0, 1, 2"
    assert rows["edited"].get() is False


def test_inspector_tool_routes_chunk_to_chunk_describer():
    from fire_engine.world.terrain.chunk import Chunk

    mgr = DevToolsManager()
    tool = InspectorTool(mgr.selection)
    chunk = Chunk((2, 3, 4))
    mgr.selection.set(chunk)
    panel = tool.build()
    assert "Chunk (2, 3, 4)" in panel.title
    assert [s.title for s in panel.sections] == ["Chunk", "Voxels"]


def test_actions_tool_add_bumps_revision():
    fired = []
    tool = ActionsTool("World", {"A": lambda: fired.append("A")})
    r0 = tool.revision
    tool.add_action("B", lambda: fired.append("B"))
    assert tool.revision == r0 + 1
    panel = tool.build()
    labels = [b.label for b in panel.buttons]
    assert labels == ["A", "B"]
    panel.buttons[1].on_click()
    assert fired == ["B"]


def test_clock_tool_formats_time_of_day():
    class FakeClock:
        game_day = 3
        game_time_of_day = 13 * 3600 + 5 * 60  # 13:05

    panel = ClockTool(FakeClock()).build()
    rows = {f.label: f.get() for f in panel.sections[0].fields}
    assert rows["day"] == 3
    assert rows["time of day"] == "13:05"


def test_callback_tool_delegates_build_and_revision():
    rev = {"n": 0}
    sec = Section("S", [Field("a", FieldKind.LABEL, lambda: 1)])
    btn = Button("go", lambda: None)
    tool = CallbackTool("env", "Env", lambda: ([sec], [btn]), revision_fn=lambda: rev["n"])
    panel = tool.build()
    assert panel.tool_id == "env" and panel.title == "Env"
    assert panel.sections == [sec] and panel.buttons == [btn]
    assert tool.revision == 0
    rev["n"] = 7
    assert tool.revision == 7


def test_manager_pick_and_select_and_remove():
    mgr = DevToolsManager()
    go = GameObject(name="Cube")
    go.transform.local_position = Vec3(0, 5, 0)
    mgr.add_selectable(go, Vec3(1, 1, 1))

    hit = mgr.pick_and_select(Vec3(0, 0, 0), Vec3(0, 1, 0))
    assert hit is go and mgr.selection.current is go

    mgr.remove_selectable(go)
    assert mgr.selectables == []
    assert mgr.selection.current is None  # cleared because the selected obj left


# ---------------------------------------------------------------------------
# Transform gizmo math
# ---------------------------------------------------------------------------


def test_gizmo_ray_geometry_primitives():
    # Ray straight down onto z=0 hits the expected point.
    o = np.array([1.0, 2.0, 5.0])
    d = np.array([0.0, 0.0, -1.0])
    hit = ray_plane_intersect(o, d, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert np.allclose(hit, [1.0, 2.0, 0.0])
    # A ray crossing the X axis at x=3 has axis param 3 and ~zero distance.
    axis_t, _, dist = closest_on_axis(
        np.array([3.0, 0.0, 5.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    assert abs(axis_t - 3.0) < 1e-9 and dist < 1e-9


def test_gizmo_pick_axis_vs_plane():
    giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
    # Ray onto the X stalk (y=z=0 at x=0.5) → X axis handle.
    h = giz.pick(Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
    assert h.type == HandleType.AXIS and h.axis == 0
    # Ray into the +X+Y quadrant of the z=0 plane → XY plane handle (normal Z=2).
    h = giz.pick(Vec3(0.3, 0.3, 5.0), Vec3(0.0, 0.0, -1.0))
    assert h.type == HandleType.PLANE and h.axis == 2


def test_gizmo_translate_axis_drag_moves_along_axis():
    giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
    handle = giz.pick(Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
    drag = giz.begin(
        handle,
        Vec3(2.0, 0.0, 5.0),
        Vec3(0.0, 0.0, -1.0),
        Vec3(0, 0, 0),
        Quat.identity(),
        Vec3(1, 1, 1),
    )
    pos, rot, scl = update_drag(drag, Vec3(5.0, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
    assert pos.approx_eq(Vec3(3.0, 0.0, 0.0), eps=1e-5)  # dragged +3 along X
    assert rot.approx_eq(Quat.identity()) and scl.approx_eq(Vec3(1, 1, 1))


def test_gizmo_rotate_ring_drag_spins_about_axis():
    giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.ROTATE)
    handle = giz.pick(Vec3(1.0, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))  # Z ring at (1,0,0)
    assert handle.type == HandleType.RING and handle.axis == 2
    drag = giz.begin(
        handle,
        Vec3(1.0, 0.0, 5.0),
        Vec3(0.0, 0.0, -1.0),
        Vec3(0, 0, 0),
        Quat.identity(),
        Vec3(1, 1, 1),
    )
    # Grab moved from (1,0,0) angle 0 to (0,1,0) angle +90° about Z.
    _, rot, _ = update_drag(drag, Vec3(0.0, 1.0, 5.0), Vec3(0.0, 0.0, -1.0))
    assert rot.approx_eq(Quat.from_axis_angle(Vec3.UP, math.pi / 2), eps=1e-4)


def test_gizmo_scale_axis_drag_scales_one_axis():
    giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.SCALE)
    handle = giz.pick(Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
    assert handle.type == HandleType.AXIS and handle.axis == 0
    drag = giz.begin(
        handle,
        Vec3(0.5, 0.0, 5.0),
        Vec3(0.0, 0.0, -1.0),
        Vec3(0, 0, 0),
        Quat.identity(),
        Vec3(1, 1, 1),
    )
    # Drag the X stalk out by one `size` (0.5 → 1.5): factor 1 + 1.0 = 2× on X only.
    _, _, scl = update_drag(drag, Vec3(1.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
    assert scl.approx_eq(Vec3(2.0, 1.0, 1.0), eps=1e-5)
