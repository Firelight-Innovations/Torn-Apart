# world — System Doc
keywords: gameobject, game object, transform, component, registry, lifecycle, awake, start, update, late_update, fixed_update, on_enable, on_disable, on_destroy, instantiate, destroy, find_with_tag, find_objects_with_tag, space, camera, app, showbase, panda3d, input, inputstate, unity, clone, hierarchy, parent, children, world matrix, dirty flag, position, rotation, forward, right, up, look_at, transform_point, inverse_transform_point, active_in_hierarchy, set_active, add_component, get_component, get_components, get_component_in_children, remove_component, compare_tag, layer, tag, uuid, batched, bucket, run_frame, fixed_steps, spiral of death, quaternion, z-up, meters, radians

> One doc per code package; filename matches the package exactly (`docs/systems/world.md` ↔ `torn_apart/world/`).

## Role

`world/` is the **World API** — Layer 3 in the engine stack.  It provides:

- **Unity-clone object model** (`GameObject`, `Component`, `Transform`): same API shape as Unity (snake_case), Z-up coordinates, quaternion-only rotations.  All authoring uses these types.
- **ComponentRegistry**: the batched frame executor that drives Unity lifecycle order across all components.
- **App** (`world/app.py`): the Panda3D ShowBase wrapper, window, frame loop, input collection, and integration hooks for Phase 3 (chunk streaming) and Phase 4 (lighting).
- **CameraComponent** (`world/camera.py`): one-way sync from a `Transform` to the Panda3D camera NodePath.

`world/` deliberately does NOT: implement game logic, generate terrain, simulate NPCs, or own any voxel data.  It is the scene-graph boundary: everything above uses the object model; everything below is contacted through direct API calls.

Only `world/app.py` and `world/camera.py` may import panda3d.  `transform.py`, `component.py`, `gameobject.py`, `registry.py`, and `player/fly_controller.py` are pure Python/numpy and fully headless-testable.

## Public API

All symbols below are re-exported from `torn_apart.world` (`__init__.py`).

### Transform (`world/transform.py`)

| Symbol | Description |
|---|---|
| `Transform()` | Per-entity TRS transform, always-present on every GameObject. |
| `Space.SELF` | Enum: operations in the local/object frame. |
| `Space.WORLD` | Enum: operations in world space. |
| `t.local_position: Vec3` | Position relative to parent (meters). Setter marks dirty. |
| `t.local_rotation: Quat` | Rotation relative to parent (unit quaternion). Setter normalises + marks dirty. |
| `t.local_scale: Vec3` | Scale relative to parent. |
| `t.parent: Transform | None` | Parent transform (read). Write via `set_parent`. |
| `t.children: tuple[Transform, ...]` | Read-only children tuple. |
| `t.set_parent(p, keep_world=True)` | Reparent; keep_world=True preserves world position/rotation. |
| `t.position: Vec3` | World-space position (derived, cached). Setter recomputes local_position. |
| `t.rotation: Quat` | World-space rotation (derived through parent chain). Setter recomputes local_rotation. |
| `t.forward: Vec3` | World-space +Y axis (facing direction). |
| `t.right: Vec3` | World-space +X axis. |
| `t.up: Vec3` | World-space +Z axis. |
| `t.translate(v, relative_to=Space.SELF)` | Move by displacement v (meters). |
| `t.rotate(q, relative_to=Space.SELF)` | Apply additional rotation. |
| `t.look_at(target, up=Vec3.UP)` | Orient forward toward target. Degrades gracefully if target == self. |
| `t.transform_point(p)` | Local → world (meters). |
| `t.inverse_transform_point(p)` | World → local (meters). Round-trips with transform_point. |

**Dirty-flag caching**: the 4×4 world matrix is recomputed lazily.  Any write to local TRS or `set_parent` marks the transform and ALL descendants dirty instantly (O(subtree)); the next read of `position`/`rotation` recomputes at most one matrix-multiply per ancestor.

### Component (`world/component.py`)

| Symbol | Description |
|---|---|
| `Component` | Base class for all components. Override any lifecycle method. |
| `c.game_object: GameObject` | Owning GameObject. Set before `awake()`. |
| `c.transform: Transform` | Convenience alias for `c.game_object.transform`. |
| `c.enabled: bool` | When False, `update/late_update/fixed_update` are skipped. `awake/start` still run. |

**Lifecycle order** (guaranteed by `ComponentRegistry.run_frame`):

1. `awake()` — immediately after construction; runs even when disabled.
2. `on_enable()` — after awake, when component is enabled.
3. `start()` — once before the first `update()`; after ALL awakes that frame.
4. `update(dt)` — every enabled frame.
5. `late_update(dt)` — after ALL updates (camera follow, IK post-processing).
6. `fixed_update(dt)` — fixed timestep (0–5× per real frame, default 50 Hz).
7. `on_disable()` — on disable or hierarchy deactivation.
8. `on_destroy()` — end-of-frame teardown.

### GameObject (`world/gameobject.py`)

| Symbol | Description |
|---|---|
| `GameObject(name, tag, layer)` | Create a new entity with an always-present Transform. |
| `go.id: UUID` | Unique identifier (uuid4). |
| `go.name: str` | Human-readable name (not unique). |
| `go.tag: str` | Single primary tag for find_with_tag lookups. |
| `go.layer: int` | Render/physics layer mask index. |
| `go.active_self: bool` | Local active flag. Write via `set_active()`. |
| `go.active_in_hierarchy: bool` | True only when this and all ancestors are active. |
| `go.transform: Transform` | Always present. Cannot be removed. |
| `go.add_component(t, **kw) -> T` | Construct, attach, schedule awake/start. |
| `go.get_component(t) -> T | None` | First component of exact type t. |
| `go.get_components(t) -> list[T]` | All components of type t. |
| `go.get_component_in_children(t) -> T | None` | BFS over transform-child hierarchy. |
| `go.remove_component(c)` | Detach; schedules on_disable + on_destroy end-of-frame. |
| `go.set_active(value)` | Sets active_self; cascades on_enable/on_disable down hierarchy. |
| `go.compare_tag(tag) -> bool` | Exact tag match. |

### ComponentRegistry (`world/registry.py`)

| Symbol | Description |
|---|---|
| `ComponentRegistry` | Module-level singleton. |
| `ComponentRegistry.run_frame(clock)` | Drive one full frame (awake/start/update/late/fixed/destroy). |
| `ComponentRegistry.clear()` | Reset all state (test isolation). |
| `instantiate(template, position, rotation, parent) -> GameObject` | Create + register. |
| `destroy(obj_or_component, delay=0.0)` | Deferred teardown (end of current frame). |
| `find_with_tag(tag) -> GameObject | None` | First registered object with tag. |
| `find_objects_with_tag(tag) -> list[GameObject]` | All registered objects with tag. |

**Execution order** within `run_frame(clock)`:
1. ALL pending `awake()` + `on_enable()` (snapshot before iterating — additions go to next frame)
2. ALL pending `start()` (snapshot before iterating)
3. `update(dt)` per type bucket
4. `late_update(dt)` per type bucket
5. `fixed_update(fixed_dt)` × N (driven by `clock.fixed_steps()`)
6. Deferred destroy flush

### App (`world/app.py`)

| Symbol | Description |
|---|---|
| `App(config, clock, event_bus)` | Panda3D ShowBase wrapper. 1280×720, vsync, FPS meter. |
| `app.input_state: InputState` | Read each frame by FlyController. |
| `app.camera_go: GameObject` | The camera entity. |
| `app.camera_comp: CameraComponent` | Camera sync component. |
| `InputState` | Dataclass: move_forward/backward/left/right/up/down, sprint, mouse_dx/dy, mouse_captured, escape_pressed. |

**Integration hooks** in `_frame_task` (search for `# integration hook`):
- `# --- integration hook: chunk streaming (Phase 3) ---` — after `run_frame`, before `event_bus.drain()`.
- `# --- integration hook: lighting dirty work (Phase 4) ---` — after chunk streaming.

### CameraComponent (`world/camera.py`)

| Symbol | Description |
|---|---|
| `CameraComponent(base)` | Syncs owning Transform → `base.camera` NodePath each frame. |
| `cam.sync_to_panda()` | Called by App after all components updated. Math3d → Panda3D conversion happens here only. |

## Imports Allowed

`world/transform.py`, `world/component.py`, `world/gameobject.py`, `world/registry.py`:
- `torn_apart.core` (Vec3, Quat, etc.)
- Python standard library, `numpy`
- **No panda3d imports**

`world/app.py`, `world/camera.py`:
- All of the above PLUS `panda3d.*`, `direct.*`

Per ARCHITECTURE.md §4a.2, `world/` may also import: `resources/`, `terrain/`, `lighting/`, `procedural/`, `core/`.

## Events

### Published
None directly from world/.  App calls `event_bus.drain()` each frame; the bus carries events from terrain/lighting/etc.

### Subscribed
None directly.  Terrain/lighting integrate via the App integration hooks, not event subscriptions.

## Units & Invariants

### Coordinate System
- **Z-up** (Panda3D native): `forward = +Y`, `right = +X`, `up = +Z`.
- **Not Unity's Y-up** — the API shape is Unity's; the axes are ours.
- All distances in **meters**.
- Rotations in **radians** throughout.

### Quaternion Composition (mouse-look)
The FlyController (player/fly_controller.py) composes:
```python
q_yaw   = Quat.from_axis_angle(Vec3.UP,   yaw)    # world Z rotation
q_pitch = Quat.from_axis_angle(Vec3.RIGHT, pitch)  # local X rotation
transform.local_rotation = (q_yaw * q_pitch).normalized()
```
Yaw and pitch are **accumulated floats** (radians), NOT integrated incrementally into the quaternion.  This avoids the roll-drift trap documented in DEVELOPMENT_PLAN.md Known Traps.  Pitch is clamped to ±89°.

### Transform Cache Invariant
`transform.position` and `transform.rotation` are always consistent with the parent chain.  After any write to local TRS or `set_parent`, all descendants are immediately marked dirty.  A stale world matrix is never returned; the cache is recomputed on the next read.

### Lifecycle Ordering Invariant
ALL `awake()` calls within a frame complete before ANY `start()` call begins — even when multiple GameObjects are added in the same frame.  Components added during `run_frame` iteration are deferred to the next frame's awake queue.

### No Euler State
Rotations are stored **only as quaternions**.  `Quat.from_euler` and `as_euler` exist solely as a presentation layer; they are never used as stored state.

## Examples

### Creating an entity with a component
```python
from torn_apart.world import instantiate, ComponentRegistry
from torn_apart.world.component import Component
from torn_apart.core.math3d import Vec3, Quat
from torn_apart.core.clock import Clock
from torn_apart.core.event_bus import EventBus

class Rotator(Component):
    def update(self, dt: float) -> None:
        from math import pi
        self.transform.rotate(Quat.from_axis_angle(Vec3.UP, pi * dt))

clock = Clock(fixed_dt=0.02, bus=EventBus())
go = instantiate(position=Vec3(0, 0, 5))
go.add_component(Rotator)

clock.update(0.016)
ComponentRegistry.run_frame(clock)  # awake + start + update
```

### Transform parent/child hierarchy
```python
from torn_apart.world.transform import Transform, Space
from torn_apart.core.math3d import Vec3, Quat
from math import pi

parent = Transform()
child  = Transform()
child.set_parent(parent, keep_world=False)

parent.local_position = Vec3(10, 0, 0)
parent.local_rotation = Quat.from_axis_angle(Vec3.UP, pi / 2)
child.local_position  = Vec3(0, 2, 0)

# Child world position: parent rotated its +Y by 90°, so child is at (-2, 0, 0) + (10, 0, 0)
print(child.position)   # Vec3(8, 0, 0)  (approximately)
```

### look_at
```python
cam = Transform()
cam.local_position = Vec3(0, -20, 10)
cam.look_at(Vec3.ZERO)
# cam.forward now points from (0,-20,10) toward the origin
```

### App setup (game entry point)
```python
from torn_apart.core import load_config, Clock, EventBus
from torn_apart.world.app import App

cfg   = load_config()
bus   = EventBus()
clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
app   = App(cfg, clock, bus)

# Add player with fly controller
from torn_apart.world import instantiate
from torn_apart.player import FlyController
from torn_apart.core.math3d import Vec3
player_go = instantiate(position=Vec3(0, 0, 20))
player_go.add_component(FlyController, move_speed=10.0)

app.run()
```

## Gotchas

1. **Components added during run_frame go to NEXT frame.** If you call `go.add_component(...)` inside an `update()`, the new component's `awake()` fires in the next `run_frame` call, not the current one.  This is intentional (Unity semantics) and prevents iterator-mutation bugs.

2. **Transform dirty propagation is O(subtree size).** Setting a parent's position marks every descendant dirty.  For deep hierarchies with many descendants, prefer batch-updating leaf positions directly rather than moving the root repeatedly.

3. **`destroy()` is deferred to end-of-frame.** A destroyed component or object continues to exist in memory and will NOT be updated after the destroy call, but `on_destroy()` doesn't fire until the flush at the end of `run_frame`.

4. **Euler angles are not state.** Never try to read back Euler angles from a Transform to modify them incrementally — use the quaternion API directly.  The FlyController correctly accumulates yaw/pitch as floats and recomputes the quaternion each frame.

5. **`active_in_hierarchy` walks the parent chain.** It's O(depth), called per-component per-frame for the `_is_active` guard.  Keep hierarchies shallow for performance.

6. **App integration hooks are comments, not callbacks.** The chunk streaming and lighting hooks in `app._frame_task` are comment placeholders guarded by `if hasattr(self, '_chunk_manager')` patterns.  Phase 3 and 4 agents assign `self._chunk_manager` and `self._light_grid` directly on the App instance.

7. **Mouse delta is in pixels (normalised by half-window-size).** The FlyController uses `mouse_sensitivity` in radians/pixel.  At 1280×720, a full-screen drag is ±640px/±360px — calibrate sensitivity accordingly.
