# player — System Doc
keywords: fly_controller, flycontroller, fly controller, camera, wasd, mouse look, mouse-look, yaw, pitch, sprint, speed, input state, inputstate, keyboard, mouse, panda3d free, component, update, look around, movement, horizontal, vertical, space, ctrl, shift, esc, escape, mouse capture, quaternion, roll drift, clamp, sensitivity, radians, meters per second

> One doc per code package; filename matches the package exactly (`docs/systems/player.md` ↔ `fire_engine/player/`).

## Role

`player/` is the **Player API** — Layer 4 in the engine stack.  It gives a human the same interface as an NPC agent: input → the same action calls NPCs will use (Session 4+).

**Session 1 scope:** free-fly camera controller only.  The player is not yet an embodied agent — there is no collision, gravity, or inventory.  The `FlyController` drives a `Transform` via keyboard + mouse without imposing any physics.

`player/` is intentionally panda3d-free.  Input arrives via `InputState` (set by `App` before `registry.run_frame()`); the controller operates purely on `Transform`/`Quat`/`Vec3`.

`player/` does NOT: render anything, query terrain/lighting, or issue Panda3D calls.

## Public API

All symbols below are re-exported from `fire_engine.player` (`__init__.py`).

### FlyController (`player/fly_controller.py`)

| Symbol | Description |
|---|---|
| `FlyController(move_speed, sprint_mult, mouse_sensitivity)` | Component. |
| `ctrl.move_speed: float` | Movement speed in **meters per second** (default 10.0 m/s). |
| `ctrl.sprint_mult: float` | Speed multiplier when Shift is held (default 5.0 → 50 m/s sprint). |
| `ctrl.mouse_sensitivity: float` | Mouse look sensitivity in **radians per pixel** (default 0.003). |
| `ctrl.yaw: float` | Accumulated heading angle in **radians** about world +Z. Wraps freely. |
| `ctrl.pitch: float` | Accumulated pitch angle in **radians** about local +X. Clamped ±89°. |
| `ctrl.set_input_state(state: InputState)` | Called by `App._push_input_to_controllers()` before each `run_frame`. |

**Controls:**

| Key | Action |
|---|---|
| `W` | Move forward along horizontal component of `transform.forward` |
| `S` | Move backward |
| `A` | Strafe left |
| `D` | Strafe right |
| `Space` or `E` | Move up (+Z, world) |
| `Ctrl` or `Q` | Move down (−Z, world) |
| `Shift` | Sprint: 5× move_speed |
| Mouse move | Yaw (left/right) + pitch (up/down) when mouse is captured |
| `ESC` | Toggle mouse capture (handled by App; controller reads `InputState.mouse_captured`) |

**Mouse-look quaternion composition** (see DEVELOPMENT_PLAN.md Known Traps):
```python
q_yaw   = Quat.from_axis_angle(Vec3.UP,   self.yaw)    # world Z
q_pitch = Quat.from_axis_angle(Vec3.RIGHT, self.pitch)  # local X
transform.local_rotation = (q_yaw * q_pitch).normalized()
```
Yaw and pitch are accumulated **as floats** each frame — NOT integrated into the quaternion incrementally.  This is critical: integrating raw mouse deltas into a quaternion (e.g. `q = q * small_delta`) accumulates floating-point error and produces roll drift.  The float accumulation + reconstruction approach is always roll-free.

**Horizontal movement**: only the XY projection of `transform.forward` and `transform.right` is used for WASD; the Z-component is zeroed.  This prevents WASD from flying the camera up/down when looking steeply.

## Imports Allowed

`player/` may import:
- `fire_engine.world` (Component, Transform, etc.)
- `fire_engine.core` (Vec3, Quat, etc.)
- Python standard library, `numpy`

**No panda3d imports** — controller is headless-testable.

Per ARCHITECTURE.md §4a.2, `player → world` is the only allowed dependency within the engine stack (player is Layer 4, world is Layer 3).

## Events

### Published
None. `FlyController` does not publish events; it drives the Transform directly.

### Subscribed
None. Input arrives via `InputState` injection, not the event bus.

## Units & Invariants

- `move_speed`: **meters per second**.  Default 10 m/s walking, 50 m/s sprint.
- `mouse_sensitivity`: **radians per pixel** (screen pixel, not normalised device coordinates).  Default 0.003 rad/px ≈ 0.17°/px.
- `yaw`: radians, unbounded (wraps freely — no normalisation needed since cos/sin are periodic).
- `pitch`: radians, hard-clamped to `[−π/2 + ε, +π/2 − ε]` where ε = 1° = π/180.  Prevents gimbal singularity at exactly ±90°.
- Movement delta per frame: `direction.normalized() * move_speed * dt` (in meters).  Only non-zero when at least one movement key is held.
- Rotation is **replaced** each frame (not accumulated): `local_rotation = q_yaw * q_pitch`.

## Examples

### Attach to a camera GameObject
```python
from fire_engine.world import instantiate, ComponentRegistry
from fire_engine.player import FlyController
from fire_engine.core.math3d import Vec3
from fire_engine.core.clock import Clock
from fire_engine.core.event_bus import EventBus

clock = Clock(fixed_dt=0.02, bus=EventBus())

camera_go = instantiate(name="MainCamera", position=Vec3(0, -20, 10))
ctrl = camera_go.add_component(FlyController, move_speed=15.0)

# Simulate one frame with W held and a small mouse delta:
from fire_engine.world.app import InputState
inp = InputState(
    move_forward=True,
    mouse_captured=True,
    mouse_dx=10.0,   # 10 pixels right
    mouse_dy=-5.0,   # 5 pixels up
)
ctrl.set_input_state(inp)
clock.update(0.016)
ComponentRegistry.run_frame(clock)
```

### Wiring in App (done automatically)
App calls `_push_input_to_controllers()` before `registry.run_frame()`:
```python
# In App._push_input_to_controllers:
for ctrl in bucket[FlyController]:
    ctrl.set_input_state(self.input_state)
```
The controller reads the state on its next `update(dt)`.

### Testing the controller headlessly
```python
from fire_engine.world.gameobject import GameObject
from fire_engine.player.fly_controller import FlyController
from fire_engine.world.registry import ComponentRegistry
from fire_engine.core.clock import Clock
from fire_engine.core.event_bus import EventBus
from fire_engine.world.app import InputState
from fire_engine.core.math3d import Vec3

ComponentRegistry.clear()
clock = Clock(fixed_dt=0.02, bus=EventBus())
go = GameObject(name="Camera")
ComponentRegistry._schedule_awake.__func__  # not needed — use add_component
ctrl = go.add_component(FlyController)

inp = InputState(move_forward=True, mouse_captured=False)
ctrl.set_input_state(inp)
clock.update(0.1)
ComponentRegistry.run_frame(clock)
assert go.transform.local_position.y > 0   # moved forward (+Y)
ComponentRegistry.clear()
```

## Gotchas

1. **Do NOT integrate mouse deltas into the quaternion directly.** This causes roll drift.  Always accumulate yaw/pitch as scalars and reconstruct `q_yaw * q_pitch` each frame.

2. **`set_input_state` must be called BEFORE `run_frame`** in the same frame.  If it's called after, the controller reads stale input for that frame.  App ensures the correct order in `_frame_task`.

3. **Sprint only multiplies speed, not sensitivity.** Mouse-look speed is unchanged when sprinting.

4. **Horizontal projection degenerates when looking straight up/down.** `_horizontal(forward)` falls back to `Vec3.FORWARD` if the projected length is near zero.  This means at exactly ±90° pitch, WASD still moves forward along the world +Y axis rather than stopping.

5. **`awake()` reads the initial transform rotation** to initialise yaw/pitch.  If you set `go.transform.local_rotation` before the first `run_frame`, those values are correctly imported as the starting yaw/pitch.

6. **No gravity or collision in Session 1.** The controller lets the camera pass through terrain.  Embodied player physics are deferred to Session 4.
