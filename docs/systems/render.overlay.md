# render.overlay — System Doc
keywords: render overlay, devtools overlay, DevOverlay, devtools_overlay, overlay, inspector, gizmo, outline, panels, spawn, weather, environment, F1, DirectGUI, in-game overlay, editor overlay, pick, select, click, wireframe box, ray AABB, camera ray, world ray, panel, field, revision, mouse cursor, free mouse, fly controller, spawn menu, live inspector, live edit, overlay environment, overlay gizmo, overlay outline, overlay panels, overlay spawn, overlay weather

> One doc per code package; filename matches the package exactly (`docs/systems/render.overlay.md` ↔ `fire_engine/render/overlay/`).

## Role

`render/overlay/` is the **Panda3D renderer for the in-game developer overlay** (toggled with
**F1**).  It is the sole panda3d-touching half of the dev tools system: the headless logic
(tool discovery, field definitions, panel data, ray-based object picking) lives in
`fire_engine/devtools/`; this package turns that data into DirectGUI widgets and handles mouse /
keyboard interaction.

Modules:

- `devtools_overlay.py` — the public `DevOverlay` class.  Builds and refreshes DirectGUI
  panels from each tool's `Panel` (rebuilt only when a tool's `revision` changes; values refreshed
  every frame to avoid polling lag).  Converts a mouse click to a world-space ray and asks
  `DevToolsManager` to pick the object under the cursor (ray/AABB, headless).  Draws a bright
  wireframe box around the selected object.
- `_overlay_environment.py` — private panel rendering for environment/time-of-day controls.
- `_overlay_gizmo.py` — private gizmo / transform-handle rendering (move/rotate handles).
- `_overlay_outline.py` — private selection-outline box builder (wireframe bounding box around
  the selected object).
- `_overlay_panels.py` — private shared helpers for building DirectGUI panel widgets (labels,
  sliders, entry fields, buttons).
- `_overlay_spawn.py` — private spawn-menu panel rendering (place primitive props).
- `_overlay_weather.py` — private weather-control panel rendering (rain intensity, storm toggle).

`render/overlay/` deliberately does NOT: implement any editor logic, compute ray–geometry
intersections, or own the dev tool state.  Swapping this for a Dear ImGui backend would not touch
`fire_engine/devtools/` at all.

## Public API

All symbols are exported from `fire_engine.render.overlay` (`__init__.py`) and re-exported from
`fire_engine.render` for convenience.

| Symbol | Description |
|---|---|
| `DevOverlay(app, manager=None)` | Construct the overlay renderer.  `app` = the `App` instance; `manager` = the headless `DevToolsManager` (or `None` to create a default one).  Panda3D-backed; `None` when panda3d is absent. |
| `DevOverlay.toggle()` | Show/hide the overlay; frees the mouse cursor when visible, re-captures for free-look when hidden. |
| `DevOverlay.update(dt)` | Per-frame refresh — poll tool revisions, rebuild changed panels, sync value displays, advance gizmos. Called by `App._frame_task`. |

The private `_overlay_*.py` modules are **not public API** — they are called from
`devtools_overlay.py` only.

## Imports Allowed

- `panda3d.*`, `direct.*` (Hard Rule 1: `render/` is the render bridge)
- `fire_engine.core` (math3d, event bus)
- `fire_engine.devtools` (headless `DevToolsManager`, `Panel`, `Field` types)
- `fire_engine.render.app` (type annotation for `App`)
- Python standard library, `numpy`

## Events

Published: none directly.  Object-pick actions are routed through `DevToolsManager` which may
publish scene-state events on the bus.

Subscribed: none directly.  `DevOverlay.update(dt)` is driven by `App._frame_task`.

## Units & Invariants

- Mouse-to-world ray uses the Panda3D camera lens; result is in world meters, Z-up.
- The selection wireframe box is sized from the picked object's AABB (meters).
- Panel refresh is guarded by `tool.revision` — panels are only rebuilt when the tool's data
  changes, not every frame (avoids DirectGUI widget churn).
- Coordinate conversions (math3d ↔ Panda3D) happen only here and in `render/app.py` /
  `render/camera.py`, per the world-layer boundary.

## Examples

```python
from fire_engine.render import App, DevOverlay
from fire_engine.core.event_bus import EventBus
from fire_engine.core.clock import Clock

cfg = ...
clock = Clock(fixed_dt=0.02, bus=EventBus())
app = App(cfg, clock, clock.bus)
overlay = DevOverlay(app)          # F1 toggles; left-click selects

# In the frame loop (handled by App automatically when overlay is wired):
overlay.update(dt)
```

## Gotchas

- **F1 toggles the overlay and mouse-capture mode simultaneously** — while the overlay is open
  and the cursor is free, left-click selects an object; while flying (cursor captured) left-click
  keeps its normal in-game meaning.  Do not intercept mouse events in the overlay when
  `input_state.mouse_captured` is True.
- DirectGUI panels are rebuilt from scratch whenever a tool's `revision` increments — avoid
  incrementing revision every frame or it will cause constant widget churn.
- `DevOverlay` must be constructed after `App.__init__` completes (the DirectGUI system needs an
  active window).
- `_overlay_*.py` modules are private; import from `devtools_overlay` or `render.overlay` only.
