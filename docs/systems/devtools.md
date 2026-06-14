# devtools — System Doc
keywords: dev tools, developer overlay, debug menu, debug overlay, in-game editor, imgui, dear imgui, directgui, inspector, hierarchy, selection, picking, outline, gizmo, spawn, performance stats, fps overlay, noclip, dev camera, panel, field, tool, DevToolsManager, DevOverlay, describe_object, ray_aabb, editor scene objects, authored objects, gizmo write-back

> One doc per code package, but this system spans **two**: the headless brain
> `fire_engine/devtools/` and its Panda3D renderer `fire_engine/render/devtools_overlay.py`
> (the only panda3d-touching half, per CLAUDE.md hard rule 1). They are documented
> together here because they are one feature; `world.md` cross-links back.
>
> Editor-authored scene objects (loaded via `fire_engine.scene.SceneRuntime`,
> tag `editor_scene`) register as overlay selectables too — F1 → click
> inspects/moves them like dev-spawned cubes. Gizmo edits on them WRITE BACK
> into the authored store (`world/scene_visuals.py` sync task), so F5 persists
> the moved transform rather than the stale authored one.

## Role
The **in-game developer overlay**: a debug menu the owner flies the dev noclip
camera in (the existing `FlyController` *is* the noclip cam). It shows live
performance stats, lets you click an object in the world to select it (bright
wireframe outline) and inspect/edit its GameObject + Transform + every
Component's tunable fields live, and exposes action buttons to spawn props and
fire events. Clicking where no dev object is hit falls back to a **terrain voxel
raycast**: the chunk under the cursor is selected, outlined as its full 16 m
cube, and its read-only voxel stats appear in the Inspector (`describe_chunk`). It is **modular**: a new panel is a `DevTool` subclass registered
on the `DevToolsManager`; a new action is one `add_action(label, fn)` call.

A selected object also gets a **Unity-style transform gizmo** (`devtools/gizmo.py`
math + renderer): drag the per-axis arrows / two-axis plane squares to translate,
the three rings to rotate, or the per-axis stalks + centre cube to scale. The
**Gizmo** panel's Move / Rotate / Scale / Off buttons switch tool. Object
properties (name/tag/layer, Transform, every Component field) are editable in the
Inspector, including the **active** toggle and each Component's **enabled** toggle
(click-to-enable/disable).

It deliberately does **not**: replace the external *Fire Editor* (`editor/`,
EDITOR_PRD.md — that runs in VS Code with the game closed; this runs inside the
live window); persist anything (it edits live engine state only — saves still go
through `SaveManager`); or do retained scene authoring beyond AABB picking +
outline + the move/rotate/scale gizmo (gizmo picking is CPU ray-vs-handle, world
axes only — local-pivot mode and snapping are future additions).

The split is the design: **all editor logic is headless** (`devtools/`, no
panda3d, fully unit-tested) and the renderer only turns `Panel` data into
DirectGUI widgets and mouse events into world rays. Swapping the renderer (e.g.
to real Dear ImGui later) touches nothing in `devtools/`.

## Public API
From `fire_engine/devtools/__init__.py`:
- `DevToolsManager` — the hub: holds `tools`, `selection`, `selectables`, `enabled`; `register_tool`, `panels`, `add_selectable`, `remove_selectable`, `find_selectable`, `pick`, `pick_and_select`.
- `DevTool` — base class for a panel; subclass + implement `build() -> Panel`, expose a `revision` that bumps on structural change.
- `PerformanceTool(providers)` — live stats from a `{label: callable}` map.
- `InspectorTool(selection)` — reflected, editable view of the selected GameObject.
- `ActionsTool(title, actions)` — button grid; `add_action(label, fn)`.
- `ClockTool(clock)` — game-calendar read-out (future home of day/night controls).
- `Selection` — current selection + `revision` counter; `set`, `clear`, `on_change`, `current`.
- `Selectable` — a GameObject + local AABB `half_extents`; `world_aabb()`.
- `ray_aabb(origin, direction, box_min, box_max)` / `pick(origin, direction, selectables)` — CPU picking.
- `describe_object(go) -> list[Section]` — reflect a GameObject into editable inspector sections.
- `describe_chunk(chunk) -> list[Section]` — reflect a terrain `Chunk` into **read-only** inspector sections (coord, world origin, size, solid/total/fill, material ids, dirty/edited). Duck-typed — imports no terrain.
- `is_chunk(obj) -> bool` — duck-typed test (`materials`/`coord`/`chunk_meters`) used by the Inspector to route a picked chunk to `describe_chunk` instead of `describe_object`.
- `Field`, `FieldKind`, `Section`, `Button`, `Panel` — the declarative panel model the renderer consumes.
- `GizmoMode` (TRANSLATE/ROTATE/SCALE), `HandleType` (AXIS/PLANE/RING/UNIFORM), `Handle`, `DragState` — the gizmo's data model.
- `Gizmo(pivot, size, mode)` — `pick(ray_o, ray_d) -> Handle | None` (hit-test handles) and `begin(handle, ray_o, ray_d, pos, rot, scale) -> DragState`.
- `update_drag(state, ray_o, ray_d) -> (Vec3, Quat, Vec3)` — resolve a live drag into the object's new local position / rotation / scale (absolute from the captured reference).

From `fire_engine/render/__init__.py` (panda3d-backed; `None` if panda3d missing):
- `DevOverlay(app, manager=None)` — the DirectGUI renderer. `toggle()` (bind to F1), `set_enabled(bool)`, `handle_world_click() -> bool`, `end_gizmo_drag()` (bind to `mouse1-up`), `spawn_cube()`, and `.actions` (the "World" `ActionsTool`) / `.manager`.

## Imports Allowed
- `fire_engine/devtools/` may import: `core` (math3d, config, clock — duck-typed) and `numpy`. It reads chunks **duck-typed** (`describe_chunk`/`is_chunk` touch only `materials`/`coord`/`chunk_meters`/`world_origin`/`dirty`/`edited`) — it does **not** import `terrain`. **Not** `world` at runtime (TYPE_CHECKING only), so it and its tests never pull panda3d into the import graph. **Never** panda3d, lighting, save.
- `fire_engine/render/devtools_overlay.py` may import: panda3d (`direct.gui`, `panda3d.core`), `core`, `devtools`, `terrain` (`raycast_voxel`, for chunk picking — an allowed downward dep), and `world.registry` (`instantiate`). It is a `world/` module, so panda3d is allowed here and nowhere else for this feature.

## Events
Published: none. The overlay drives engine state through direct public-API calls
(brush/explosion via `main.py`'s action; `instantiate` for spawns), never the bus.
Subscribed: none. (Per ARCHITECTURE.md §4 rule 3, this UI-tick path stays off the
Event Bus — it reads live state each frame and writes through setters.)

## Units & Invariants
- All world space in **meters**, Z-up (forward +Y, right +X, up +Z) — same as the engine.
- Picking AABBs are **world-axis-aligned**; rotation is ignored for v1 picking (boxes stay axis-aligned). `half_extents` are pre-scale; `world_aabb()` multiplies by `abs(local_scale)`.
- Rotation in the inspector is shown/edited in **degrees** (HPR) but stored as a `Quat` (ARCHITECTURE.md §5.4 — Euler is a display view only).
- `Field.set is None` ⇒ read-only. Editing applies through the engine's public setters/attributes only.
- A tool's `revision` bumps only on **structural** change (sections/fields/buttons appear or vanish — e.g. selection change). The renderer rebuilds widgets on a revision change and otherwise just polls `Field.get` each frame, so a value edit never forces a rebuild.
- Headless determinism: `devtools/` has no randomness and no panda3d; `tests/test_devtools.py` runs in the standard headless suite.

## Examples
Register a custom tool and an action (headless side):
```python
from fire_engine.devtools import DevToolsManager, DevTool, Panel, Section, Field, FieldKind

class WeatherTool(DevTool):
    tool_id, title = "weather", "Weather"
    def __init__(self, sky): self._sky = sky
    def build(self) -> Panel:
        return Panel(self.tool_id, self.title, [Section("Sky", [
            Field("rain", FieldKind.FLOAT,
                  lambda: self._sky.rain, lambda v: setattr(self._sky, "rain", v)),
        ])])

mgr = DevToolsManager()
mgr.register_tool(WeatherTool(sky))
```

Wire the overlay into the app (in `main.py`, after the App + camera exist):
```python
from fire_engine.render import DevOverlay
overlay = DevOverlay(app)
overlay.actions.add_action("Fire Explosion", fire_explosion)
app.accept("f1", overlay.toggle)
# left-click: overlay.handle_world_click() picks when the menu is open + cursor free
```

## Gotchas
- **F1 also frees/captures the mouse.** Opening the overlay releases the cursor (so panels/objects are clickable); closing it re-captures for free-look. A click over a DirectGui region is treated as UI (no world pick); a click on empty world deselects.
- **Click is shared with gameplay.** `main.py`'s `mouse1` handler calls `overlay.handle_world_click()` first; it returns `True` (consumed) only when the overlay is open with a free cursor. Flying (cursor captured) keeps left-click = explosion.
- **The camera has no AABB**, so clicking can't select it and it shows no outline; it is the default selection so the inspector isn't empty on first open. Editing the camera's rotation won't stick — `FlyController` rewrites it from yaw/pitch each frame. Spawn a cube to see edits persist.
- **Spawned cubes need their NodePath synced** — the renderer mirrors each spawned GameObject's transform onto its `models/misc/rgbCube` NodePath every frame (`_sync_spawned`). A spawned object with no component never ticks the ComponentRegistry but is still registered for tag lookups via `instantiate`.
- **Entries commit on Enter.** Editable float/vec3/string fields write on Enter; the renderer live-refreshes them only while *unfocused* so it never stomps mid-typing.
- **`devtools/` must stay panda3d-free.** Keep panda3d-reading values (FPS, draw counts) as callables supplied by the renderer; never import panda3d (or `world` at runtime) from this package, or the headless suite breaks.
- **Object pick beats terrain pick.** `handle_world_click` first ray/AABB-picks registered dev objects; only on a full object miss does it voxel-raycast terrain (`_pick_chunk`, ≤200 m). The picked chunk is the manager's cached `Chunk` (same object per coord), so re-clicking it is selection-stable — no inspector rebuild. A chunk is read-only in the inspector (edit voxels with the brush); clicking empty sky deselects.
- **Action buttons use `relief=FLAT` + a thin `borderWidth`.** DirectGui's default is a 0.1-unit raised bevel; on an unscaled `DirectButton` that dwarfs the ~0.036-tall label and overlaps the text. Any new button row must override the border, or it renders as a giant box swallowing its caption.
- **Mouse capture survives ESC + alt-tab.** Opening the overlay frees the cursor and closing it recaptures (`set_enabled` → `App._set_mouse_capture`). Separately, `App.windowEvent` reasserts the hidden/relative-mouse window props on focus regain (the OS drops them on alt-tab), and `main.py`'s `on_click` re-captures when the cursor is free and the overlay is closed — so clicking the window after ESC resumes free-look instead of firing.
- **Editable entries commit on Enter *and* click-off, and don't self-stomp.** Each float/int/string/vec3 `DirectEntry` wires both `command` and `focusOutCommand` to the same submit, and the per-frame value refresh skips a focused entry (`_is_focused` → `entry.guiItem.get_focus()` — **not** the nonexistent `is_focused()`, which silently returned False and let the refresh overwrite every keystroke). So typing is preserved until you press Enter or click away, then it applies.
- **The gizmo drag is press→drag→release across three bindings.** `mouse1` press → `handle_world_click` tries `_begin_gizmo` *before* selection picking (a grabbed handle consumes the click); the per-frame `_update_gizmo` resolves the live cursor ray each frame and writes the new pose; `mouse1-up` → `end_gizmo_drag`. The gizmo only shows for a selected *pickable GameObject* (not the camera — no AABB, `FlyController` rewrites its rotation — and not chunks). Switching mode or losing focus cancels the drag. Geometry is rebuilt per frame (like the outline), drawn depth-test-off in `fixed` bin 110 (above the outline's 100).
- **Gizmo math uses world axes and resolves drags absolutely.** `devtools/gizmo.py` is panda3d-free and unit-tested: handle picking is CPU ray-vs-segment / ray-vs-plane / ray-vs-ring, and `update_drag` computes the new pose from the reference captured at `begin` (axis param / plane point / ring angle / radial distance) — never incrementally — so a paused cursor leaves the object still and re-grabbing is exact. Axes are global (index 0/1/2 = X/Y/Z = `Vec3.RIGHT/FORWARD/UP`); local-pivot mode is a future addition.
