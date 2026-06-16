# render._impl — System Doc
keywords: render _impl, app_input, app_profiler, app_terrain, quad, transform_math, private implementation, App helpers, input collection, profiler setup, terrain render integration, billboard quad, unit quad, additive instanced node, TRS matrix, transform math, setup_profiler, setup_terrain_rendering, build_unit_quad, setup_additive_instanced_node, trs_matrix

> One doc per code package; filename matches the package exactly (`docs/systems/render._impl.md` ↔ `fire_engine/render/_impl/`).

## Role

`render/_impl/` is the **private implementation helpers** package for `fire_engine.render`.

It exists purely to satisfy the ≤500-line module limit (Hard Rule 8): functions extracted from fat
public classes in `render/` live here, taking the owning instance as their first argument
(`self_obj`) and being called from the class as `_func(self, ...)`.  This package is **not a
public API** — nothing outside `render/` should import from it directly.

Modules:

- `app_input.py` — input-collection helpers for `App` (mouse capture, `InputState` assembly,
  window-properties writes).
- `app_profiler.py` — profiler setup and per-frame snapshot helpers for `App`
  (`setup_profiler`, snapshot the top scopes for the F3 overlay).
- `app_terrain.py` — terrain render-integration helpers for `App`
  (`setup_terrain_rendering`, `stream_and_upload_terrain`).
- `quad.py` — shared unit billboard-quad geometry builder (`build_unit_quad`) and additive
  particle node setup helper (`setup_additive_instanced_node`), reused by rain, mote, and leaf-
  litter renderers to eliminate duplication.
- `transform_math.py` — pure-numpy TRS matrix math (`trs_matrix`) extracted from
  `render/transform.py`.

`render/_impl/` deliberately does NOT: export symbols to users of `fire_engine.render`, define
public classes, or hold any game logic.

## Public API

No public API — all symbols are private (`_`-prefixed callers or internal to `render/`).

The following are the notable free functions (internal use only):

| Module | Function | Description |
|---|---|---|
| `app_input` | `collect_input(self_obj)` | Update `App.input_state` from Panda3D key map each frame. |
| `app_profiler` | `setup_profiler(self_obj)` | Create `ProfilerOverlay` + `PStatsBridge`; no-op when `profiler_enabled` is off. |
| `app_terrain` | `setup_terrain_rendering(self_obj, ...)` | Wire the terrain render path (ground texture, material map, `set_light_off`). |
| `app_terrain` | `stream_and_upload_terrain(self_obj)` | Per-frame terrain chunk stream + Geom upload into `terrain_root`. |
| `quad` | `build_unit_quad(name)` | Build a shared 4-vertex / 2-triangle unit billboard quad Geom. |
| `quad` | `setup_additive_instanced_node(node, geom_node)` | Apply additive blending, disable depth write, set infinite BoundingBox + `set_final`. |
| `transform_math` | `trs_matrix(pos, rot, scale) -> np.ndarray` | Build a 4×4 TRS matrix from Vec3 pos, Quat rot, Vec3 scale (float64). |

## Imports Allowed

- `panda3d.*`, `direct.*` (all modules here are in `render/` — Hard Rule 1 allows panda3d)
- `fire_engine.core` (math3d types, profiler, etc.)
- `fire_engine.render.*` public modules (via `TYPE_CHECKING` guards to avoid circular imports)
- Python standard library, `numpy`

## Events

Published: none.

Subscribed: none.  Event handling lives in the public component classes, not in `_impl/`.

## Units & Invariants

- All spatial values in world meters (Z-up).
- `trs_matrix` returns a `(4, 4) float64` ndarray — multiply column-vectors on the right.
- `build_unit_quad` produces a ±0.5 m unit quad centred on the origin in the XZ plane; callers
  scale via `set_instance_count` + vertex-shader offsets.
- `setup_additive_instanced_node` sets `BoundingBox((-1e6,−1e6,−1e6),(1e6,1e6,1e6))` so Panda3D
  never frustum-culls the instanced particles.

## Examples

```python
# Internal usage only — do not import from render._impl directly.
# From render/app.py:
from fire_engine.render._impl.app_profiler import setup_profiler
setup_profiler(self)  # self = App instance

from fire_engine.render._impl.quad import build_unit_quad, setup_additive_instanced_node
quad_geom = build_unit_quad("rain_quad")
setup_additive_instanced_node(rain_node, rain_geom_node)
```

## Gotchas

- All functions take the **owning instance as first arg** — they are free functions, not methods;
  call them as `func(self, ...)`, not `self.func(...)`.
- `TYPE_CHECKING` guards prevent circular imports: modules in `_impl/` reference public classes
  only under `if TYPE_CHECKING:`.
- `transform_math.trs_matrix` returns `float64`; Panda3D expects `float32` — convert before
  uploading to the scene graph (done in `render/transform.py`).
