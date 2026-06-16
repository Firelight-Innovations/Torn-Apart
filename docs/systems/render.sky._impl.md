# render.sky._impl — System Doc
keywords: sky impl, sky build, sky geom, sky update, sky geometry, dome builder, cloud builder, build_dome, build_clouds, update_dome, update_clouds, update_fog_and_light, update_shooting_star, rain build, build_particles, build_cylinders, update_cylinders, lightning bolt, upload_bolt, advance_bolt, bolt_envelope, bolt_sky_flash, add_flash_light, refresh_cover, cover_z, sky_build, sky_geom, sky_update, rain_build, lightning_bolt, private implementation, 500 line split, C0302

> One doc per code package; filename matches the package exactly (`docs/systems/render.sky._impl.md` <-> `fire_engine/render/sky/_impl/`).

## Role

`render/sky/_impl/` is a **private implementation package** containing method clusters extracted from the sky-package render components to keep each top-level module under 500 lines (Hard Rule 8 / C0302 compliance). None of these modules are a public API; callers should import from the parent package (`fire_engine.render.sky.*`).

Five modules live here:

- **`sky_build.py`** — `build_dome` and `build_clouds`: the `start()`-time geometry/shader/texture builders for `SkyRendererComponent`. Geometry helpers and constants are in `sky_geom` (no circular dependency).
- **`sky_geom.py`** — geometry builders, texture helpers, and physical constants (`_DOME_RADIUS_M`, `_VCLOUD_ALT_M`, `_VCLOUD_THICK_M`, etc.) shared between `sky_build` and `sky_renderer`. Also `_build_dome_node`, `_clamp01`, `_fallback_moon`, `_fallback_star_cube`, `_load_or_bake_cloud_noise`, `_make_geom_node`, `_sky_texture`.
- **`sky_update.py`** — `update_dome`, `update_shooting_star`, `update_clouds`, `update_fog_and_light`: the per-frame uniform writers called from `SkyRendererComponent.late_update`.
- **`rain_build.py`** — `build_particles`, `build_cylinders`, `update_cylinders`: geometry builders and the per-frame cylinder scroll update for `RainRendererComponent`.
- **`lightning_bolt.py`** — `upload_bolt`, `advance_bolt`, `bolt_envelope`, `bolt_sky_flash`, `add_flash_light`, `refresh_cover`, `cover_z`: bolt geometry upload, envelope animation, and roof-aware strike Z helpers for `LightningRendererComponent`.

**Dependency rule**: these modules import FROM their parent component modules only under `TYPE_CHECKING` (for type annotations). They do NOT import at runtime from the component that calls them — only the reverse is allowed (component -> _impl). This prevents circular imports.

## Public API

All symbols in this package are private (`_`-prefixed or internal). There is no public API — do not import from `render.sky._impl` outside of `render/sky/`.

For reference, the callable entry points used by the parent modules:

| Symbol | Module | Description |
|---|---|---|
| `build_dome(component, star_count)` | `sky_build` | Build the sky dome geom, shader, and textures into `component`. |
| `build_clouds(component)` | `sky_build` | Build the volumetric cloud dome geom and shader into `component`. |
| `update_dome(component, st, cx, cy, cz)` | `sky_update` | Write per-frame sky dome uniforms from `SkyState` `st` and camera position. |
| `update_shooting_star(component, st, dt)` | `sky_update` | Advance the deterministic shooting-star slot and animate the streak. |
| `update_clouds(component, st, cx, cy, cz)` | `sky_update` | Write per-frame cloud uniforms (sun/moon radiance, coverage, wind drift). |
| `update_fog_and_light(component, st)` | `sky_update` | Set exponential fog density/colour and terrain colour-scale (CPU backend only). |
| `build_particles(component, cfg)` | `rain_build` | Build the GPU-instanced rain particle node into `component`. |
| `build_cylinders(component, cfg)` | `rain_build` | Build the low-preset rain cylinder nodes into `component`. |
| `update_cylinders(component, cam, dt)` | `rain_build` | Scroll the cylinder UV offsets per frame. |
| `upload_bolt(component, bolt, geom, intensity, seed)` | `lightning_bolt` | Write bolt segment geometry to a pooled dynamic-geometry node. |
| `advance_bolt(bolt, dt)` | `lightning_bolt` | Step a bolt's envelope animation (leader->return->afterglow->restrikes). |
| `bolt_envelope(bolt)` | `lightning_bolt` | Compute the current `(reveal, flash)` pair from the bolt's age and phase. |
| `bolt_sky_flash(bolt)` | `lightning_bolt` | Return the scalar sky/cloud flash contribution for this bolt (0..1). |
| `add_flash_light(component, pos, intensity)` | `lightning_bolt` | Add a transient point-light at the strike position via the lighting pipeline. |
| `refresh_cover(component)` | `lightning_bolt` | Recenter the rain-cover heightmap if the player has moved past the threshold. |
| `cover_z(component, gx, gy)` | `lightning_bolt` | Look up the roof-aware Z at world XY (gx, gy) from the cover heightmap. |

## Imports Allowed

Same set as the parent `render/sky/` package (see `docs/systems/render.sky.md`):

- `panda3d.*` — all modules here are inside `render/` (the panda3d bridge).
- `fire_engine.core` — `get_logger`, `rng.for_domain`.
- `fire_engine.render.component`, `fire_engine.render._impl.quad` — shared component and geometry helpers.
- `fire_engine.render.sky.{sky_shaders,rain_shaders,lightning_shaders}` — GLSL sources (for `Shader.make`).
- `fire_engine.world.terrain` — `RainCoverField` (headless).
- `fire_engine.world.weather` — `generate_bolt` (headless).
- Python standard library; `numpy`.
- Sibling `_impl` modules (`sky_geom` imported by `sky_build`, etc.).
- Parent component modules ONLY under `TYPE_CHECKING` (never at runtime).

## Events

None. These are pure implementation helpers; event subscriptions and publications belong to the parent component classes.

## Units & Invariants

All units and coordinate conventions are inherited from the parent package (see `docs/systems/render.sky.md`):

- World space: meters, Z-up.
- Functions that accept a `SkyRendererComponent` or `RainRendererComponent` instance as their first argument treat all position/distance fields as meters.
- `_clamp01(x)` in `sky_geom.py` and `sky_update.py` is a scalar (0..1) clamp helper, not a vectorised numpy operation.

## Examples

```python
# Internal usage pattern (sky_renderer.py calls sky_build):
from fire_engine.render.sky._impl.sky_build import build_dome, build_clouds

def start(self) -> None:
    build_dome(self, star_count)   # populates self._dome_np, shaders, textures
    build_clouds(self)             # populates self._cloud_np

# Internal usage pattern (sky_renderer.py calls sky_update in late_update):
from fire_engine.render.sky._impl.sky_update import (
    update_dome, update_shooting_star, update_clouds, update_fog_and_light,
)

def late_update(self, dt: float) -> None:
    update_dome(self, st, cx, cy, cz)
    update_shooting_star(self, st, dt)
    update_clouds(self, st, cx, cy, cz)
    update_fog_and_light(self, st)
```

## Gotchas

- **No circular imports at runtime.** The `TYPE_CHECKING` guard is mandatory when a `_impl` module needs to reference the parent component class in a type annotation. Without it the import cycle crashes on startup.
- **`sky_geom.py` re-exports private `_`-prefixed symbols** through `__all__` because `sky_renderer.py` re-exports them for backward compat. This is the only case where `__all__` lists private names; do not replicate this pattern elsewhere.
- These modules all import `panda3d` and therefore must only be used inside `render/` (Hard Rule 1). They are NOT headless-importable.
- `bolt_envelope` in `lightning_bolt.py` is a pure helper (no panda3d calls); it is separated from `advance_bolt` so unit tests can exercise the envelope math headlessly if needed.
