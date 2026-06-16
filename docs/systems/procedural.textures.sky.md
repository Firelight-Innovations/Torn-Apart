# procedural.textures.sky — System Doc
keywords: sky texture, night sky, night_sky, night_sky_cube, moon, moon_surface, rain, rain_streak, star field, stars, galaxy, equirect, equirectangular, cube map, cube-map, NightSkyDef, NightSkyCubeDef, MoonSurfaceDef, RainStreakDef, atmospheric texture, sky dome, additive blend, alpha luminance, cube face, GL face order, maria, crater, lunar, tileable, tiling, pixel art

> One doc per code package; filename matches the package exactly (`docs/systems/procedural.textures.sky.md` ↔ `fire_engine/procedural/textures/sky/`).

## Role

`procedural/textures/sky/` is the **sky and atmospheric texture definitions** sub-package of `procedural/textures/`. It registers four built-in sky/atmospheric texture defs at import time:

- `"night_sky"` — 1024×512 RGBA equirectangular star field with galaxy band; alpha = luminance for additive blending.
- `"night_sky_cube"` — (6, 512, 512, 4) RGBA cube-map version of the same star field; no pole distortion; GL face order.
- `"moon_surface"` — 256×256 RGBA lunar disc with pale regolith, dark maria and a vectorised crater field; alpha 255 inside the unit disc.
- `"rain_streak"` — 128×512 RGBA tiling rain streak texture; alpha = streak intensity; tileable in both U and V.

Shared helper functions for the cube-map and night-sky generation (`cube_face_directions`, `_dirs_to_face_pixels`, `_hash3i`, `_value_noise_3d`, `_upsample2_faces`, `_ramp_rgb`) live in `_night_sky_helpers.py` (private) and are re-exported from `night_sky.py`.

This is a private organisational sub-package of `fire_engine.procedural.textures`; callers should import the parent package or `fire_engine.procedural` rather than this directly.

## Public API

Exported from `fire_engine.procedural.textures.sky` (`__init__.py`) as module references:

| Module reference | Registered name(s) | Output | Notes |
|---|---|---|---|
| `moon_surface` | `"moon_surface"` | `(256,256,4)` uint8 | Alpha 255 inside the disc; seeded per world — each world grows a different moon. Params: `size`, `crater_count`. |
| `night_sky` | `"night_sky"`, `"night_sky_cube"` | equirect `(512,1024,4)` uint8; cube-map `(6,512,512,4)` uint8 | `"night_sky"` alpha = luminance (additive-blend mask). +Z pole at v=1 (array row 0 = zenith). Params: `width`, `height`, `star_count`. |
| `rain_streak` | `"rain_streak"` | `(512,128,4)` uint8 | Alpha = streak intensity. Tileable in U and V. Params: `width`, `height`, `streak_count`. |

## Imports Allowed

`procedural/textures/sky/` may only import:
- Python standard library (`math`, `typing`, ...)
- `numpy`
- `fire_engine.procedural.defs` (for `register_def`)
- `fire_engine.procedural.textures.base` (for `ProceduralTextureDef`, `value_noise`, `pixel_noise`)

Private module `_night_sky_helpers.py` (prefixed `_`) is imported by `night_sky.py` and `night_sky_cube.py` internally — this is fine and not a dependency violation.

**No panda3d imports.** Never import from `render/`, `world/`, `simulation/`, `lighting/`, or any higher layer.

## Events

### Published
None.

### Subscribed
None.

## Units & Invariants

- `"night_sky"` and `"night_sky_cube"` use a **+Z pole at v=1** convention: array row 0 = zenith. The sky-dome shader must account for this when sampling.
- `"night_sky"` alpha channel = luminance (additive blend with `alpha * luminance`). Alpha is NOT an opacity mask for these textures.
- `"moon_surface"` alpha = 255 only inside the circular disc; transparent outside. The sky shader samples the disc using a disc-local UV and applies the phase terminator dynamically — it is NOT baked into the texture.
- `"rain_streak"` is tileable in BOTH U and V (seamless). Streak alpha = streak intensity; fully transparent where there is no rain.
- `"night_sky_cube"` uses GL face order: `(+X, -X, +Y, -Y, +Z, -Z)` as face indices `(0..5)`.
- All sky textures are seeded from the world seed — different worlds have different star fields and moon surfaces.

## Examples

### Get sky textures
```python
from fire_engine.core.rng import set_world_seed
from fire_engine.procedural import get
import numpy as np

set_world_seed(1337)
sky   = get("night_sky")           # (512, 1024, 4) uint8
cube  = get("night_sky_cube")      # (6, 512, 512, 4) uint8
moon  = get("moon_surface")        # (256, 256, 4) uint8
rain  = get("rain_streak")         # (512, 128, 4) uint8

# sky alpha = luminance — additive blend only
assert sky[..., 3].max() <= 255
# moon alpha = disc mask
assert (moon[..., 3] == 0).any()   # transparent outside disc
```

## Gotchas

1. **`"night_sky"` alpha is NOT opacity.** It is a luminance mask for additive blending (`color.rgb * alpha / 255`). Do not use it as a transparency map — stars will vanish.
2. **`"night_sky"` registers TWO names.** Importing `night_sky.py` registers both `"night_sky"` (equirect) and `"night_sky_cube"` (via `NightSkyCubeDef` in `night_sky_cube.py` which is imported by `night_sky.py`). Calling `get("night_sky_cube")` does NOT require separate import.
3. **Pole convention is +Z up, row 0 = zenith.** Array row 0 maps to v=1 in the shader's UV (the sky dome maps V=1 to the zenith pole). If the night sky is upside-down in-game, verify the dome shader's UV flip direction, not the texture.
4. **Moon is seeded per world.** Each world generates a different crater layout. The phase terminator (day/night shadow) is applied by the sky shader at runtime — do not bake it into the texture.
