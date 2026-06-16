# procedural.textures.ground — System Doc
keywords: ground texture, ground, dirt, grass, wasteland, plaster, wall, dirt_ground, grass_ground, wasteland_ground, plaster_wall, DirtGroundDef, GrassGroundDef, WastelandGroundDef, PlasterWallDef, DIRT_PALETTE, DIRT_THRESHOLDS, GRASS_PALETTE, GRASS_THRESHOLDS, PLASTER_PALETTE, PLASTER_THRESHOLDS, pixel_noise, value_noise, posterise, posterize, palette, ground LUT, ground_lut, cracked earth, dead earth, post-apocalyptic, pixel art, register_def

> One doc per code package; filename matches the package exactly (`docs/systems/procedural.textures.ground.md` ↔ `fire_engine/procedural/textures/ground/`).

## Role

`procedural/textures/ground/` is the **ground-surface texture definitions** sub-package of `procedural/textures/`. It collects the four built-in ground and wall texture defs:

- `"dirt_ground"` — 64×64 dry post-apocalyptic dirt with dark clod clusters.
- `"grass_ground"` — 64×64 weathered living grass, crisp pixel-art palette.
- `"wasteland_ground"` — 256×256 cracked dead earth (smooth `value_noise`, bilinear).
- `"plaster_wall"` — 64×64 weathered lime-plaster wall albedo for buildings.

`grass_ground` and `dirt_ground` export their palette constants (`GRASS_PALETTE`, `GRASS_THRESHOLDS`, `DIRT_PALETTE`, `DIRT_THRESHOLDS`). The GPU terrain shader bakes them into the world-space palette LUT via `procedural.textures.ground_lut.build_ground_lut` so the in-engine non-repeating ground matches the baked textures exactly.

This is a private organisational sub-package of `fire_engine.procedural.textures`; callers should import the parent package or `fire_engine.procedural` rather than this directly.

`procedural/textures/ground/` does NOT: render anything, upload textures, or contain shared machinery (noise helpers, base classes).

## Public API

Exported from `fire_engine.procedural.textures.ground` (`__init__.py`) as module references:

| Module reference | Registered name | Output | Key exports |
|---|---|---|---|
| `dirt_ground` | `"dirt_ground"` | `(64,64,4)` uint8 dry dirt with dark clods; 6-colour posterised palette | `DIRT_PALETTE`, `DIRT_THRESHOLDS` |
| `grass_ground` | `"grass_ground"` | `(64,64,4)` uint8 weathered pixel-art grass; 8-colour posterised palette | `GRASS_PALETTE`, `GRASS_THRESHOLDS` |
| `wasteland_ground` | `"wasteland_ground"` | `(256,256,4)` uint8 cracked dead earth (smooth bilinear noise) | — |
| `plaster_wall` | `"plaster_wall"` | `(64,64,4)` uint8 lime-plaster wall with hairline cracks | `PLASTER_PALETTE`, `PLASTER_THRESHOLDS` |

## Imports Allowed

`procedural/textures/ground/` may only import:
- Python standard library
- `numpy`
- `fire_engine.procedural.defs` (for `register_def`)
- `fire_engine.procedural.textures.base` (for `ProceduralTextureDef`, `pixel_noise`, `value_noise`)

**No panda3d imports.** Never import from `render/`, `world/`, `simulation/`, `lighting/`, or any higher layer.

## Events

### Published
None.

### Subscribed
None.

## Units & Invariants

- All defs return `(H, W, 4)` uint8, alpha = 255 (fully opaque).
- `GRASS_PALETTE` / `GRASS_THRESHOLDS`, `DIRT_PALETTE` / `DIRT_THRESHOLDS`, `PLASTER_PALETTE` / `PLASTER_THRESHOLDS` are the single source of truth for each material's colour ramp. The GPU ground LUT pipeline (`procedural/textures/ground_lut.py`) imports these constants directly — change them here and the shader's world-space procedural ground updates automatically.
- All ground defs use `pixel_noise` (block-edged texels) except `wasteland_ground` which uses `value_noise` (smooth bilinear) because it represents large-scale cracked earth, not a pixel-art surface.

## Examples

### Get a ground texture
```python
from fire_engine.core.rng import set_world_seed
from fire_engine.procedural import get

set_world_seed(99)
grass = get("grass_ground")        # (64, 64, 4) uint8
dirt  = get("dirt_ground")         # (64, 64, 4) uint8
waste = get("wasteland_ground")    # (256, 256, 4) uint8
wall  = get("plaster_wall")        # (64, 64, 4) uint8
```

### Access palette constants for the ground LUT
```python
from fire_engine.procedural.textures.ground.grass_ground import (
    GRASS_PALETTE, GRASS_THRESHOLDS,
)
from fire_engine.procedural.textures.ground.dirt_ground import (
    DIRT_PALETTE, DIRT_THRESHOLDS,
)
from fire_engine.procedural.textures.ground_lut import build_ground_lut

entries = {0: (GRASS_PALETTE, GRASS_THRESHOLDS),
           1: (DIRT_PALETTE,  DIRT_THRESHOLDS)}
lut = build_ground_lut(entries)    # (2, 256, 4) uint8
```

## Gotchas

1. **Import via parent.** `fire_engine.procedural.textures.__init__` imports this sub-package to trigger `@register_def`. Do not import `procedural.textures.ground` directly in game code — import `fire_engine.procedural` or `fire_engine.procedural.textures` instead.
2. **Palette constants are in the leaf module, not `__init__`.** `from fire_engine.procedural.textures.ground.grass_ground import GRASS_PALETTE` is the correct import path. The `__init__.py` re-exports module references, not the palette arrays.
3. **`wasteland_ground` uses `value_noise`, not `pixel_noise`.** Its output is smooth and bilinear. If you need pixel-art ground, use `grass_ground` or `dirt_ground`.
