# procedural.textures — System Doc
keywords: textures, ProceduralTextureDef, ProceduralDef, value_noise, pixel_noise, wasteland_ground, night_sky, night_sky_cube, rain_streak, grass_ground, dirt_ground, moon_surface, grass_tuft, dust_mote, leaf_sprite, flower_sprite, plaster_wall, register_def, get, RGBA, uint8, pixel art, posterise, palette, noise, ground texture, sky texture, sprite texture, alpha cutout, texture def, texture registry, texture authoring

> One doc per code package; filename matches the package exactly (`docs/systems/procedural.textures.md` ↔ `fire_engine/procedural/textures/`).

## Role

`procedural/textures/` is the **texture-generation sub-package** of `procedural/`. It provides the `ProceduralTextureDef` base class, shared noise helpers (`value_noise`, `pixel_noise`), and registers all built-in texture definitions at import time.

Built-in textures are organised into three category sub-packages for the deep-and-narrow structure rule:

- `ground/` — ground-surface textures (dirt, grass, wasteland, plaster wall).
- `sky/` — atmospheric/sky textures (moon, rain streaks, night sky equirect, night sky cube-map).
- `sprites/` — vegetation sprite / particle textures (grass tuft, flower, leaf litter, dust mote).

Importing `fire_engine.procedural.textures` (or `fire_engine.procedural`) triggers all `@register_def` decorators in all three sub-packages, registering every built-in name in the global procedural registry.

`procedural/textures/` does NOT: render anything, upload to Panda3D, store game-world state, or do per-pixel Python loops.

## Public API

All symbols below are re-exported from `fire_engine.procedural.textures` (`__init__.py`).

### Base class and noise helpers (`textures/base.py`)

| Symbol | Description |
|---|---|
| `ProceduralTextureDef` | `ProceduralDef` subclass; override `generate(rng, **params) -> np.ndarray (H,W,4) uint8`. Decorated with `@register_def` to auto-register at import. |
| `value_noise(rng, shape, octaves, persistence, lacunarity, base_freq)` | Layered 2-D value noise → `float32 (H,W)` in `[0,1]`. Bilinear upsampling — smooth, no visible texels. |
| `pixel_noise(rng, shape, octaves, persistence, lacunarity, base_freq)` | Layered 2-D pixel (nearest-neighbour) noise → `float32 (H,W)` in `[0,1]`. Crisp block-edged texels, retro pixel-art look. |

### Built-in texture modules (re-exported as module references)

| Re-export | Module | Registered name(s) | Output |
|---|---|---|---|
| `wasteland_ground` | `ground/wasteland_ground.py` | `"wasteland_ground"` | `(256,256,4)` uint8 cracked dead earth |
| `grass_ground` | `ground/grass_ground.py` | `"grass_ground"` | `(64,64,4)` uint8 weathered grass; exports `GRASS_PALETTE`, `GRASS_THRESHOLDS` |
| `dirt_ground` | `ground/dirt_ground.py` | `"dirt_ground"` | `(64,64,4)` uint8 dry dirt with dark clods; exports `DIRT_PALETTE`, `DIRT_THRESHOLDS` |
| `plaster_wall` | `ground/plaster_wall.py` | `"plaster_wall"` | `(64,64,4)` uint8 weathered lime-plaster wall; exports `PLASTER_PALETTE`, `PLASTER_THRESHOLDS` |
| `moon_surface` | `sky/moon_surface.py` | `"moon_surface"` | `(256,256,4)` uint8 lunar disc with maria and craters |
| `night_sky` | `sky/night_sky.py` | `"night_sky"`, `"night_sky_cube"` | equirect `(512,1024,4)` uint8 star field; cube-map `(6,512,512,4)` uint8 |
| `rain_streak` | `sky/rain_streak.py` | `"rain_streak"` | `(512,128,4)` uint8 tiling rain streaks (alpha = streak intensity) |
| `grass_tuft` | `sprites/grass_tuft.py` | `"grass_tuft"` | `(32,32,4)` uint8 binary-alpha grass blade silhouette |
| `flower_sprite` | `sprites/flower_sprite.py` | `"flower_sprite"` | `(32,128,4)` uint8 wildflower atlas, 4 hue variants |
| `leaf_sprite` | `sprites/leaf_sprite.py` | `"leaf_sprite"` | `(32,96,4)` uint8 leaf-litter atlas, 3 hue variants |
| `dust_mote` | `sprites/dust_mote.py` | `"dust_mote"` | `(32,32,4)` uint8 soft radial dust/pollen speck |

## Imports Allowed

`procedural/textures/` may only import:
- Python standard library (`math`, ...)
- `numpy`
- `fire_engine.procedural.defs` (for `ProceduralDef`, `register_def`)
- `fire_engine.core` (for `for_domain`)

**No panda3d imports.** Never import from `render/`, `world/`, `simulation/`, `lighting/`, or any higher layer. The Panda3D bridge lives exclusively in `render/texture_bridge.py`.

## Events

### Published
None. `procedural/textures/` is a pure generation layer.

### Subscribed
None. Textures are generated on demand via `procedural.get()` — no event triggers.

## Units & Invariants

- All `ProceduralTextureDef.generate()` implementations must return shape `(H, W, 4)`, dtype `uint8`, channel order RGBA.
- Alpha = 255 (fully opaque) unless explicitly documented. Documented exceptions: `"night_sky"` (alpha = luminance for additive blending), `"rain_streak"` (alpha = streak intensity), sprite textures (binary alpha for discard rendering).
- `value_noise` and all built-in texture defs use only numpy array expressions — no per-pixel Python loops (Hard Rule 4).
- Determinism: `get(name, **params)` with the same `(name, world_seed, params)` always produces byte-identical output across processes and restarts.

## Examples

### Get a built-in texture
```python
from fire_engine.core.rng import set_world_seed
from fire_engine.procedural import get
import numpy as np

set_world_seed(1337)
arr = get("wasteland_ground")       # (256, 256, 4) uint8
assert arr.shape == (256, 256, 4)
assert arr.dtype == np.uint8
assert (arr[..., 3] == 255).all()   # fully opaque
```

### Author a new texture
```python
# fire_engine/procedural/textures/ground/cracked_rock.py
import numpy as np
from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, value_noise

@register_def
class CrackedRockDef(ProceduralTextureDef):
    """256×256 grey cracked rock surface.

    Docs: docs/systems/procedural.textures.ground.md
    """
    name = "cracked_rock"

    def generate(self, rng, **params):
        W = int(params.get("width", 256))
        H = int(params.get("height", 256))
        base = value_noise(rng, (H, W), octaves=4)
        rgba = np.empty((H, W, 4), dtype=np.uint8)
        grey = (base * 140 + 40).astype(np.uint8)
        rgba[..., 0:3] = grey[..., None]
        rgba[..., 3] = 255
        return rgba

# Import in fire_engine/procedural/textures/ground/__init__.py
# Add determinism test in tests/
# Preview: python tools/preview_texture.py cracked_rock
```

## Gotchas

1. **Sub-package import order matters for registration.** `fire_engine.procedural.textures.__init__` imports `ground`, `sky`, `sprites` (triggering their `@register_def` calls) before re-exporting module references. Adding a new texture module requires adding an import in the correct sub-package `__init__.py` AND in `procedural/textures/__init__.py`.
2. **Re-exports are module objects, not def instances.** `from fire_engine.procedural.textures import grass_ground` gives you the module, not the `GrassGroundDef` instance. To get the texture array call `get("grass_ground")`.
3. **Panda3D vertical flip.** `render/texture_bridge.py` flips arrays vertically on upload (OpenGL UV origin is bottom-left). Sprite textures whose bases sit on the bottom image row (V=0 in the array) appear at V=0 in the shader — which IS the bottom of the quad. This is correct; do not compensate in the texture generator.
4. **`pixel_noise` vs `value_noise`.** `pixel_noise` gives crisp pixel-art block edges (ground textures, posterised palettes). `value_noise` gives smooth continuous fields (heightmaps, fog). Mixing them in the wrong context looks wrong — use `pixel_noise` for any texture intended to be posterised.
