# procedural.textures.sprites — System Doc
keywords: sprite, sprite texture, atlas, grass_tuft, dust_mote, leaf_sprite, flower_sprite, GrassTuftDef, DustMoteDef, LeafSpriteDef, FlowerSpriteDef, alpha cutout, binary alpha, additive blend, smooth alpha, grass blade, blade silhouette, leaf litter, wildflower, flower atlas, pollen, dust mote, particle texture, billboard, crossed quads, wind sway, hue variant, cell, sprite atlas, pixel art, register_def

> One doc per code package; filename matches the package exactly (`docs/systems/procedural.textures.sprites.md` ↔ `fire_engine/procedural/textures/sprites/`).

## Role

`procedural/textures/sprites/` is the **vegetation sprite and particle texture definitions** sub-package of `procedural/textures/`. It registers four built-in sprite/particle texture defs at import time:

- `"grass_tuft"` — 32×32 RGBA pixel-art grass-blade alpha cutout; binary alpha (0 or 255).
- `"dust_mote"` — 32×32 RGBA soft radial dust/pollen speck with smooth alpha falloff for additive blending.
- `"leaf_sprite"` — 32×96 RGBA leaf-litter atlas, three 32×32 cells with three autumn hue variants.
- `"flower_sprite"` — 32×128 RGBA wildflower atlas, four 32×32 cells with four muted hue variants.

The GPU vegetation/flora renderers (`render/grass_renderer.py`, `render/flora_renderer.py`, `render/mote_renderer.py`) map these onto billboard quads. Texture bases sit on the **bottom image row** (V=0 after the texture bridge's upload flip), matching the quads whose V=0 edge is at ground level.

This is a private organisational sub-package of `fire_engine.procedural.textures`; callers should import the parent package or `fire_engine.procedural` rather than this directly.

`procedural/textures/sprites/` does NOT render, upload to Panda3D, or contain shared noise helpers.

## Public API

Exported from `fire_engine.procedural.textures.sprites` (`__init__.py`) as module references:

| Module reference | Registered name | Output | Alpha mode | Notes |
|---|---|---|---|---|
| `grass_tuft` | `"grass_tuft"` | `(32,32,4)` uint8 | Binary (0 or 255) — render with discard | ~9 curved pixel-art blades; blade bases on bottom row. Param: `blades`. |
| `dust_mote` | `"dust_mote"` | `(32,32,4)` uint8 | Smooth radial falloff — additive blend | Soft warm off-white pollen speck; alpha IS the additive mask. |
| `leaf_sprite` | `"leaf_sprite"` | `(32,96,4)` uint8 | Soft rim / near-binary — discard | 3 cells (green/ochre/russet); serrated elliptical silhouette with midrib. |
| `flower_sprite` | `"flower_sprite"` | `(32,128,4)` uint8 | Binary (0 or 255) — discard | 4 cells (off-white/yellow/violet/red); petal rosette + stem + seed-head. |

### Atlas layout

- `"leaf_sprite"`: 3 cells, each 32×32. Cell `k` spans texels `[k*32, (k+1)*32)` in width. Shader samples `u = (k + frac_u) / 3`.
- `"flower_sprite"`: 4 cells, each 32×32. Cell `k` spans `[k*32, (k+1)*32)`. Shader samples `u = (k + frac_u) / 4`.

## Imports Allowed

`procedural/textures/sprites/` may only import:
- Python standard library (`math`, `typing`, ...)
- `numpy`
- `fire_engine.procedural.defs` (for `register_def`)
- `fire_engine.procedural.textures.base` (for `ProceduralTextureDef`, `value_noise`, `pixel_noise`)

**No panda3d imports.** Never import from `render/`, `world/`, `simulation/`, `lighting/`, or any higher layer.

## Events

### Published
None.

### Subscribed
None.

## Units & Invariants

- All defs return `(H, W, 4)` uint8 RGBA.
- `"grass_tuft"` and `"flower_sprite"`: alpha is strictly binary — every texel is either 0 or 255. Render with alpha-test / discard; never alpha-blend.
- `"dust_mote"`: alpha is a smooth falloff for additive blending. RGB is a constant warm off-white; alpha carries the entire intensity. Do NOT render with discard — use additive blend with depth-write off.
- `"leaf_sprite"`: alpha is nearly binary (sharp leaf mask) with a 1-texel soft rim near the edge. Render with a discard threshold (e.g. `alpha < 0.5`) rather than full alpha-blend.
- Blade/stem bases sit on the **bottom image row of the array**. After `render/texture_bridge.py` flips vertically on upload (OpenGL UV origin is bottom-left), the bases land at V=0 in the shader — matching quads whose V=0 edge is at ground level. Do NOT compensate in the generator.
- Determinism: all defs consume the injected `rng` only; same seed → byte-identical textures.
- Per-blade / per-cell iteration is allowed (O(blades) or O(variants), ≤9 or ≤4 iterations). Per-pixel Python loops are prohibited (Hard Rule 4).

## Examples

### Get sprite textures

```python
from fire_engine.core.rng import set_world_seed
from fire_engine.procedural import get
import numpy as np

set_world_seed(1337)
tuft   = get("grass_tuft")        # (32, 32, 4) uint8, binary alpha
mote   = get("dust_mote")         # (32, 32, 4) uint8, smooth alpha
leaves = get("leaf_sprite")       # (32, 96, 4) uint8, 3-cell atlas
flower = get("flower_sprite")     # (32, 128, 4) uint8, 4-cell atlas

# grass_tuft alpha is binary:
assert ((tuft[..., 3] == 0) | (tuft[..., 3] == 255)).all()

# dust_mote centre is brighter than corner:
assert mote[16, 16, 3] > mote[0, 0, 3]

# leaf_sprite variant 0 centre is opaque:
assert leaves[16, 16, 3] == 255
```

### Access a specific atlas cell

```python
# leaf_sprite variant k=1 (ochre):
cell_ochre = leaves[:, 1*32:(1+1)*32, :]   # (32, 32, 4)

# flower_sprite variant k=2 (faded violet):
cell_violet = flower[:, 2*32:(2+1)*32, :]  # (32, 32, 4)
```

## Gotchas

1. **`"dust_mote"` is additive, not cutout.** Do NOT render it with discard — the smooth alpha falloff requires additive blending (depth-write off). Rendering it as a cutout produces a hard ugly circle.
2. **`"grass_tuft"` binary alpha is load-bearing.** The grass renderer discards at `alpha < 0.5` using the full texture; no per-vertex alpha fade. The blades' silhouette IS the geometry.
3. **Vertical flip on upload.** `render/texture_bridge.py` flips sprite textures vertically so their V=0 (bottom array row) aligns with the quad's ground-level V=0 edge in the shader. Never pre-flip in the generator.
4. **Atlas UV contract.** Leaf and flower variant selection is `u = (k + frac_u) / num_variants` in the shader. Adding or removing variants changes the UV scale and breaks the renderers — bump `_VARIANTS` and the relevant shader uniform together.
5. **Import via parent.** `fire_engine.procedural.textures.__init__` imports this sub-package to trigger `@register_def`. Do not import `procedural.textures.sprites` directly in game code.
