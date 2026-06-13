# procedural â€” System Doc
keywords: procedural, ProceduralDef, ProceduralTextureDef, register, get, clear_cache, reset_registry, value_noise, pixel_noise, wasteland_ground, night_sky, rain_streak, grass_ground, dirt_ground, grass, dirt, pixel art, pixelated, pixel_noise, posterize, posterise, palette, stars, star field, galaxy, equirect, equirectangular, sky texture, rain texture, tileable, seamless, texture, noise, registry, cache, determinism, world_seed, params_digest, for_domain, biome, building, content, authoring, RGBA, uint8, octaves, persistence, lacunarity, base_freq, layered, heightmap, moon_surface, moon, crater, maria, lunar, normal map, emission map, derive_normal_map, flat_normal_map, black_emission_map, sobel, maps, ground_lut, build_ground_lut, palette LUT, GRASS_PALETTE, DIRT_PALETTE, GRASS_THRESHOLDS, DIRT_THRESHOLDS, ground palette, posterised ramp, world-space ground, flower_sprite, plaster_wall, plaster, wall texture, building texture, render, lime wash, PLASTER_PALETTE, PLASTER_THRESHOLDS, flower, bush, shrub, tree, flora, sprite atlas, atlas, variants, wildflower, canopy, trunk, alpha cutout, 3D tree, dynamic trees, TreeSpeciesDef, TreeVariantSet, TreeSkeleton, SkeletonBuilder, branches, leaves, Leaves, leaves_at_tips, cellular automaton, CA, hydration, individual leaves, leaf card, TreeMesh, mesh_branches, mesh_leaves, merge_parts, validate_skeleton, bark_texture, leaf_texture, compose_atlas, AtlasLayout, rasterize_impostor, impostor_atlas, impostor, billboard LOD, species, grow, palettes, tree_gnarled_oak, tree_dead, bush_scrub, bush_berry, gnarled oak, dead tree, scrub bush, berry bush, variant pool, sway weight, pitch_set, yaw_mode, spiral, preview_tree

> One doc per code package; filename matches the package exactly (`docs/systems/procedural.md` â†” `fire_engine/procedural/`).

## Role

`procedural/` is the **content-authoring Foundation layer** â€” pure Python/numpy, zero panda3d imports, callable from every other layer.  It provides:

- A **`ProceduralDef` base class** and registry so any layer can request generated content by name without knowing which module produced it.
- A **`ProceduralTextureDef` domain subclass** that generates `(H, W, 4) uint8` RGBA arrays.
- **Shared layered noise helpers** (`value_noise` for smooth fields; `pixel_noise` for crisp pixel-art block noise) used by textures and reusable by Phase 3 terrain.
- The **`procedural/flora/` subpackage** — 3-D tree/bush generation: branch skeletons, leaf clusters, mesh building, species atlases and far-LOD impostor sprites, all per-species Python scripts on a `TreeSpeciesDef` base.  See `docs/content/tree_species_authoring.md` for the authoring guide.
- A **deterministic cache** keyed by `(def_name, world_seed, sorted_params_digest)` so identical calls always return the same object.
- Auto-registration of built-in content at import time (textures: `"wasteland_ground"`, `"night_sky"`, `"rain_streak"`, `"grass_ground"`, `"dirt_ground"`, `"moon_surface"`, `"grass_tuft"`, `"dust_mote"`, `"leaf_sprite"`, `"flower_sprite"`, `"plaster_wall"`; tree species: `"tree_gnarled_oak"`, `"tree_dead"`, `"bush_scrub"`, `"bush_berry"`).

`procedural/` deliberately does NOT: render anything, touch the Panda3D scene graph, store game-world state, or do any per-pixel Python loops.

The bridge from procedural arrays to Panda3D Texture lives in `world/texture_bridge.py` (the only coupling point).

## Public API

All symbols below are re-exported from `fire_engine.procedural` (`__init__.py`).

### Base classes (`procedural/defs.py`, `procedural/textures/base.py`)

| Symbol | Description |
|---|---|
| `ProceduralDef` | Abstract base class for all content defs.  Set `name` (str), override `generate(rng, **params)`. |
| `ProceduralTextureDef` | `ProceduralDef` subclass; `generate` returns `np.ndarray (H,W,4) uint8`. |
| `register_def` | Class decorator: instantiates the class and registers it immediately at import time. |

### Registry (`procedural/registry.py`)

| Symbol | Description |
|---|---|
| `register(def_instance)` | Register a `ProceduralDef` instance by `def_instance.name`. |
| `get(name, **params)` | Generate (or return cached) content for the named def. |
| `clear_cache()` | Flush the generated-result cache; def registry is preserved. |
| `reset_registry()` | Flush both the def registry and the cache (tests only). |

### Noise helpers (`procedural/textures/base.py`)

| Symbol | Description |
|---|---|
| `value_noise(rng, shape, octaves, persistence, lacunarity, base_freq)` | Layered 2-D value noise â†’ `float32 (H,W)` in `[0,1]`. Bilinear upsampling â€” smooth, no visible texels. Use for heightmaps, fog density, continuous fields. Pure numpy. |
| `pixel_noise(rng, shape, octaves, persistence, lacunarity, base_freq)` | Layered 2-D pixel (nearest-neighbour) noise â†’ `float32 (H,W)` in `[0,1]`. Each octave upsampled with integer index math (`np.repeat`-equivalent) â€” crisp block-edged texels, retro pixel-art look. Use for ground textures, posterised surfaces. Pure numpy. |

### Built-in texture definitions

| Registered name | File | Output |
|---|---|---|
| `"wasteland_ground"` | `procedural/textures/wasteland_ground.py` | `(256,256,4) uint8` RGBA dirt/dead-grass (smooth `value_noise`, bilinear) |
| `"night_sky"` | `procedural/textures/night_sky.py` | `(512,1024,4) uint8` equirect star field + galaxy band (+Z pole at v=1, U-seamless, alpha = luminance for additive blending).  Params: `width`, `height`, `star_count` (pass `Config.sky_star_count`). |
| `"rain_streak"` | `procedural/textures/rain_streak.py` | `(512,128,4) uint8` sparse vertical rain streaks, tileable in U and V, alpha = streak intensity.  Params: `width`, `height`, `streak_count`. |
| `"grass_ground"` | `procedural/textures/grass_ground.py` | `(64,64,4) uint8` crisp pixel-art weathered grass; 8-colour posterised palette; `pixel_noise`-based.  Params: `width`, `height`.  Exports `GRASS_PALETTE` / `GRASS_THRESHOLDS` (the colour ramp) for the GPU ground LUT. |
| `"dirt_ground"` | `procedural/textures/dirt_ground.py` | `(64,64,4) uint8` crisp pixel-art dry dirt with dark clod clusters; 6-colour posterised palette; `pixel_noise`-based.  Params: `width`, `height`.  Exports `DIRT_PALETTE` / `DIRT_THRESHOLDS`. |
| `"moon_surface"` | `procedural/textures/moon_surface.py` | `(256,256,4) uint8` lunar disc: pale regolith + dark maria + vectorised crater field (bright rims, shadowed floors); alpha 255 inside the unit disc.  Seeded per world — every world grows a different moon.  Params: `size`, `crater_count`.  Sampled disc-locally by the sky-dome shader; phase terminator applied dynamically (not baked). |
| `"grass_tuft"` | `procedural/textures/grass_tuft.py` | `(32,32,4) uint8` pixel-art grass-blade silhouette with **binary alpha** (255 on blades, 0 elsewhere — render with discard, never blend).  ~9 leaning blades, dark base → pale dried tip, blade bases on the bottom image row (V=0 after the upload flip).  Mapped onto the GPU grass tuft quads (`world/grass_renderer.py`).  Params: `width`, `height`, `blades`. |
| `"flower_sprite"` | `procedural/textures/flower_sprite.py` | `(32,128,4) uint8` wildflower **atlas** — one row of 4 hue variants (yarrow white / dusty yellow / faded violet / washed red), each a leaning stem + leaf nubs + chunky petal rosette around a dark seed-head.  Binary alpha; stem bases on the bottom row.  Cell `k` samples `u = (k + frac_u) / 4`.  Mapped onto the flora crossed quads (`world/flora_renderer.py`).  Params: `width`, `height`. |
| `"plaster_wall"` | `procedural/textures/plaster_wall.py` | `(64,64,4) uint8` weathered lime-plaster wall albedo for buildings — low-contrast off-white with subtle tonal drift, hairline cracks and flake pocks; 6-colour posterised palette; `pixel_noise`-based.  Sampled as the building wall/slab albedo (`world/building_renderer.py`); shader falls back to flat albedo if absent.  Params: `width`, `height`.  Exports `PLASTER_PALETTE` / `PLASTER_THRESHOLDS`. |
(Tree and bush sprite defs were retired when 3-D trees shipped — billboards survive only as the flora subpackage's far-LOD impostors below.)

### 3-D tree/bush generation (`procedural/flora/`)

Species are **Python scripts** subclassing `TreeSpeciesDef` (the "node graph in code" model — full authoring guide with annotated examples: `docs/content/tree_species_authoring.md`).  `get("tree_gnarled_oak")` returns a `TreeVariantSet`: a per-world-seed **pool of unique meshes** (oak 8, others 6) + one bark/leaf atlas + one impostor sprite strip; instances draw from the pool (`world/tree_renderer.py`).

| Symbol | File | Description |
|---|---|---|
| `TreeSkeleton` | `flora/skeleton.py` | Branch-segment SoA: `parent`, `start`/`end` (tree-local m, Z-up, base at origin), `radius_start`/`radius_end`, `depth`, `sway` (0 trunk base → ≈1 tips). |
| `SkeletonBuilder` | `flora/skeleton.py` | The species-script helper library: `trunk(height_m, base_radius_m, segments, wobble_m, lean_rad)` → ids; `branches(parents, count, t_range, pitch_set, yaw_mode="spiral"\|"opposite"\|"random", length_ratio\|length_m, length_scale_by_height, radius_ratio, upturn_rad, droop_rad, bend_rad, segments)` → ids; `skeleton()` finalises + computes sway. |
| `validate_skeleton(sk)` | `flora/skeleton.py` | Machine check: every child `start` ON its parent segment (the floating-canopy bug class), radii non-increasing, sway monotone in [0,1].  `TreeSpeciesDef.generate` runs it on every variant. |
| `Leaves`, `leaves_at_tips(sk, ids, rng, cell_m, rounds, density, per_cell, leaf_size_m, max_leaves, ...)` | `flora/leaves.py` | INDIVIDUAL leaves grown by a cellular automaton seeded at branch tips: hydration spreads `rounds − 1` cells from each tip, surviving cells sprout leaf cards — the canopy shape emerges from the wood (Dynamic-Trees leaf rule). |
| `TreeMesh`, `mesh_branches(sk, sides=4, uv_rect, tint)`, `mesh_leaves(leaves, rng, tilt_range_rad, ...)`, `merge_parts(*parts)` | `flora/mesher.py` | Tapered square prisms + one small upward-biased-oriented quad PER LEAF → V3N3T2C4 arrays (`geometry_bridge.to_geom`-compatible); hundreds of leaves batch into the variant's single mesh.  **`colors[:,3]` is the per-vertex sway weight**, not alpha. |
| `AtlasLayout`, `bark_texture`, `leaf_texture`, `compose_atlas` | `flora/atlas.py` | 64×64 species atlas: left half tileable bark (opaque), right half ONE binary-alpha pixel-art leaf (teardrop + midrib) every leaf card cuts out; rng-consuming helpers, palette-driven. |
| `rasterize_impostor(sk, leaves, ...)`, `impostor_atlas(cells)` | `flora/impostor.py` | Headless software-rasterised far-LOD billboard cells (XZ orthographic; branch capsules + the leaf point-cloud scattered as dilated dots; deterministic, pytest-testable). |
| `TreeSpeciesDef`, `TreeVariantSet` | `flora/species_def.py` | Base class (override `grow(rng, variant)` + optionally `palettes(rng)`; class attrs `variants`, palettes, berry/hole knobs) and the cached result bundle (`meshes`, `atlas`, `impostors`, `max_height_m`, `max_radius_m`, `impostor_width_m/height_m`). |
| `GnarledOakDef` `"tree_gnarled_oak"` (8 variants), `DeadTreeDef` `"tree_dead"`, `ScrubBushDef` `"bush_scrub"`, `BerryBushDef` `"bush_berry"` | `flora/species/*.py` | The built-in species scripts — also the reference content for authoring new ones. |

Preview workflow: `python tools/preview_tree.py <species> --obj --png` dumps per-variant OBJs + atlas/impostor PNGs to `tools/out/trees/`.

### Derived maps (`procedural/maps.py`)

| Symbol | Description |
|---|---|
| `derive_normal_map(rgba, strength=1.4) -> (H,W,4) uint8` | Tangent-space normal map from a texture's luminance gradient (wrap-padded Sobel; brightness-as-height).  Used by main.py to build the GPU terrain shader's per-material normal maps — no hand-authored maps, keeping textures 100 % procedural. |
| `flat_normal_map(size=4)` | All-(128,128,255) flat normal map (placeholder stage). |
| `black_emission_map(size=4)` | All-black emission map (placeholder stage). |

### Ground palette LUT (`procedural/textures/ground_lut.py`)

| Symbol | Description |
|---|---|
| `build_ground_lut(entries, levels=256) -> (rows, levels, 4) uint8` | Bake a per-material posterised colour ramp into a lookup texture: `entries` maps a **material id** to `(palette, thresholds)` (the constants the ground defs export); row `m`, column `v` holds the colour `_posterise` assigns to noise value `(v+0.5)/levels`.  `rows = max(material_id)+1`; alpha always 255.  Uses the same `searchsorted(..., side="right")` rule as the defs, so the GPU shader's world-space procedural ground matches the baked previews bucket-for-bucket.  Uploaded via `world/texture_bridge.to_field_texture` and bound as `u_ground_lut` (see `world/terrain_shader.apply_terrain_shader`).  Pure numpy/headless. |

## Imports Allowed

`procedural/` may only import:
- Python standard library (`hashlib`, `abc`, ...)
- `numpy`
- `fire_engine.core` (for `for_domain`, `set_world_seed`)

**No panda3d imports.** Never import from `world/`, `terrain/`, `lighting/`, or any higher layer.

## Events

### Published
None.  `procedural/` is a pure function layer; it does not publish events.

### Subscribed
None.  Content generation is triggered by direct `get()` calls, not events.

## Units & Invariants

### Determinism Guarantee
`get(name, **params)` with the same `(name, world_seed, params)` tuple always produces **byte-identical** output across separate Python processes and interpreter restarts.  This is guaranteed because:
1. `core.rng.for_domain("procedural", name, params_digest)` is cross-process deterministic (blake2b, not `hash()`).
2. `value_noise` uses only numpy array operations seeded entirely from the injected `rng`.
3. The registry cache key includes `world_seed` (read at call time from `core.rng._world_seed`).

### Cache Key
```
(name: str, world_seed: int, params_digest: str)
```
`params_digest` is `blake2b-8(repr(sorted(params.items()))).hex()` â€” stable across processes.

If `set_world_seed()` is called with a new seed between two `get()` calls, the cache key differs and fresh content is generated.

### Texture Invariants
All `ProceduralTextureDef.generate()` implementations must produce:
- Shape `(H, W, 4)`, dtype `uint8`.
- Channel order: RGBA (red=0, green=1, blue=2, alpha=3).
- Alpha channel = 255 (fully opaque) unless the definition explicitly documents partial transparency.  Documented exceptions: `"night_sky"` (alpha = luminance â€” additive-blend mask) and `"rain_streak"` (alpha = streak intensity).

### No Per-Pixel Python Loops
`value_noise` and all built-in texture defs use only numpy array expressions.  A Python loop over individual pixels is a correctness violation (performance cliff at 256Â² = 65 k iterations, much worse at 512Â²).

## Examples

### Get a pre-registered texture
```python
from fire_engine.core.rng import set_world_seed
from fire_engine.procedural import get
import numpy as np

set_world_seed(1337)
arr = get("wasteland_ground")      # np.ndarray (256, 256, 4) uint8
assert arr.shape == (256, 256, 4)
assert arr.dtype == np.uint8
assert (arr[..., 3] == 255).all()  # fully opaque
```

### Cache identity
```python
arr1 = get("wasteland_ground")
arr2 = get("wasteland_ground")
assert arr1 is arr2               # same cached object

from fire_engine.procedural import clear_cache
clear_cache()
arr3 = get("wasteland_ground")
assert arr3 is not arr1           # freshly generated after cache clear
```

### Use value_noise directly (Phase 3 terrain heightmap pattern)
```python
from fire_engine.core.rng import set_world_seed, for_domain
from fire_engine.procedural import value_noise
import numpy as np

set_world_seed(42)
rng = for_domain("terrain", "height", (cx, cy))  # per-chunk RNG
heights = value_noise(rng, shape=(32, 32), octaves=6, base_freq=2)
# Scale to amplitude in voxels: heights * 48  (â‰ˆ24 m at 0.5 m/voxel)
assert heights.shape == (32, 32)
assert heights.dtype == np.float32
assert 0.0 <= heights.min() and heights.max() <= 1.0
```

### Author a new texture (AI-agent authoring guide)
```python
# 1. Create fire_engine/procedural/textures/cracked_rock.py:

import numpy as np
from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, value_noise

@register_def
class CrackedRockDef(ProceduralTextureDef):
    """256Ã—256 grey cracked rock texture."""
    name = "cracked_rock"

    def generate(self, rng: np.random.Generator, **params) -> np.ndarray:
        W = int(params.get("width",  256))
        H = int(params.get("height", 256))
        # Two noise fields: base tone + crack pattern
        base  = value_noise(rng, (H, W), octaves=4, base_freq=4)
        crack = value_noise(rng, (H, W), octaves=6, persistence=0.3, base_freq=8)
        # Map to grey rock palette
        grey = (base * 120 + 60).astype(np.float32)     # [60..180]
        darken = (1.0 - crack * 0.4)                     # cracks darken
        rgb = np.clip(grey[..., None] * darken[..., None], 0, 255)
        rgba = np.empty((H, W, 4), dtype=np.uint8)
        rgba[..., :3] = rgb.astype(np.uint8)
        rgba[..., 3] = 255
        return rgba

# 2. Import in fire_engine/procedural/textures/__init__.py:
#    from fire_engine.procedural.textures import cracked_rock

# 3. Add determinism test in tests/test_procedural.py.

# 4. Preview: python tools/preview_texture.py cracked_rock
```

### Bridge to Panda3D (in world/ only)
```python
# Only in world/ â€” do not call this from procedural/ or tests/
from fire_engine.procedural import get
from fire_engine.world.texture_bridge import to_panda_texture

arr = get("wasteland_ground")      # (256,256,4) uint8
tex = to_panda_texture(arr)        # panda3d.core.Texture, nearest-neighbour
node_path.set_texture(tex)
```

## Gotchas

1. **Import order for registration**: `@register_def` calls `registry.register()` at decoration time (module import).  The texture module must be imported before `get()` is called.  `fire_engine/procedural/__init__.py` does this automatically by importing `fire_engine.procedural.textures`, which in turn imports each texture module (`wasteland_ground`, `night_sky`, `rain_streak`, `grass_ground`, `dirt_ground`).  If you add a new texture module, add an import line in `procedural/textures/__init__.py`.

2. **`reset_registry()` is tests-only**: it drops all registered defs.  After calling it, you must re-register any defs you need (or re-import the package).  Never call it in production code.

3. **Cache key includes world seed**: if you call `set_world_seed()` between two `get()` calls without `clear_cache()`, the old cached entry is *not* returned (the key differs).  Stale entries accumulate until `clear_cache()` is called.  This is intentional: it prevents cross-seed contamination.

4. **`value_noise` consumes `octaves` draws from `rng`**: each octave calls `rng.random((freq_h+1, freq_w+1))`.  The same `rng` object passed to a texture def must not be used for other purposes before or after calling `value_noise`, or determinism breaks within that def.  The registry always provides a fresh `rng` per `get()` call, so this is only relevant if you call `value_noise` outside the registry (e.g. directly in terrain generation with a `for_domain` RNG).

5. **No per-pixel loops**: even `dtype` conversions like `.astype(np.uint8)` are fine; it is the Python `for pixel in array` pattern that is banned.  numpy ufuncs, indexing, broadcasting â€” all fine.

6. **Panda3D vertical flip**: `world/texture_bridge.py` flips the array vertically before upload (OpenGL UV origin is bottom-left).  If a texture looks upside-down in-engine but correct in the PNG preview, the flip is the cause â€” it is correct behaviour, not a bug.
