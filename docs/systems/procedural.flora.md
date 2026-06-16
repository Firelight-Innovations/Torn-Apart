# procedural.flora — System Doc
keywords: flora, tree, bush, 3D tree, 3-D tree, TreeSpeciesDef, TreeVariantSet, TreeSkeleton, SkeletonBuilder, validate_skeleton, Leaves, leaves_at_tips, TreeMesh, mesh_branches, mesh_leaves, merge_parts, mesh_leaf_area_m2, AtlasLayout, bark_texture, leaf_texture, compose_atlas, rasterize_impostor, impostor_atlas, species script, grow, palettes, branch skeleton, branch, trunk, limbs, twigs, impostor, billboard LOD, far-LOD sprite, canopy, individual leaves, leaf card, cellular automaton, CA, hydration, sway, sway weight, wind, pixel art, atlas, bark, leaf, species, gnarled_oak, dead_tree, scrub_bush, berry_bush, variant pool, variants, pixel_noise, determinism

> One doc per code package; filename matches the package exactly (`docs/systems/procedural.flora.md` ↔ `fire_engine/procedural/flora/`).

## Role

`procedural/flora/` is the **3-D tree and bush generation sub-package** of `procedural/`. It is a pure Python/numpy pipeline — zero panda3d imports, headless and deterministically testable — that converts a species-script recipe into renderable mesh arrays and far-LOD sprite atlases.

The pipeline has five stages:

1. A **species script** (in `flora/species/*.py`) subclasses `TreeSpeciesDef` and uses `SkeletonBuilder` to grow a `TreeSkeleton`: a struct-of-arrays of tapering branch segments rooted at the origin.
2. `leaves_at_tips` runs a cellular automaton (the *Dynamic Trees* leaf rule) seeded at branch tips to grow hundreds of **individual leaves** (`Leaves` struct-of-arrays) — the canopy shape emerges from the wood.
3. `mesh_branches` / `mesh_leaves` / `merge_parts` emit `TreeMesh` arrays in the engine's V3N3T2C4 interleaved vertex layout with per-vertex sway weights baked into `color.a`.
4. `bark_texture` / `leaf_texture` / `compose_atlas` produce the species' 64×64 pixel-art texture atlas (bark left half, single leaf card right half).
5. `rasterize_impostor` / `impostor_atlas` software-rasterise far-LOD billboard sprite cells for every variant — deterministic, no GPU bake needed.

`procedural.get("tree_gnarled_oak")` returns a registry-cached `TreeVariantSet` containing all meshes, the atlas and the impostor atlas for one world seed.

`procedural/flora/` does NOT: render anything, touch the Panda3D scene graph, manage per-instance placement, or do any per-pixel or per-vertex Python loops.

## Public API

All symbols below are re-exported from `fire_engine.procedural.flora` (`__init__.py`).

### Species base classes (`species_def.py`, `types.py`)

| Symbol | Description |
|---|---|
| `TreeSpeciesDef` | Abstract base — subclass, set `name`, `variants`, `BARK_PALETTE`, `LEAF_PALETTE`; implement `grow(rng, variant) -> (TreeSkeleton, Leaves)`. `generate()` drives the full pipeline. |
| `TreeVariantSet` | Registry-cached result: `meshes` (tuple of `TreeMesh`), `atlas` (64×64 uint8), `impostors` (H×W·n uint8), `max_height_m`, `max_radius_m`, `impostor_width_m`, `impostor_height_m`. |

### Skeleton (`skeleton.py`, `types.py`)

| Symbol | Description |
|---|---|
| `TreeSkeleton` | Finalized branch struct-of-arrays: `parent`, `start`/`end` (float32 (S,3) m, Z-up), `radius_start`/`radius_end` (float32 (S,) m), `depth` (int32 (S,)), `sway` (float32 (S,) in [0,1]). Properties: `n_segments`, `sway_start()`, `tip_ids(ids)`. |
| `SkeletonBuilder` | Species-script "node-graph in code": `trunk(height_m, base_radius_m, segments, wobble_m, lean_rad)` → ids; `branches(parents, count, t_range, pitch_set, yaw_mode, length_ratio, ...)` → ids; `skeleton()` finalises arrays + sway. |
| `validate_skeleton(sk, atol)` | Machine-check: every child `start` ON its parent segment, radii taper, sway monotone in [0,1]. Raises `ValueError` on violation. |

### Leaves (`leaves.py`)

| Symbol | Description |
|---|---|
| `Leaves` | Per-leaf struct-of-arrays: `center` (float32 (L,3) m), `radius` (float32 (L,) m), `sway` (float32 (L,) in [0.85,1]). Properties: `n_leaves`, `empty()`. |
| `leaves_at_tips(sk, ids, rng, cell_m, rounds, density, per_cell, leaf_size_m, sway_min, max_leaves)` | Cellular automaton seeded at tip segments: hydration radiates `rounds` cells outward, surviving cells sprout leaf cards. Returns `Leaves`. |

### Mesher (`mesher.py`)

| Symbol | Description |
|---|---|
| `TreeMesh` | V3N3T2C4 mesh arrays: `positions` (float32 (N,3) m), `normals` (float32 (N,3)), `uvs` (float32 (N,2)), `colors` (float32 (N,4) — A = sway weight), `indices` (uint32 (M,)). `height_m`, `radius_m`. |
| `mesh_branches(sk, sides, uv_rect, tint, cap_tips)` | Tapered `sides`-gon prisms for every skeleton segment, flat-shaded. `sides=4` gives the blocky pixel-art look. |
| `mesh_leaves(leaves, rng, uv_rect, tilt_range_rad, size_jitter, tint)` | One small oriented quad per individual leaf; random yaw + tilt for per-leaf Lambert dappling. |
| `merge_parts(*parts)` | Concatenate mesh parts (re-offset indices) into one draw-ready `TreeMesh`. |
| `mesh_leaf_area_m2(mesh)` | Total one-sided leaf area (m²) — leaf triangles identified by atlas UV x >= 0.5. Used by lighting occluders. |

### Atlas (`atlas.py`)

| Symbol | Description |
|---|---|
| `AtlasLayout` | UV contract: `width`, `height`, `bark_rect`, `leaf_rect` — pass `bark_rect`/`leaf_rect` as `mesh_branches`/`mesh_leaves` `uv_rect`. Default 64×64, bark left half, leaf right half. |
| `bark_texture(rng, width, height, palette, striation_freq, streak_px, shade_side)` | Vertically-striated posterised bark, fully opaque (H, W, 4) uint8. |
| `leaf_texture(rng, width, height, palette, hole_thresh, clump_freq, berry_color, berry_density)` | One pixel-art teardrop leaf, binary-alpha (H, W, 4) uint8. |
| `compose_atlas(layout, bark_rgba, leaf_rgba)` | Assemble the 64×64 species atlas: bark into left half, leaf into right half. |

### Impostor (`impostor.py`)

| Symbol | Description |
|---|---|
| `rasterize_impostor(sk, leaves, bark_palette, leaf_palette, rng, cell_wh, hole_thresh, px_per_m)` | Orthographic XZ software-rasterise of skeleton + individual-leaf point cloud → (H, W, 4) uint8, binary alpha. |
| `impostor_atlas(cells)` | Lay variant sprite cells left→right into one (H, W·n, 4) uint8 strip. |

## Imports Allowed

`procedural/flora/` may only import:
- Python standard library (`math`, `dataclasses`, ...)
- `numpy`
- `fire_engine.core` (for `for_domain`)
- `fire_engine.procedural.defs` (for `ProceduralDef`, `register_def`)
- `fire_engine.procedural.textures.base` (for `pixel_noise`)

**No panda3d imports.** Never import from `render/`, `world/`, `simulation/`, `lighting/`, or any higher layer.

## Events

### Published
None. `procedural/flora/` is a pure generation pipeline; it does not publish events.

### Subscribed
None. Tree variant sets are generated on demand via `procedural.get()` — no event triggers.

## Units & Invariants

- **Meters, Z-up, tree-local space.** The trunk base is always at `(0, 0, 0)`; the renderer translates instances onto the terrain.
- **Radii** are half-thickness in meters; the mesher renders a square cross-section of side `2 × radius`.
- **Sway** weights are in `[0, 1]`: 0 at the trunk base, rising monotonically along every branch path to ≈1 at the outermost tips. Baked into `mesh.colors[:, 3]` (not alpha — the tree shader reads it as wind weight).
- **Determinism**: same `(name, world_seed)` always returns a byte-identical `TreeVariantSet`. Per-variant child RNGs are derived deterministically from the registry-injected rng so variant count changes don't affect earlier variants.
- **No per-vertex/per-pixel Python loops.** The per-segment loop in `mesh_branches` is over tens of segments with vectorised numpy inside each call. The per-blade loop in `rasterize_impostor` is bounded (tens of segments). Per-leaf mesh work in `mesh_leaves` is fully vectorised over all L leaves at once.
- `validate_skeleton` is always run by `TreeSpeciesDef.generate` — the "floating canopy" class of bugs is caught before any mesh is built.

## Examples

### Get the gnarled oak variant pool
```python
from fire_engine.core.rng import set_world_seed
from fire_engine.procedural import get

set_world_seed(1337)
oaks = get("tree_gnarled_oak")   # TreeVariantSet, 8 unique meshes
mesh = oaks.meshes[0]            # TreeMesh for variant 0
print(mesh.positions.shape)      # (N, 3) float32 tree-local meters
print(mesh.colors[:, 3].max())   # ≈ 1.0 — leaf sway weights
```

### Author a new species
```python
# fire_engine/procedural/flora/species/my_tree.py
import math, numpy as np
from fire_engine.procedural.defs import register_def
from fire_engine.procedural.flora.leaves import leaves_at_tips
from fire_engine.procedural.flora.skeleton import SkeletonBuilder, TreeSkeleton
from fire_engine.procedural.flora.species_def import TreeSpeciesDef

@register_def
class MyTreeDef(TreeSpeciesDef):
    """Stub species — copy from gnarled_oak.py for a real recipe."""
    name = "tree_my_tree"
    variants = 6
    BARK_PALETTE = np.array([(40, 30, 20), (60, 46, 30), (80, 62, 40)], dtype=np.uint8)
    LEAF_PALETTE = np.array([(30, 45, 22), (55, 75, 38), (80, 100, 50)], dtype=np.uint8)

    def grow(self, rng, variant):
        sb = SkeletonBuilder(rng)
        trunk = sb.trunk(height_m=5.0, base_radius_m=0.25, segments=3)
        limbs = sb.branches(trunk, count=(2, 3), pitch_set=(math.radians(85),))
        sk = sb.skeleton()
        leaves = leaves_at_tips(sk, limbs, rng, rounds=3, density=0.7)
        return sk, leaves

# Add import in fire_engine/procedural/flora/species/__init__.py
# Preview: python tools/preview_tree.py tree_my_tree --obj --png
```

## Gotchas

1. **`grow()` must be overridden.** The base `TreeSpeciesDef.grow()` raises `NotImplementedError` — copy `gnarled_oak.py` as your starting point.
2. **`validate_skeleton` fires on every variant.** A floating-canopy bug (child `start` not on its parent segment) raises `ValueError` immediately in `generate()`. Build `start` from `SkeletonBuilder.branches()` — never compute endpoints manually.
3. **Sway in `colors[:, 3]`, not alpha.** The tree fragment shader reads `color.a` as the wind-sway weight. Renderers that assume `color.a` is transparency will see wrong behaviour — use `world/tree_renderer.py` only.
4. **Per-variant child RNGs are independent.** `TreeSpeciesDef.generate` derives child rngs via `np.random.default_rng(int(grow_seeds[v]))` — one per variant. Consuming the parent rng after `grow_seeds = rng.integers(...)` does NOT destabilize variant 0.
5. **`mesh_leaf_area_m2` uses the UV x >= 0.5 rule.** This matches `AtlasLayout.leaf_rect = (0.5, 0.0, 1.0, 1.0)`. If you change the atlas layout, `mesh_leaf_area_m2` (and lighting occluders) will give wrong results.
6. **Impostor scale is pool-wide.** All variant cells share one `px_per_m` derived from the tallest/widest mesh so the renderer's single billboard quad size fits every variant. Pass `px_per_m=None` only for single-tree previews.
