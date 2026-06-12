# Authoring Tree & Bush Species
keywords: tree, bush, species, authoring, grow, SkeletonBuilder, trunk, branches, pitch_set, yaw_mode, spiral, opposite, random, length_ratio, length_m, length_scale_by_height, radius_ratio, upturn, droop, wobble, lean, leaves_at_tips, cellular automaton, CA, hydration, rounds, cell_m, density, per_cell, leaf_size_m, max_leaves, individual leaves, leaf card, TreeSpeciesDef, TreeVariantSet, BARK_PALETTE, LEAF_PALETTE, berry, BERRY_COLOR, LEAF_HOLE_THRESH, variants, variant pool, preview_tree, OBJ, impostor, determinism, validate_skeleton, floating canopy, species_mix, register_def, dynamic trees, 3D tree, node graph

This is the guide for writing a **new tree or bush species** for Torn Apart.
A species is one Python script in `fire_engine/procedural/flora/species/`
that describes how the plant *grows* — the engine turns it into a pool of
unique 3-D meshes per world, a pixel-art bark/leaf atlas, and far-LOD
impostor billboards, then plants it wherever a zone volume asks for it.

The model is **Unreal's node-graph editor, but in straight Python**: a small
library of helper calls (`trunk`, `branches`, `leaves_at_tips`)
chained however you like, with full `if`/loop/rng freedom between calls.
Trees and bushes are the SAME system — a bush is a species whose trunk is a
0.15 m stub.  Leaves are **individual cards** grown by a cellular automaton
seeded at the branch tips (the Dynamic-Trees leaf rule), all batched into
the variant's single mesh — hundreds of leaves, still one draw.

## The 60-second version

Copy `species/gnarled_oak.py`, change the knobs, register, preview:

```python
# fire_engine/procedural/flora/species/willow_snag.py
import math
import numpy as np
from fire_engine.procedural.defs import register_def
from fire_engine.procedural.flora.leaves import Leaves, leaves_at_tips
from fire_engine.procedural.flora.skeleton import SkeletonBuilder, TreeSkeleton
from fire_engine.procedural.flora.species_def import TreeSpeciesDef

_D = math.radians

@register_def
class WillowSnagDef(TreeSpeciesDef):
    """Drooping half-dead willow, 4–6 m."""
    name = "tree_willow_snag"
    variants = 6

    BARK_PALETTE = np.array([(44, 38, 30), (62, 54, 42), (84, 74, 58)], dtype=np.uint8)
    LEAF_PALETTE = np.array([(38, 48, 28), (54, 66, 36), (74, 86, 44),
                             (96, 106, 54), (118, 126, 66)], dtype=np.uint8)

    def grow(self, rng, variant) -> tuple[TreeSkeleton, Leaves]:
        sb = SkeletonBuilder(rng)
        trunk = sb.trunk(height_m=4.5 + float(rng.uniform(0.0, 1.5)),
                         base_radius_m=0.22, segments=4, wobble_m=0.3)
        limbs = sb.branches(trunk, count=(1, 2), pitch_set=(_D(70), _D(85)),
                            yaw_mode="spiral", length_ratio=(0.4, 0.6),
                            droop_rad=_D(30), segments=2)     # willow droop
        sk = sb.skeleton()
        leaves = leaves_at_tips(sk, limbs, rng,
                                cell_m=0.25, rounds=2, density=0.7)
        return sk, leaves
```

Then:

1. Add `from fire_engine.procedural.flora.species import willow_snag` (and the
   re-export) to `species/__init__.py`.
2. Preview headlessly: `python tools/preview_tree.py tree_willow_snag --obj --png`
   → per-variant OBJs + atlas/impostor PNGs in `tools/out/trees/`.
3. Add a determinism test (copy any species block in `tests/test_tree_species.py`
   — or just add the name to its `SPECIES`/`POOL_SIZES` tables).
4. Plant it: `zone_store.add("trees", lo, hi, params={"species": "tree_willow_snag"})`
   or mix it — `params={"species_mix": "tree_gnarled_oak:3,tree_willow_snag:1"}`.

Naming: `tree_*` species go in `"trees"` volumes, `bush_*` in `"bushes"`
(the prefix is convention; the volume kind picks the config table —
density, spacing, fade distances).

## How a species becomes pixels

```
grow(rng, variant)  ── per variant ──►  TreeSkeleton + Leaves
        │                                    │ validate_skeleton()  (hard fail on floating branches)
        │                                    ▼
        │                    mesh_branches + mesh_leaves → TreeMesh   (pool of `variants` meshes)
palettes(rng) ───────►  bark_texture | leaf_texture → 64×64 species atlas
        └────────────►  rasterize_impostor per variant → impostor strip (far-LOD billboards)
                                     all bundled into a cached TreeVariantSet
```

- `generate()` (the base class) runs all of this; you normally override ONLY
  `grow` (+ optionally `palettes`).  Each variant gets its **own child rng**,
  so variants are mutually distinct but byte-identical per world seed.
- The renderer (`world/tree_renderer.py`) instances the pool over CPU-baked
  placements (`zones/tree_placement.py`) and crossfades to the impostor
  strip past the mesh fade distance.  You never touch panda3d.

## SkeletonBuilder — the helper library

Everything is **meters and radians**, Z-up, tree-local with the trunk base
at the origin.  Use `math.radians` (the species scripts alias it `_D`).

### `sb.trunk(...) -> ids`

| Knob | Meaning |
|---|---|
| `height_m` | total trunk height. **Bush = `height_m≈0.15, segments=1`.** |
| `base_radius_m`, `tip_radius_m` | half-thickness at ground / top (taper). `tip_radius_m=None` → 25 % of base. |
| `segments` | stacked sub-segments — more = wigglier trunk (3–5 for trees). |
| `wobble_m` | per-joint sideways meander (0 = arrow-straight, 0.35 = gnarled). |
| `lean_rad` | whole-trunk lean off vertical. |

### `sb.branches(parents, ...) -> ids`

Grows children ON the segments in `parents` (the array a previous call
returned).  Children *start on their parent by construction* — a floating
canopy is structurally impossible, and `validate_skeleton` double-checks.

| Knob | Meaning |
|---|---|
| `count=(lo, hi)` | branches per parent segment (inclusive range; `(0, 2)` gives sparse/dead looks). |
| `t_range=(0.45, 0.95)` | where along the parent they sprout (0 base → 1 tip). |
| `pitch_set=(angle, ...)` | **the signature knob** — the set of angles off the parent's axis each branch picks from. `(90°,)` = the blocky Dynamic-Trees right angle; `(80°, 95°)` = gnarled tiers; `(35°, 55°)` = upswept bush stems. |
| `pitch_jitter_rad` | random spread around the picked pitch. |
| `yaw_mode` | `"spiral"` (golden-angle around the trunk — natural even fill), `"opposite"` (paired 180°), `"random"`. |
| `length_ratio=(lo, hi)` | child length as a fraction of the parent's chain length — OR `length_m=(lo, hi)` for absolute meters (bush stems). |
| `length_scale_by_height=(base, crown)` | multiplier ramp by sprout height — `(1.0, 0.45)` makes crown branches half as long (the classic tapering silhouette). |
| `radius_ratio`, `min_radius_m` | child thickness vs parent (radii can never grow — validated). |
| `upturn_rad` / `droop_rad` | tip curl up (living reach) / down (willow, dead weight). |
| `bend_rad`, `segments` | per-sub-segment crook + how many sub-segments per branch. |

Returns ALL created sub-segment ids — feed them back in as the next level's
`parents`, or into `leaves_at_tips`.

### `leaves_at_tips(sk, ids, rng, ...)` — the leaf cellular automaton

Grows INDIVIDUAL leaves around the **tip** segments among `ids` (segments
nothing grew from).  Each tip seeds "hydration" into a coarse cell grid;
each CA round hydration spreads one cell to the 6 axis neighbours losing a
level, then every surviving cell sprouts leaf cards — so the canopy shape
IS the branch structure, voxel-grown like Dynamic Trees.

| Knob | Meaning |
|---|---|
| `cell_m` (0.25) | CA cell edge — the canopy's "leaf voxel" size. Smaller = tighter packing. |
| `rounds` (3) | hydration radius in cells.  **`rounds=1` = a single-cell tuft** (no spread — seeds start at hydration 1); 2 = a hand-sized clump; 3 = a ~0.8 m dome per tip. |
| `density` (0.6) | base leaf probability per hydrated cell, falling toward the canopy rim — interiors fill, silhouettes stay ragged. |
| `per_cell` (1, 2) | leaves per surviving cell. |
| `leaf_size_m` (0.09, 0.14) | per-leaf half-size — cards are 2× this across. |
| `sway_min` (0.85) | wind-weight floor (foliage always rides gusts harder than wood). |
| `max_leaves` (600) | deterministic thinning cap — YOUR vertex budget lever (4 verts/leaf). |

A leafless species returns `Leaves.empty()`.  The mesher gives every leaf
its own upward-biased random orientation, so sunlight dapples the canopy
leaf by leaf.

### `sb.skeleton() -> TreeSkeleton`

Finalize.  Computes per-segment **sway weights** (cumulative path length /
longest path — trunk base pinned at 0, outer tips ≈1).  The mesher bakes
these into vertex `color.a`; the shader squares them so trunks stand firm
while canopies bend.  You never set sway by hand.

## Class-attribute knobs (texture side)

| Attr | Default | Meaning |
|---|---|---|
| `variants` | 6 | mesh-pool size per world (oak uses 8). More = less repetition, more VRAM. |
| `BARK_PALETTE` | brown ramp | 3 colours dark→lit; bark is vertical striations, left half shaded a tier darker. |
| `LEAF_PALETTE` | olive ramp | 5 colours dark→pale; the single-leaf texture posterises through it (midrib a tier darker, right side a tier lighter). |
| `LEAF_HOLE_THRESH` | 0.18 | leaf-edge raggedness — higher = more chewed/dying (dead tufts use 0.30). |
| `BERRY_COLOR` / `BERRY_DENSITY` | None / 0.0 | speckle pass over the leaf texture (`bush_berry` uses `(168, 86, 72)` / `0.035`). |
| `TINT_RANGE` | (0.92, 1.08) | per-variant brightness jitter baked into vertex colours. |
| `impostor_cell` | (64, 96) | impostor raster cell px (w, h) — bushes use (48, 48). |

Override `palettes(rng) -> (bark, leaf)` for per-world colour drift
(`berry_bush.py` shifts its greens ±8 per world).

## The 4 built-in species (reference reading order)

| Script | Demonstrates |
|---|---|
| `species/gnarled_oak.py` | the canonical tree: tiered near-90° limbs, spiral yaw, `length_scale_by_height` crown taper, twig level, `rounds=3` CA canopy |
| `species/dead_tree.py` | leafless/sparse: `count=(0, 2)`, wide `pitch_set`, droop + bend, hand-thinned tip subset + `rounds=2` micro-tufts, some variants `Leaves.empty()` |
| `species/scrub_bush.py` | the bush pattern: stub trunk, splayed absolute-length stems (`length_m`), random yaw, small-cell CA |
| `species/berry_bush.py` | texture customisation: berry speckles, `palettes()` override, high-density CA dome |

## Checklist before you ship a species

1. **Determinism** — never `random.*` or unseeded numpy; use ONLY the `rng`
   passed to `grow`/`palettes` (Hard Rule 2). Same world seed must reproduce
   the species byte-for-byte (`tests/test_tree_species.py` style test).
2. **`validate_skeleton` passes** — `generate()` runs it per variant and
   raises on floating branches / growing radii; if your script trips it,
   you edited arrays by hand instead of using the builder.
3. **Scale sanity** — trees 3–8 m tall, ≤ ~0.35 m base radius; bushes ≤ 1.5 m.
   The renderer pads culling bounds from `max_height_m`/`max_radius_m`, so
   a 40 m monster "works" but wrecks culling and the impostor raster scale.
4. **Budget** — keep a variant under ~2 500 vertices (oak peaks ≈ 2 480).
   Vertex count ≈ `segments × 16 + leaves × 4`; `max_leaves` is the lever.
   Instanced thousands of times, it adds up.
5. **Preview** — `python tools/preview_tree.py <name> --obj --png`; open a
   couple of OBJs, check the atlas reads at 64×64 and every impostor cell
   has a grounded trunk.
6. **Register + export** — `@register_def` on the class, import line +
   re-export in `species/__init__.py`. Done — `species_mix` params can use
   it immediately, no engine changes.

## Gotchas

- `grow` is called once per variant with a variant-specific rng — draw all
  randomness inside, never at module scope.
- `branches` returns sub-segment ids of EVERY branch it grew (a 2-segment
  branch contributes 2 ids).  `leaves_at_tips` filters to tips itself, so
  passing the whole array is correct and idiomatic.
- **`rounds=1` does not spread** — seeds start at hydration `rounds`, and a
  neighbour gets `rounds − 1`; at 1 that's zero, so you get one cell per
  tip.  Use `rounds=2` with a small `cell_m` for visible tufts.
- Bark UVs live in the atlas's left half, the single leaf in the right —
  handled by `mesh_branches`/`mesh_leaves` defaults; don't pass custom
  `uv_rect`s unless you also change `AtlasLayout`.
- Don't call `registry.get()` from inside a species (cache recursion);
  compose helpers like `bark_texture` directly — they consume your rng.
- The impostor is rasterised from the SAME skeleton/leaves/palettes as
  the mesh (the leaf point-cloud scatters as dilated dots), with one shared
  meters-per-pixel across the whole pool — that's what makes the LOD
  crossfade seamless.  No work needed, just don't rasterise variants
  yourself.
