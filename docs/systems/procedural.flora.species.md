# procedural.flora.species -- System Doc
keywords: flora species, built-in species, tree species, bush species, GnarledOakDef, DeadTreeDef, ScrubBushDef, BerryBushDef, tree_gnarled_oak, tree_dead, bush_scrub, bush_berry, gnarled oak, dead tree, scrub bush, berry bush, species script, grow, palettes, BARK_PALETTE, LEAF_PALETTE, BERRY_COLOR, BERRY_DENSITY, register_def, variant, trunk, limbs, twigs, stems, SkeletonBuilder, leaves_at_tips, along-wood leaves, leaves_per_m, max_leaves, twiglet, sub-stem, twig level, stub trunk, bush path, leafless, near-leafless, see-through foliage, wasteland, post-apocalyptic, pixel art species, species authoring, reference species

> One doc per code package; filename matches the package exactly (`docs/systems/procedural.flora.species.md` -- `fire_engine/procedural/flora/species/`).

## Role

`procedural/flora/species/` contains the **built-in tree and bush species scripts** -- one file per species.  Each file subclasses `TreeSpeciesDef`, sets class-attribute palette and geometry knobs, implements `grow(rng, variant)` with a `SkeletonBuilder` recipe, and decorates the class with `@register_def` so the species is registered in the procedural registry at import time.

Importing this package (which `fire_engine.procedural.flora.__init__` does automatically) registers all four built-in species:

| Registered name | Class | Description |
|---|---|---|
| `"tree_gnarled_oak"` | `GnarledOakDef` | 5-7 m crooked oak, blocky tiered limbs (limbs → twigs → fine twiglets), full dense olive crown (~1.3-1.4 k leaf cards/variant).  8 variants per world. |
| `"tree_dead"` | `DeadTreeDef` | 6-9 m bare snag, sparse drooping limbs, at most two small dry leaf tufts.  6 variants per world. |
| `"bush_scrub"` | `ScrubBushDef` | ~1 m dry scrub with 4-7 splayed stems + short sub-stems and dusty, deliberately see-through olive foliage.  6 variants per world. |
| `"bush_berry"` | `BerryBushDef` | Full living-green dome (stems + fine sub-stems) speckled with washed-red berries.  6 variants per world. |

These files also serve as **reference examples** for authoring new species (see `gnarled_oak.py` -- every knob is annotated; full guide: `docs/content/tree_species_authoring.md`).

This package does NOT: grow any meshes or textures at import time (generation is deferred to `get()` calls), store any game state, or render anything.

## Public API

| Symbol | File | Description |
|---|---|---|
| `GnarledOakDef` | `gnarled_oak.py` | Wasteland oak -- crooked trunk + near-90-degree limbs (`pitch_set=(80, 95) degrees`) + upturned twigs + a 3rd fine **twiglet** level, all foliated along the wood into a dense crown (`leaves_per_m=90, max_leaves=1200`).  The annotated reference species for authoring. |
| `DeadTreeDef` | `dead_tree.py` | Bare snag -- tall lean trunk + sparse drooping/bent limbs + optional 0-2 small dry leaf tufts (`max_leaves=36`; `rounds=2, density=0.7` is a legacy-compatible call -- those CA knobs are accepted but ignored).  Demonstrates the near-leafless path. |
| `ScrubBushDef` | `scrub_bush.py` | Scrub bush -- stub trunk (`height_m=0.15`) + splayed stems using `length_m` (absolute, not ratio) + short sub-stems; foliage kept see-through.  Demonstrates the bush path. |
| `BerryBushDef` | `berry_bush.py` | Berry bush -- dome of upcurled stems + fine sub-stems foliated into a full dome + `BERRY_COLOR` / `BERRY_DENSITY` leaf speckles + per-world `palettes()` hue drift. |

## Imports Allowed

`procedural/flora/species/` may only import:
- Python standard library (`math`)
- `numpy`
- `fire_engine.procedural.defs` (for `register_def`)
- `fire_engine.procedural.flora.leaves` (for `Leaves`, `leaves_at_tips`)
- `fire_engine.procedural.flora.skeleton` (for `SkeletonBuilder`, `TreeSkeleton`)
- `fire_engine.procedural.flora.species_def` (for `TreeSpeciesDef`)

**No panda3d imports.**  Never import from `render/`, `world/`, `simulation/`, `lighting/`, or any higher layer.

## Events

### Published
None.  Species scripts are pure data/logic definitions; they publish no events.

### Subscribed
None.  Registration happens at import time; generation is triggered by `procedural.get()`.

## Units & Invariants

- All length arguments are in **meters**.  Trunk heights are world-scale (oak 5-7 m; snag 6-9 m; bushes 0.5-1 m).
- `grow(rng, variant)` must consume `rng` (a `numpy.random.Generator`) for ALL randomness -- never call `random.*` or unseeded `np.random.*` (Hard Rule 2).  The registry injects a per-variant child rng derived deterministically from the world seed.
- `grow` must return `(TreeSkeleton, Leaves)`.  Leafless species return `Leaves.empty()`.
- `@register_def` at class decoration time registers the species by `name`.  A typo in `name` creates a wrong registry key silently -- verify with `procedural.get(name)` in the determinism test.
- **`BARK_PALETTE` / `LEAF_PALETTE`**: `uint8 (T, 3)` arrays, shadow tone first.  T controls the number of posterisation tiers.  Palettes override the base class defaults per species.
- **Bush path**: a bush uses `height_m = 0.12-0.15`, `segments=1`, and `length_m=(min, max)` (absolute stem lengths) instead of `length_ratio`, because ratios of a 10-15 cm trunk are meaningless.

## Examples

### Get a built-in species

```python
from fire_engine.core.rng import set_world_seed
from fire_engine.procedural import get

set_world_seed(42)
oaks = get("tree_gnarled_oak")    # TreeVariantSet, 8 unique meshes
snags = get("tree_dead")          # TreeVariantSet, 6 unique meshes
scrub = get("bush_scrub")         # TreeVariantSet, 6 unique meshes
berries = get("bush_berry")       # TreeVariantSet, 6 unique meshes
# Preview any of these:
# python tools/preview_tree.py tree_gnarled_oak --obj --png
```

### Add a new species

```python
# 1. Create fire_engine/procedural/flora/species/my_pine.py:
import math
import numpy as np
from fire_engine.procedural.defs import register_def
from fire_engine.procedural.flora.leaves import Leaves, leaves_at_tips
from fire_engine.procedural.flora.skeleton import SkeletonBuilder, TreeSkeleton
from fire_engine.procedural.flora.species_def import TreeSpeciesDef

@register_def
class MyPineDef(TreeSpeciesDef):
    """Prototype pine -- straight trunk, spiralled limbs, sparse foliage."""
    name = "tree_my_pine"
    variants = 6
    BARK_PALETTE = np.array([(42, 32, 22), (60, 48, 33), (80, 64, 45)], dtype=np.uint8)
    LEAF_PALETTE = np.array([(24, 38, 22), (36, 54, 30), (50, 70, 40)], dtype=np.uint8)
    LEAF_HOLE_THRESH = 0.25

    def grow(self, rng: np.random.Generator, variant: int) -> tuple[TreeSkeleton, Leaves]:
        sb = SkeletonBuilder(rng)
        trunk = sb.trunk(height_m=6.0 + float(rng.uniform(-1.0, 1.0)),
                         base_radius_m=0.20, segments=5, wobble_m=0.1)
        limbs = sb.branches(trunk, count=(1, 2),
                             pitch_set=(math.radians(80),),
                             yaw_mode="spiral",
                             length_ratio=(0.4, 0.6),
                             length_scale_by_height=(1.0, 0.3))
        twigs = sb.branches(limbs, count=(2, 3),
                             pitch_set=(math.radians(80),),
                             length_ratio=(0.4, 0.6))         # more leaf-bearing wood
        sk = sb.skeleton()
        # Leaves grow ALONG the branch wood; count = density · leaves_per_m ·
        # Σ length, capped at max_leaves.  Denser = more twigs + higher caps,
        # NOT bigger leaves.
        leaves = leaves_at_tips(sk, np.concatenate([limbs, twigs]), rng,
                                density=0.65, leaves_per_m=70, max_leaves=600)
        return sk, leaves

# 2. Import it in fire_engine/procedural/flora/species/__init__.py:
#    from fire_engine.procedural.flora.species.my_pine import MyPineDef

# 3. Add a determinism test in tests/procedural/flora/species/test_my_pine.py.
# 4. Preview: python tools/preview_tree.py tree_my_pine --obj --png
```

## Gotchas

1. **`@register_def` fires at import.** Adding a new species file is not enough -- you must import the class in `fire_engine/procedural/flora/species/__init__.py`; otherwise `get("tree_my_species")` raises `KeyError`.
2. **`grow()` is required.** The base `TreeSpeciesDef.grow()` raises `NotImplementedError`.  Copy `gnarled_oak.py` (the annotated reference) -- do not start from a blank file.
3. **Bush path uses `length_m`, not `length_ratio`.** For a stub trunk (~0.12-0.15 m), `length_ratio` would produce stems of only a few centimetres.  Pass `length_m=(min_m, max_m)` as absolute stem lengths instead (see `scrub_bush.py`, `berry_bush.py`).
4. **Dead-tree / near-leafless path.** When `n_tufted == 0`, return `Leaves.empty()` directly rather than relying on `leaves_at_tips` with `density=0`.  (`density <= 0` or `rounds <= 0` does return empty, but skipping the call is clearer.)  See `dead_tree.py` for the correct pattern.
5. **`palettes(rng)` override for per-world hue drift.** If you want the species foliage colour to vary per world, override `palettes(rng)` returning `{"bark": ..., "leaf": ...}`.  Consume `rng` inside this method -- see `berry_bush.py` for the drift pattern.
6. **`name` is the registry key.** Convention: `"tree_<species>"` or `"bush_<species>"`.  A mismatch between the `name` attribute and the import in `__init__.py` is a silent failure -- `get()` raises `KeyError` at runtime.