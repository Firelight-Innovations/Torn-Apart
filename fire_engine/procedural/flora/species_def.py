"""
procedural/flora/species_def.py — TreeSpeciesDef base + TreeVariantSet.

A tree/bush **species** is a registered :class:`ProceduralDef` whose
``generate`` returns a :class:`TreeVariantSet`: a per-world-seed pool of
unique 3-D variant meshes + the species texture atlas + the far-LOD
impostor atlas.  The registry caches the whole set by
``(name, world_seed, params_digest)`` — every world grows its own gnarled
oaks, at zero per-instance cost and zero save bytes.

Authoring a species (full guide: ``docs/content/tree_species_authoring.md``)
----------------------------------------------------------------------------
Subclass, set the class attributes, implement :meth:`grow`::

    from fire_engine.procedural.defs import register_def
    from fire_engine.procedural.flora import (SkeletonBuilder,
        TreeSpeciesDef, leaves_at_tips)

    @register_def
    class MyTreeDef(TreeSpeciesDef):
        name = "tree_my_tree"
        variants = 8
        BARK_PALETTE = ...   # uint8 (T, 3) ramps, shadow tone first
        LEAF_PALETTE = ...

        def grow(self, rng, variant):
            sb = SkeletonBuilder(rng)
            trunk = sb.trunk(height_m=5.0, base_radius_m=0.25)
            limbs = sb.branches(trunk, ...)
            sk = sb.skeleton()
            return sk, leaves_at_tips(sk, limbs, rng)

Then ``procedural.get("tree_my_tree")`` does the rest: per-variant child
rngs, skeleton validation, meshing, atlas composition, impostor raster.
Preview with ``python tools/preview_tree.py tree_my_tree --obj --png``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fire_engine.procedural.defs import ProceduralDef
from fire_engine.procedural.flora.atlas import (
    AtlasLayout,
    bark_texture,
    compose_atlas,
    leaf_texture,
)
from fire_engine.procedural.flora.impostor import (
    impostor_atlas,
    rasterize_impostor,
)
from fire_engine.procedural.flora.leaves import Leaves
from fire_engine.procedural.flora.mesher import (
    TreeMesh,
    merge_parts,
    mesh_branches,
    mesh_leaves,
)
from fire_engine.procedural.flora.skeleton import (
    TreeSkeleton,
    validate_skeleton,
)

__all__ = ["TreeSpeciesDef", "TreeVariantSet"]


@dataclass(frozen=True)
class TreeVariantSet:
    """
    Everything the renderer needs to draw one species — registry-cached.

    Attributes
    ----------
    name : str
        The species def name (``"tree_gnarled_oak"`` …).
    meshes : tuple[TreeMesh, ...]
        The variant pool — unique per world seed; instances draw from it.
    atlas : numpy.ndarray
        ``(H, W, 4) uint8`` species texture (bark left half opaque, leaf
        right half binary alpha) — bind once per species.
    impostors : numpy.ndarray
        ``(Hc, Wc × n_variants, 4) uint8`` far-LOD sprite atlas, one cell
        per variant, trunk bases on the bottom row.  All cells share ONE
        meters-per-texel scale (see ``impostor_height_m``).
    max_height_m / max_radius_m : float
        Pool-wide extents (m) — pad render bounds with these × max scale.
    impostor_width_m / impostor_height_m : float
        World size (m) of the impostor billboard quad: because the cells
        share one scale, a quad of exactly this size, base at the trunk
        base, overlays every variant's mesh at the crossfade.
    """

    name: str
    meshes: tuple[TreeMesh, ...]
    atlas: np.ndarray
    impostors: np.ndarray
    max_height_m: float
    max_radius_m: float
    impostor_width_m: float
    impostor_height_m: float

    @property
    def n_variants(self) -> int:
        """Variant pool size."""
        return len(self.meshes)


class TreeSpeciesDef(ProceduralDef):
    """
    Base class for 3-D tree/bush species (see module docstring to author).

    Class attributes (override per species)
    ---------------------------------------
    name : str
        Registry key — convention ``"tree_<species>"`` / ``"bush_<species>"``.
    variants : int
        Mesh pool size per world.  Default 6.
    BARK_PALETTE / LEAF_PALETTE : numpy.ndarray
        ``uint8 (T, 3)`` posterise ramps, shadow tone first.
    LEAF_HOLE_THRESH : float
        Foliage raggedness (higher = scragglier).  Default 0.18.
    BERRY_COLOR / BERRY_DENSITY
        Optional fruit speckles in the leaf texture.
    TINT_RANGE : tuple[float, float]
        Per-variant albedo drift baked into vertex colors.  Default
        (0.92, 1.08).

    Hooks
    -----
    grow(rng, variant) -> (TreeSkeleton, Leaves)
        REQUIRED — the species recipe (SkeletonBuilder calls).
    palettes(rng) -> dict
        Optional — return ``{"bark": ..., "leaf": ...}`` ramps; override to
        drift hues per world.  Default returns the class palettes.
    """

    variants: int = 6
    atlas_layout: AtlasLayout = AtlasLayout()
    impostor_cell: tuple[int, int] = (64, 96)

    BARK_PALETTE = np.array([(38, 30, 22), (56, 44, 32), (76, 60, 42)], dtype=np.uint8)
    LEAF_PALETTE = np.array(
        [(30, 44, 26), (44, 62, 34), (60, 80, 42), (80, 98, 52), (104, 116, 64)], dtype=np.uint8
    )
    LEAF_HOLE_THRESH: float = 0.18
    BERRY_COLOR: tuple[int, int, int] | None = None
    BERRY_DENSITY: float = 0.0
    TINT_RANGE: tuple[float, float] = (0.92, 1.08)

    # ------------------------------------------------------------------
    # Species hooks
    # ------------------------------------------------------------------

    def grow(self, rng: np.random.Generator, variant: int) -> tuple[TreeSkeleton, Leaves]:
        """
        Grow one variant — THE species recipe.  Override this.

        Parameters
        ----------
        rng : numpy.random.Generator
            Per-variant child generator (derived deterministically from the
            registry rng) — consume it for ALL randomness.
        variant : int
            Variant index ``0 … variants-1`` (vary structure by index if
            you want, e.g. variant 0 always full-crowned).

        Returns
        -------
        (TreeSkeleton, Leaves)
            From ``SkeletonBuilder.skeleton()`` and ``leaves_at_tips``
            (``Leaves.empty()`` for leafless species).
        """
        raise NotImplementedError(
            f"{type(self).__name__}.grow() not implemented — see "
            "docs/content/tree_species_authoring.md"
        )

    def palettes(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """``{"bark", "leaf"}`` ramps; override for per-world hue drift."""
        return {"bark": self.BARK_PALETTE, "leaf": self.LEAF_PALETTE}

    # ------------------------------------------------------------------
    # Shared pipeline (species rarely override below here)
    # ------------------------------------------------------------------

    def generate(self, rng: np.random.Generator, **params) -> TreeVariantSet:
        """
        Build the full variant set (registry-cached; do not call directly —
        use ``procedural.get(self.name)``).

        Parameters
        ----------
        rng : numpy.random.Generator
            Injected by the registry (Hard Rule 2).
        **params : any
            ``variants=<int>`` overrides the pool size.

        Returns
        -------
        TreeVariantSet
        """
        n = max(1, int(params.get("variants", self.variants)))
        layout = self.atlas_layout
        pal = self.palettes(rng)

        hw, hh = layout.half_px
        atlas = compose_atlas(
            layout,
            bark_texture(rng, hw, hh, pal["bark"]),
            leaf_texture(
                rng,
                hw,
                hh,
                pal["leaf"],
                hole_thresh=self.LEAF_HOLE_THRESH,
                berry_color=self.BERRY_COLOR,
                berry_density=self.BERRY_DENSITY,
            ),
        )

        # Per-variant child rngs chained off the injected rng — deterministic
        # and independent of how many draws each grow() consumes.  Separate
        # seeds for growth and impostor noise so the two-pass build below
        # (grow all → raster all at the POOL-COMMON scale) stays stable.
        grow_seeds = rng.integers(0, 2**63, size=n)
        imp_seeds = rng.integers(0, 2**63, size=n)
        meshes: list[TreeMesh] = []
        grown: list = []
        for v in range(n):  # pool-size loop (≤ 8)
            vrng = np.random.default_rng(int(grow_seeds[v]))
            sk, leaves = self.grow(vrng, v)
            validate_skeleton(sk)
            tint = float(vrng.uniform(*self.TINT_RANGE))
            wood = mesh_branches(sk, uv_rect=layout.bark_rect, tint=(tint, tint, tint))
            foliage = mesh_leaves(leaves, vrng, uv_rect=layout.leaf_rect, tint=(tint, tint, tint))
            meshes.append(merge_parts(wood, foliage))
            grown.append((sk, leaves))

        # Pool-common impostor scale: one meters-per-texel for every cell so
        # the renderer's single billboard quad overlays each variant exactly.
        max_h = max(m.height_m for m in meshes)
        max_r = max(m.radius_m for m in meshes)
        cw, ch = self.impostor_cell
        px_per_m = min(
            (cw - 1) / (2.0 * max(max_r, 0.25) * 1.05), (ch - 1) / (max(max_h, 0.5) * 1.02)
        )
        cells = [
            rasterize_impostor(
                sk,
                leaves,
                pal["bark"],
                pal["leaf"],
                np.random.default_rng(int(imp_seeds[v])),
                cell_wh=self.impostor_cell,
                hole_thresh=self.LEAF_HOLE_THRESH,
                px_per_m=px_per_m,
            )
            for v, (sk, leaves) in enumerate(grown)
        ]

        return TreeVariantSet(
            name=self.name,
            meshes=tuple(meshes),
            atlas=atlas,
            impostors=impostor_atlas(cells),
            max_height_m=max_h,
            max_radius_m=max_r,
            impostor_width_m=cw / px_per_m,
            impostor_height_m=ch / px_per_m,
        )
