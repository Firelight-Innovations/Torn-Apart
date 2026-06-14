"""
lighting/glsl.py — GLSL compute-shader sources for the GPU lighting pipeline.

The GLSL source now lives in `lighting/shaders/*.comp` (loaded verbatim via
`core.shader_source.load_glsl`) so editors get syntax highlighting + LSP; this
module just exposes them as plain string constants.  It imports nothing
GPU-side, so it stays importable headless; only `lighting/gpu.py` compiles
them via panda3d.

Pipeline shape (per radiance cascade)
-------------------------------------
1. **INJECT** (runs when the volume, sun, sky or lights change) — per cell:
   - march toward the sun / moon through occupancy → celestial visibility
     (``u_vis``: R = sun, G = moon, B = sky), the source of all voxel shadows;
   - first-bounce sunlight: an air cell next to a sunlit solid neighbour
     receives the neighbour's albedo × sun radiance × Lambert × 1/π (this is
     what makes a sunlit red wall glow red into the room);
   - emissive solid neighbours leak their emission into adjacent air;
   - dynamic point/area lights add windowed-inverse-square radiance with a
     short occupancy march for shadows.
   Writes ``u_source`` (surface radiosity: first bounce + emissive leak) and
   ``u_lit`` (dynamic-light direct in air).  NO skylight is injected — the
   gather integrates sky through real openings itself.

2. **GATHER** (only when INJECT runs; ``light_gi_iters`` ping-pong
   iterations) — ray-marched voxel GI: per air cell, a fibonacci-sphere fan
   of ``light_gi_rays`` rays marches the occupancy.  Escaping rays gather
   sky radiance (sky reaches interiors only through real openings); hitting
   rays gather the surface's lit source + a feedback bounce of the previous
   gather tinted by the surface albedo (multi-bounce colour bleed).  The
   result is a pure function of its inputs — no per-frame iteration, no
   convergence delay.  (Replaces the deprecated flood-fill propagate pass,
   owner decision 2026-06-12.)

2b. **SMOOTH** (after the gather iterations, ``light_gi_smooth_passes``
   times) — air-masked 3³ box filter of the RAY-GATHERED component only:
   the crisp own-cell contact term is subtracted, the remainder averaged
   over the air cells of the 3³ neighbourhood (never across solids — no
   leaks), and the contact term re-added.  Together with the gather's
   8-phase 2×2×2 ray-fan tile this completes the stratified direction set
   (8 × ``light_gi_rays`` effective rays) and kills the blotch/confetti
   noise of disagreeing neighbour fans.

Froxel fog
----------
3. **FOG_SCATTER** — per froxel (screen-aligned X/Y, exponential depth Z):
   weather density × (sun/moon Henyey-Greenstein in-scatter shadowed through
   the cascade occupancy → god rays, plus isotropic sky/GI ambient).
4. **FOG_INTEGRATE** — front-to-back accumulation along each screen ray,
   writing per-slice (accumulated in-scatter RGB, transmittance A) so any
   surface can composite fog at its own depth with one texture tap.

All units meters; world coordinates Z-up.  ``u_origin_m``/``u_cell_m``/
``u_cells`` define each cascade's window (see `lighting/volume.py`).
"""

from __future__ import annotations

from fire_engine.core.shader_source import load_glsl

__all__ = [
    "FOG_INTEGRATE_COMPUTE",
    "FOG_SCATTER_COMPUTE",
    "GATHER_COMPUTE",
    "INJECT_COMPUTE",
    "MAX_LIGHTS",
    "SHIFT_COMPUTE",
    "SMOOTH_COMPUTE",
]

# Must match Config.light_max_point_lights' upper bound and LightSet.pack.
MAX_LIGHTS = 64


# ---------------------------------------------------------------------------
# 1. Injection: direct radiance + celestial visibility
# ---------------------------------------------------------------------------

INJECT_COMPUTE = load_glsl(__file__, "inject.comp")


# ---------------------------------------------------------------------------
# 2. Gather: ray-marched voxel GI (sky through openings + surface bounce)
# ---------------------------------------------------------------------------

GATHER_COMPUTE = load_glsl(__file__, "gather.comp")


# ---------------------------------------------------------------------------
# 2b. Smooth: air-masked de-noise of the ray-gathered GI component (the
#     own-cell contact term stays voxel-crisp; solids are never crossed).
# ---------------------------------------------------------------------------

SMOOTH_COMPUTE = load_glsl(__file__, "smooth.comp")


# ---------------------------------------------------------------------------
# 2c. Radiance shift on recenter: copy the previous radiance field into the
#     write texture offset by the integer cell delta so the same-frame
#     re-gather's feedback term reads a spatially-aligned previous field.
# ---------------------------------------------------------------------------

SHIFT_COMPUTE = load_glsl(__file__, "shift.comp")


# ---------------------------------------------------------------------------
# 3. Froxel fog scatter
# ---------------------------------------------------------------------------

FOG_SCATTER_COMPUTE = load_glsl(__file__, "fog_scatter.comp")


# ---------------------------------------------------------------------------
# 4. Froxel fog integration (front-to-back along each screen ray)
# ---------------------------------------------------------------------------

FOG_INTEGRATE_COMPUTE = load_glsl(__file__, "fog_integrate.comp")
