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
     (``u_vis``: R = sun, G = moon), the source of all voxel shadows;
   - march straight up → sky visibility, injecting ``sky_ambient`` skylight
     into open-air cells;
   - first-bounce sunlight: an air cell next to a sunlit solid neighbour
     receives the neighbour's albedo × sun radiance × Lambert (this is what
     makes a sunlit red wall glow red into the room);
   - emissive solid neighbours leak their emission into adjacent air;
   - dynamic point/area lights add windowed-inverse-square radiance with a
     short occupancy march for shadows.
   Writes ``u_direct`` (rgba16f source radiance).

2. **PROPAGATE** (every frame, ``light_prop_iters`` ping-pong iterations) —
   exponential diffusion: ``next = direct·(1−decay) + decay·avg₆``, where a
   solid neighbour reflects the cell's own radiance back tinted by its
   albedo (multi-bounce approximation).  Spectral radius < 1 ⇒ stable; light
   visibly "flows" a few cells per frame after a change, converging to a
   smooth flood-fill GI field.

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
    "INJECT_COMPUTE",
    "PROPAGATE_COMPUTE",
    "SHIFT_COMPUTE",
    "FOG_SCATTER_COMPUTE",
    "FOG_INTEGRATE_COMPUTE",
    "MAX_LIGHTS",
]

# Must match Config.light_max_point_lights' upper bound and LightSet.pack.
MAX_LIGHTS = 64


# ---------------------------------------------------------------------------
# 1. Injection: direct radiance + celestial visibility
# ---------------------------------------------------------------------------

INJECT_COMPUTE = load_glsl(__file__, "inject.comp")


# ---------------------------------------------------------------------------
# 2. Propagation: exponential diffusion flood fill (one Jacobi iteration)
# ---------------------------------------------------------------------------

PROPAGATE_COMPUTE = load_glsl(__file__, "propagate.comp")


# ---------------------------------------------------------------------------
# 2b. Radiance shift on recenter: copy the previous radiance field into the
#     write texture offset by the integer cell delta so a recentered cascade
#     keeps its already-converged GI (only the newly-exposed border band needs
#     to re-propagate) instead of reading a stale, misaligned field.
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
