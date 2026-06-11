# Handoff — terrain shimmer ("z-fighting") + lighting popping/noise/GI

keywords: shimmer, z-fighting, aliasing, sparkle, crawl, normal map, mipmap,
FT_nearest, light pixel, u_quant_m, quantization, cascade popping, GI, global
illumination, assembly worker, load-in delay, terrain.frag

Written mid-session for a fresh agent picking up the lighting work. The owner
asked specifically for a **lighting-grid LOD** fix, but the headline bug
(shimmer on dirt/grass) is almost certainly a **different** cause — read the
hypotheses below before coding.

---

## RESOLVED 2026-06-11 (next session) — root cause found and fixed

A working motion-shimmer measurement tool now exists:
`tools/shimmer_probe.py` (sub-pixel camera-yaw sweep via
`FlyController.yaw`, multi-frame settle per pose, `RTM_copy_ram` capture,
built-in static + positive controls that validate the harness every run).
Metric: fraction of pixels whose value flips > 0.12 per 0.25 px yaw step.

Measured results (open ground, noon, pitch −12):

| variant                          | full    | far-ground band |
|----------------------------------|---------|-----------------|
| baseline                         | 0.00496 | 0.00805         |
| Hypothesis A (`N = n`)           | 0.00495 | 0.00804         |
| Hypothesis B (quant-grid LOD)    | 0.00497 | 0.00805         |
| constant albedo (`base=0.18`)    | ~0      | ~0              |
| **fix: posterise-per-tap**       | 0.00048 | 0.00028         |

- **Hypothesis A (normal map): REFUTED** — disabling it changed nothing.
- **Hypothesis B (light-quant grid): not the open-ground cause** — noon
  open-field lighting is uniform, so grid flips are invisible there.  (It may
  still matter where lighting has contrast; the GI room measured only 0.0008
  at a 3× more sensitive threshold, before AND after, i.e. already tiny.)
- **Actual cause: quantise-after-filter ordering in the albedo path.**
  `groundNoise` was supersampled/octave-faded (band-limited, fine) but the
  *averaged* value was then pushed through the hard posterising palette LUT
  once — re-hardening it.  Pixels whose averaged noise sat near a palette
  bucket edge popped a full palette step on every sub-pixel camera move.
  Fix in `terrain.frag`: run the LUT lookup inside the 4-tap supersample and
  average the resulting **colours** (filter after quantisation) — far-ground
  flips down ~27×, near-field pixel art unchanged (all taps share one texel).
  See `docs/systems/world.md` gotcha 21 and DECISIONS.md 2026-06-11.

Remaining residual: the **horizon/silhouette line** (terrain-vs-sky geometric
edge, no MSAA — `App.__init__` sets `M_none` for the retro look).  That is
edge aliasing, not texture shimmer; if the owner still sees twinkle it is
this, and the lever is MSAA (`framebuffer-multisample`/`multisamples` PRC +
`M_multisample`), an owner aesthetic call.  Hypothesis C (cascade handoff
popping) and symptoms 3–6 remain open.

---

## Symptoms (owner, observed in-motion in `python main.py`)

1. **"z-fighting"/shimmer in the textures as the camera moves** — on dirt,
   grass, AND the Cornell/GI room surfaces.
2. **It is WORSE on dirt/grass (open terrain) than in the Cornell room.**
   ← This is the single most important discriminator. Use it to rank causes.
3. Lighting takes **~1–2 s to "load in"** at startup.
4. **Lots of popping** in the lighting as the camera moves.
5. **Lots of noise** in the lighting.
6. **Still no visible global illumination** (bounce) in the open world.

A still frame is temporally stable (the renderer is deterministic per pose);
shimmer is **motion** aliasing. Stills only reveal *spatial* aliasing (speckle),
which correlates but is not the same thing. Final validation must be in motion
by the owner.

---

## What I changed this session (now committed)

**Albedo procedural-ground anti-aliasing** in `fire_engine/world/shaders/terrain.frag`:
- The ground albedo is a hard per-texel **world-space hash** (the pixel-art
  blocks), indexed into `u_ground_lut`. Hard hash = white noise → aliases hard
  under minification.
- Added `groundOctave()` / `groundNoise()` with a **per-octave Nyquist LOD fade**
  (fade each octave to its mean 0.5 *before* its texel reaches ~1 px) plus a
  **4-tap rotated-grid supersample** in `main()` for near-field edge AA.
- Final fade tuning: `smoothstep(0.5, 1.4, mpp * texels)` where
  `mpp * texels` = that octave's texels-per-pixel (1.0 == Nyquist).

**Result:** stills are visibly band-limited now (mid/far ground smooth, near
stays crisp — see `tools/out/diag/ground_lod.png`, `ground_lod_near.png` vs the
speckly baseline `ground_grazing.png`). **BUT the owner reports the dirt/grass
shimmer persists and is still worse than the Cornell room.** Therefore the
albedo hash is **not** the dominant remaining cause. Do not keep tuning it.

(The same uncommitted batch also carried earlier work: `extra_materials` LUT
wiring for the GI room, a coarse far **cascade-2** in the shader + `gpu.py`, the
threaded `assembly_worker.py`, and `u_quant_m` lowered to 0.0625 m. These were
the owner's / prior-session changes, committed together as a checkpoint.)

---

## Hypothesis A (most likely for dirt/grass): NORMAL-MAP ALIASING

`terrain.frag:154`:
```glsl
vec3 nm = texture(p3d_Texture1, v_uv).xyz * 2.0 - 1.0;   // tangent-space normal map
vec3 N  = normalize(t * nm.x + b * nm.y + n * max(nm.z, 0.3));
```
`fire_engine/world/texture_bridge.py:116-117` (also 161-162, 203-204) — **all**
terrain textures, including the normal map, are bound:
```python
tex.set_minfilter(SamplerState.FT_nearest)   # no mipmaps
tex.set_magfilter(SamplerState.FT_nearest)
```
A nearest-filtered, **non-mipmapped** normal map under minification: each screen
pixel fetches one random normal texel. As the camera moves sub-pixel, the
fetched texel flips → `N` flips → the Lambert `NdotL` direct term flickers →
shimmer. This is:
- **heaviest on far open ground** (most minification) = dirt/grass, and
- **~absent on the Cornell walls** (flat / no detailed normal map),

which is **exactly** symptom #2. And albedo no longer dominates because `gnoise`
is now band-limited, so the still-unfiltered normal path is what's left.

### FIRST diagnostic the next agent should run (definitive, 1-line):
Force `N = n;` (ignore the normal map) in `terrain.frag` and fly around. If the
dirt/grass shimmer **disappears**, the normal map is the cause — proceed to fix.
If it persists, the normal map is innocent (it may be a flat default) → go to
Hypothesis B / C. Also confirm whether the terrain normal map even has detail
(check where `p3d_Texture1` is assigned for terrain Geoms in
`fire_engine/world/geometry_bridge.py` and what generates it).

### Candidate fixes (in order of preference)
1. **Derivative-fade the normal-map influence** toward the geometric normal `n`
   as the UV footprint grows (mirrors the albedo octave LOD, no mip chain, keeps
   the retro look):
   ```glsl
   float uvpx = max(fwidth(v_uv.x), fwidth(v_uv.y)) * NORMAL_TEX_DIM;
   float nstr = 1.0 - smoothstep(0.5, 1.4, uvpx);   // 1 near, 0 when texel<~1px
   N = normalize(mix(n, N, nstr));
   ```
2. **Mipmap + trilinear** the normal map: `FT_linear_mipmap_linear` min filter,
   build the mip chain. Caveat: naive normal-map mips lose energy / flatten;
   renormalize per-mip, or use **Toksvig** (fold normal variance into a
   roughness/strength term). Heavier than #1 and changes the texture-bridge
   contract (which is currently "retro hard pixel" by design).

---

## Hypothesis B (the owner-requested "lighting-grid LOD"): QUANTIZATION ALIASING

`terrain.frag:161`:
```glsl
vec3 wq = (floor(v_world / u_quant_m) + 0.5) * u_quant_m;   // light-pixel grid
```
- `u_quant_m` is now **0.0625 m** (8×8×8 light pixels per 0.5 m voxel) — it was
  0.25 m. That is **4× finer**, i.e. 4× higher spatial frequency → MORE
  aliasing, and it starts closer to the camera. (config: `core/config.py` /
  `config.toml`, `light_quant_m`.)
- `sampleCascades()` then point-samples radiance/vis/occ at the quantized probe.
  Hard grid + hard shadow (`vis`) edges sweep across pixels under motion →
  sparkle/pop. Most visible where the LIGHTING has **contrast** (shadows, GI
  colour) → explains Cornell-room shimmer and shadowed-terrain shimmer, but NOT
  flat-lit open noon ground (no lighting contrast there). So B is likely a
  *secondary* contributor relative to A for dirt/grass, but is the right fix for
  the Cornell/shadow case and is what the owner explicitly asked for.

### The lighting-grid LOD fix
Clamp the effective light-pixel size so it never drops below ~1 screen pixel —
keep the chunky look up close, coarsen at distance so the grid can't sub-pixel
sparkle (and the linear-filtered cascade textures show through smoothly):
```glsl
float mpp = max(fwidth(v_world.x), max(fwidth(v_world.y), fwidth(v_world.z))); // world m / px
float eff = max(u_quant_m, mpp * 1.2);                 // LOD the grid
vec3  wq  = (floor(v_world / eff) + 0.5) * eff;
```
Compute `mpp` once near the top of `main()`. Then verify the cascade 3D textures
(`u_c*_radiance/_vis/_geom`) are **linear**-filtered in `gpu.py` (if they're
nearest, that's another aliasing source the LOD won't fully hide).

---

## Hypothesis C: cascade handoff + recenter (the popping, symptom #4)

- `sampleCascades()` (`terrain.frag:66-92`) hard-switches c0→c1→c2 by `inBox`.
  At a boundary, the resolution/quality jumps → a visible seam that pops as the
  camera moves the boundary across the world. Consider trilinear **blending**
  across the boundary band instead of a hard `if`.
- `gpu.py` commits a cascade's new origin only after the worker uploads the new
  volume (`casc.window.origin_cell = res.origin_cell`, ~L627-629). When the
  committed origin jumps, all GI/shadows shift discretely → pop. Mitigate with
  smaller recenter steps, more hysteresis, or blend old↔new.

---

## Other reported issues

- **~1–2 s load-in (symptom #3):** cascades assemble on a worker thread
  (`fire_engine/lighting/assembly_worker.py`; `gpu.py` `_schedule_assembly` /
  `_submit_assembly` / `_drain`). First volumes aren't on the GPU for the first
  second+, so the world looks unlit/half-lit at boot. Consider a **synchronous
  first assembly** of cascade-0 at spawn, or a deliberate fade-in.
- **Noise (symptom #5):** partly the A/B/C aliasing above; also check the
  radiance flood-fill (`inject.comp` / `propagate.comp`) for undersampling and
  whether any temporal accumulation exists.
- **No visible GI (symptom #6):** the GI room *did* show red/green bleed earlier
  (`tools/out/.../gi_final2.png`), but the open world shows none. Investigate:
  is radiance injection+propagation running for ordinary terrain cells? Is the
  GI term (`radiance * ao` in the `hdr` composite) swamped by direct sun +
  auto-exposure? Repro cleanly with a **night/indoor single point light** and
  confirm bounce. Check the inject/propagate dispatch in `gpu.py`,
  `EMISSION_SCALE`, and that `u_c*_radiance` is actually non-zero on the GPU.

---

## Why my automated shimmer validator FAILED (don't repeat it)

- Frame-to-frame diff at a fixed pose = 0 (renderer is deterministic; shimmer is
  motion-only).
- Supersampled-truth (jittered-average at a fixed pose) attempts:
  - **Camera-rotation jitter:** the FlyController re-derives camera rotation on
    every `taskMgr.step()` and overwrote my jitter → exact-zero diff.
  - **Lens `set_film_offset` jitter:** a *single* `taskMgr.step()` after the
    change captures a **stale** framebuffer (≥1-frame pipeline latency) → ~zero
    even for a 4 px positive-control shift.
  - A working version would need a full multi-step settle (~100 steps) after
    each jittered pose — expensive, and cascades recenter between poses.
- **Validate instead by:** (a) the `N = n` isolation test for Hypothesis A,
  (b) visual stills for spatial aliasing, (c) owner in-motion confirmation.

---

## Key files / lines

- `fire_engine/world/shaders/terrain.frag` — surface shading. Albedo LOD:
  `groundOctave`/`groundNoise`; normal map L154-155; light quant L161; cascade
  sampling L66-92; composite `hdr` ~L218.
- `fire_engine/world/texture_bridge.py:116-117, 161-162, 203-204` — `FT_nearest`,
  no mipmaps (Hypothesis A source).
- `fire_engine/world/terrain_shader.py` — compiles shader, binds ground LUT
  (`extra_materials`).
- `fire_engine/world/geometry_bridge.py` — terrain Geom texture stages
  (where `p3d_Texture1` is assigned; confirm normal-map detail here).
- `fire_engine/lighting/gpu.py` — cascades, uniforms, assembly scheduling, fog
  (cascade-2 + worker added this batch).
- `fire_engine/lighting/assembly_worker.py` — threaded volume assembly
  (load-in delay + popping).
- `fire_engine/core/config.py`, `config.toml` — `light_quant_m` (now 0.0625),
  cascade sizes/counts.
- `main.py` — `build_gi_test_room`, GI materials, the area light.

## Diagnostic commands

```
.venv\Scripts\python.exe -m pytest -q          # keep green (currently 521 passed, 1 deselected)
.venv\Scripts\python.exe tools\screenshot.py --time-of-day 12.0 --pitch -12 --no-grass --out diag\x.png
.venv\Scripts\python.exe tools\screenshot.py --gi-room --inside --time-of-day 13.0 --out diag\gi.png
```
GOTCHA: `--time-of-day` is in **HOURS (0–24)**, not a fraction. `0.5` = 00:30
(night); use `12.0` for noon, `13.0` for the GI room.
```
```
