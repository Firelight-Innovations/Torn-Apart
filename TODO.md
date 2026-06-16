# TODO — Torn Apart

Cross-session backlog of larger initiatives and handed-off work. One-off task tracking
lives in commits / `docs/sessions/`; this file is for things that outlive a single session.

---

## Optimization

Performance backlog. Diagnosis, measurements, and root-cause detail live in
`docs/sessions/graphics-optimization-session.md` (profiler runbook: `docs/systems/profiler.md`).
Tool of record: `python tools/profile_run.py --seed 1 --frames 800 --warmup 120`.

### Done
- [x] **Lightning rain-cover rebuild storm.** `LightningRendererComponent` rebuilt its entire
  `RainCoverField` (`rebuild_all`, all loaded chunks) on every `ChunkLoadedEvent`, every frame,
  on the main thread — the steady-state stutter + a large slice of the 5-FPS startup. Fixed to
  mirror `RainRendererComponent`'s budgeted incremental refold (dirty-column set + per-frame
  `rain_cover_budget_columns` budget). Measured `LateUpdate:LightningRendererComponent`
  **22.9 ms → 0.9 ms** mean; frame mean **49.6 → 25.7 ms** (~20 → ~39 FPS on the stress flight).
  Branch `perf/graphics-optimization`. (2026-06-16)

### Open — independent wins (a fresh agent can take these now)
- [ ] **Startup stall — multi-second early frames.** Frame 0 = ~6.9 s; first ~30 s well below
  target. Suspect: lazy **GPU shader compilation** (cloud raymarch, GI/lighting, sky, post-FX
  compile on first draw) + **synchronous initial chunk gen/mesh/upload**. Action: confirm with
  PStats on frame 0 (`profile_run.py --pstats`) and `py-spy dump` during the stall; then
  pre-warm shaders at boot and/or move initial chunk meshing off the main thread. Biggest
  remaining chunk of the "5 FPS for 30 s" symptom.
- [ ] **`ChunkStream` ~13.5 ms/frame while moving.** Now the dominant steady scope after the
  lightning fix. Legitimate terrain streaming on the main thread. Action: budget/thread the
  per-frame streaming work; this is also where terrain LOD (below) pays off. Terrain meshing
  threading is noted "still pending" from prior sessions.
- [ ] **`Lighting` cascade-assembly spikes (~500 ms occasional).** A worker thread was added in
  a prior session; a residual main-thread spike remains. Action: confirm what still runs on the
  main thread during a recenter and move it off.
- [ ] **HDR float-buffer fallback warning** ("RGBA16F → fixed-point, HDR will clip"). Dev iGPU
  caveat; not perf, but degrades HDR. Action: gate / document `[graphics] gfx_hdr_format`.

### Open — major initiative: Terrain LOD + universal LOD communication
Goal: render a **distant horizon**, super-optimize terrain, and have *every* world object share
one LOD authority so geometry, placement density, and shadow detail transition together with no
visual mismatch. The primitive already exists and is **specified but unwired**:
`core/lod.py::LODPolicy.band_for(distance_m)` (default bands 32/96/192/512 m); ARCHITECTURE §6
says terrain and the render object model must read the *same* policy. Today nothing consumes it
(only `core/lod.py` + `core/__init__.py` reference `LODPolicy`).

Suggested phasing (can run alongside the independent wins above):
- [ ] **P1 — Wire `LODPolicy` as the shared authority.** Single boot-time instance from config;
  expose a small per-frame "what band is this position in?" query the chunk streamer and the
  render object model both call. No behavior change yet — just the shared seam.
- [ ] **P2 — Terrain chunk LOD.** Generate merged/simplified meshes for distant chunk bands
  (ARCHITECTURE §6: "octree over chunks; LOD via merged/simplified distant meshes"). Stream
  coarse far, refine near; transition on `LODPolicy` bands. Targets the `ChunkStream` cost.
- [ ] **P3 — Distant-horizon tier.** Beyond the mesh radius, render a cheap far representation
  (heightfield horizon / imposters) so the world reads as continuous without per-chunk geometry.
- [ ] **P4 — Universal LOD comms for all world objects.** A shared contract (events/protocol or
  a `core` query) so flora, trees, buildings, NPCs, grass, etc. all pick detail/billboard/cull
  from the *same* `LODPolicy` bands as terrain. (Billboard imposters are the band-3 plan; see
  `fire_engine/procedural/flora/impostor.py` for the existing imposter primitive to generalize.)
- [ ] **P5 — Determinism + budget guards.** LOD transitions must be deterministic (seed-stable)
  and hysteresis-damped (no band thrash at a threshold); add a per-frame LOD-work budget so a
  fast flight can't spike. Add `profile_run.py` baselines per phase to prove no regression.
