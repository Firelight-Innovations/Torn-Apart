# Torn Apart · FireEngine

A fantasy post-apocalyptic sandbox RPG on a **custom, numpy-first engine** that uses Panda3D
as a rendering SDK only. Deterministic voxel terrain, GPU radiance-cascade global
illumination, a physically-based sky, volumetric weather, a living wind field, and
procedurally *grown* forests — every texture, tree, cloud, and blade of grass generated from
a single world seed. No hand-made environment art.

**Built almost entirely by AI coding agents in eight days** (June 9–16, 2026, ~130 commits),
directed by a human. The visual build log — with the development screenshots the agents took
to check their own work — is in **[docs/progression/](docs/progression/README.md)**. Start
there if you want the story; stay here if you want to run it.

![A procedural forest in the FireEngine — grown trees, dead snags, scrub bushes, wildflower meadow, and volumetric clouds](tools/out/trees/iter4_after_fulltree.png)

*Everything in this frame is procedural: trees grown per-species from Python scripts, leaves
placed by cellular automaton, instanced grass and flowers in tagged zones, ray-marched clouds
— all lit by GPU radiance cascades and swaying in a simulated wind field.*

## What the engine does today

- **Deterministic voxel world** — 0.5 m voxels in 32³ chunks, streamed and meshed around the
  camera; brush edits (left-click explosions) remesh and relight within a frame or two.
- **Delta saves** — a save is the world seed plus per-system diffs. No pickles, ever.
- **GPU volumetric lighting** — three nested radiance cascades, ray-marched voxel shadows,
  bounced global illumination, froxel fog, dynamic point/spot lights (torches, flashlight,
  explosion flashes).
- **Physical HDR sky** — Rayleigh/Mie atmosphere feeding the light grid, volumetric
  ray-marched clouds, bloom, lens flare, god rays, FXAA.
- **Spatial weather** — storm cells that drift across the map, humidity-driven fog,
  volumetric rain that respects roofs and canopy, procedural lightning and thunder.
- **Wind field** — a travelling, spatially-varying gust field driving grass, tree sway, dust
  motes, and leaf litter. Costs zero bytes in a save.
- **Procedural vegetation** — GPU-instanced grass and wildflowers in tagged zone volumes;
  3-D trees and bushes grown per-species from Python scripts with impostor LODs and
  per-meter canopy light extinction.
- **Buildings** — free-form floorplan authoring (walls, arcs, openings, auto-detected rooms,
  slabs), meshed headlessly in numpy.
- **Unity-style object model** — GameObjects, components, and lifecycle in snake_case;
  quaternions only.
- **Fire Editor** — a VS Code extension driving the live engine: scene tree, component
  inspector, gizmos, scene save, screenshot RPC.
- **Frame profiler** — headless core, in-game F3 overlay, scripted benchmark harness.

The deep simulation layers (AI, economy, politics) are still stubs — that's the next arc.

## Setup

Python 3.11+ required.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on POSIX)
pip install -r requirements.txt
python main.py
```

### Controls

| Input | Action |
|---|---|
| `W` `A` `S` `D` + mouse | Free-fly camera (`Shift` sprints, `Space`/`Ctrl` up/down) |
| `ESC` | Toggle mouse capture |
| **Left-click** | Fire an explosion — carves a crater and flashes a point light |
| `F` | Toggle camera flashlight |
| `L` / `K` | Drop a torch / clear dynamic lights |
| `G` | Build a Cornell-style GI test room ahead of the camera |
| `F1` | Developer overlay |
| `F5` / `F9` | Quick save / quick load (delta saves to `saves/quick.ta`) |
| `F6` | Cycle forced weather (clear → cloudy → overcast → fog → rain → storm → natural) |
| `F7` / `F8` | Fast-forward time scale / jump the clock +6 hours |

## Testing

```bash
pytest -q                 # full headless suite (no window / GPU required)
pytest -m window          # tests that need a real Panda3D window
pytest -q tests/standards/  # the machine-enforced standards gate
```

The headless suite (236 test files) runs without a GPU because **only `render/` and
`lighting/` may import panda3d** — everything else is pure Python/numpy. The standards gate
enforces structure limits, lint, strict types, docs-link resolution, and per-module test
presence; a violation fails the build like any failing test
([docs/systems/standards.md](docs/systems/standards.md)).

### Offscreen tools (no window needed)

```bash
python tools/screenshot.py --out shot.png          # offscreen screenshot of the demo world
python tools/preview_texture.py wasteland_ground   # render a ProceduralTextureDef → PNG
python tools/preview_tree.py                       # preview a tree species' variants
python tools/verify_weather.py                     # summon a thunderstorm overhead
python tools/profile_run.py                        # scripted performance benchmark
python tools/dump_save.py saves/quick.ta           # inspect a save's per-system deltas
```

These tools are how the AI agents verified their own visual work — the screenshots they
saved along the way became the [progression docs](docs/progression/README.md).

## Architecture at a glance

The engine is layered: downward calls are direct imports, upward/sideways notifications go
through an event bus, and only the render bridge touches the rendering SDK. Full design
authority: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
simulation/ (player, ai*, economy*, politics*)          * = stubs
    │
world/ (terrain, weather, wind, sky)   buildings/  zones/  scene/
    │
lighting/ ── render/            ← the only packages that may import panda3d
    │
core/  procedural/  resources/  save/   ← pure Python/numpy foundation
```

- **`docs/` is grep-first** — it's the AI search index. Every package has a doc in
  [`docs/systems/`](docs/systems/) with identical headings, so agents grep docs before
  reading code. Stale docs fail the build.
- **Determinism:** the world is a pure function of `world_seed`
  ([`config.toml`](config.toml)). All randomness flows through `core.rng.for_domain(*keys)`.
  Same seed → byte-identical world; that's what makes delta saves and bug repro possible.

## Repo layout

```
main.py            # demo entry point (python main.py [--load save.ta])
config.toml        # world_seed + all sizes/distances (no magic numbers in code)
fire_engine/       # the engine: core/ procedural/ resources/ save/ lighting/ render/
                   #   world/ simulation/ buildings/ zones/ scene/ devtools/
editor/            # Fire Editor VS Code extension + Python daemon
docs/              # grep-first knowledge base: ARCHITECTURE, systems/, sessions/,
                   #   content/, progression/  ← the visual build log
tests/             # headless suite + standards gate (tests/standards/)
tools/             # screenshot / preview / profiling / save-inspection utilities
assets/            # hand-crafted files only (env textures are procedural — never here)
```

See [DECISIONS.md](DECISIONS.md) for the dated decision log,
[docs/sessions/](docs/sessions/) for per-session handoff notes, and
[docs/progression/](docs/progression/README.md) for the illustrated eight-day build story.
