# CLAUDE.md — Torn Apart

Fantasy post-apocalyptic sandbox RPG. Custom engine on Panda3D (rendering SDK only). Python 3.11+, numpy-first.
**Read `docs/ARCHITECTURE.md` (design authority) and `docs/DEVELOPMENT_PLAN.md` (sequencing authority) before any work.**

## Repo Layout
```
CLAUDE.md  README.md  DECISIONS.md  requirements.txt  config.toml  main.py
docs/            # grep-first knowledge base — search HERE before reading code
  ARCHITECTURE.md  DEVELOPMENT_PLAN.md
  systems/       # one doc per code package, filename == package name (_TEMPLATE.md defines the format)
  content/       # authoring guides for AI content agents (textures, biomes, buildings...)
  sessions/      # per-session handoff notes (session-01.md, ...)
fire_engine/      # the code package (core/ procedural/ world/ terrain/ lighting/ ...)
tests/  tools/   # headless suite; preview/dump/screenshot utilities
assets/          # hand-crafted only: models/ audio/ textures/ (env textures are procedural — never put them here)
saves/           # gitignored
```

## Docs Are Grep-First (the AI search index)
- **Before exploring code, grep `docs/`:** `grep -ril "<topic>" docs/` → read the hit. Every `docs/systems/*.md` uses identical H2 headings (Role / Public API / Imports Allowed / Events / Units & Invariants / Examples / Gotchas), so structured queries work: `grep -rA5 "## Events" docs/systems/`.
- Filenames mirror the package tree: `docs/systems/terrain.md` documents `fire_engine/terrain/`; a deeper leaf package `fire_engine/procedural/textures/` → `docs/systems/procedural/textures.md`. Every leaf package gets one doc (enforced by the docs gate).
- Each doc opens with a `keywords:` line of synonyms — extend it when you notice a missed grep.
- Spell code identifiers in docs exactly as in code (`apply_brush`, `for_domain`) so one grep hits both.
- **Any commit that changes a public API must update the matching `docs/systems/<package>.md` in the same commit.** Stale docs are bugs — they poison every future agent's context.

## Commands
```
.venv\Scripts\activate            # Windows venv
pip install -r requirements-dev.txt          # standards-gate toolchain (one-time)
pre-commit install                           # run the gate on every commit (one-time)
python main.py                    # run the game
pytest -q                         # full headless test suite — must pass before every commit
pytest -q tests/standards/        # the standards gate (lint/type/structure/docs)
pytest -m coverage                # branch-coverage ratchet gate (CI/nightly; heavy)
mkdocs serve                      # browse the docs site locally
python tools/preview_texture.py <def_name>   # render a ProceduralTextureDef to PNG
python tools/dump_save.py <save_file>        # inspect a save
```

## Hard Rules (violations are bugs)
1. **panda3d may only be imported in `world/` and `lighting/`.** Everything else stays headless-testable. Bridges (`world/texture_bridge.py`, geometry upload) exist for this.
2. **All randomness goes through `core.rng.for_domain(*keys)`.** Never `random.*`, never unseeded `np.random.*`. Same seed must always produce the same world — determinism is what makes delta saves and bug repro possible.
3. **No pickle, anywhere, ever** (owner decision 2026-06-09). Saves are seed + per-system deltas via the `Saveable` protocol (`save/saveable.py`); deltas are dicts of primitives/numpy arrays only — no live object references.
4. **No per-voxel/per-vertex Python loops.** Bulk work is numpy array expressions. If you can't vectorize it, flag it in the commit message rather than shipping a Python loop over 32³ elements.
5. **Event Bus is for state-change notifications (upward/sideways), never per-frame data plumbing.** Downward calls are direct imports of the lower layer's public API. Render/terrain/lighting hot paths never publish per-element events.
6. **Only the World API issues render commands** (Lighting API excepted for light-grid GPU work).
7. **Render commands and scene-graph writes happen via bulk operations** — build arrays, write once.
8. **Deep & narrow structure** (machine-enforced): ≤5 sub-folders and ≤10 modules per folder (`__init__.py` excluded), ≤500 lines per module. Bumping a limit means introduce a new sub-package — never cram. Limits live in `pyproject.toml [tool.firelight]`.
9. **One public class per module** — a file holds that class and its tightly-bound `_`-prefixed helpers. Trivial support types (`@dataclass`, `Enum`, `Protocol`, the `*Event` frozen dataclasses) may be grouped in a dedicated `events.py`/`types.py`/`enums.py`/`protocols.py`/`constants.py`.
10. **Code points to its docs** — every public module/class/function docstring carries a resolvable `Docs: docs/systems/<path>.md[#anchor]` line; dead doc links fail the build.
11. **Every module has a matching test** (`tests/<mirrored-path>/test_<stem>.py`); branch coverage is enforced on a ratchet floor that only ever rises.

> Rules 8–11 are enforced by `tests/standards/` (and `.pre-commit-config.yaml`) — a violation fails the build like any test. See `docs/systems/standards.md`. The gate **delegates** fixes to scoped sub-agents; do not bulk-fix inline.

## Conventions
- **Docstrings are product, not decoration.** Every public class/function gets a docstring with types, units (meters, seconds, voxels), and a usage example for content base classes (`ProceduralDef`, `BiomeDef`, `BuildingDef`, `NPCArchetype`...). These docs are the prompt context for AI content agents — write them as if the reader has never seen the codebase.
- Type hints mandatory on all public APIs. Events are `@dataclass(frozen=True)` named `*Event`.
- **Object model is a Unity API clone** (ARCHITECTURE.md §5.4): Unity names/semantics in snake_case, Unity lifecycle order. Rotations are **quaternions only** (`core.math3d.Quat`); never store Euler angles or use panda3d math types outside `world/`.
- Units: world space in **meters**; voxel = 0.5 m; chunk = 32³ voxels = 16 m; light cell = 1 m. Z-up (Panda3D native). Chunk coords are integer `(cx, cy, cz)`.
- Config values come from `core.config` — no magic numbers for sizes/distances/scales.
- Package-level documentation lives in `docs/systems/<package>.md` (not per-package READMEs — one canonical place to grep). `__init__.py` exports the public API explicitly.
- Stubs raise `NotImplementedError` with a message pointing at the relevant ARCHITECTURE.md section — never silent `pass`.

## Testing
- Headless suite (`tests/`) must run without a window or GPU; anything importing panda3d is excluded from it by the import rule above.
- Required test categories per system: determinism (same seed → identical output), correctness fixtures (e.g., mesher face counts), and round-trips (save/load, register/get).
- New `ProceduralDef` content needs a determinism test + a `preview_texture.py`/fixture entry. Visual changes: state in the commit message what to look at in-game.

## Workflow
- Work the phases in `docs/DEVELOPMENT_PLAN.md` in order; one commit per phase minimum, message prefixed `phase N:`. Each phase commit includes the updated `docs/systems/` doc(s) for the packages it touched.
- If behind schedule, cut stretch scope (Phase 6) — never skip tests or the Final Verification checklist.
- When a design question isn't answered by ARCHITECTURE.md, prefer the smallest decision that doesn't close doors, and record it in a `DECISIONS.md` log (date, question, choice, why).
