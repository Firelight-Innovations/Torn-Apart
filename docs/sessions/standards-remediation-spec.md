# Standards-Gate Remediation Spec (overnight agent brief)

> **Goal for the autonomous run:** make `pytest -q` exit **0** — i.e. drive the
> 6 failing standards-gate tests to green **without breaking any of the 2660
> logic tests that currently pass.** This is a *conformance* refactor, not a
> behavioural one. If a logic test flips from pass→fail, you broke something;
> revert and rethink.

This document is the single source of truth for *what the rules are* and *how to
fix each violation*. The rules themselves live in code — this just collects and
explains them. Authoritative sources, in priority order:

1. `CLAUDE.md` — the 11 Hard Rules + conventions (NEVER violate these to pass a gate).
2. `docs/systems/standards.md` — the gate's own system doc.
3. `pyproject.toml` `[tool.firelight]` / `[tool.ruff]` / `[tool.mypy]` / `[tool.pylint]` / `[tool.vulture]` — every numeric limit and tool config.
4. `tools/check_repo_structure.py`, `tools/check_docs.py`, `tools/standards_config.py` — the custom checkers (pure AST/text walks, zero magic numbers).

---

## 0. Baseline (measured 2026-06-15)

```
pytest -q  →  6 failed, 2660 passed, 4 deselected   (~3m20s)
```

**Passing already (do not regress):** `test_ruff_format`, `test_mkdocs_build_strict`,
`test_git_hygiene`, and the entire functional suite.

**The 6 failing gates:**

| Test (in `tests/standards/`) | Tool | Violations | Nature |
| --- | --- | --- | --- |
| `test_repo_structure::test_repo_structure` | `tools/check_repo_structure.py` | **146** | missing test mirrors + >1 public class per module |
| `test_docs::test_doc_pointers_and_schema` | `tools/check_docs.py` | **884** | missing package docs + missing `Docs:` pointers |
| `test_code_quality::test_ruff_lint` | `ruff check .` | **~265** | unused vars, import placement, `zip(strict=)`, etc. |
| `test_code_quality::test_mypy_strict` | `mypy` | **1 (blocking)** | dup module name halts mypy before real checking |
| `test_code_quality::test_pylint_duplication_and_module_length` | `pylint` R0801/C0302 | dup-code | copy-pasted blocks ≥8 lines |
| `test_code_quality::test_vulture_dead_code` | `vulture` | **2** | unused locals |

**Root cause:** the engine was reorganised (the `world/` regroup — see the
package-reorg session) and new subsystems (`world/{terrain,weather,wind,sky}`,
`simulation/*`, `procedural/{flora,textures}`, `buildings`, `zones`) were shipped
faster than their **test mirrors** and **system docs** were backfilled. The
structure and docs gates are layout-driven, so every new module/package
immediately demands its mirror test and its doc.

---

## 1. Inviolable constraints (these OUTRANK passing any gate)

From `CLAUDE.md` "Hard Rules". A refactor that breaks one of these is a bug even
if the gate goes green:

1. **`panda3d` may only be imported in `render/` and `lighting/`.** Everything
   else stays headless. When you split a module, do not pull a panda3d import
   into a headless package. The headless test suite excludes anything importing
   panda3d — moving an import can silently drop tests.
2. **All randomness via `core.rng.for_domain(*keys)`.** Never `random.*`, never
   unseeded `np.random.*`. (Ruff `NPY002` enforces the numpy half.)
3. **No `pickle`, ever.** Saves are seed + per-system deltas (`Saveable`).
4. **No per-voxel / per-vertex Python loops** — bulk numpy only.
5. **Event Bus is for state-change notifications, not per-frame data.**
6. **Only `render/` issues render commands** (lighting light-grid excepted).
7. **Render/scene writes are bulk array ops.**
8–11. Structure / one-class / docs-pointer / test-mirror — these *are* the gates
   you're fixing; see below.

**Determinism is sacred:** same seed → identical world. Many logic tests assert
this. Never reorder RNG draws or change `for_domain` key tuples when splitting a
module.

---

## 2. The gates, rule-by-rule, with fix recipes

### 2.1 Repo structure — `check_repo_structure.py` (standards 6, 7, 10, 11, 17)

Walks every non-excluded dir under `source_roots = ["fire_engine"]`. Limits from
`[tool.firelight]`:

| Rule | Limit | Check |
| --- | --- | --- |
| **6** max sub-folders per folder | `max_subdirs = 5` | `len(child dirs) > 5` |
| **7** max modules per folder | `max_modules_per_dir = 10` (excl. `__init__.py`) | `len(*.py) > 10` |
| **10** one public top-level class per module | 1 | count of `class Name` not starting with `_` |
| **11** one responsibility per module | — | module docstring required |
| **17** every module has a matching test | — | mirror path must exist |

**Rule 10 — "one public class per module" (the bulk of structural failures).**
A module may define **at most one** public (non-`_`) top-level class. Known
offenders and the intended split:

| Module | Public classes found | Fix |
| --- | --- | --- |
| `world/sky/sky_state.py` | `SkyState`, `SkySystem` | split `SkySystem` into `world/sky/system.py` (or similar) |
| `world/terrain/brush.py` | `BrushMode`, `SphereBrush`, `BoxBrush`, `CylinderBrush` | `BrushMode` is an `Enum` → move to `enums.py`; each brush shape → its own module, OR keep shapes together only if they collapse to one public class + `_` helpers |
| `world/weather/cells.py` | `CellKind`, `Regime`, `StormCell` | `CellKind`/`Regime` (Enums) → `enums.py`; keep `StormCell` |
| `world/weather/clouds.py` | `CloudGenus`, `CloudBand`, `CloudLayers` | `CloudGenus` (Enum) → `enums.py`; `CloudBand` (dataclass) → `types.py`; keep `CloudLayers` |
| `world/weather/system.py` | `LocalWeather`, `WeatherSystem` | `LocalWeather` (dataclass) → `types.py`; keep `WeatherSystem` |
| `world/wind/field.py` | `WindSnapshot`, `WindField` | `WindSnapshot` (dataclass) → `types.py`; keep `WindField` |
| `world/wind/modifiers.py` | `WindModifier`, `GustFront` | split the second into its own module along responsibility |
| `world/wind/worker.py` | `VenturiJob`, `VenturiResult`, `VenturiWorker` | `VenturiJob`/`VenturiResult` (dataclasses) → `types.py`; keep `VenturiWorker` |

> **The grouping-module escape hatch (rule 11 exemption).** Files named exactly
> `events.py`, `types.py`, `constants.py`, `enums.py`, or `protocols.py`
> (`grouping_modules` in config) are **exempt from the one-public-class limit** —
> they're allowed to hold several trivial support types. So the canonical fix for
> "module defines an `Enum`/`@dataclass`/`Protocol` plus its real class" is:
> **move the trivial support type(s) into the package's `types.py` / `enums.py`,
> leave the one real class behind.** This is almost always cleaner than inventing
> a new single-class module. Update `__init__.py` re-exports and all importers.

**Rule 17 — every module needs a test mirror.** For
`fire_engine/<a>/<b>/mod.py` the checker wants
`tests/<a>/<b>/test_mod.py` (the leading `fire_engine/` is dropped). A legacy
flat `tests/test_mod.py` is also accepted. ~40 modules are missing mirrors,
e.g. `world/terrain/{chunk,chunk_manager,generation,meshing}.py`,
`world/sky/{atmosphere,celestial,sky_state,weather}.py`,
`world/weather/*`, `world/wind/*`, `procedural/textures/{night_sky,rain_streak,wasteland_ground}.py`,
`procedural/flora/species/{berry_bush,dead_tree,gnarled_oak,scrub_bush}.py`.

> **Tests must be REAL, not stubs.** A test mirror that exists but asserts
> nothing will satisfy the structure gate but is dishonest and won't help the
> coverage ratchet. Each new test should cover the module's required categories
> (CLAUDE.md "Testing"): **determinism** (same seed → identical output),
> **correctness fixtures**, and **round-trips** (save/load, register/get) where
> applicable. Headless only — if the module imports panda3d it belongs to
> `render/`/`lighting/` and is excluded from the headless suite; do not write a
> panda3d-importing test into `tests/`.

**Rules 6 & 7 (subdir/module counts):** when splitting creates an 11th module in
a folder or a 6th subfolder, **introduce a new sub-package** rather than widening.
Check the current counts before adding modules.

**Verify:** `pytest -q tests/standards/test_repo_structure.py`
(or `python tools/check_repo_structure.py` for the full list).

### 2.2 Documentation — `check_docs.py` (standards 12, 13)

**Standard 13 — every package has a schema-valid doc.** For each directory with
an `__init__.py` under `fire_engine/` (the root `fire_engine/` itself excepted),
there must be `docs/systems/<dotted.path>.md`:

- `fire_engine/world/terrain/` → `docs/systems/world.terrain.md`
- `fire_engine/simulation/ai/` → `docs/systems/simulation.ai.md`
- `fire_engine/procedural/flora/species/` → `docs/systems/procedural.flora.species.md`

Each doc MUST contain a `keywords:` line **and all seven H2 headings, spelled
exactly** (order per template): `## Role`, `## Public API`, `## Imports Allowed`,
`## Events`, `## Units & Invariants`, `## Examples`, `## Gotchas`. Use
`docs/systems/_TEMPLATE.md` as the skeleton. Missing packages reported include:
`world.md`, `simulation.md`, `simulation.ai.md`, `simulation.economy.md`,
`simulation.politics.md`, `scene.md`, `procedural.flora.md`,
`procedural.flora.species.md`, `procedural.textures.md` (and likely more — run
the checker for the live list).

**Standard 12 — code points to its docs.** Every public (non-`_`) **module,
class, method, and function** docstring must carry a resolvable pointer matching
`Docs:\s*(docs/systems/<path>.md)(#anchor)?`. Rules:

- A public target with **no docstring at all** fails ("has no docstring") — give
  it one. (This is the bulk of the 884: methods like `BuildingDef.generate` with
  no docstring.)
- The pointed-to `.md` file must **exist**, and if a `#anchor` is given it must
  resolve to a real heading slug in that file (slug = lowercase, non-word chars
  stripped, spaces/underscores → `-`). A dead anchor fails.
- Anchor is optional; `Docs: docs/systems/world.terrain.md` alone is valid.

> **Order of operations matters:** create/repair the package doc (std 13) FIRST,
> then add `Docs:` pointers (std 12) that reference it — otherwise every pointer
> is a "missing doc" violation. Spell code identifiers in the docs exactly as in
> code so a single grep hits both (CLAUDE.md "Docs Are Grep-First").

**Verify:** `pytest -q tests/standards/test_docs.py`
(`python tools/check_docs.py` lists the first 40 + a total count). The
`test_mkdocs_build_strict` half currently passes — keep doc links valid so it
stays green (nav is auto-generated; broken in-page links are what would fail it).

### 2.3 Ruff lint — `ruff check .` (standards 1–5 hygiene)

Config in `[tool.ruff]`: line-length 100, py311, rule families
`E,F,W,I,UP,B,C90,PERF,NPY,ERA,SIM,RUF` (ANN intentionally OFF — mypy owns
typing; `RUF001/2/3` ignored for em-dash house style). Per-file ignores:
`tests/**` waive `C901,PERF`; `tools/**` waive `C901`.

Measured breakdown (~265 hits): `E402` 71 (import not at top), `F841` 38 (unused
local), `E741` 23 (ambiguous name `l`/`I`/`O`), `B905` 20 (`zip()` without
`strict=`), `B007` 18 (unused loop var), `E501` 17 (line >100), `B904` 12
(`raise ... from`), `C901` 11 (complexity >12), `E712` 9 (`== True`), `F821` 3
(undefined name — **investigate, may be a real bug**), `B023` 2 (loop-var
closure), `B017` 1.

> **Big lever:** ~111 of these live in `tools/out/diag/` — throwaway diagnostic
> dump scripts, not maintained code. **Recommended:** add `tools/out` to
> `[tool.ruff] extend-exclude` (and ideally gitignore `tools/out/`). That single
> config change clears ~40% of ruff hits legitimately, because scratch dumps were
> never meant to be linted. Confirm with the owner-style intent: `tools/out/` is
> generated output, not source. **Do NOT** blanket-add `# noqa` to silence real
> findings in `fire_engine/`.

Fix recipes: `F841`/`B007` → delete the unused binding (or prefix `_`). `B905` →
add `strict=True`/`strict=False` per intent. `E712` → `if x:` / `if not x:`.
`E402` → move imports to top (the diag scripts interleave imports with prints).
`E741` → rename. `C901` → extract helpers (mind the `panda3d` boundary).
**`F821` (undefined name): treat as a possible real bug — read the code, don't
just silence it.**

**Verify:** `pytest -q tests/standards/test_code_quality.py::test_ruff_lint`.
`ruff check --fix .` safely auto-fixes a large share (unused imports, `==True`,
import sort) — run it, then review the diff before committing.

### 2.4 mypy --strict — `mypy` (standard 1)  ← cheapest win, do FIRST

Currently **one blocking error** halts all type checking:

```
tools/standards_config.py: error: Source file found twice under different module
names: "standards_config" and "tools.standards_config"
```

`[tool.mypy]` sets `files = ["fire_engine"]`, but the gate invokes bare `mypy`
which is picking up `tools/standards_config.py` under two module paths (it's
imported both as `tools.standards_config` and directly). Fix options, simplest
first:

1. Ensure `tools/` is a proper package (`tools/__init__.py` exists) **and** that
   mypy resolves it one way — usually adding `explicit_package_bases = true` +
   `mypy_path`/`namespace_packages` settings, or
2. Add `tools/out` and the duplicate-resolution to `[tool.mypy] exclude`, or
3. Scope the invocation. **Whatever you choose, the fix is in `pyproject.toml` /
   a `tools/__init__.py`, not in deleting code.** Once mypy runs clean past this
   error, it may surface *real* strict-typing errors in `fire_engine` that were
   hidden behind the early halt — fix those with proper annotations (never by
   weakening `strict`, and never widen the `panda3d.*` override beyond the
   existing narrow one).

**Verify:** `pytest -q tests/standards/test_code_quality.py::test_mypy_strict`.

### 2.5 pylint — duplicate-code + too-many-lines (standards 3, 8)

Invoked `pylint --disable=all --enable=duplicate-code,too-many-lines fire_engine`.
`R0801` fires on ≥8 similar lines across files (`min-similarity-lines = 8`,
ignoring comments/docstrings/imports/signatures). `C0302` fires at >500 lines/module.

The reported dup centres on **repeated procedural-texture docstring/boilerplate**
(surfaced via `zones/__init__.py` vs a textures module). Fix by **extracting the
shared block** into one helper/base and referencing it — do not paper over it.
Note the config ignores docstrings/imports for similarity, so the real dup is
*code*, not the doc prose; read the actual flagged span.

> A pylint run also hit a `cp1252` encode crash writing its report on Windows —
> that's a console-encoding artifact, not the violation. The checkers call
> `force_utf8()`; if pylint's own output crashes, set `PYTHONUTF8=1` in the
> environment for the run. The underlying R0801 finding is real.

**Verify:** `pytest -q tests/standards/test_code_quality.py::test_pylint_duplication_and_module_length`.

### 2.6 vulture — dead code (standard 2)  ← trivial, do FIRST

Two 100%-confidence unused locals:

- `fire_engine/buildings/meshing.py:304` → `outward`
- `fire_engine/render/resource_adapter.py:207` → `resource_manager`

Read each — if genuinely unused, delete the assignment (and any now-dead
upstream). If it's reached only via dynamic dispatch (Unity lifecycle / Saveable
protocol), the *correct* place is `vulture_whitelist.py` — but for plain locals
that's almost never the case; prefer deletion. **Never whitelist to hide real
dead code.**

**Verify:** `pytest -q tests/standards/test_code_quality.py::test_vulture_dead_code`.

---

## 3. Recommended execution plan for the autonomous run

**Guardrail first.** Before any change, confirm the logic suite is green and keep
it as a tripwire after every batch:

```
pytest -q -p no:cacheprovider -q  --ignore=tests/standards     # 2660 must stay green
```

**Suggested order (cheap unblockers → bulk mechanical → bulk authoring):**

1. **vulture (2.6)** — 2 deletions. Minutes.
2. **mypy config (2.4)** — unblock mypy; then fix whatever real strict errors it
   reveals. Config-level, then per-package.
3. **ruff (2.3)** — `ruff check --fix .`, exclude `tools/out/`, then hand-fix the
   residue in `fire_engine/`. Investigate `F821`.
4. **pylint dup (2.5)** — extract the shared block.
5. **Repo structure (2.1)** — the one-public-class splits + the ~40 missing test
   mirrors. **Fan out: one sub-agent per package** (`world/sky`, `world/terrain`,
   `world/weather`, `world/wind`, `procedural/textures`, `procedural/flora`, …).
   Each agent: split modules using the `types.py`/`enums.py` grouping escape
   hatch, write real mirror tests, update `__init__.py` re-exports + importers,
   re-run `pytest -q tests/standards/test_repo_structure.py` AND the affected
   logic tests, return only when both are green.
6. **Docs (2.2)** — largest count. **Fan out: one sub-agent per package.** Each:
   author `docs/systems/<dotted>.md` from `_TEMPLATE.md` (all 7 H2s + keywords),
   then add `Docs:` pointers to every public docstring in that package. Do docs
   for a package in the **same** sub-agent that just split it (step 5), so the
   doc reflects the final module layout — splitting after documenting means
   re-documenting.

**Why this order:** structure changes (5) move classes between modules, which
changes what docstrings (6) and tests (5) must exist — so settle layout before
mass-authoring docs. mypy/ruff/vulture (1–4) are independent of layout and bank
quick green.

**Commit discipline:** one focused commit per package/gate, message describing
the seam you split along. **Any commit that changes a public API must update the
matching `docs/systems/<package>.md` in the same commit** (CLAUDE.md). Stay on a
working branch (not `master`) per the git-hygiene gate, and delete merged
branches when done.

**Definition of done:**

```
pytest -q          →  0 failed   (all standards gates + all logic tests green)
```

Optionally also confirm the heavy coverage gate doesn't regress:
`pytest -m coverage` (it has its own ratchet floor; new real tests should help,
never hurt).

---

## 4. Quick reference — check any single gate

```bash
# structure / docs (full violation lists)
python tools/check_repo_structure.py
python tools/check_docs.py

# code quality (each independent)
ruff check .
ruff format --check .
mypy
pylint --disable=all --enable=duplicate-code,too-many-lines fire_engine
vulture fire_engine vulture_whitelist.py

# the gate as pytest sees it
pytest -q tests/standards/                      # all gates except heavy coverage
pytest -q tests/standards/test_repo_structure.py
pytest -q tests/standards/test_docs.py
pytest -q tests/standards/test_code_quality.py
```

All limits are in `pyproject.toml [tool.firelight]`; never hard-code a number
elsewhere. The checkers are deterministic AST/text walks — same tree, same
verdict.
