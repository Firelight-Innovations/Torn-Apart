# standards — System Doc
keywords: standards gate, code quality, ruff, mypy, pylint, vulture, coverage, pytest-cov, branch coverage, hypothesis, mkdocs, mkdocstrings, structure check, deep narrow, max modules, max subdirs, max lines, one public class per module, docs pointer, Docs:, dead doc link, keywords line, H2 schema, pre-commit, ratchet, check_repo_structure, check_docs, check_git_hygiene, git hygiene, stale branches, merged branch cleanup, tidy git log, standards_config, firelight, vulture_whitelist, delegate sub-agent

> The machine-enforced standards gate. One doc per code package; this one
> documents `tools/check_repo_structure.py`, `tools/check_docs.py`,
> `tools/standards_config.py`, and the `tests/standards/` pytest gates.

## Role

`standards` is the **self-enforcing cleanliness layer**. It makes the repo stay
clean automatically so any agent can drop into any folder and understand it
without holding the whole tree in their head. Every standard is wired into the
headless pytest suite (`tests/standards/`) and into `.pre-commit-config.yaml`, so
a standards violation fails the build exactly like a failing test.

It deliberately does NOT auto-fix anything. When a gate fails it prints a
**delegate-to-sub-agent** message naming the offending paths; a scoped sub-agent
does the cleanup so the orchestrator's context stays lean.

The enforced standards, by group:

- **Code quality** — everything typed, no dead code, no duplicate code,
  consistent formatting, performance hygiene.
- **Repo structure** — deep & narrow: ≤5 sub-folders and ≤10 modules per folder,
  ≤500 lines per module.
- **Per-file content** — one public class / one responsibility per module.
- **Documentation** — code points to its docs; every package has a schema-valid
  `docs/systems/` doc; the docs site builds `--strict`.
- **Testing & coverage** — every module has a matching test; branch coverage is
  enforced on a ratchet.

## Public API

Custom checks (no off-the-shelf tool covers these):

- `tools/standards_config.py` — `load_config()` → `StandardsConfig`; `force_utf8()`.
  Single source of truth, read from `[tool.firelight]` in `pyproject.toml`.
- `tools/check_repo_structure.py` — `collect_violations(cfg)`, `main()`.
  Enforces standards 6, 7, 10, 11, 17.
- `tools/check_docs.py` — `collect_violations(cfg)`, `main()`. Enforces 12, 13.
- `tools/check_git_hygiene.py` — `collect_violations(cfg)`, `main()`. Flags stale
  merged branches (returns `None` when it skips). Enforces git hygiene.

Pytest gates (each fails independently):

- `tests/standards/test_code_quality.py` — ruff lint, ruff format, mypy, pylint, vulture.
- `tests/standards/test_repo_structure.py` — runs `check_repo_structure.py`.
- `tests/standards/test_docs.py` — runs `check_docs.py` + `mkdocs build --strict`.
- `tests/standards/test_git_hygiene.py` — runs `check_git_hygiene.py`.
- `tests/standards/test_coverage.py` — branch-coverage ratchet (marker `coverage`).

### Toolchain

Pinned in `requirements-dev.txt`. Off-the-shelf: **Ruff** (lint + the sole
formatter), **mypy --strict** (typing — Ruff `ANN` stays off to avoid
double-reporting), **pylint** (narrow: `duplicate-code` + `too-many-lines`),
**vulture** (cross-module dead code; see `vulture_whitelist.py`), **pytest-cov**
/ **coverage** (branch mode), **pytest-xdist**, **hypothesis**, **mkdocs** +
**mkdocstrings[python]** + **mkdocs-material**.

Type checking and linting are different analyses — neither replaces the other.
Ruff + mypy is the core; pylint/vulture/the custom checks fill the gaps Ruff
can't cover (cross-file duplication, cross-module dead code, structure, docs).

## Imports Allowed

The custom checks and gates are **headless tooling**, not engine code: they may
import only the standard library (`ast`, `re`, `subprocess`, `tomllib`,
`pathlib`, `dataclasses`) and `pytest`. They MUST NOT import `panda3d`,
`fire_engine`, or any GPU/window code — the standards suite has to run without a
window or GPU like the rest of `tests/`. The checks analyse source statically
(AST + text), never by importing it.

## Events

Published: none. Subscribed: none. The gate is invoked by the test runner and
by pre-commit, not via the Event Bus.

## Units & Invariants

All limits live in `[tool.firelight]` (`pyproject.toml`) — no magic numbers in
the checks.

### Configuration

| key | default | standard |
| --- | --- | --- |
| `max_subdirs` | 5 | 6 — sub-folders per folder |
| `max_modules_per_dir` | 10 | 7 — modules per folder (excl. `__init__.py`; 5 is the stretch target) |
| `max_lines` | 500 | 8 — source lines per module (pylint `C0302`) |
| `coverage_fail_under` | ratchet | 18 — branch-coverage floor (only ever raised) |
| `source_roots` | `["fire_engine"]` | which trees are walked |
| `grouping_modules` | `events.py, types.py, …` | §C exemption from one-public-class |
| `exclude` | `.git, .venv, saves, assets, …` | paths skipped by all custom checks |
| `git.default_branches` | `["master", "main"]` | git hygiene — merge baseline, never flagged |
| `git.check_remotes` | `true` | git hygiene — also flag merged `origin/*` refs |
| `git.protected` | `[]` | git hygiene — extra branch globs never flagged |

Invariants: the checks are pure `O(tree)` walks with no side effects; same tree
→ same verdict (deterministic). The structure/docs mirroring is **layout-driven**
— it adapts automatically as packages are split or reorganised, so it keeps
working across package reorgs without edits.

## Examples

### Code quality

```bash
pip install -r requirements-dev.txt      # one-time
ruff check . && ruff format --check .     # lint + format
mypy                                      # strict typing (fire_engine)
pylint --disable=all --enable=duplicate-code,too-many-lines fire_engine
vulture fire_engine vulture_whitelist.py
```

### Repo structure

```bash
python tools/check_repo_structure.py      # ≤5 subdirs, ≤10 modules, 1 public class, test mirror
```

A module's test mirror is `tests/<path-under-source-root>/test_<stem>.py`
(legacy flat `tests/test_<stem>.py` is also accepted), e.g.
`fire_engine/core/rng.py` → `tests/core/test_rng.py`.

### Documentation

Every public module/class/function docstring carries a resolvable pointer:

```python
def apply_brush(...) -> None:
    """Carve terrain under a brush.

    Docs: docs/systems/world.terrain.md#public-api
    """
```

A package's doc is its dotted full path: `fire_engine/world/terrain/` →
`docs/systems/world.terrain.md`, carrying the full H2 schema and a
`keywords:` line.

### Git hygiene

```bash
python tools/check_git_hygiene.py         # no stale merged branches lingering
```

Keeps the branch list tidy so the next agent isn't wading through dead branches.
A branch is flagged when it is fully merged into the default branch — by ancestry
(ordinary merge / fast-forward) **or** by patch-id (`git cherry`, which catches
squash- and rebase-merges that leave no ancestry link) — yet still exists. The
current branch and `git.default_branches` are never flagged; add `git.protected`
globs (e.g. `release/*`) for long-lived branches that should survive merge. The
gate prints `SKIP` and passes outside a git work tree, on a shallow clone, or
when no default branch resolves (so a CI checkout never fails spuriously). Fix by
deleting the branches:

```bash
git branch -d <branch>            # local (already merged)
git push origin --delete <branch> # remote
git remote prune origin           # drop stale remote-tracking refs
```

### Integration

```bash
pytest -q tests/standards/                # all gates except the heavy coverage one
pytest -m coverage                        # branch-coverage ratchet (CI/nightly)
pre-commit install                        # run the same checks on every commit
mkdocs serve                              # browse the docs locally
```

### Coverage

The floor is a **ratchet**: set to the current measured branch coverage (rounded
down) and only ever raised. Standard 17 (every module ships a test) keeps new
code honest while the floor climbs. Raise `coverage_fail_under` in
`[tool.firelight]` once the new level is comfortably met — never lower it.

## Gotchas

- **The gate does not auto-fix.** Failures delegate to a scoped sub-agent on
  purpose; fixing inline blows the orchestrator's context at scale.
- **Don't weaken `mypy --strict` globally** to silence Panda3D — there is a
  narrow per-module override for `panda3d.*` / `direct.*` only.
- **Ruff `ANN` is intentionally off** — mypy owns typing; enabling both
  double-reports.
- **The coverage gate is heavy** (it re-runs the suite under coverage), so it is
  the `coverage` marker, deselected from the default run. pre-commit runs on
  changed files only; the full suite + coverage runs in CI/nightly.
- **Windows cp1252** mangles the emoji/em-dash in check output — the checks call
  `force_utf8()`; keep it when editing `main()`.
- **`vulture_whitelist.py`** is for names reached only via dynamic dispatch (the
  Unity lifecycle, the `Saveable` protocol). Never add a name there to hide real
  dead code.
