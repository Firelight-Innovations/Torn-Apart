# core._impl — System Doc
keywords: config_loader, load_config, resolve_graphics_preset, GRAPHICS_PRESETS, quat, Quat, quaternion, worker, QueueWorker, worker_pool, WorkerPool, thread pool, n_workers, profiler_scope, NullScope, _ScopeCtx, _alloc_profiler_buffers, profiler_report, frame_time_stats, build_snapshot, write_profiler_snapshot, commit_frame, private implementation, overflow split, module count limit

> One doc per code package; filename matches the package exactly (`docs/systems/core._impl.md` ↔ `fire_engine/core/_impl/`).

## Role

Private implementation helpers for `fire_engine/core/` — not a public API.

`core/_impl/` exists solely to satisfy the ≤10-module-per-directory rule (enforced by `tests/standards/`).  When `fire_engine/core/` reached its module-count ceiling, large implementation blocks were extracted here.  All exported names are re-exported from the appropriate top-level `core/` module; callers must **never** import from `fire_engine.core._impl` directly.

This sub-package deliberately does NOT: define any public API, introduce new functionality, or export symbols that are not already available through the top-level `core.*` modules.

Modules:

| Module | Contents | Re-exported from |
|---|---|---|
| `config_loader.py` | `GRAPHICS_PRESETS`, `load_config`, `resolve_graphics_preset` | `fire_engine.core.config` |
| `quat.py` | `Quat` unit-quaternion class | `fire_engine.core.math3d`, `fire_engine.core` |
| `worker.py` | `QueueWorker` generic single-background-thread base | imported directly by `world/wind/` and `lighting/` subclasses |
| `worker_pool.py` | `WorkerPool` generic N-thread pool base | imported directly by the terrain LOD subclass |
| `profiler_scope.py` | `NullScope`, `_ScopeCtx`, `_alloc_profiler_buffers`, `_NULL_SCOPE` | `fire_engine.core.profiler` |
| `profiler_report.py` | `frame_time_stats`, `build_snapshot`, `write_profiler_snapshot`, `commit_frame` | `fire_engine.core.profiler` |

## Public API

This package has **no public API of its own**.  All symbols below are public only as re-exports from their parent `core/` modules.  See `docs/systems/core.md` for the full public API documentation.

Re-exported symbols (for grep convenience):

- `GRAPHICS_PRESETS` — `dict[str, dict]` mapping preset name (`"off"`, `"low"`, `"medium"`, `"high"`) to flat `gfx_*` kwargs.
- `load_config(path="config.toml") -> Config` — load engine config from TOML; returns `Config()` defaults when the file is absent or unreadable.
- `resolve_graphics_preset(table) -> dict` — expand a `[graphics]` TOML table into flat `gfx_*` kwargs using the `preset` key; falls back to `"high"` on unknown preset names (never raises).
- `Quat(w, x, y, z)` — unit quaternion, scalar-first float32 numpy storage.
- `QueueWorker` — generic single-daemon-thread worker; subclasses implement `_process`.
- `WorkerPool` — generic N-daemon-thread pool variant of `QueueWorker` (one shared in/out queue fanned across `n_workers` threads), used by the terrain LOD system to parallelise independent mesh-build/decimation jobs; subclasses implement `_process` and must keep it pure (it may run concurrently). `stop()` enqueues one `None` sentinel per thread.
- `NullScope` — shared no-op timing scope for a disabled `Profiler`.
- `_ScopeCtx` — pooled timing scope context manager (private helper).
- `_alloc_profiler_buffers(prof)` — allocate numpy ring-buffer arrays on a `Profiler` (private helper).
- `_NULL_SCOPE` — module-level `NullScope` singleton (private helper).
- `frame_time_stats(frames_ms, budget_ms) -> dict` — vectorised per-frame statistics (mean, median, p99, p999, fps_mean, over_budget_pct).
- `build_snapshot(prof) -> dict` — assemble the versioned plain-dict performance summary from a live `Profiler`.
- `write_profiler_snapshot(prof, path)` — atomically write `build_snapshot` output as JSON (tmp → `os.replace`).
- `commit_frame(prof, total_ms)` — commit a frame's per-scope accumulators into the ring buffer + run hitch detection.

## Imports Allowed

Same constraints as `fire_engine/core/`:

- Python standard library only (`math`, `hashlib`, `json`, `os`, `queue`, `threading`, `tempfile`, `abc`, `contextlib`, `datetime`, `typing`, `warnings`, `tomllib`).
- `numpy`.
- `fire_engine.core.*` — allowed (sibling modules within the `core/` package), but only via `TYPE_CHECKING` guards or bottom-of-file imports to avoid circular dependencies.

**No panda3d imports.  No imports from any `fire_engine.*` package outside `core/`.**

## Events

Published: none — `core/_impl/` emits no events.

Subscribed: none.

## Units & Invariants

- `Quat` components are **float32**; rotation axes are in **radians**.
- `frame_time_stats` expects `frames_ms` in **milliseconds** and `budget_ms` in **milliseconds**; returns floats only (JSON-serializable).
- `QueueWorker` uses a daemon thread — a missed `stop()` never blocks process exit.
- `WorkerPool` uses `n_workers` daemon threads (clamped to ≥1) sharing one in/out queue; `_pending` is mutated only on the main thread (`submit`/`drain_results`) so it needs no lock, but `_process` may run on several threads at once and must be pure.
- `_alloc_profiler_buffers` sets all numpy arrays to zero on construction; the ring buffer is correctly sized to `0` when the profiler is disabled (no memory used).
- `write_profiler_snapshot` uses `os.replace` (atomic on POSIX; near-atomic on Windows via tmp file in the same directory) so readers never see a half-written JSON file.

## Examples

```python
# config_loader — reached via fire_engine.core.config, never imported directly:
from fire_engine.core.config import load_config, resolve_graphics_preset, GRAPHICS_PRESETS

cfg = load_config("config.toml")          # fallback-safe
gfx = resolve_graphics_preset({"preset": "low", "gfx_fxaa": True})

# Quat — reached via fire_engine.core.math3d or fire_engine.core:
from fire_engine.core.math3d import Vec3, Quat
from math import pi
q = Quat.from_axis_angle(Vec3.UP, pi / 2)

# QueueWorker subclass — imported directly by world/wind and lighting:
from fire_engine.core._impl.worker import QueueWorker

class MyWorker(QueueWorker[bytes, dict]):
    def _process(self, job: bytes) -> dict:
        return {"size": len(job)}

w = MyWorker("MyWorker")
w.start()
w.submit(b"hello")
results = w.drain_results()
w.stop()
```

## Gotchas

1. **Never import from `fire_engine.core._impl` in production code.** Import from the appropriate top-level `core/` module instead (`core.config`, `core.math3d`, `core.profiler`).  The internal layout of `_impl/` can change at any time to satisfy module-count limits.

2. **`config_loader.py` has a safe circular import**: `config.py` imports `config_loader` at the bottom of the file (after `Config` is defined), and `config_loader` imports `Config` from `config` at module level.  This is intentional and safe — do not move the import in `config.py` to the top.

3. **`Quat` uses call-time imports for `Vec3`** in `rotate()` and `from_euler()` to avoid a circular module-level dependency between `quat.py` and `math3d.py`.  This is correct and expected.

4. **`_alloc_profiler_buffers` must be called after scalar fields are set** on the `Profiler`; calling it before `history_frames`, `max_scopes`, etc. are assigned will allocate wrong-sized arrays.

5. **`write_profiler_snapshot` is a no-op when `prof.enabled` is False** — it returns immediately without touching the filesystem.
