"""
core/profiler.py — engine-agnostic frame profiler (no panda3d).

Per-frame hierarchical scope timer backed by a preallocated numpy ring buffer.
Computes percentile / hitch statistics and emits a plain-dict ``snapshot()``
for overlay and AI-agent consumption.  Headless and observational only.

Siblings:
  - :mod:`fire_engine.core._impl.profiler_scope`  — NullScope, _ScopeCtx, buffer alloc
  - :mod:`fire_engine.core._impl.profiler_report` — frame_time_stats, build_snapshot

Example
-------
    from fire_engine.core.profiler import get_profiler

    prof = get_profiler()
    prof.begin_frame()
    with prof.scope("Update:Weather"):
        advance_weather()
    prof.end_frame()
    snap = prof.snapshot()   # plain dict, JSON-serializable
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from fire_engine.core._impl.profiler_report import (
    build_snapshot,
    commit_frame,
    frame_time_stats,
    write_profiler_snapshot,
)
from fire_engine.core._impl.profiler_scope import (
    _NULL_SCOPE,
    NullScope,
    _alloc_profiler_buffers,
    _ScopeCtx,
)
from fire_engine.core.log import get_logger

if TYPE_CHECKING:
    from fire_engine.core.config import Config

__all__ = [
    "SCHEMA_VERSION",
    "NullScope",
    "Profiler",
    "frame_time_stats",
    "get_profiler",
    "init_profiler",
]

_log = get_logger("profiler")

#: JSON snapshot schema version.  Bump on breaking dict-shape changes only.
SCHEMA_VERSION = 1


class Profiler:
    """
    Per-frame hierarchical timer with a numpy ring buffer + hitch detection.

    Use :func:`get_profiler` / :func:`init_profiler` for the process-wide
    singleton.  Construct directly only for headless tests.

    See ``docs/systems/profiler.md`` for the full field / behaviour reference.

    Docs: docs/systems/core.md
    """

    _CPU_MS_COUNTER = "frame_cpu_ms"

    # Buffer/state attributes initialised by _alloc_profiler_buffers (profiler_scope.py);
    # declared here so the class is the single source of truth for their types.
    _scope_ids: dict[str, int]
    _scope_names: list[str]
    _counter_ids: dict[str, int]
    _counter_names: list[str]
    _capacity_warned: bool
    _ctx_pool: list[_ScopeCtx]
    _frame_ms: np.ndarray
    _scope_ms: np.ndarray
    _scope_calls: np.ndarray
    _counter_val: np.ndarray
    _counter_seen_any: np.ndarray
    _write_index: int
    _frames_written: int
    _cur_scope_ns: np.ndarray
    _cur_scope_calls: np.ndarray
    _cur_counter: np.ndarray
    _cur_counter_seen: np.ndarray
    _active_depth: np.ndarray
    _active_start_ns: np.ndarray
    _stack: list[int]
    _recent_ms: np.ndarray
    _recent_index: int
    _recent_count: int
    _hitches: list[dict[str, Any]]
    _hitch_count: int
    _last_threshold_ms: float
    _frame_start_ns: int
    _pending: bool
    _cpu_ms: float
    _start_wall: int
    _frame_cpu_ms_seen: bool

    def __init__(
        self,
        *,
        enabled: bool = False,
        frame_budget_ms: float = 5.0,
        history_frames: int = 1024,
        hitch_abs_ms: float = 8.0,
        hitch_rel_mult: float = 1.5,
        hitch_window: int = 120,
        max_scopes: int = 64,
        max_counters: int = 32,
        recent_hitches: int = 16,
        time_source: Callable[[], int] | None = None,
    ) -> None:
        self._time = time_source or time.perf_counter_ns
        self._observers: list[tuple[Callable[[str], None], Callable[[str], None]]] = []
        self._counter_observers: list[Callable[[str, float], None]] = []
        self._configure(
            enabled=enabled,
            frame_budget_ms=frame_budget_ms,
            history_frames=history_frames,
            hitch_abs_ms=hitch_abs_ms,
            hitch_rel_mult=hitch_rel_mult,
            hitch_window=hitch_window,
            max_scopes=max_scopes,
            max_counters=max_counters,
            recent_hitches=recent_hitches,
        )

    def _configure(
        self,
        *,
        enabled: bool,
        frame_budget_ms: float,
        history_frames: int,
        hitch_abs_ms: float,
        hitch_rel_mult: float,
        hitch_window: int,
        max_scopes: int,
        max_counters: int,
        recent_hitches: int,
    ) -> None:
        """Set scalar config attrs, then delegate buffer allocation to
        :func:`~fire_engine.core.profiler_scope._alloc_profiler_buffers`."""
        self.enabled = bool(enabled)
        self.frame_budget_ms = float(frame_budget_ms)
        self.history_frames = int(history_frames)
        self.hitch_abs_ms = float(hitch_abs_ms)
        self.hitch_rel_mult = float(hitch_rel_mult)
        self.hitch_window = max(1, int(hitch_window))
        self.max_scopes = int(max_scopes)
        self.max_counters = int(max_counters)
        self.recent_hitches = int(recent_hitches)
        _alloc_profiler_buffers(self)

    def configure_from_config(self, config: Config) -> Profiler:
        """
        (Re)configure in place from a :class:`Config`.  Mutates self so
        existing references stay valid.  Returns self.
        """
        self._configure(
            enabled=bool(getattr(config, "profiler_enabled", False)),
            frame_budget_ms=float(getattr(config, "profiler_frame_budget_ms", 5.0)),
            history_frames=int(getattr(config, "profiler_history_frames", 1024)),
            hitch_abs_ms=float(getattr(config, "profiler_hitch_abs_ms", 8.0)),
            hitch_rel_mult=float(getattr(config, "profiler_hitch_rel_mult", 1.5)),
            hitch_window=int(getattr(config, "profiler_hitch_window", 120)),
            max_scopes=int(getattr(config, "profiler_max_scopes", 64)),
            max_counters=int(getattr(config, "profiler_max_counters", 32)),
            recent_hitches=int(getattr(config, "profiler_recent_hitches", 16)),
        )
        return self

    def add_observer(
        self,
        on_start: Callable[[str], None],
        on_stop: Callable[[str], None],
    ) -> None:
        """Register start/stop callbacks (outermost scope enter/exit).
        Used by the PStats bridge in ``render/profiler_bridge.py``."""
        self._observers.append((on_start, on_stop))

    def add_counter_observer(self, on_counter: Callable[[str, float], None]) -> None:
        """Register a callback invoked when a counter is set/added this frame."""
        self._counter_observers.append(on_counter)

    # ------------------------------------------------------------------
    # Scope API
    # ------------------------------------------------------------------

    def scope(self, name: str) -> NullScope | _ScopeCtx:
        """
        Return a context manager timing *name* (PStats-style compound names).

        Returns a shared no-op when the profiler is disabled (single bool
        check, no allocation).

        Example
        -------
            with profiler.scope("Update:Weather"):
                weather.update(...)
        """
        if not self.enabled:
            return _NULL_SCOPE
        sid = self._sid(name)
        if sid < 0:
            return _NULL_SCOPE
        ctx = self._ctx_pool.pop() if self._ctx_pool else _ScopeCtx(self)
        ctx._sid = sid
        return ctx

    def profiled(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator timing a whole function/method under *name*."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if not self.enabled:
                    return fn(*args, **kwargs)
                with self.scope(name):
                    return fn(*args, **kwargs)

            return wrapper

        return decorator

    def start(self, name: str) -> None:
        """Manually open a scope (pair with :meth:`stop`)."""
        if not self.enabled:
            return
        sid = self._sid(name)
        if sid >= 0:
            self._start_sid(sid)

    def stop(self, name: str) -> None:
        """Manually close a scope opened with :meth:`start`.
        Raises ``ValueError`` on mismatch."""
        if not self.enabled:
            return
        sid = self._scope_ids.get(name, -1)
        if sid < 0 or not self._stack or self._stack[-1] != sid:
            innermost = self._scope_names[self._stack[-1]] if self._stack else None
            raise ValueError(
                f"profiler.stop({name!r}) does not match the innermost open scope ({innermost!r})"
            )
        self._stop_sid(sid)

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def set_counter(self, name: str, value: float) -> None:
        """Set a per-frame counter (mirrors PStats ``set_level``)."""
        if not self.enabled:
            return
        cid = self._cid(name)
        if cid < 0:
            return
        self._cur_counter[cid] = float(value)
        self._cur_counter_seen[cid] = True
        for cb in self._counter_observers:
            cb(name, float(value))

    def add_counter(self, name: str, delta: float = 1.0) -> None:
        """Add *delta* to a per-frame counter (mirrors PStats ``add_level``)."""
        if not self.enabled:
            return
        cid = self._cid(name)
        if cid < 0:
            return
        self._cur_counter[cid] += float(delta)
        self._cur_counter_seen[cid] = True
        for cb in self._counter_observers:
            cb(name, float(self._cur_counter[cid]))

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def begin_frame(self) -> None:
        """Open a new frame.  Commits the previous frame into the ring."""
        if not self.enabled:
            return
        now = self._time()
        if self._pending:
            total_ms = (now - self._frame_start_ns) / 1e6
            commit_frame(self, total_ms)
        self._frame_start_ns = now
        self._cur_scope_ns.fill(0.0)
        self._cur_scope_calls.fill(0)
        self._cur_counter.fill(0.0)
        self._cur_counter_seen.fill(False)
        self._active_depth.fill(0)
        self._stack.clear()
        self._cpu_ms = 0.0
        self._frame_cpu_ms_seen = False
        self._pending = True

    def end_frame(self) -> None:
        """Close the main-loop body.  Records ``frame_cpu_ms`` counter."""
        if not self.enabled or not self._pending:
            return
        self._cpu_ms = (self._time() - self._frame_start_ns) / 1e6
        if self._stack:
            leaked = [self._scope_names[s] for s in self._stack]
            _log.error("profiler: %d scope(s) never stopped this frame: %s", len(leaked), leaked)
            self._stack.clear()
            self._active_depth.fill(0)
        self.set_counter(self._CPU_MS_COUNTER, self._cpu_ms)

    # ------------------------------------------------------------------
    # Internal scope timing (called by _ScopeCtx / start / stop)
    # ------------------------------------------------------------------

    def _start_sid(self, sid: int) -> None:
        self._stack.append(sid)
        d = int(self._active_depth[sid])
        if d == 0:
            self._active_start_ns[sid] = self._time()
            name = self._scope_names[sid]
            for on_start, _ in self._observers:
                on_start(name)
        self._active_depth[sid] = d + 1
        self._cur_scope_calls[sid] += 1

    def _stop_sid(self, sid: int) -> None:
        if self._stack and self._stack[-1] == sid:
            self._stack.pop()
        elif sid in self._stack:
            _log.error("profiler: out-of-order stop of scope %r", self._scope_names[sid])
            self._stack.remove(sid)
        d = int(self._active_depth[sid]) - 1
        if d < 0:
            d = 0
        self._active_depth[sid] = d
        if d == 0:
            elapsed = self._time() - self._active_start_ns[sid]
            self._cur_scope_ns[sid] += float(elapsed)
            name = self._scope_names[sid]
            for _, on_stop in self._observers:
                on_stop(name)

    def _release_ctx(self, ctx: _ScopeCtx) -> None:
        ctx._sid = -1
        self._ctx_pool.append(ctx)

    # ------------------------------------------------------------------
    # Name → column id registries
    # ------------------------------------------------------------------

    def _sid(self, name: str) -> int:
        sid = self._scope_ids.get(name)
        if sid is not None:
            return sid
        if len(self._scope_names) >= self.max_scopes:
            self._warn_capacity("scope", name)
            return -1
        sid = len(self._scope_names)
        self._scope_ids[name] = sid
        self._scope_names.append(name)
        return sid

    def _cid(self, name: str) -> int:
        cid = self._counter_ids.get(name)
        if cid is not None:
            return cid
        if len(self._counter_names) >= self.max_counters:
            self._warn_capacity("counter", name)
            return -1
        cid = len(self._counter_names)
        self._counter_ids[name] = cid
        self._counter_names.append(name)
        return cid

    def _warn_capacity(self, kind: str, name: str) -> None:
        if not self._capacity_warned:
            _log.warning(
                "profiler: %s capacity reached — %r and further new %ss are "
                "DROPPED (raise profiler_max_%ss in config). A dropped %s reads "
                "as zero cost, not free.",
                kind,
                name,
                kind,
                kind,
                kind,
            )
            self._capacity_warned = True

    # ------------------------------------------------------------------
    # Stats / snapshot (commit + hitch + snapshot impl in profiler_report.py)
    # ------------------------------------------------------------------

    def _valid_slice(self) -> tuple[slice, int]:
        valid = min(self._frames_written, self.history_frames)
        if valid == 0:
            return slice(0, 0), 0
        if self._frames_written <= self.history_frames:
            return slice(0, self._write_index), valid
        return slice(0, self.history_frames), valid

    def frame_count(self) -> int:
        """Number of frames currently held in the ring buffer."""
        return min(self._frames_written, self.history_frames)

    def recent_frame_ms(self, n: int) -> np.ndarray:
        """Return the last *n* frame times (oldest→newest) for the overlay graph."""
        if not self.enabled or self._recent_count == 0:
            return np.zeros(0, dtype=np.float64)
        n = min(n, self._recent_count)
        if self._recent_count < self.hitch_window:
            chrono = self._recent_ms[: self._recent_count]
        else:
            chrono = np.concatenate(
                (self._recent_ms[self._recent_index :], self._recent_ms[: self._recent_index])
            )
        return chrono[-n:].copy()

    def snapshot(self) -> dict[str, Any]:
        """Plain-dict performance summary (AI-agent / overlay contract).
        Delegates to :func:`~fire_engine.core.profiler_report.build_snapshot`."""
        return build_snapshot(self)

    def write_snapshot(self, path: str) -> None:
        """Atomically write :meth:`snapshot` to *path* as JSON.  No-op when disabled."""
        write_profiler_snapshot(self, path)

    # ------------------------------------------------------------------
    # Convenience for the overlay
    # ------------------------------------------------------------------

    @property
    def last_frame_ms(self) -> float:
        """Most recently committed full frame time in ms (0 if none)."""
        if not self.enabled or self._frames_written == 0:
            return 0.0
        last = (self._write_index - 1) % self.history_frames
        return float(self._frame_ms[last])

    @property
    def hitch_count(self) -> int:
        """Total hitches detected since (re)configuration."""
        return self._hitch_count

    @property
    def recent_hitch(self) -> dict[str, Any] | None:
        """The most recent hitch record (or None)."""
        return self._hitches[0] if self._hitches else None


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_PROFILER = Profiler(enabled=False)


def get_profiler() -> Profiler:
    """Return the process-wide :class:`Profiler` singleton (disabled until boot)."""
    return _PROFILER


def init_profiler(config: Config) -> Profiler:
    """Configure the singleton from :class:`Config` and return it.  Call once at boot."""
    return _PROFILER.configure_from_config(config)
