"""
core/profiler.py — the engine-agnostic performance profiler (no panda3d).

This is the headless heart of Fire Engine's performance instrumentation.  It
times named, hierarchical scopes per frame, rolls them into a preallocated
numpy ring buffer, computes percentile / hitch statistics on demand, and emits
a plain-dict ``snapshot()`` that an AI coding agent can read from a JSON file to
see exactly which engine stage is slow or stuttering.  It imports only ``core``
siblings + numpy and has zero knowledge of rendering — the Panda3D PStats mirror
and the in-game overlay live in ``world/`` and attach via the observer hook.

Why it exists
-------------
The target is **200+ FPS at max settings → a 5 ms total-frame budget**, and the
owner prefers a steady 150 FPS over a 220 FPS average that hitches.  So this
module makes two things first-class:

  1. **Milliseconds against the 5 ms budget**, not just FPS.
  2. **Stutter / hitch detection** — every spike is counted, timestamped, and
     attributed to the scope that contributed the most to *that* frame.

Frame-time convention
----------------------
``frame_ms`` is the **wall-clock time between two successive ``begin_frame()``
calls** — the true full frame, including the GPU render / flip / vsync wait that
happens after the main-loop body returns.  That is the honest number to compare
against a *total*-frame budget.  The per-scope sums are the CPU/Python stage
costs measured inside the loop body; the gap ``frame_ms − sum(top-level
scopes)`` is render + overhead (also visible in PStats as Cull/Draw/Flip).
``end_frame()`` additionally records the loop-body (CPU) time as the
``frame_cpu_ms`` counter, so the CPU-vs-total split is always available.

Determinism
-----------
The profiler never calls ``core.rng``, introduces no randomness, and never
touches simulation state or save deltas — timing is observational only.  A
``time_source`` callable (default :func:`time.perf_counter_ns`) is injectable so
tests can feed synthetic timestamps and get exact answers.

Cost
----
Nearly free enabled, truly free disabled.  Timing uses ``perf_counter_ns()``
integer adds into a preallocated numpy array — no per-frame heap allocation
(scope context objects are pooled).  When disabled, ``scope()`` returns a shared
no-op object, so ``with profiler.scope(...)`` is a single boolean check.

Example
-------
    from fire_engine.core.profiler import get_profiler

    prof = get_profiler()
    prof.begin_frame()
    with prof.scope("Update:Weather"):
        advance_weather()
    prof.set_counter("draw_calls", 412)
    prof.end_frame()
    # ... many frames later ...
    snap = prof.snapshot()           # plain dict, JSON-serializable
    print(snap["frame_ms"]["p99"], snap["hitches"]["count"])
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np

from fire_engine.core.log import get_logger

__all__ = [
    "SCHEMA_VERSION",
    "NullScope",
    "Profiler",
    "frame_time_stats",
    "get_profiler",
    "init_profiler",
]

_log = get_logger("profiler")

#: JSON snapshot schema version.  Bump only on a breaking change to the dict
#: shape — the baseline-diff tool and any AI agent parse against this.
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Pure stats helper (unit-testable in isolation against known arrays)
# ---------------------------------------------------------------------------

def frame_time_stats(frames_ms: np.ndarray, budget_ms: float) -> dict:
    """
    Compute frame-time summary statistics from an array of per-frame ms.

    Vectorized single pass (numpy) — no Python loop over frames.  All values
    are plain floats so the result is directly JSON-serializable.

    Parameters
    ----------
    frames_ms : numpy.ndarray
        1-D array of per-frame durations in **milliseconds**.  May be empty.
    budget_ms : float
        The per-frame budget in milliseconds (e.g. 5.0 for a 200 FPS target);
        used to compute ``over_budget_pct``.

    Returns
    -------
    dict
        Keys: ``mean``, ``median``, ``min``, ``max``, ``p99`` (1% low — the
        frame time the worst 1% of frames exceed), ``p999`` (0.1% low),
        ``fps_mean``, ``over_budget_pct``.  ``p99``/``p999`` are the 99th /
        99.9th percentile frame times (high = bad: long frames).

    Example
    -------
    >>> import numpy as np
    >>> s = frame_time_stats(np.array([4.0, 5.0, 6.0, 40.0]), budget_ms=5.0)
    >>> round(s["max"], 1)
    40.0
    """
    n = int(frames_ms.size)
    if n == 0:
        return {
            "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0,
            "p99": 0.0, "p999": 0.0, "fps_mean": 0.0, "over_budget_pct": 0.0,
        }
    mean = float(frames_ms.mean())
    # p99 / p999 are the *high* percentiles (slow frames) — the 1% / 0.1% lows.
    p99, p999, median = np.percentile(frames_ms, [99.0, 99.9, 50.0])
    over = float(np.count_nonzero(frames_ms > budget_ms)) / n * 100.0
    return {
        "mean": mean,
        "median": float(median),
        "min": float(frames_ms.min()),
        "max": float(frames_ms.max()),
        "p99": float(p99),
        "p999": float(p999),
        "fps_mean": (1000.0 / mean) if mean > 0.0 else 0.0,
        "over_budget_pct": over,
    }


# ---------------------------------------------------------------------------
# Scope context objects
# ---------------------------------------------------------------------------

class NullScope:
    """
    Shared no-op scope returned by a disabled :class:`Profiler`.

    ``with profiler.scope(...)`` then costs one boolean check + this object's
    trivial enter/exit — no timing, no allocation.
    """

    __slots__ = ()

    def __enter__(self) -> NullScope:
        return self

    def __exit__(self, *exc) -> bool:
        return False


#: Module-level singleton no-op scope (never holds state).
_NULL_SCOPE = NullScope()


class _ScopeCtx:
    """
    Pooled context manager for one active timing scope.

    Reused via the profiler's free-list so steady-state scope entry allocates
    nothing.  Holds only the owning profiler + the integer scope id.
    """

    __slots__ = ("_prof", "_sid")

    def __init__(self, prof: Profiler) -> None:
        self._prof = prof
        self._sid = -1

    def __enter__(self) -> _ScopeCtx:
        self._prof._start_sid(self._sid)
        return self

    def __exit__(self, *exc) -> bool:
        self._prof._stop_sid(self._sid)
        self._prof._release_ctx(self)
        return False


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class Profiler:
    """
    Per-frame hierarchical timer with a numpy ring buffer + hitch detection.

    Construct directly for headless tests, or use the process-wide singleton
    via :func:`get_profiler` / :func:`init_profiler` (the engine wires the
    singleton from :class:`~fire_engine.core.config.Config` at boot).

    Parameters
    ----------
    enabled : bool
        Master switch.  When False the profiler allocates no buffers and every
        ``scope`` is a no-op.
    frame_budget_ms : float
        Per-frame budget in ms (200 FPS = 5.0).  Drives ``over_budget_pct`` and
        the overlay colour bands.
    history_frames : int
        Ring-buffer length: how many recent frames percentile stats span.
    hitch_abs_ms : float
        Absolute floor for the hitch threshold in ms.
    hitch_rel_mult : float
        Relative multiplier: a frame is a hitch when
        ``frame_ms > max(hitch_abs_ms, hitch_rel_mult * rolling_median)``.
    hitch_window : int
        How many recent frames the rolling median (the relative threshold) is
        computed over.
    max_scopes / max_counters : int
        Capacity of the preallocated per-scope / per-counter columns.  Names
        beyond the cap are dropped with a one-time logged warning (never
        silently — a dropped scope would read as "free").
    recent_hitches : int
        How many of the most-recent hitches to keep in the snapshot.
    time_source : Callable[[], int] | None
        Monotonic nanosecond clock; defaults to ``time.perf_counter_ns``.
        Injectable for deterministic tests.

    Units & invariants
    -------------------
    - All stored times are **milliseconds** (float64); timing is taken in
      integer nanoseconds and converted once per frame.
    - ``frame_ms`` is wall-clock between successive ``begin_frame()`` calls.
    - Scope times are **inclusive** (a parent's time includes its children),
      matching PStats collectors; re-entering an already-active scope does not
      double-count (only the outermost span is measured).
    """

    # Reserved counters the profiler maintains itself.
    _CPU_MS_COUNTER = "frame_cpu_ms"

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

    # ------------------------------------------------------------------
    # Configuration / (re)allocation
    # ------------------------------------------------------------------

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
        """(Re)allocate all buffers and reset state.  Used by __init__ + the
        singleton's :func:`init_profiler`."""
        self.enabled = bool(enabled)
        self.frame_budget_ms = float(frame_budget_ms)
        self.history_frames = int(history_frames)
        self.hitch_abs_ms = float(hitch_abs_ms)
        self.hitch_rel_mult = float(hitch_rel_mult)
        self.hitch_window = max(1, int(hitch_window))
        self.max_scopes = int(max_scopes)
        self.max_counters = int(max_counters)
        self.recent_hitches = int(recent_hitches)

        # Scope / counter name registries (assigned lazily on first sight).
        self._scope_ids: dict[str, int] = {}
        self._scope_names: list[str] = []
        self._counter_ids: dict[str, int] = {}
        self._counter_names: list[str] = []
        self._capacity_warned = False

        # Pool of reusable scope context objects (steady-state alloc-free).
        self._ctx_pool: list[_ScopeCtx] = []

        # When disabled, skip all the (potentially large) buffer allocation.
        H = self.history_frames if self.enabled else 0
        S = self.max_scopes if self.enabled else 0
        C = self.max_counters if self.enabled else 0

        # Ring buffers (history_frames rows).  Row order is irrelevant to
        # percentile/mean (computed over the whole valid slice); recency is
        # tracked separately for hitch detection.
        self._frame_ms = np.zeros(H, dtype=np.float64)
        self._scope_ms = np.zeros((H, S), dtype=np.float64)
        self._scope_calls = np.zeros((H, S), dtype=np.int32)
        self._counter_val = np.zeros((H, C), dtype=np.float64)
        self._counter_seen_any = np.zeros(C, dtype=bool)

        self._write_index = 0
        self._frames_written = 0

        # Per-frame accumulators (reset each begin_frame).
        self._cur_scope_ns = np.zeros(S, dtype=np.float64)
        self._cur_scope_calls = np.zeros(S, dtype=np.int32)
        self._cur_counter = np.zeros(C, dtype=np.float64)
        self._cur_counter_seen = np.zeros(C, dtype=bool)

        # Active-scope bookkeeping (by scope id).
        self._active_depth = np.zeros(S, dtype=np.int32)
        self._active_start_ns = np.zeros(S, dtype=np.int64)
        self._stack: list[int] = []

        # Recent-frame ring for the rolling-median hitch threshold.
        W = self.hitch_window if self.enabled else 0
        self._recent_ms = np.zeros(W, dtype=np.float64)
        self._recent_index = 0
        self._recent_count = 0

        # Hitch records (most-recent-first list, capped at recent_hitches).
        self._hitches: list[dict] = []
        self._hitch_count = 0
        self._last_threshold_ms = 0.0

        # Frame timing state.
        self._frame_start_ns = 0
        self._pending = False          # a frame is open / awaiting commit
        self._cpu_ms = 0.0             # loop-body time recorded by end_frame
        self._start_wall = self._time()  # for hitches-per-second
        self._frame_cpu_ms_seen = False

    def configure_from_config(self, config) -> Profiler:
        """
        (Re)configure this profiler in place from a :class:`Config`.

        Reads the flat ``profiler_*`` fields.  Mutates in place so existing
        references (held by the registry, app, etc.) stay valid.

        Parameters
        ----------
        config : fire_engine.core.config.Config

        Returns
        -------
        Profiler — self.
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

    # ------------------------------------------------------------------
    # Observer hook (the panda3d-free seam for the PStats bridge / overlay)
    # ------------------------------------------------------------------

    def add_observer(
        self,
        on_start: Callable[[str], None],
        on_stop: Callable[[str], None],
    ) -> None:
        """
        Register start/stop callbacks invoked on the *outermost* enter/exit of
        every scope (1:1 with PStats collector start/stop).

        ``world/profiler_bridge.py`` uses this to mirror scopes into Panda3D
        PStatCollectors without ``core/`` ever importing panda3d.  Callbacks
        receive the compound scope name (e.g. ``"Update:Weather"``).
        """
        self._observers.append((on_start, on_stop))

    def add_counter_observer(self, on_counter: Callable[[str, float], None]) -> None:
        """Register a callback invoked when a counter is set/added this frame
        (mirrors PStats ``set_level``)."""
        self._counter_observers.append(on_counter)

    # ------------------------------------------------------------------
    # Scope API
    # ------------------------------------------------------------------

    def scope(self, name: str):
        """
        Return a context manager timing the named (compound) scope.

        Use PStats-style compound names: ``"Update:Terrain:Mesh"``, ``"Draw"``.
        Nesting accumulates inclusively into each named scope.

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

    def profiled(self, name: str):
        """
        Decorator timing a whole function/method under the named scope.

        Example
        -------
            @profiler.profiled("Weather:Update")
            def update(self, dt): ...
        """
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return fn(*args, **kwargs)
                with self.scope(name):
                    return fn(*args, **kwargs)
            return wrapper
        return decorator

    def start(self, name: str) -> None:
        """
        Manually open a scope (pair with :meth:`stop`).

        Prefer the ``with`` form; this exists for the rare case where start and
        stop straddle a callback boundary.  A mismatched :meth:`stop` raises.
        """
        if not self.enabled:
            return
        sid = self._sid(name)
        if sid >= 0:
            self._start_sid(sid)

    def stop(self, name: str) -> None:
        """
        Manually close a scope opened with :meth:`start`.

        Raises
        ------
        ValueError
            If *name* is not the innermost open scope (a start/stop mismatch) —
            never silently mismatches.
        """
        if not self.enabled:
            return
        sid = self._scope_ids.get(name, -1)
        if sid < 0 or not self._stack or self._stack[-1] != sid:
            innermost = (self._scope_names[self._stack[-1]]
                         if self._stack else None)
            raise ValueError(
                f"profiler.stop({name!r}) does not match the innermost open "
                f"scope ({innermost!r})")
        self._stop_sid(sid)

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def set_counter(self, name: str, value: float) -> None:
        """
        Set a per-frame counter (mirrors PStats ``set_level``).

        Counters track non-time metrics that often explain a spike — chunks
        meshed this frame, draw calls, triangles, active component count, bytes
        uploaded.  Snapshot reports each counter's mean over the history.
        """
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
        """Add to a per-frame counter (mirrors PStats ``add_level``)."""
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
        """
        Open a new frame.  Call once at the very top of the main-loop body.

        Commits the *previous* frame into the ring (its full duration is now
        known: this ``begin_frame`` minus the last one), then resets the
        per-frame accumulators.
        """
        if not self.enabled:
            return
        now = self._time()
        if self._pending:
            total_ms = (now - self._frame_start_ns) / 1e6
            self._commit_frame(total_ms)
        self._frame_start_ns = now
        # Reset accumulators for the new frame.
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
        """
        Close the main-loop body.  Call once at the very bottom of the loop.

        Records the loop-body (CPU) time as the ``frame_cpu_ms`` counter — the
        full ``frame_ms`` (incl. the render/flip that follows) is finalized at
        the next :meth:`begin_frame`.  Logs (and resets) any unbalanced scope
        stack rather than silently mismatching.
        """
        if not self.enabled or not self._pending:
            return
        self._cpu_ms = (self._time() - self._frame_start_ns) / 1e6
        if self._stack:
            leaked = [self._scope_names[s] for s in self._stack]
            _log.error("profiler: %d scope(s) never stopped this frame: %s",
                       len(leaked), leaked)
            self._stack.clear()
            self._active_depth.fill(0)
        # Record the CPU-frame counter (reserved name).
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
        # LIFO validation: pop the matching id.
        if self._stack and self._stack[-1] == sid:
            self._stack.pop()
        elif sid in self._stack:
            # Out-of-order stop — unwind to it but warn (never silent).
            _log.error("profiler: out-of-order stop of scope %r",
                       self._scope_names[sid])
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
                "as zero cost, not free.", kind, name, kind, kind, kind)
            self._capacity_warned = True

    # ------------------------------------------------------------------
    # Commit + hitch detection
    # ------------------------------------------------------------------

    def _commit_frame(self, total_ms: float) -> None:
        """Write the just-finished frame's totals into the ring and run hitch
        detection.  ``total_ms`` is the full frame wall time."""
        i = self._write_index
        self._frame_ms[i] = total_ms
        self._scope_ms[i, :] = self._cur_scope_ns / 1e6
        self._scope_calls[i, :] = self._cur_scope_calls
        self._counter_val[i, :] = self._cur_counter
        self._counter_seen_any |= self._cur_counter_seen

        # Hitch detection BEFORE pushing this frame into the rolling window, so
        # a spike never inflates its own threshold.
        self._detect_hitch(total_ms)

        # Push into the recent-frame ring (rolling-median source).
        ri = self._recent_index
        self._recent_ms[ri] = total_ms
        self._recent_index = (ri + 1) % self.hitch_window
        if self._recent_count < self.hitch_window:
            self._recent_count += 1

        # Advance the main ring.
        self._write_index = (i + 1) % self.history_frames
        self._frames_written += 1

    def _rolling_median(self) -> float:
        if self._recent_count == 0:
            return 0.0
        return float(np.median(self._recent_ms[:self._recent_count]))

    def _detect_hitch(self, total_ms: float) -> None:
        median = self._rolling_median()
        threshold = max(self.hitch_abs_ms, self.hitch_rel_mult * median)
        self._last_threshold_ms = threshold
        if median <= 0.0:
            # Not enough history yet to judge relative spikes; only the
            # absolute floor applies.
            threshold = self.hitch_abs_ms
        if total_ms <= threshold:
            return
        # Prime suspect = the scope with the largest delta ABOVE its own
        # rolling mean this frame (not the largest absolute scope — else a
        # heavy-but-steady stage like Draw always wins).
        suspect = self._prime_suspect()
        rec = {
            "frame": int(self._frames_written),
            "ms": float(total_ms),
            "prime_suspect": suspect,
        }
        self._hitch_count += 1
        self._hitches.insert(0, rec)
        if len(self._hitches) > self.recent_hitches:
            self._hitches.pop()

    def _prime_suspect(self) -> str | None:
        ns = len(self._scope_names)
        if ns == 0:
            return None
        valid = min(self._frames_written, self.history_frames)
        cur = self._cur_scope_ns[:ns] / 1e6
        if valid > 0:
            mean = self._scope_ms[:valid, :ns].mean(axis=0)
        else:
            mean = np.zeros(ns, dtype=np.float64)
        delta = cur - mean
        idx = int(np.argmax(delta))
        if delta[idx] <= 0.0:
            # No scope rose above its mean (e.g. spike was render/vsync) — pick
            # the largest absolute scope this frame so attribution is non-null.
            idx = int(np.argmax(cur))
            if cur[idx] <= 0.0:
                return None
        return self._scope_names[idx]

    # ------------------------------------------------------------------
    # Stats / snapshot
    # ------------------------------------------------------------------

    def _valid_slice(self):
        """Return the populated rows of the ring (order-independent)."""
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
        """
        Return the last *n* frame times (oldest→newest) for the overlay graph.

        Reads the recent-frame ring; clamped to what's available.  Returns an
        empty array when disabled / no data.
        """
        if not self.enabled or self._recent_count == 0:
            return np.zeros(0, dtype=np.float64)
        n = min(n, self._recent_count)
        # Reconstruct chronological order from the ring.
        if self._recent_count < self.hitch_window:
            chrono = self._recent_ms[:self._recent_count]
        else:
            chrono = np.concatenate(
                (self._recent_ms[self._recent_index:],
                 self._recent_ms[:self._recent_index]))
        return chrono[-n:].copy()

    def snapshot(self) -> dict:
        """
        Build a plain-dict performance summary (the AI-agent / overlay contract).

        Primitives + lists only — no live refs — so ``json.dumps(snapshot())``
        round-trips losslessly.  Schema is stable + versioned
        (``schema_version``); see ``docs/systems/profiler.md`` for the layout.

        Returns
        -------
        dict
        """
        sl, valid = self._valid_slice()
        frames = self._frame_ms[sl]
        fstats = frame_time_stats(frames, self.frame_budget_ms)

        # Per-scope means/maxes over the valid slice (vectorized).
        ns = len(self._scope_names)
        scopes: list[dict] = []
        if valid > 0 and ns > 0:
            sm = self._scope_ms[sl, :ns]
            sc = self._scope_calls[sl, :ns]
            scope_mean = sm.mean(axis=0)
            scope_max = sm.max(axis=0)
            calls_mean = sc.mean(axis=0)
            frame_mean = fstats["mean"]
            for sid in range(ns):
                m = float(scope_mean[sid])
                pct = (m / frame_mean * 100.0) if frame_mean > 0.0 else 0.0
                scopes.append({
                    "name": self._scope_names[sid],
                    "mean_ms": m,
                    "max_ms": float(scope_max[sid]),
                    "pct_of_frame": pct,
                    "calls_per_frame": float(calls_mean[sid]),
                })
            scopes.sort(key=lambda s: s["mean_ms"], reverse=True)

        # Counters: mean over frames where the counter was ever recorded.
        counters: dict[str, float] = {}
        if valid > 0:
            for cid, cname in enumerate(self._counter_names):
                if not self._counter_seen_any[cid]:
                    continue
                counters[cname + "_mean"] = float(self._counter_val[sl, cid].mean())

        elapsed_s = max(1e-9, (self._time() - self._start_wall) / 1e9)
        snap = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "frames_measured": int(valid),
            "budget_ms": self.frame_budget_ms,
            "frame_ms": {
                "mean": fstats["mean"], "median": fstats["median"],
                "min": fstats["min"], "max": fstats["max"],
                "p99": fstats["p99"], "p999": fstats["p999"],
                "fps_mean": fstats["fps_mean"],
            },
            "over_budget_pct": fstats["over_budget_pct"],
            "hitches": {
                "count": int(self._hitch_count),
                "per_second": float(self._hitch_count) / elapsed_s,
                "threshold_ms": float(self._last_threshold_ms),
                "recent": [dict(h) for h in self._hitches],
            },
            "scopes": scopes,
            "counters": counters,
        }
        return snap

    def write_snapshot(self, path: str) -> None:
        """
        Atomically write :meth:`snapshot` to *path* as JSON (tmp file →
        ``os.replace``), so a reader never sees a half-written file.

        Creates parent directories as needed.  No-op when disabled.
        """
        if not self.enabled:
            return
        snap = self.snapshot()
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(snap, fh, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Convenience for the overlay
    # ------------------------------------------------------------------

    @property
    def last_frame_ms(self) -> float:
        """The most recently committed full frame time in ms (0 if none)."""
        if not self.enabled or self._frames_written == 0:
            return 0.0
        last = (self._write_index - 1) % self.history_frames
        return float(self._frame_ms[last])

    @property
    def hitch_count(self) -> int:
        """Total hitches detected since (re)configuration."""
        return self._hitch_count

    @property
    def recent_hitch(self) -> dict | None:
        """The most recent hitch record (or None)."""
        return self._hitches[0] if self._hitches else None


# ---------------------------------------------------------------------------
# Process-wide singleton (mirrors the ComponentRegistry singleton pattern)
# ---------------------------------------------------------------------------

_PROFILER = Profiler(enabled=False)


def get_profiler() -> Profiler:
    """
    Return the process-wide :class:`Profiler` singleton.

    Starts disabled (a no-op) until :func:`init_profiler` wires it from config
    at boot.  Modules call this at use time so they always see the configured
    instance.
    """
    return _PROFILER


def init_profiler(config) -> Profiler:
    """
    Configure the singleton profiler from a :class:`Config` and return it.

    Mutates the existing singleton in place, so references grabbed earlier via
    :func:`get_profiler` stay valid.  Call once at boot (after config load).
    """
    return _PROFILER.configure_from_config(config)
