"""
Frame-time statistics and snapshot serialisation for the Torn Apart profiler.

Moved to :mod:`fire_engine.core._impl.profiler_report` to satisfy the
per-directory module-count limit.  Contains:

- ``frame_time_stats`` — pure vectorised numpy summary over a 1-D array of
  per-frame millisecond durations, unit-testable in isolation.
- ``build_snapshot`` — assembles a plain-dict performance summary from a live
  :class:`~fire_engine.core.profiler.Profiler` (the AI-agent / overlay contract).
- ``write_profiler_snapshot`` — atomically writes the snapshot to a JSON file.

Docs: docs/systems/core.md
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from fire_engine.core.profiler import Profiler

__all__ = ["build_snapshot", "commit_frame", "frame_time_stats", "write_profiler_snapshot"]


def commit_frame(prof: Profiler, total_ms: float) -> None:
    """Commit the open frame's accumulators into *prof*'s ring buffer + run
    hitch detection.  Called by :meth:`Profiler.begin_frame` for the prior frame."""
    i = prof._write_index
    prof._frame_ms[i] = total_ms
    prof._scope_ms[i, :] = prof._cur_scope_ns / 1e6
    prof._scope_calls[i, :] = prof._cur_scope_calls
    prof._counter_val[i, :] = prof._cur_counter
    prof._counter_seen_any |= prof._cur_counter_seen
    _detect_hitch(prof, total_ms)
    ri = prof._recent_index
    prof._recent_ms[ri] = total_ms
    prof._recent_index = (ri + 1) % prof.hitch_window
    if prof._recent_count < prof.hitch_window:
        prof._recent_count += 1
    prof._write_index = (i + 1) % prof.history_frames
    prof._frames_written += 1


def _rolling_median(prof: Profiler) -> float:
    if prof._recent_count == 0:
        return 0.0
    return float(np.median(prof._recent_ms[: prof._recent_count]))


def _detect_hitch(prof: Profiler, total_ms: float) -> None:
    median = _rolling_median(prof)
    threshold = max(prof.hitch_abs_ms, prof.hitch_rel_mult * median)
    prof._last_threshold_ms = threshold
    if median <= 0.0:
        threshold = prof.hitch_abs_ms
    if total_ms <= threshold:
        return
    suspect = _prime_suspect(prof)
    rec = {"frame": int(prof._frames_written), "ms": float(total_ms), "prime_suspect": suspect}
    prof._hitch_count += 1
    prof._hitches.insert(0, rec)
    if len(prof._hitches) > prof.recent_hitches:
        prof._hitches.pop()


def _prime_suspect(prof: Profiler) -> str | None:
    ns = len(prof._scope_names)
    if ns == 0:
        return None
    valid = min(prof._frames_written, prof.history_frames)
    cur = prof._cur_scope_ns[:ns] / 1e6
    mean = prof._scope_ms[:valid, :ns].mean(axis=0) if valid > 0 else np.zeros(ns)
    delta = cur - mean
    idx = int(np.argmax(delta))
    if delta[idx] <= 0.0:
        idx = int(np.argmax(cur))
        if cur[idx] <= 0.0:
            return None
    return prof._scope_names[idx]


def frame_time_stats(frames_ms: np.ndarray, budget_ms: float) -> dict[str, float]:
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
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p99": 0.0,
            "p999": 0.0,
            "fps_mean": 0.0,
            "over_budget_pct": 0.0,
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


def build_snapshot(prof: Profiler) -> dict[str, Any]:
    """
    Build a plain-dict performance summary from a live :class:`Profiler`.

    Primitives + lists only — no live refs — so ``json.dumps(build_snapshot(prof))``
    round-trips losslessly.  Schema is stable + versioned
    (``schema_version``); see ``docs/systems/profiler.md`` for the layout.

    This function is called by :meth:`Profiler.snapshot` — call that method in
    application code; use this function directly only for unit tests that need
    to inspect the snapshot logic in isolation.

    Parameters
    ----------
    prof : Profiler

    Returns
    -------
    dict
    """
    from fire_engine.core.profiler import SCHEMA_VERSION

    sl, valid = prof._valid_slice()
    frames = prof._frame_ms[sl]
    fstats = frame_time_stats(frames, prof.frame_budget_ms)

    # Per-scope means/maxes over the valid slice (vectorized).
    ns = len(prof._scope_names)
    scopes: list[dict[str, Any]] = []
    if valid > 0 and ns > 0:
        sm = prof._scope_ms[sl, :ns]
        sc = prof._scope_calls[sl, :ns]
        scope_mean = sm.mean(axis=0)
        scope_max = sm.max(axis=0)
        calls_mean = sc.mean(axis=0)
        frame_mean = fstats["mean"]
        for sid in range(ns):
            m = float(scope_mean[sid])
            pct = (m / frame_mean * 100.0) if frame_mean > 0.0 else 0.0
            scopes.append(
                {
                    "name": prof._scope_names[sid],
                    "mean_ms": m,
                    "max_ms": float(scope_max[sid]),
                    "pct_of_frame": pct,
                    "calls_per_frame": float(calls_mean[sid]),
                }
            )
        scopes.sort(key=lambda s: s["mean_ms"], reverse=True)

    # Counters: mean over frames where the counter was ever recorded.
    counters: dict[str, float] = {}
    if valid > 0:
        for cid, cname in enumerate(prof._counter_names):
            if not prof._counter_seen_any[cid]:
                continue
            counters[cname + "_mean"] = float(prof._counter_val[sl, cid].mean())

    elapsed_s = max(1e-9, (prof._time() - prof._start_wall) / 1e9)
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frames_measured": int(valid),
        "budget_ms": prof.frame_budget_ms,
        "frame_ms": {
            "mean": fstats["mean"],
            "median": fstats["median"],
            "min": fstats["min"],
            "max": fstats["max"],
            "p99": fstats["p99"],
            "p999": fstats["p999"],
            "fps_mean": fstats["fps_mean"],
        },
        "over_budget_pct": fstats["over_budget_pct"],
        "hitches": {
            "count": int(prof._hitch_count),
            "per_second": float(prof._hitch_count) / elapsed_s,
            "threshold_ms": float(prof._last_threshold_ms),
            "recent": [dict(h) for h in prof._hitches],
        },
        "scopes": scopes,
        "counters": counters,
    }


def write_profiler_snapshot(prof: Profiler, path: str) -> None:
    """
    Atomically write the :func:`build_snapshot` dict for *prof* to *path* as
    JSON (tmp file → ``os.replace``), so a reader never sees a half-written
    file.

    Creates parent directories as needed.  No-op when disabled.

    Parameters
    ----------
    prof : Profiler
    path : str
    """
    if not prof.enabled:
        return
    snap = build_snapshot(prof)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(snap, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
