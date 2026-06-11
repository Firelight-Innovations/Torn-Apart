"""
tools/profile_stream.py — Per-substep frame profiler for the fly-around stutter.

DEV-ONLY diagnostic (Task 0 of the terrain-threading plan).  Boots the real
demo (``main.build_demo``), scripts the camera on a fast straight + turning
flight path so new terrain streams in and the GPU-lighting cascade windows
recenter, then times the main-thread cost of each per-frame substep and prints
p50/p95/p99 per substep.

The goal is to answer ONE question before any threading work: *what actually
burns the 200 ms/frame when flying?*  Candidate substeps timed separately:

  - ``stream_frame``        terrain gen + mesh (CPU, terrain/)
  - ``terrain_upload``      pending_meshes -> GeomNode -> scene graph (CPU, world/)
  - ``lighting.update``     GPU-lighting per-frame driver (cascade reassembly,
                            injection dispatch) — split into:
      - ``assemble_geometry`` CPU voxel gather into the cascade volume
      - ``volume_upload``     3-D texture upload to the GPU
      - ``exposure_meter``    per-frame chunk walk for auto-exposure
  - ``surface_inputs``      lighting.update_surface_inputs (terrain uniforms)
  - ``frame_total``         wall time of the whole taskMgr.step()

Run:  python tools/profile_stream.py [num_frames]

Opens a window briefly, flies the scripted path, prints a table, exits.  Not
imported by the engine; lives in tools/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Repo root on sys.path so ``import main`` / ``fire_engine`` resolve when run
# as ``python tools/profile_stream.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fire_engine.core.math3d import Vec3


class _Stat:
    """Accumulates per-call millisecond samples for one labelled substep."""

    __slots__ = ("samples", "count")

    def __init__(self) -> None:
        self.samples: list[float] = []
        self.count: int = 0

    def add(self, ms: float) -> None:
        self.samples.append(ms)
        self.count += 1

    def pct(self, p: float) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        i = min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))
        return s[i]

    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    def total(self) -> float:
        return sum(self.samples)


_STATS: dict[str, _Stat] = {}


def _stat(label: str) -> _Stat:
    s = _STATS.get(label)
    if s is None:
        s = _STATS[label] = _Stat()
    return s


def _wrap(obj, attr: str, label: str) -> None:
    """Monkeypatch ``obj.attr`` with a wall-clock timer feeding ``_stat(label)``."""
    orig = getattr(obj, attr)

    def timed(*a, _orig=orig, _label=label, **kw):
        t0 = time.perf_counter()
        try:
            return _orig(*a, **kw)
        finally:
            _stat(_label).add((time.perf_counter() - t0) * 1000.0)

    setattr(obj, attr, timed)


def _install_probes(app) -> None:
    """Wrap the per-frame substeps we want to time on the real app."""
    cm = app.chunk_manager
    _wrap(cm, "stream_frame", "stream_frame")

    # Upload cost = _stream_and_upload_terrain minus the stream_frame it calls.
    # We time the whole method and subtract stream_frame at report time.
    _wrap(app, "_stream_and_upload_terrain", "stream_and_upload")

    pipe = getattr(app, "lighting_pipeline", None)
    if pipe is not None:
        _wrap(pipe, "update", "lighting_update")
        _wrap(pipe, "update_surface_inputs", "surface_inputs")
        if hasattr(pipe, "exposure_meter"):
            _wrap(pipe.exposure_meter, "update", "exposure_meter")
        # Module-level helpers used inside lighting_update.
        import fire_engine.lighting.gpu as gpu_mod
        if hasattr(gpu_mod, "assemble_geometry"):
            _wrap(gpu_mod, "assemble_geometry", "assemble_geometry")
        if hasattr(gpu_mod, "_upload_volume"):
            _wrap(gpu_mod, "_upload_volume", "volume_upload")


def _flight_path(i: int, n: int) -> Vec3:
    """
    Scripted camera position for frame ``i``: a fast sprint out from spawn with
    a couple of direction changes, climbing slightly — guaranteed to cross many
    fresh chunks and recenter every cascade window repeatedly.
    """
    import math
    # ~50 m/s feel: advance ~0.8 m per frame, with gentle XY turning + climb.
    t = i * 0.8
    x = math.sin(i * 0.01) * t * 0.5
    y = t
    z = 12.0 + math.sin(i * 0.02) * 6.0
    return Vec3(float(x), float(y), float(z))


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    warmup = 60

    print(f"[profile_stream] booting demo, will fly {n} frames "
          f"(after {warmup} warmup)...")
    import main as demo
    app = demo.build_demo()
    _install_probes(app)

    cam = app.camera_go.transform

    # Warmup: let prewarm settle and JIT/driver caches warm before timing.
    for i in range(warmup):
        cam.local_position = _flight_path(i, n)
        app.taskMgr.step()
    _STATS.clear()  # discard warmup samples

    t_start = time.perf_counter()
    for i in range(warmup, warmup + n):
        cam.local_position = _flight_path(i, n)
        f0 = time.perf_counter()
        app.taskMgr.step()
        _stat("frame_total").add((time.perf_counter() - f0) * 1000.0)
    wall = time.perf_counter() - t_start

    # Derived: terrain upload = stream_and_upload - stream_frame (per total).
    su = _stat("stream_and_upload").total()
    sf = _stat("stream_frame").total()
    upload_total = max(0.0, su - sf)

    ft = _stat("frame_total")
    avg_frame = ft.mean()
    fps = 1000.0 / avg_frame if avg_frame > 0 else 0.0

    print(f"\n[profile_stream] {n} frames in {wall:.2f}s — "
          f"avg {avg_frame:.1f} ms/frame (~{fps:.0f} fps)\n")
    hdr = f"{'substep':<22}{'calls':>7}{'p50 ms':>10}{'p95 ms':>10}" \
          f"{'p99 ms':>10}{'total ms':>11}{'% frame':>9}"
    print(hdr)
    print("-" * len(hdr))

    frame_total_ms = ft.total()

    def row(label: str, total_ms: float, calls: int,
            p50: float, p95: float, p99: float) -> None:
        pct = (total_ms / frame_total_ms * 100.0) if frame_total_ms else 0.0
        print(f"{label:<22}{calls:>7}{p50:>10.2f}{p95:>10.2f}"
              f"{p99:>10.2f}{total_ms:>11.1f}{pct:>8.1f}%")

    order = [
        "stream_frame", "lighting_update", "assemble_geometry",
        "volume_upload", "exposure_meter", "surface_inputs", "frame_total",
    ]
    for label in order:
        s = _STATS.get(label)
        if s is None:
            continue
        row(label, s.total(), s.count, s.pct(50), s.pct(95), s.pct(99))
    # Derived terrain upload row (no per-call percentiles available).
    su_stat = _STATS.get("stream_and_upload")
    if su_stat is not None:
        row("terrain_upload(der.)", upload_total, su_stat.count, 0.0, 0.0, 0.0)

    print("\n[profile_stream] Note: 'lighting_update' is the umbrella; "
          "'assemble_geometry' + 'volume_upload' + 'exposure_meter' are its "
          "CPU sub-costs.  A substep dominating '% frame' is the real bottleneck.")

    # Clean exit without entering the blocking main loop.
    try:
        app.userExit()
    except Exception:
        pass


if __name__ == "__main__":
    main()
