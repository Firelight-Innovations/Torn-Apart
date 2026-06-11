"""
tools/shimmer_probe.py — quantify MOTION shimmer ("z-fighting" sparkle) in the
terrain renderer, headless-ish (uses a real GL window like screenshot.py).

Why this exists
---------------
Shimmer is *temporal* aliasing: it only shows up when the camera moves
sub-pixel amounts between frames.  A still frame cannot show it, and the
naive validators tried before all failed (documented in
docs/sessions/lighting-shimmer-gi-handoff.md):

  * frame-to-frame diff at a fixed pose is exactly 0 (deterministic renderer);
  * jittering ``transform.local_rotation`` is overwritten every frame by
    FlyController.update() (fly_controller.py:169 recomposes the rotation
    from its internal ``yaw``/``pitch`` floats);
  * capturing one ``taskMgr.step()`` after a lens change grabs a STALE
    framebuffer (>=1 frame of pipeline latency).

This probe avoids all three:

  1. it rotates the camera by writing ``FlyController.yaw`` directly
     (the one channel the controller itself honours);
  2. it steps several frames per pose so the captured frame is fresh and
     auto-exposure has settled;
  3. it carries two built-in controls that VALIDATE THE HARNESS each run:
       - static control: two captures at the identical pose must match
         almost exactly (catches real-time animation polluting the metric);
       - positive control: a multi-pixel rotation must produce a large
         diff (catches stale-framebuffer capture — the old failure).

Metric
------
The camera sweeps ``--poses`` poses separated by ``--step-px`` screen pixels
of yaw (default 0.25 px — genuine sub-pixel motion).  For each consecutive
pair we compute the per-pixel max-channel absolute difference and report the
fraction of pixels whose value flips by more than ``--threshold`` (default
0.12 in [0,1] units).  A band-limited image shifted by a quarter pixel only
changes proportionally to its (bounded) gradient, so its flip fraction is
tiny; hard hash/normal/shadow texels flip whole palette steps and score high.
The fraction is reported for the full frame and for three horizontal bands
(bottom/mid/top thirds: near ground / far ground / horizon+sky at a
downward pitch).

Usage
-----
    .venv\\Scripts\\python.exe tools\\shimmer_probe.py --out diag\\probe_baseline
    .venv\\Scripts\\python.exe tools\\shimmer_probe.py --gi-room --inside \\
        --time-of-day 13.0 --out diag\\probe_giroom

Outputs (under tools/out/<out>/):
    report.txt           the printed table
    heatmap.png          per-pixel max diff across the sub-pixel sweep
    frame_first.png / frame_last.png   the first/last captured poses

The final line on stdout is ``SHIMMER_RESULT <flip fraction>`` for scripting.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_GAME_DAY_S = 24.0 * 3600.0


def _grab(cap_tex, w: int, h: int) -> np.ndarray:
    """Copy the RTM_copy_ram capture texture to a float32 (H, W, 3) array."""
    buf = cap_tex.get_ram_image_as("RGB")
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    return arr.astype(np.float32) / 255.0


def _flip_fraction(a: np.ndarray, b: np.ndarray, threshold: float):
    """(full, bottom, mid, top) fraction of pixels with max-channel |diff|
    above *threshold*.  Row 0 of the GL RAM image is the BOTTOM of the
    screen, so band 0 = bottom third = nearest ground at a downward pitch."""
    d = np.abs(a - b).max(axis=2)
    h = d.shape[0]
    bands = [d[i * h // 3:(i + 1) * h // 3] for i in range(3)]
    return (float((d > threshold).mean()),
            *[float((bb > threshold).mean()) for bb in bands])


def probe(args) -> float:
    import main as demo
    from fire_engine.core.math3d import Vec3, Quat
    from fire_engine.player.fly_controller import FlyController
    from panda3d.core import Texture, GraphicsOutput
    from tools.screenshot import _apply_sky_settings

    app = demo.build_demo()

    # No physical-mouse interference during the unattended run.
    app.input_state.mouse_captured = False
    app._set_mouse_capture(False)

    # Initial pose BEFORE the first step: FlyController.awake() reads it.
    yaw0 = math.radians(args.yaw)
    app.camera_go.transform.local_rotation = (
        Quat.from_axis_angle(Vec3.UP, yaw0)
        * Quat.from_axis_angle(Vec3.RIGHT, math.radians(args.pitch))
    ).normalized()
    if args.height is not None:
        pos = app.camera_go.transform.position
        app.camera_go.transform.position = Vec3(pos.x, pos.y, float(args.height))

    _apply_sky_settings(app, args.time_of_day, args.weather)

    if args.explode:
        # Carve an open SURFACE crater 10 m ahead (centre at ground height so
        # the bowl opens to the sky — screenshot.py's cam.z-8 placement carves
        # a sealed underground cave that never shows) so the probe sweeps the
        # shadowed dirt walls the owner sees shimmering after shooting.
        from fire_engine.terrain import SphereBrush, BrushMode, apply_brush
        cam = app.camera_go.transform.position
        gz = float(app._config.ground_height_m)
        apply_brush(SphereBrush(3.0), Vec3(cam.x, cam.y + 10.0, gz),
                    BrushMode.REMOVE, material=1,
                    chunk_provider=app.chunk_manager.get_or_create,
                    bus=app._event_bus)

    if args.gi_room:
        cx, cy, z0 = demo.build_gi_test_room(app)
        if args.inside:
            app.camera_go.transform.position = Vec3(cx, cy - 4.2, z0 + 2.4)
            app.camera_go.transform.local_rotation = Quat.from_axis_angle(
                Vec3.RIGHT, math.radians(-7.0)).normalized()

    # Continuous RAM copy of every rendered frame — no save/load round trip,
    # no stale ``get_screenshot`` between flips.
    cap = Texture("probe_capture")
    app.win.add_render_texture(cap, GraphicsOutput.RTM_copy_ram)

    hold_tod = float(app._clock.game_time_of_day)

    def step(n: int) -> None:
        for _ in range(n):
            app.taskMgr.step()
            app._clock.game_time_of_day = hold_tod   # freeze the sun

    # Warmup: stream chunks, assemble cascades, settle auto-exposure.
    step(args.frames)

    # Grass sways per-frame; it would swamp the metric.  Hide it (the root
    # only exists after the first frame, hence after the warmup).
    grass_go = getattr(app, "grass_go", None)
    if grass_go is not None:
        from fire_engine.world.grass_renderer import GrassRendererComponent
        gc = grass_go.get_component(GrassRendererComponent)
        if gc is not None and getattr(gc, "_root", None) is not None:
            gc._root.hide()
            step(2)

    fly = app.camera_go.get_component(FlyController)
    if fly is None:
        raise SystemExit("no FlyController on the camera GameObject")

    w, h = app.win.get_x_size(), app.win.get_y_size()
    fov_x = float(app.camLens.get_fov()[0])
    deg_per_px = fov_x / float(w)
    step_rad = math.radians(deg_per_px * args.step_px)

    lines: list[str] = []

    def say(msg: str) -> None:
        print(msg)
        lines.append(msg)

    say(f"window {w}x{h}  fov_x {fov_x:.2f} deg  "
        f"({deg_per_px * 60.0:.2f} arcmin/px)  "
        f"step {args.step_px} px  settle {args.settle} frames/pose  "
        f"threshold {args.threshold}")

    # ---- static control: same pose twice -> must be ~identical -----------
    step(args.settle)
    f_a = _grab(cap, w, h)
    step(args.settle)
    f_b = _grab(cap, w, h)
    static_frac = _flip_fraction(f_a, f_b, args.threshold)[0]
    static_mean = float(np.abs(f_a - f_b).mean())
    say(f"static control  flip fraction {static_frac:.5f}  "
        f"mean |diff| {static_mean:.6f} "
        f"(must be ~0; nonzero = real-time animation polluting the metric)")

    # ---- positive control: an 8 px rotation -> must be clearly visible ---
    # Gate on MEAN |diff| (not the flip fraction): a well-filtered or dim
    # scene legitimately has few above-threshold flips even for a multi-pixel
    # shift, but a real shift always moves the mean; a stale capture moves
    # neither.
    fly.yaw = fly.yaw - math.radians(deg_per_px * 8.0)
    step(args.settle)
    f_c = _grab(cap, w, h)
    pos_frac = _flip_fraction(f_b, f_c, args.threshold)[0]
    pos_mean = float(np.abs(f_b - f_c).mean())
    say(f"positive control (8 px)  flip fraction {pos_frac:.5f}  "
        f"mean |diff| {pos_mean:.6f} "
        f"(mean must be >> static; ~static = STALE CAPTURE, harness broken)")
    if pos_mean < max(1e-4, static_mean * 4.0):
        say("HARNESS BROKEN: positive control did not register — aborting")
        return -1.0
    fly.yaw = fly.yaw + math.radians(deg_per_px * 8.0)   # restore
    step(args.settle)

    # ---- the sub-pixel sweep ---------------------------------------------
    frames: list[np.ndarray] = []
    for k in range(args.poses):
        fly.yaw = fly.yaw - (step_rad if k > 0 else 0.0)
        step(args.settle)
        frames.append(_grab(cap, w, h))

    fulls, b0s, b1s, b2s = [], [], [], []
    heat = np.zeros((h, w), dtype=np.float32)
    for a, b in zip(frames, frames[1:]):
        full, bot, mid, top = _flip_fraction(a, b, args.threshold)
        fulls.append(full); b0s.append(bot); b1s.append(mid); b2s.append(top)
        heat = np.maximum(heat, np.abs(a - b).max(axis=2))

    say("")
    say(f"sub-pixel sweep ({args.poses} poses x {args.step_px} px):")
    say(f"  flip fraction  full   {np.mean(fulls):.5f}")
    say(f"  flip fraction  bottom {np.mean(b0s):.5f}   (near ground)")
    say(f"  flip fraction  middle {np.mean(b1s):.5f}   (far ground)")
    say(f"  flip fraction  top    {np.mean(b2s):.5f}   (horizon/sky)")
    say(f"  static floor          {static_frac:.5f}")

    # ---- artifacts ---------------------------------------------------------
    out_dir = _REPO_ROOT / "tools" / "out" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(
        (np.clip(heat * 3.0, 0.0, 1.0) * 255).astype(np.uint8)[::-1]
    ).save(out_dir / "heatmap.png")
    Image.fromarray(
        (frames[0] * 255).astype(np.uint8)[::-1]).save(out_dir / "frame_first.png")
    Image.fromarray(
        (frames[-1] * 255).astype(np.uint8)[::-1]).save(out_dir / "frame_last.png")
    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    say(f"artifacts in {out_dir}")

    return float(np.mean(fulls))


def main() -> None:
    p = argparse.ArgumentParser(description="Measure terrain motion shimmer.")
    p.add_argument("--frames", type=int, default=240,
                   help="warmup frames (default 240)")
    p.add_argument("--poses", type=int, default=13,
                   help="poses in the sub-pixel sweep (default 13)")
    p.add_argument("--step-px", type=float, default=0.25,
                   help="yaw step between poses, screen pixels (default 0.25)")
    p.add_argument("--settle", type=int, default=8,
                   help="frames to settle per pose (default 8)")
    p.add_argument("--threshold", type=float, default=0.12,
                   help="per-pixel flip threshold in [0,1] (default 0.12)")
    p.add_argument("--time-of-day", type=float, default=12.0)
    p.add_argument("--weather", default="clear")
    p.add_argument("--pitch", type=float, default=-12.0)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--height", type=float, default=None)
    p.add_argument("--explode", action="store_true",
                   help="carve a crater ahead of the camera before probing")
    p.add_argument("--gi-room", action="store_true")
    p.add_argument("--inside", action="store_true")
    p.add_argument("--out", default="diag/probe",
                   help="artifact dir under tools/out/ (default diag/probe)")
    args = p.parse_args()

    score = probe(args)
    print(f"SHIMMER_RESULT {score:.5f}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
