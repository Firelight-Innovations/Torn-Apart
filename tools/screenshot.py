"""
tools/screenshot.py — boot the demo, render N frames, save a PNG, exit.

A CI / verification smoke test for the render path: it boots the full demo
(`main.build_demo`), optionally fires a few explosions and moves the camera,
steps Panda3D's task manager for a fixed number of frames so terrain streams
and lights, captures the framebuffer to `tools/out/<name>.png`, and exits
cleanly without entering the blocking main loop.

Usage
-----
    python tools/screenshot.py                       # default demo shot
    python tools/screenshot.py --frames 240 --out spawn.png
    python tools/screenshot.py --explode             # carve a crater first

Notes
-----
- Requires a graphics pipe (a window is created off to the side; we render
  into it and grab the framebuffer).  On a truly headless box without GL this
  will fail at window creation — that is expected; this is a `window`-class
  tool, not part of the headless pytest suite.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the repo root importable when run as `python tools/screenshot.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def capture(frames: int, out_name: str, explode: bool) -> Path:
    """
    Build the demo, step `frames` frames, write the framebuffer to a PNG.

    Parameters
    ----------
    frames : int
        Number of frames to step before capturing (gives streaming + lighting
        time to settle).
    out_name : str
        File name (under tools/out/) to write.
    explode : bool
        If True, fire a downward explosion near spawn before capturing so the
        screenshot shows a crater + the light shaft into it (the "money shot").

    Returns
    -------
    Path
        The written PNG path.
    """
    import main as demo
    from torn_apart.core.math3d import Vec3
    from torn_apart.terrain import SphereBrush, BrushMode, apply_brush

    app = demo.build_demo()

    # Point the camera somewhat downward so terrain fills the frame.
    from torn_apart.core.math3d import Quat
    import math
    app.camera_go.transform.local_rotation = Quat.from_axis_angle(
        Vec3.RIGHT, math.radians(-35.0)
    )

    if explode:
        # Carve a crater straight below the camera so the relit interior shows.
        cam = app.camera_go.transform.position
        center = Vec3(cam.x, cam.y + 10.0, cam.z - 8.0)
        apply_brush(
            SphereBrush(3.0),
            center,
            BrushMode.REMOVE,
            material=1,
            chunk_provider=app.chunk_manager.get_or_create,
            bus=app._event_bus,
        )

    # Step the task manager so chunks stream, remesh, and relight.  Each step
    # runs the frame task AND flips the window, so the framebuffer is valid.
    for _ in range(frames):
        app.taskMgr.step()

    out_dir = _REPO_ROOT / "tools" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    # Capture the framebuffer into a PNMImage and write it (more reliable than
    # win.save_screenshot, which can no-op if called between flips).
    from panda3d.core import PNMImage, Filename
    img = PNMImage()
    ok = app.win.get_screenshot(img)
    if ok:
        img.write(Filename.from_os_specific(str(out_path)))
    else:
        # Fallback to the ShowBase helper.
        app.screenshot(str(out_path), defaultFilename=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a demo screenshot.")
    parser.add_argument("--frames", type=int, default=180,
                        help="frames to step before capture (default 180)")
    parser.add_argument("--out", default="demo.png",
                        help="output PNG name under tools/out/")
    parser.add_argument("--explode", action="store_true",
                        help="carve a crater before capturing")
    args = parser.parse_args()

    path = capture(args.frames, args.out, args.explode)
    # Report on stdout for CI.
    size = os.path.getsize(path) if path.exists() else 0
    print(f"SCREENSHOT_RESULT wrote {path} ({size} bytes)")
    sys.stdout.flush()
    # Force-exit so the lingering Panda3D window / OpenAL device don't hang us.
    os._exit(0)


if __name__ == "__main__":
    main()
