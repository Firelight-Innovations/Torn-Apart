"""
render/_impl/offscreen.py — render a world to a PNG with no visible window.

This is the render subprocess behind the editor's ``world.screenshot`` RPC (and a
handy standalone "render this save from here" tool).  The panda3d-free editor
daemon temp-saves its live session, then spawns::

    python -m fire_engine.render._impl.offscreen \\
        --save <tmp.ta> --seed <session.seed> \\
        --px 0 --py -20 --pz 12 [--yaw 0 --pitch -20] \\
        --width 1280 --height 720 --frames 180 --out shot.png

We set ``window-type offscreen`` (+ ``win-size``) via ``loadPrcFileData`` BEFORE
ShowBase is constructed — Panda3D reads those at GSG-creation time — so the engine
renders into an offscreen GraphicsBuffer instead of a window.  ``build_demo`` is
then driven exactly like ``tools/screenshot.py`` (step N frames so chunks stream
and lighting/sky settle, capture the framebuffer), but headless and parameterised
by a save + seed so the result is the *current live-edited* world.

Requires a real GL context (a GPU on the host).  On success the last stdout line
is ``SCREENSHOT_RESULT {json}`` and the process force-exits 0; on failure it exits
non-zero with the traceback on stderr (the daemon turns that into an RpcError).

Docs: docs/systems/render.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Make ``fire_engine`` (repo root) importable when run as ``python -m`` from any
# cwd, mirroring tools/screenshot.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_FRAMES = 180


def _look_at_origin_angles(px: float, py: float, pz: float) -> tuple[float, float]:
    """Yaw/pitch (degrees) that aim a camera at ``px,py,pz`` toward the origin.

    Matches the camera convention used by ``tools/screenshot.py``: forward at
    ``yaw=pitch=0`` is +Y (north); yaw rotates about world +Z, pitch about +X
    (negative pitch looks down).  Falls back to a gentle downward framing when
    the camera sits on the origin (direction undefined).

    Docs: docs/systems/render.md
    """
    dx, dy, dz = -px, -py, -pz
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-6:
        return 0.0, -20.0
    dx, dy, dz = dx / length, dy / length, dz / length
    yaw = math.degrees(math.atan2(-dx, dy))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, dz))))
    return yaw, pitch


def render_offscreen(
    save_path: str,
    seed: int,
    px: float,
    py: float,
    pz: float,
    out_path: str,
    yaw: float | None = None,
    pitch: float | None = None,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    frames: int = _DEFAULT_FRAMES,
) -> Path:
    """Render ``save_path`` (regenerated at ``seed``) to ``out_path`` offscreen.

    Parameters
    ----------
    save_path : str
        ``.ta`` save to load (terrain deltas + authored scene objects).
    seed : int
        World seed the save was written with — passed to ``build_demo`` so the
        baseline world matches and ``SaveManager.load`` accepts the header.
    px, py, pz : float
        Camera world position in meters.
    out_path : str
        Where to write the PNG.
    yaw, pitch : float | None
        Look direction in degrees (yaw about +Z, pitch about +X; negative pitch
        looks down).  When BOTH are None the camera looks at the world origin.
    width, height : int
        Framebuffer size in pixels (set via the ``win-size`` PRC).
    frames : int
        Frames to step before capture so chunks stream and lighting/sky settle.

    Returns
    -------
    Path
        The written PNG path.

    Docs: docs/systems/render.md
    """
    # MUST precede ShowBase construction (build_demo -> App -> ShowBase).
    from panda3d.core import Filename, PNMImage, loadPrcFileData

    loadPrcFileData(
        "torn-apart-offscreen",
        f"window-type offscreen\nwin-size {int(width)} {int(height)}",
    )

    import main as demo
    from fire_engine.core.math3d import Quat, Vec3

    app = demo.build_demo(load_path=save_path, seed=int(seed), headless=True)

    if yaw is None and pitch is None:
        yaw, pitch = _look_at_origin_angles(px, py, pz)
    else:
        yaw = 0.0 if yaw is None else yaw
        pitch = 0.0 if pitch is None else pitch

    app.camera_go.transform.position = Vec3(float(px), float(py), float(pz))
    app.camera_go.transform.local_rotation = (
        Quat.from_axis_angle(Vec3.UP, math.radians(yaw))
        * Quat.from_axis_angle(Vec3.RIGHT, math.radians(pitch))
    ).normalized()

    # Step the task manager so chunks stream + remesh + relight and the sky
    # settles; each step also flips the buffer so the framebuffer is valid.
    for _ in range(int(frames)):
        app.taskMgr.step()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = PNMImage()
    if app.win.get_screenshot(img):
        img.write(Filename.from_os_specific(str(out)))
    else:
        # Fallback to the ShowBase helper (can no-op between flips, hence the
        # get_screenshot path first).
        app.screenshot(str(out), defaultFilename=False)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry: parse args, render, print ``SCREENSHOT_RESULT {json}``.

    Docs: docs/systems/render.md
    """
    parser = argparse.ArgumentParser(
        prog="fire_engine.render._impl.offscreen",
        description="Render a world save to a PNG offscreen (no window).",
    )
    parser.add_argument("--save", required=True, help=".ta save to load")
    parser.add_argument(
        "--seed", type=int, required=True, help="world seed the save was written with"
    )
    parser.add_argument("--px", type=float, required=True, help="camera X (meters)")
    parser.add_argument("--py", type=float, required=True, help="camera Y (meters)")
    parser.add_argument("--pz", type=float, required=True, help="camera Z (meters)")
    parser.add_argument(
        "--yaw", type=float, default=None, help="yaw degrees (default: look at origin)"
    )
    parser.add_argument(
        "--pitch", type=float, default=None, help="pitch degrees (default: look at origin)"
    )
    parser.add_argument("--width", type=int, default=_DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=_DEFAULT_HEIGHT)
    parser.add_argument("--frames", type=int, default=_DEFAULT_FRAMES)
    parser.add_argument("--out", required=True, help="output PNG path")
    args = parser.parse_args(argv)

    path = render_offscreen(
        save_path=args.save,
        seed=args.seed,
        px=args.px,
        py=args.py,
        pz=args.pz,
        out_path=args.out,
        yaw=args.yaw,
        pitch=args.pitch,
        width=args.width,
        height=args.height,
        frames=args.frames,
    )
    size = path.stat().st_size if path.exists() else 0
    print(
        "SCREENSHOT_RESULT "
        + json.dumps({"path": str(path), "width": args.width, "height": args.height, "bytes": size})
    )
    sys.stdout.flush()
    # Force-exit so the lingering Panda3D buffer / OpenAL device don't hang the
    # daemon's proc.wait().
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
