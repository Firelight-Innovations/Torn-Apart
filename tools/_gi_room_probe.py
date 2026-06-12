"""
tools/_gi_room_probe.py — scratch diagnostic: GI numbers inside the test room.

Boots the demo, builds the Cornell-style GI test room (main.build_gi_test_room),
settles N frames, then reads cascade-0 radiance/geometry back off the GPU and
reports the numbers the eye can't be trusted with under auto-exposure:

- interior air radiance (mean rgb + blue/red ratio — the "sky-blue floor fill"
  check: skylight should only enter through the roof hole and doorway);
- air radiance within ~1.5 m of the GREEN wall and the RED wall vs the room
  centre (colour-bleed check: g/r ratio should rise near the green wall and
  drop near the red wall).

Run the same script in a baseline worktree for a true A/B.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _tex_to_numpy(app, tex):
    """Pull a 3-D texture off the GPU -> float32 ndarray [z, y, x, rgba]."""
    gsg = app.win.get_gsg()
    if not app.graphicsEngine.extract_texture_data(tex, gsg):
        raise RuntimeError(f"extract_texture_data failed for {tex.get_name()}")
    n = tex.get_z_size()
    raw = tex.get_ram_image()
    if tex.get_component_type() == 2:        # T_float
        arr = np.frombuffer(raw, dtype=np.float32).copy()
    else:                                     # T_unsigned_byte
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0
    arr = arr.reshape(n, n, n, 4)
    return arr[..., [2, 1, 0, 3]]            # panda ram order is BGRA -> RGBA


def _report(tag: str, rad: np.ndarray, mask: np.ndarray) -> None:
    if not mask.any():
        print(f"{tag}: EMPTY MASK")
        return
    r = rad[mask]
    m = r[:, :3].mean(axis=0)
    print(f"{tag}: n={mask.sum()}  mean rgb ({m[0]:.4f}, {m[1]:.4f}, "
          f"{m[2]:.4f})  b/r {m[2] / max(m[0], 1e-6):.2f}  "
          f"g/r {m[1] / max(m[0], 1e-6):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--time-of-day", type=float, default=16.0)
    args = ap.parse_args()

    import main as demo
    from fire_engine.core.math3d import Vec3, Quat

    sys.path.insert(0, str(_REPO_ROOT / "tools"))
    from screenshot import _apply_sky_settings

    app = demo.build_demo()
    app.input_state.mouse_captured = False
    app._set_mouse_capture(False)
    app.camera_go.transform.local_rotation = Quat.from_axis_angle(
        Vec3.RIGHT, math.radians(-20.0)).normalized()
    _apply_sky_settings(app, args.time_of_day, "clear")

    cx, cy, z0 = demo.build_gi_test_room(app)
    print(f"ROOM at ({cx:.1f}, {cy:.1f}, {z0:.1f})")

    hold = float(app._clock.game_time_of_day)
    for _ in range(args.frames):
        app.taskMgr.step()
        app._clock.game_time_of_day = hold

    pipe = app.lighting_pipeline
    casc = pipe.cascades[0]
    ox, oy, oz = casc.origin_m()
    cell = casc.cell_m
    n = casc.cells
    geom = _tex_to_numpy(app, casc.geom)        # [z, y, x, rgba]
    rad = _tex_to_numpy(app, casc.radiance_current)

    ks = np.arange(n)
    wz = oz + (ks + 0.5) * cell
    wy = oy + (ks + 0.5) * cell
    wx = ox + (ks + 0.5) * cell
    WZ, WY, WX = np.meshgrid(wz, wy, wx, indexing="ij")

    occ = geom[..., 3]
    air = occ < 0.5
    interior = (air
                & (np.abs(WX - cx) < 3.4) & (np.abs(WY - cy) < 3.4)
                & (WZ > z0 + 0.3) & (WZ < z0 + 4.2))
    _report("interior air (all)", rad, interior)

    # Locate the coloured walls by their albedo in the geometry volume.
    solid = occ > 0.5
    g_wall = solid & (geom[..., 1] > 0.35) \
        & (geom[..., 1] > 2.0 * geom[..., 0]) \
        & (geom[..., 1] > 2.0 * geom[..., 2])
    r_wall = solid & (geom[..., 0] > 0.35) \
        & (geom[..., 0] > 2.0 * geom[..., 1]) \
        & (geom[..., 0] > 2.0 * geom[..., 2])
    print(f"green wall cells: {g_wall.sum()}   red wall cells: {r_wall.sum()}")

    # Air within ~1.5 m (3 cells) of each coloured wall: dilate the wall mask.
    def near(mask, cells=3):
        out = mask.copy()
        for ax in range(3):
            for sh in range(1, cells + 1):
                out |= np.roll(mask, sh, axis=ax) | np.roll(mask, -sh, axis=ax)
        return out

    _report("air near GREEN wall", rad, interior & near(g_wall))
    _report("air near RED wall  ", rad, interior & near(r_wall))

    centre = interior & (np.abs(WX - cx) < 1.2) & (np.abs(WY - cy) < 1.2)
    _report("room centre column ", rad, centre)

    # Outdoor reference: open air band 1-2 cells above ground, outside the room.
    outside = air & ((np.abs(WX - cx) > 8.0) | (np.abs(WY - cy) > 8.0)) \
        & (WZ > z0 - 0.5) & (WZ < z0 + 1.0)
    _report("outdoor ground air ", rad, outside)


if __name__ == "__main__":
    main()
