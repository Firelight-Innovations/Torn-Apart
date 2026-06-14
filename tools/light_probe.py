"""
tools/light_probe.py — read back the GPU lighting cascades and report ground truth.

Boots the demo exactly like tools/screenshot.py (window-class tool, not part of
the headless suite), spawns a 3 m occluder cube ahead of the camera, settles N
frames, then pulls the cascade textures (vis / radiance / geom) off the GPU via
``extract_texture_data`` and reports what the *data* actually contains:

- per-cascade horizontal slices of sun visibility at ground height, saved as
  upscaled PNGs (``tools/out/diag/probe_c{i}_sunvis.png``) — shows the true
  resolution of the computed shadow field, independent of the surface shader;
- a numeric sun-vis profile through the occluder shadow (edge-crispness check);
- radiance + geom-albedo statistics in the air band just above the ground
  (is GI bounce energy present in the volume at all?).

Usage
-----
    python tools/light_probe.py                  # 16:00 clear, occluder cube
    python tools/light_probe.py --time-of-day 12 --frames 300
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

_OUT = _REPO_ROOT / "tools" / "out" / "diag"


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


def _save_gray(img01: np.ndarray, path: Path, scale: int = 6) -> None:
    from PIL import Image
    a = np.clip(img01, 0.0, 1.0)
    a = np.kron(a, np.ones((scale, scale)))
    Image.fromarray((a * 255).astype(np.uint8), "L").save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--time-of-day", type=float, default=16.0)
    args = ap.parse_args()

    import main as demo
    from fire_engine.core.math3d import Vec3, Quat
    from fire_engine.world.terrain import raycast_voxel

    sys.path.insert(0, str(_REPO_ROOT / "tools"))
    from screenshot import _apply_sky_settings

    app = demo.build_demo()
    app.input_state.mouse_captured = False
    app._set_mouse_capture(False)
    app.camera_go.transform.local_rotation = Quat.from_axis_angle(
        Vec3.RIGHT, math.radians(-30.0)).normalized()
    _apply_sky_settings(app, args.time_of_day, "clear")

    # 3 m occluder cube ~6 m ahead (same path as screenshot.py --occluder).
    cam = app.camera_go.transform.position
    fwd = app.camera_go.transform.forward
    gx, gy = cam.x + fwd.x * 6.0, cam.y + fwd.y * 6.0
    hit = raycast_voxel(Vec3(gx, gy, cam.z + 40.0), Vec3(0.0, 0.0, -1.0),
                        app.chunk_manager.get_or_create, max_distance_m=90.0)
    gz = hit.point.z if hit is not None else 8.0
    go = app.dev_overlay.spawn_cube()
    go.transform.position = Vec3(gx, gy, gz + 1.5)
    go.transform.local_scale = Vec3(3.0, 3.0, 3.0)
    print(f"OCCLUDER at ({gx:.1f}, {gy:.1f}, {gz + 1.5:.1f}), ground z={gz:.2f}")

    hold = float(app._clock.game_time_of_day)
    for _ in range(args.frames):
        app.taskMgr.step()
        app._clock.game_time_of_day = hold

    pipe = app.lighting_pipeline
    if pipe is None:
        print("NO GPU LIGHTING PIPELINE (cpu backend?) - nothing to probe")
        return
    _OUT.mkdir(parents=True, exist_ok=True)

    for casc in pipe.cascades:
        i = casc.index
        cell = casc.cell_m
        ox, oy, oz = casc.origin_m()
        vis = _tex_to_numpy(app, casc.vis)
        rad = _tex_to_numpy(app, casc.radiance_current)
        geom = _tex_to_numpy(app, casc.geom)

        # Horizontal slice one cell above the ground under the occluder.
        kz = int((gz + 0.6 * cell - oz) / cell)
        kz = max(0, min(casc.cells - 1, kz))
        sun = vis[kz, :, :, 0]                       # [y, x] sun visibility
        _save_gray(sun, _OUT / f"probe_c{i}_sunvis.png",
                   scale=max(2, int(12 * cell)))
        print(f"\n=== cascade {i} (cell {cell} m, origin "
              f"({ox:.1f},{oy:.1f},{oz:.1f}), slice z-index {kz} "
              f"= world z {oz + (kz + 0.5) * cell:.2f}) ===")

        # Sun-vis profile along x through the occluder centre line.
        jy = int((gy - oy) / cell)
        ix = int((gx - ox) / cell)
        if 0 <= jy < casc.cells:
            lo, hi = max(0, ix - 12), min(casc.cells, ix + 13)
            prof = " ".join(f"{v:.2f}" for v in sun[jy, lo:hi])
            print(f"sun-vis x-profile at occluder row (x {lo}..{hi - 1}): {prof}")

        # Air band just above ground across the whole window: GI energy check.
        gz0 = int((gz + 0.5 * cell - oz) / cell)
        band = slice(max(0, gz0), min(casc.cells, gz0 + 2))
        occ = geom[band, :, :, 3]
        air = occ < 0.5
        if air.any():
            r = rad[band][air]
            print(f"radiance in ground air band: mean rgb "
                  f"({r[:, 0].mean():.4f}, {r[:, 1].mean():.4f}, "
                  f"{r[:, 2].mean():.4f})  max {r[:, :3].max():.4f}")
        solid_band = slice(max(0, gz0 - 2), max(1, gz0))
        socc = geom[solid_band, :, :, 3] > 0.5
        if socc.any():
            alb = geom[solid_band][socc]
            print(f"ground albedo in geom: mean rgb "
                  f"({alb[:, 0].mean():.3f}, {alb[:, 1].mean():.3f}, "
                  f"{alb[:, 2].mean():.3f})")

    print("\nslice PNGs in", _OUT)


if __name__ == "__main__":
    main()
