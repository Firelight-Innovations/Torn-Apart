"""
tools/_noise_probe.py — scratch diagnostic: GI gather noise statistics.

Boots the demo, builds the GI test room (main.build_gi_test_room), settles N
frames, reads cascade-0 radiance/geometry back off the GPU and reports the
STDDEV of radiance luminance among air cells in:

(a) a thin band of open ground away from the room (pure sky gather — any
    variance here is ray-fan disagreement, the terrain is flat); and
(b) the air band hugging the room's exterior walls (high-contrast visible
    sources: white walls + doorway light spill — where the blotches live).

Also prints the interior-air mean luminance (regression guard: de-noising
must not shift the overall level) per band.  Run in a baseline tree first,
then after each de-noise candidate, and compare stddevs.
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
    if tex.get_component_type() == 2:  # T_float
        arr = np.frombuffer(raw, dtype=np.float32).copy()
    else:  # T_unsigned_byte
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0
    arr = arr.reshape(n, n, n, 4)
    return arr[..., [2, 1, 0, 3]]  # panda ram order is BGRA -> RGBA


def _lum(rad: np.ndarray) -> np.ndarray:
    return 0.2126 * rad[..., 0] + 0.7152 * rad[..., 1] + 0.0722 * rad[..., 2]


def _report(tag: str, lum: np.ndarray, mask: np.ndarray, hf: np.ndarray) -> None:
    if not mask.any():
        print(f"{tag}: EMPTY MASK")
        return
    v = lum[mask]
    h = hf[mask]
    print(
        f"{tag}: n={mask.sum()}  mean {v.mean():.4f}  "
        f"std {v.std():.4f}  cv {v.std() / max(v.mean(), 1e-6):.3f}  "
        f"hf-std {h.std():.4f}"
    )


def _hf_residual(lum: np.ndarray, air: np.ndarray) -> np.ndarray:
    """
    Luminance minus the air-masked 3^3 local mean (computed among air cells
    only — solids excluded, same support as smooth.comp).  This isolates the
    cell-to-cell ray-fan disagreement (the blotch/confetti noise) from real
    spatial lighting gradients, which survive a local mean.
    """
    a = air.astype(np.float32)
    la = lum * a
    s = np.zeros_like(lum)
    c = np.zeros_like(lum)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                s += np.roll(la, (dz, dy, dx), axis=(0, 1, 2))
                c += np.roll(a, (dz, dy, dx), axis=(0, 1, 2))
    return lum - s / np.maximum(c, 1.0)


def _dilate(mask: np.ndarray, cells: int) -> np.ndarray:
    out = mask.copy()
    for ax in range(3):
        for sh in range(1, cells + 1):
            out |= np.roll(mask, sh, axis=ax) | np.roll(mask, -sh, axis=ax)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=120)
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
        Vec3.RIGHT, math.radians(-20.0)
    ).normalized()
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
    geom = _tex_to_numpy(app, casc.geom)  # [z, y, x, rgba]
    rad = _tex_to_numpy(app, casc.radiance_current)
    lum = _lum(rad)
    # Ray-gathered component only: subtract the own-cell contact term
    # (u_source + u_lit), which is crisp BY DESIGN and must not be counted
    # as noise.  This is exactly the field smooth.comp filters.
    own = _tex_to_numpy(app, casc.source)[..., :3] + _tex_to_numpy(app, casc.lit)[..., :3]
    ray_lum = np.maximum(
        lum - (0.2126 * own[..., 0] + 0.7152 * own[..., 1] + 0.0722 * own[..., 2]), 0.0
    )

    ks = np.arange(n)
    wz = oz + (ks + 0.5) * cell
    wy = oy + (ks + 0.5) * cell
    wx = ox + (ks + 0.5) * cell
    WZ, WY, WX = np.meshgrid(wz, wy, wx, indexing="ij")

    occ = geom[..., 3]
    air = occ < 0.5
    solid = occ > 0.5
    cheby = np.maximum(np.abs(WX - cx), np.abs(WY - cy))
    hf = _hf_residual(ray_lum, air)
    print(f"gi_smooth_passes = {getattr(pipe, '_gi_smooth', 'MISSING')}")
    print("NOTE: hf-std is computed on the RAY component (rad - own term).")

    # (a) Open ground far from the room: thin air band just above ground,
    # at least 10 m from the room centre (outer wall is at 5.5 m).  The
    # terrain there is flat grass — variance is ray-fan noise.
    ground = air & (cheby > 10.0) & (WZ > z0 - 0.5) & (WZ < z0 + 1.0)
    _report("open ground band   ", lum, ground, hf)

    # (b) Air hugging the room's EXTERIOR walls: within 1 m (2 cells) of a
    # solid cell, just outside the outer wall plane (5.5 m), wall-height z.
    near_solid = _dilate(solid, 2)
    wall_band = air & near_solid & (cheby > 5.4) & (cheby < 7.0) & (WZ > z0 + 0.2) & (WZ < z0 + 5.0)
    _report("exterior wall band ", lum, wall_band, hf)

    # (c) Air above the roof (the blotchy roof in the night shot).
    roof_band = air & near_solid & (cheby < 5.4) & (WZ > z0 + 5.4) & (WZ < z0 + 7.0)
    _report("roof band          ", lum, roof_band, hf)

    # Regression guards: interior mean must hold (±15 %), and the doorway
    # spill region (rainbow confetti in the night shot) gets its own line.
    interior = air & (cheby < 3.4) & (WZ > z0 + 0.3) & (WZ < z0 + 4.2)
    _report("interior air (all) ", lum, interior, hf)

    door = air & (cheby > 5.4) & (cheby < 9.0) & (WZ > z0 - 0.5) & (WZ < z0 + 1.5) & ~ground
    _report("doorway apron band ", lum, door, hf)


if __name__ == "__main__":
    main()
