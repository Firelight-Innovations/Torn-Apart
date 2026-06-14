import sys
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))
import numpy as np, main as demo
from panda3d.core import Texture
import os

app = demo.build_demo()
app.input_state.mouse_captured = False
app._set_mouse_capture(False)
app._clock.game_time_of_day = 12.0 * 3600.0
hold = app._clock.game_time_of_day
pipe = app.lighting_pipeline
for _ in range(200):
    app.taskMgr.step()
    app._clock.game_time_of_day = hold


def rb(tex):
    app.graphicsEngine.extract_texture_data(tex, app.win.get_gsg())
    pta = tex.get_ram_image()
    z, y, x = tex.get_z_size(), tex.get_y_size(), tex.get_x_size()
    nbytes = len(pta)
    bpc = nbytes // (z * y * x * 4)  # bytes per channel
    dt = {1: np.uint8, 2: np.float16, 4: np.float32}[bpc]
    arr = np.frombuffer(memoryview(pta), dtype=dt).astype(np.float32).reshape(z, y, x, 4)
    return arr  # page-major (z,y,x), BGRA channel order


for casc in pipe.cascades:
    geom = rb(casc.geom)
    vis = rb(casc.vis)
    om = casc.window.world_origin_m
    cm = casc.cell_m
    n = casc.cells

    # ground surface ~ z=8m. find k for z just below(solid) and just above(air)
    def kz(zw):
        return int((zw - om[2]) / cm)

    # column at window center x,y
    cx = n // 2
    cy = n // 2
    print(f"\ncascade {casc.index} cell_m={cm} origin_z={om[2]:.1f}")
    for zw in [6.0, 7.5, 8.0, 8.5, 9.0, 10.0, 12.0, 16.0]:
        k = kz(zw)
        if not (0 <= k < n):
            continue
        g = geom[k, cy, cx]
        v = vis[k, cy, cx]
        # geom BGRA uint8: occ=A=idx3. vis BGRA float16: sun=R=idx2, sky=B=idx0
        print(f"  z={zw:5.1f} k={k:3d} occ={g[3]:.0f} sunVis(R)={v[2]:.3f} skyVis(B)={v[0]:.3f}")
sys.stdout.flush()
os._exit(0)
